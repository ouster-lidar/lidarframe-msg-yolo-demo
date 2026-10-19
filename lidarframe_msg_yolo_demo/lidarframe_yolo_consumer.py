#!/usr/bin/env python3
"""
Subscribe to LidarFrame (planar / non-interleaved layout), decode selected
channels (including the FLOAT16 x 3 RGB channel when present) to BGR, run
Ultralytics YOLO, publish annotated images and a (coloured) point cloud.
"""

from __future__ import annotations

import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy

try:
    import cv2
except Exception as exc:  # noqa: BLE001 — catch NumPy/OpenCV ABI mismatch
    sys.stderr.write(
        '\nouster_yolo_consumer: importing cv2 failed. This is usually caused by '
        'NumPy 2.x with an OpenCV wheel built for NumPy 1.x.\n'
        'Fix (same venv as this node):\n'
        '  pip install "numpy>=1.23.2,<2.0" --force-reinstall\n'
        '  pip install --force-reinstall "opencv-python-headless>=4.8,<5"\n'
        f'Original error: {exc!r}\n\n'
    )
    raise

from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
from lidar_msgs.msg import LidarChannel, LidarFrame, LidarInfo

# C++ implementation of the frame -> xyz(+rgb) float32 projection, exposed via pybind11
# from the lidar-conversions package. Imported lazily so the node still loads
# if the binding is missing — the point cloud topic will just stay silent.
try:
    import lidar_conversions_py as _lidar_conversions_py
except Exception as _lc_exc:  # noqa: BLE001
    _lidar_conversions_py = None
    _lc_import_error = _lc_exc
else:
    _lc_import_error = None

# LidarChannel datatype IDs: sensor_msgs/PointField numeric types + FLOAT16.
_POINT_INT8 = 1
_POINT_UINT8 = 2
_POINT_INT16 = 3
_POINT_UINT16 = 4
_POINT_INT32 = 5
_POINT_UINT32 = 6
_POINT_FLOAT32 = 7
_POINT_FLOAT64 = 8
_POINT_FLOAT16 = LidarChannel.FLOAT16

# LidarChannel datatype -> NumPy base dtype character (endianness prepended at use).
_DTYPE_CHAR: Dict[int, str] = {
    _POINT_INT8: 'i1',
    _POINT_UINT8: 'u1',
    _POINT_INT16: 'i2',
    _POINT_UINT16: 'u2',
    _POINT_INT32: 'i4',
    _POINT_UINT32: 'u4',
    _POINT_FLOAT32: 'f4',
    _POINT_FLOAT64: 'f8',
    _POINT_FLOAT16: 'f2',
}


def _channel_dtype(datatype: int, is_bigendian: bool) -> np.dtype:
    if datatype not in _DTYPE_CHAR:
        raise ValueError(f'unsupported channel datatype {datatype}')
    endian = '>' if is_bigendian else '<'
    return np.dtype(endian + _DTYPE_CHAR[datatype])


def _extract_channel(frame: LidarFrame, channel: LidarChannel) -> np.ndarray:
    """Return the channel as an HxWxcount array in its native dtype.

    With the planar layout each channel is a contiguous block of
    height * width * count * sizeof(datatype) bytes at LidarChannel.offset, so
    decoding is a single np.frombuffer + reshape — no per-pixel Python loop.
    """
    h, w = int(frame.height), int(frame.width)
    count = max(int(channel.count), 1)
    offset = int(channel.offset)
    dtype = _channel_dtype(int(channel.datatype), bool(frame.is_bigendian))

    block_elems = h * w * count
    block_bytes = block_elems * dtype.itemsize
    if offset < 0 or offset + block_bytes > len(frame.data):
        raise ValueError(
            f'channel "{channel.name}" extends past data[] '
            f'(offset={offset}, needs {block_bytes} B, have {len(frame.data)} B)'
        )

    arr = np.frombuffer(frame.data, dtype=dtype, count=block_elems, offset=offset)
    return arr.reshape(h, w, count)


def _extract_channel_2d(frame: LidarFrame, channel: LidarChannel) -> np.ndarray:
    """Return an HxW float array for the channel's first sub-element."""
    return _extract_channel(frame, channel)[..., 0].astype(np.float64, copy=False)


def _rgb_to_bgr_visual(
    rgb: np.ndarray, percentile: float = 99.0, gamma: float = 2.2
) -> np.ndarray:
    """Tone-map an HxWx3 linear (HDR) RGB array to uint8 BGR.

    Ouster RGB is unbounded linear float16; scale so the given percentile of
    finite values maps to full scale, clamp, gamma-encode. UINT8 input is
    assumed display-ready and only reordered to BGR.
    """
    if rgb.dtype == np.uint8:
        return np.ascontiguousarray(rgb[..., ::-1])
    lin = rgb.astype(np.float32)
    finite = lin[np.isfinite(lin) & (lin > 0)]
    scale = float(np.percentile(finite, percentile)) if finite.size else 1.0
    if scale <= 0.0:
        scale = 1.0
    lin = np.nan_to_num(lin, nan=0.0, posinf=scale, neginf=0.0)
    norm = np.clip(lin / scale, 0.0, 1.0) ** (1.0 / gamma)
    return np.ascontiguousarray((norm[..., ::-1] * 255.0 + 0.5).astype(np.uint8))


def _to_bgr_visual(
    arr: np.ndarray, mode: str, clip_percentile: Tuple[float, float] = (2.0, 98.0)
) -> np.ndarray:
    """Normalize scalar field to uint8 BGR (3-channel) for YOLO and display."""
    flat = arr[np.isfinite(arr)].astype(np.float64)
    if flat.size == 0:
        return np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
    lo, hi = np.percentile(flat, clip_percentile)
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    gray = (norm * 255.0).astype(np.uint8)
    if mode == 'range':
        cmap = cv2.COLORMAP_VIRIDIS
    elif mode == 'near_ir':
        cmap = cv2.COLORMAP_INFERNO
    else:
        cmap = cv2.COLORMAP_MAGMA
    color = cv2.applyColorMap(gray, cmap)
    return color


def _build_pointcloud2(
    header: Header,
    data: bytes,
    height: int,
    width: int,
    is_dense: bool,
    point_step: int,
) -> PointCloud2:
    """Wrap the xyz(+rgb) float32 buffer returned by the C++ binding as PointCloud2."""
    msg = PointCloud2()
    msg.header = header
    msg.height = height
    msg.width = width
    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    if point_step >= 16:
        # packed 0x00RRGGBB in a float32 slot (PCL / RViz RGB8 convention)
        msg.fields.append(
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1)
        )
    msg.is_bigendian = False
    msg.point_step = point_step
    msg.row_step = msg.point_step * width
    msg.data = data
    msg.is_dense = bool(is_dense)
    return msg


def _bgr_to_ros_image(header: Header, bgr: np.ndarray) -> Image:
    msg = Image()
    msg.header = header
    msg.height, msg.width = bgr.shape[:2]
    msg.encoding = 'bgr8'
    msg.is_bigendian = 0
    msg.step = msg.width * 3
    msg.data = bgr.tobytes()
    return msg


class YoloLidarConsumer(Node):
    """Runs YOLO on RANGE / NEAR_IR / REFLECTIVITY (+ RGB) LidarFrame channels."""

    CHANNEL_SPECS = (
        ('range', 'range'),
        ('near_ir', 'near_ir'),
        ('reflectivity', 'reflectivity'),
        ('rgb', 'rgb'),
    )
    # Channels only some sensors provide; their absence is not an error.
    OPTIONAL_CHANNELS = frozenset({'rgb'})

    def __init__(self) -> None:
        super().__init__('yolo_lidar_consumer')
        self.declare_parameter('model_name', 'yolo11n.pt')
        self.declare_parameter('confidence', 0.25)
        self.declare_parameter('device', '')
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('lidar_info_topic', '/ouster/lidar_info')
        self.declare_parameter('lidar_frame_topic', '/ouster/lidar_frame')
        self.declare_parameter('points_topic', '/ouster_yolo/points')
        self.declare_parameter('points_organized', True)
        self.declare_parameter('range_channel_name', 'range')
        # Colour the point cloud from this channel when the frame has it ('' = never).
        self.declare_parameter('color_channel_name', 'rgb')
        self.declare_parameter('rgb_auto_percentile', 99.0)
        self.declare_parameter('rgb_gamma', 2.2)

        model_name = self.get_parameter('model_name').get_parameter_value().string_value
        self._conf = float(
            self.get_parameter('confidence').get_parameter_value().double_value
        )
        self._device = self.get_parameter('device').get_parameter_value().string_value
        self._imgsz = int(self.get_parameter('imgsz').value)

        self.get_logger().info(f'Loading Ultralytics YOLO model: {model_name}')
        from ultralytics import YOLO

        self._yolo = YOLO(model_name)

        info_topic = self.get_parameter('lidar_info_topic').value
        frame_topic = self.get_parameter('lidar_frame_topic').value

        info_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        frame_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )

        self._latest_info: Optional[LidarInfo] = None
        self._logged_info_sync = False
        self.create_subscription(LidarInfo, info_topic, self._on_info, info_qos)
        self._logged_missing: set = set()
        self.create_subscription(LidarFrame, frame_topic, self._on_frame, frame_qos)

        self._pubs: Dict[str, Publisher] = {}
        for _, suffix in self.CHANNEL_SPECS:
            topic = f'/ouster_yolo/{suffix}_annotated'
            self._pubs[suffix] = self.create_publisher(
                Image,
                topic,
                QoSProfile(
                    depth=5,
                    reliability=ReliabilityPolicy.RELIABLE,
                    history=HistoryPolicy.KEEP_LAST,
                ),
            )

        points_topic = self.get_parameter('points_topic').value
        self._points_organized = bool(
            self.get_parameter('points_organized').value
        )
        self._range_channel_name = str(
            self.get_parameter('range_channel_name').value
        )
        self._color_channel_name = str(
            self.get_parameter('color_channel_name').value
        ).lower()
        self._rgb_auto_percentile = float(self.get_parameter('rgb_auto_percentile').value)
        self._rgb_gamma = float(self.get_parameter('rgb_gamma').value)
        self._points_pub = self.create_publisher(
            PointCloud2,
            points_topic,
            QoSProfile(
                depth=5,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
            ),
        )

        if _lidar_conversions_py is None:
            self.get_logger().warning(
                'lidar_conversions_py binding not importable; '
                f'{points_topic} will not be published. Error: {_lc_import_error!r}'
            )

        self.get_logger().info(
            'YOLO lidar consumer ready (RANGE, NEAR_IR, REFLECTIVITY, RGB if present). '
            f'Publishing annotated images under /ouster_yolo/*_annotated and '
            f'projected points on {points_topic}.'
        )

    def _on_info(self, msg: LidarInfo) -> None:
        self._latest_info = msg

    def _maybe_log_info_sync(self, frame: LidarFrame) -> None:
        info = self._latest_info
        if self._logged_info_sync or info is None:
            return
        self._logged_info_sync = True
        va = len(info.vertical_angles)
        ha = len(info.horizontal_angles)
        self.get_logger().info(
            f'LidarInfo received (vertical_angles={va}, horizontal_angles={ha}); '
            f'frame grid {frame.width}x{frame.height}; channels: '
            + ', '.join(
                f'{c.name}(dt={c.datatype}, count={c.count})' for c in frame.channels
            )
        )

    def _find_channels(self, frame: LidarFrame) -> Dict[str, LidarChannel]:
        by_name: Dict[str, LidarChannel] = {}
        for ch in frame.channels:
            by_name[ch.name.lower()] = ch
        return by_name

    def _publish_pointcloud(
        self, frame: LidarFrame, by_name: Dict[str, LidarChannel]
    ) -> None:
        """Project frame -> PointCloud2 via the C++ binding and publish."""
        if _lidar_conversions_py is None:
            return

        info = self._latest_info
        if info is None:
            self.get_logger().warn(
                'Waiting for LidarInfo before publishing point cloud...',
                throttle_duration_sec=1.0
            )
            return

        # Only request colour when the frame actually carries the channel, so
        # non-RGB sensors still get a plain XYZ cloud.
        color = (
            self._color_channel_name if self._color_channel_name in by_name else ''
        )
        try:
            # TODO[unaal]: Modify _lidar_conversions_py to produce a PointCloud2 message directly.
            data, h, w, is_dense, point_step = (
                _lidar_conversions_py.lidar_frame_to_pointcloud2_data(
                    info, frame,
                    self._range_channel_name,
                    self._points_organized,
                    color_channel_name=color,
                    color_auto_percentile=self._rgb_auto_percentile,
                    color_gamma=self._rgb_gamma,
                )
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(
                f'LidarFrameToPointCloud failed: {exc}',
                throttle_duration_sec=1.0
            )
            return

        self._points_pub.publish(
            _build_pointcloud2(frame.header, data, h, w, is_dense, point_step)
        )

    def _on_frame(self, frame: LidarFrame) -> None:
        if frame.height == 0 or frame.width == 0 or not frame.data:
            return
        self._maybe_log_info_sync(frame)
        by_name = self._find_channels(frame)
        self._publish_pointcloud(frame, by_name)
        header = frame.header

        for logical, suffix in self.CHANNEL_SPECS:
            ch = by_name.get(logical)
            pub = self._pubs[suffix]
            if ch is None:
                if logical not in self.OPTIONAL_CHANNELS:
                    self.get_logger().warning(
                        f'Missing LidarFrame channel "{logical}"; skipping {suffix}',
                        throttle_duration_sec=5.0,
                    )
                elif logical not in self._logged_missing:
                    self._logged_missing.add(logical)
                    self.get_logger().info(
                        f'LidarFrame has no "{logical}" channel; {suffix} disabled'
                    )
                continue
            try:
                if logical == 'rgb':
                    if int(ch.count) != 3:
                        raise ValueError(f'expected count == 3, got {ch.count}')
                    bgr = _rgb_to_bgr_visual(
                        _extract_channel(frame, ch),
                        self._rgb_auto_percentile,
                        self._rgb_gamma,
                    )
                else:
                    arr = _extract_channel_2d(frame, ch)
                    bgr = _to_bgr_visual(arr, logical)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(f'{logical}: decode failed: {exc}')
                continue

            try:
                device = self._device or None
                results = self._yolo.predict(
                    source=bgr,
                    conf=self._conf,
                    imgsz=self._imgsz,
                    verbose=False,
                    device=device,
                )
                annotated = results[0].plot()
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(f'{logical}: YOLO inference failed: {exc}')
                annotated = bgr

            pub.publish(_bgr_to_ros_image(header, annotated))


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = YoloLidarConsumer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
