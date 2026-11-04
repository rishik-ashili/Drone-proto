"""Gazebo camera -> cv_node frame source (gz-transport Python bindings).

The downward camera of the ``x500_flow_cam_down`` model (scripts/export_gazebo.py)
publishes gz.msgs.Image on ``/gdnav/down_camera/image``. This class subscribes to
it and hands the latest frame to scripts/run_px4.py, so perception runs on
Gazebo's rendered images while PX4 SITL flies the drone.

Requires the Gazebo Harmonic Python bindings (Ubuntu):
    sudo apt install python3-gz-transport13 python3-gz-msgs10
"""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np

DEFAULT_TOPIC = "/gdnav/down_camera/image"


def _import_gz():
    for transport, msgs in (("gz.transport13", "gz.msgs10"), ("gz.transport14", "gz.msgs11")):
        try:
            t = __import__(transport, fromlist=["Node"])
            m = __import__(f"{msgs}.image_pb2", fromlist=["Image"])
            return t.Node, m.Image
        except ImportError:
            continue
    raise ImportError("Gazebo Python bindings not found: sudo apt install python3-gz-transport13 python3-gz-msgs10")


class GzCamera:
    """Subscribes to a Gazebo camera topic; calling the object returns the latest BGR frame."""
    def __init__(self, topic: str = DEFAULT_TOPIC, size: tuple[int, int] = (640, 480)):
        Node, Image = _import_gz()
        self.size = size
        self._lock = threading.Lock()
        self._img: np.ndarray | None = None
        self._stamp = 0.0
        self.count = 0
        self.node = Node()
        if not self.node.subscribe(Image, topic, self._on_image):
            raise RuntimeError(f"could not subscribe to {topic}")

    def _on_image(self, msg) -> None:
        # pixel_format_type 3 = RGB_INT8 (R8G8B8), 6 = BGR_INT8, 1 = L_INT8
        buf = np.frombuffer(msg.data, np.uint8)
        if msg.pixel_format_type == 1:
            img = cv2.cvtColor(buf.reshape(msg.height, msg.width), cv2.COLOR_GRAY2BGR)
        else:
            img = buf.reshape(msg.height, msg.width, 3)
            if msg.pixel_format_type != 6:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if (img.shape[1], img.shape[0]) != self.size:
            img = cv2.resize(img, self.size)
        with self._lock:
            self._img = img
            self._stamp = time.monotonic()
            self.count += 1

    def __call__(self) -> np.ndarray | None:
        """Frame-source callable for Px4MavsdkVehicle: latest frame not older than 0.5 s."""
        with self._lock:
            if self._img is None or time.monotonic() - self._stamp > 0.5:
                return None
            return self._img.copy()
