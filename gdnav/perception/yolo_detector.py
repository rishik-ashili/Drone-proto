"""YOLO landmark detector running an exported ONNX model.

Works with Ultralytics YOLOv8 / YOLO11 ONNX exports (output shape
[1, 4 + num_classes, N]). Uses onnxruntime when installed (CPU, CUDA or
TensorRT execution providers) and falls back to OpenCV DNN otherwise, so the
same file runs on a laptop, a Raspberry Pi or a Jetson.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from gdnav.perception.detections import Detection


class YoloDetector:
    def __init__(self, model_path: str | Path, class_names: list[str] | None = None,
                 imgsz: int = 320, conf: float = 0.45, iou: float = 0.5):
        self.path = Path(model_path)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.backend = ""
        self.class_names = class_names or self._read_names()
        self.last_ms = 0.0
        try:
            import onnxruntime as ort

            providers = [p for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider",
                                     "CPUExecutionProvider") if p in ort.get_available_providers()]
            so = ort.SessionOptions()
            so.intra_op_num_threads = 2
            self.sess = ort.InferenceSession(str(self.path), so, providers=providers)
            self.input_name = self.sess.get_inputs()[0].name
            shape = self.sess.get_inputs()[0].shape
            if isinstance(shape[2], int):
                self.imgsz = shape[2]
            self.backend = "onnxruntime:" + self.sess.get_providers()[0].replace("ExecutionProvider", "")
            self._names_from_metadata()
        except ImportError:
            self.sess = None
            self.net = cv2.dnn.readNetFromONNX(str(self.path))
            self.backend = "opencv-dnn"

    def _read_names(self) -> list[str]:
        meta = self.path.with_suffix(".json")
        if meta.exists():
            return json.loads(meta.read_text())["names"]
        return []

    def _names_from_metadata(self) -> None:
        if self.class_names:
            return
        m = self.sess.get_modelmeta().custom_metadata_map
        if "names" in m:  # Ultralytics stores "{0: 'person', ...}"
            import ast
            d = ast.literal_eval(m["names"])
            self.class_names = [d[k] for k in sorted(d)]

    def _letterbox(self, img: np.ndarray):
        h, w = img.shape[:2]
        s = self.imgsz / max(h, w)
        nh, nw = int(round(h * s)), int(round(w * s))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        top, left = (self.imgsz - nh) // 2, (self.imgsz - nw) // 2
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = resized
        return canvas, s, left, top

    def detect(self, bgr: np.ndarray) -> list[Detection]:
        t0 = time.perf_counter()
        canvas, s, left, top = self._letterbox(bgr)
        blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, swapRB=True)
        if self.sess is not None:
            out = self.sess.run(None, {self.input_name: blob})[0]
        else:
            self.net.setInput(blob)
            out = self.net.forward()
        pred = out[0]
        if pred.shape[0] > pred.shape[1]:
            pred = pred.T
        boxes, scores = pred[:4].T, pred[4:].T
        cls_ids = scores.argmax(1)
        confs = scores[np.arange(len(scores)), cls_ids]
        keep = confs >= self.conf
        boxes, confs, cls_ids = boxes[keep], confs[keep], cls_ids[keep]
        dets: list[Detection] = []
        if len(boxes):
            xywh = boxes.copy()
            xywh[:, 0] -= xywh[:, 2] / 2
            xywh[:, 1] -= xywh[:, 3] / 2
            idx = cv2.dnn.NMSBoxesBatched(xywh.tolist(), confs.tolist(), cls_ids.tolist(), self.conf, self.iou)
            for i in np.array(idx).ravel():
                x, y, w, h = xywh[i]
                x1, y1 = (x - left) / s, (y - top) / s
                x2, y2 = (x + w - left) / s, (y + h - top) / s
                cid = int(cls_ids[i])
                name = self.class_names[cid] if cid < len(self.class_names) else str(cid)
                dets.append(Detection(name, float(confs[i]), (float(x1), float(y1), float(x2), float(y2)), "yolo"))
        self.last_ms = (time.perf_counter() - t0) * 1000
        return dets
