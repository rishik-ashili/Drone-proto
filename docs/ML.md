# ML notes: landmark detector

## What it does

YOLO outputs boxes, classes and confidences, **not** a position (plan section 2). The detector finds
landmarks; `landmark_localizer` converts the box centre into a metric offset using altitude and
camera intrinsics, and the estimator fuses the implied drone position.

Classes: `helipad`, `pad_red`, `pad_blue`, `cross_yellow`, `aruco_tag`. YOLO also learns `aruco_tag`
so it can find tags too blurred to decode. When ArUco decodes the ID, the decoded tag wins.

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1. Synthetic dataset | `scripts/generate_dataset.py --train 1600 --val 300` | `datasets/landmarks/` (YOLO format, exact auto-labels) |
| 2. Transfer learning from COCO yolov8n | `scripts/train_landmark_model.py --epochs 30` | `runs/landmarks/`, `models/landmarks_yolov8n.{onnx,pt,json}` |
| 3. Evaluation on the held-out split | same script | precision, recall, mAP50, mAP50-95 in `models/landmarks_yolov8n.json` |
| 4. Jetson | `scripts/trt_engine_builder.py --fp16` | TensorRT engine + benchmark |

Randomised in every image: floor type (concrete, grass, mat), colour cast, light level (0.35–1.35),
height (1–4.5 m), roll/pitch (±12°), yaw, landmark size and rotation, painted distractors, sensor
noise, motion blur, defocus and JPEG compression. The validation split uses different random seeds
(different scenes), standing in for "a held-out set recorded on a different day".

## Before flying the real drone (plan section 8)

1. Print the landmarks. The sim versions are drawn in `gdnav/sim/world.py:render_landmark`; ArUco
   uses DICT_4X4_50.
2. Record video from the real camera at flight height over the real floor, in varied light, then
   label 1,000–3,000 images per class (CVAT, Label Studio or Roboflow).
3. Merge them with the synthetic set (keep both in `data.yaml`) and retrain. Evaluate on real
   images only.
4. Add every in-flight failure case back to the dataset.
5. After INT8 quantisation on the Jetson, re-measure accuracy, not only speed.

## Inference backends

`gdnav/perception/yolo_detector.py` loads the ONNX model with ONNX Runtime (TensorRT, then CUDA,
then CPU execution providers, whichever are installed) and falls back to OpenCV DNN. The CPU-only
fallback with no model is HSV colour segmentation (`color_detector.py`), which cannot recognise the
helipad.

## Licence

Ultralytics YOLOv8/YOLO11 is **AGPL-3.0**. Shipping it inside a proprietary product requires an
Ultralytics Enterprise licence. Alternatives named in the plan:

- **YOLOX-nano/tiny** (Apache-2.0). Export it to ONNX and set `perception.yolo_model`. The decoder
  in `yolo_detector.py` expects the YOLOv8 output layout `[1, 4 + classes, N]`; YOLOX outputs
  `[1, N, 5 + classes]` with objectness, so it needs a small adapter.
- **NanoDet** (Apache-2.0) or MobileNet-SSD for Pi-class boards.

The pretrained weights used as a starting point (`yolov8n.pt`, COCO) are downloaded from the
Ultralytics GitHub releases by `scripts/download_models.py`.
