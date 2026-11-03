# Landmark detector: model card

`landmarks_yolov8n.onnx` (in use) = **v2**. The previous model is kept in `v1/`.

| | v1 | v2 (current) |
|---|---|---|
| Start | COCO yolov8n | v1 |
| Data | 1,600 synthetic (scripts/generate_dataset.py) | + 200 MIT helipad images (datasets/public/mit_helipad) |
| Epochs | 30 | +12 fine-tune |
| Synthetic held-out: P / R / mAP50 / mAP50-95 | 0.973 / 0.984 / 0.994 / 0.893 | 0.978 / 0.973 / 0.993 / 0.904 |
| External helipad test (20 images, different style): mAP50 / mAP50-95 | 0.995 / 0.834 | 0.995 / 0.913 |

- **Classes:** helipad, pad_red, pad_blue, cross_yellow, aruco_tag. Input 320×320, ONNX opset 12, 3.1 M parameters.
- **Limits:** trained on rendered and synthetic images only. No real photographs yet. Expect a drop on
  real camera frames until real images are added (docs/ML.md).
- **Licences:** Ultralytics (AGPL-3.0) code and base weights; MIT helipad data (MIT licence).
