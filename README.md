# Face Detection & Face Mesh

Real-time face detection with confidence scores and dense face-mesh landmark tracking, built on MediaPipe and OpenCV and packaged as reusable detector classes.

![Python](https://img.shields.io/badge/python-3.11-blue)
![MediaPipe](https://img.shields.io/badge/mediapipe-0.10.8-orange)
![OpenCV](https://img.shields.io/badge/opencv-4.11-green)
![License](https://img.shields.io/badge/license-MIT-blue)

## What's inside

| Script | What it does |
| --- | --- |
| [`face_detection_module.py`](face_detection_module.py) | `FaceDetection` class: bounding boxes, confidence percentage and optional key points, plus an FPS counter |
| [`face_mesh_module.py`](face_mesh_module.py) | `FaceMeshDetector` class: draws the full MediaPipe face mesh and returns per-face landmark pixel coordinates |
| [`face_detection_basics.py`](face_detection_basics.py) | Minimal face detection loop (75% confidence threshold) |
| [`face_mesh_basics.py`](face_mesh_basics.py) | Minimal face mesh loop |

The two modules are meant to be imported by other projects; each also has a `main()` demo.

## Quick start

```bash
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python face_detection_module.py   # runs on videos/7.mp4; press q to quit
python face_mesh_module.py        # uses the webcam
```

To switch a script between the bundled clips and the webcam, change its `cv2.VideoCapture(...)` line (`0` = default webcam).

## Using the modules

```python
import cv2
from face_detection_module import FaceDetection
from face_mesh_module import FaceMeshDetector

img = cv2.imread("photo.jpg")
img, boxes = FaceDetection(detection_confidence=0.5).findFaces(img)   # [[id, (x, y, w, h), score], ...]
img, faces = FaceMeshDetector(max_num_faces=3).findFaceMesh(img)      # one landmark list per face
```

## Project structure

```
face_detection_module.py   reusable face detector
face_mesh_module.py        reusable face-mesh detector
face_detection_basics.py   minimal examples
face_mesh_basics.py
videos/                    sample clips used by the demos
```

## Status

All four scripts run end to end in a headless harness on the bundled clips and a still frame.

## License

[MIT](LICENSE)
