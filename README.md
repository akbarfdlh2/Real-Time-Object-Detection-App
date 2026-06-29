# Real-Time Object Detection App

Real-time object detection from a browser webcam using Streamlit, WebRTC,
OpenCV, and YOLOv8 small.

## Live Demo

Open the public app:

https://real-time-object-detection-app-createdbyakbarfdlh.streamlit.app/

## Tech Stack

- Python
- Streamlit
- streamlit-webrtc
- YOLOv8 small from Ultralytics
- OpenCV
- PyTorch CPU

## Features

- Real-time webcam detection in the browser
- Upload image mode for static image detection
- COCO object detection across 80 classes
- Bounding boxes with class labels and confidence scores
- Per-category object counter in the sidebar
- Webcam frame capture with PNG download
- Adjustable confidence threshold from the sidebar
- CPU-only PyTorch setup to avoid large CUDA downloads
- Streamlit Cloud-ready system packages via `packages.txt`
- Optional STUN configuration for cloud WebRTC connections

## Project Structure

```text
.
├── app.py
├── processor.py
├── requirements.txt
├── packages.txt
├── README.md
└── docs/
    └── plan/
        └── PLAN_object_detection_app.md
```

## Run Locally

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the app:

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

On Windows, activate the virtual environment with:

```powershell
.venv\Scripts\activate
```

The first run downloads `yolov8s.pt` automatically.

## WebRTC Notes

For local development, STUN is disabled by default to avoid noisy UDP retry
errors from the WebRTC ICE layer.

For cloud deployment, enable STUN:

```bash
WEBRTC_USE_STUN=true streamlit run app.py
```

You can also override the STUN server:

```bash
WEBRTC_USE_STUN=true WEBRTC_STUN_URL=stun:stun.l.google.com:19302 streamlit run app.py
```

## Deploy to Streamlit Cloud

1. Push this project to GitHub.
2. Open https://streamlit.app.
3. Create a new app from the repository.
4. Set the main file path to `app.py`.
5. Deploy.

The app uses `yolov8s.pt` for better accuracy. If memory or startup time is too
tight on free-tier cloud resources, switch `MODEL_NAME` in `processor.py` back
to `yolov8n.pt`.

## Author

Created by [Akbar Fadilah](https://muhamadakbarfadilah.my.id/) for
[Afda Technology Solutions](https://afdatech.com/).
