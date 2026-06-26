# Real-Time Object Detection App

Streamlit app for real-time object detection from a browser webcam using
YOLOv8 nano, OpenCV, and streamlit-webrtc.

## Features

- Browser webcam streaming with WebRTC
- YOLOv8 COCO detection for 80 object classes
- Bounding boxes with labels and confidence scores
- Adjustable confidence threshold from the sidebar
- Streamlit Cloud deployment support via `packages.txt`
- CPU-only PyTorch pins to avoid downloading CUDA packages
- Optional STUN configuration for cloud deployments

## Project Structure

```text
.
├── app.py
├── processor.py
├── requirements.txt
├── packages.txt
└── docs/plan/PLAN_object_detection_app.md
```

## Run Locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On Windows, activate the virtual environment with:

```powershell
venv\Scripts\activate
```

The first run downloads `yolov8n.pt` automatically.

For local runs, STUN is disabled by default to avoid noisy UDP retry errors from
the WebRTC ICE layer. For cloud deployment, enable it with:

```bash
WEBRTC_USE_STUN=true streamlit run app.py
```

## Deploy to Streamlit Cloud

1. Push this project to GitHub.
2. Open https://streamlit.app and create a new app.
3. Select the repository.
4. Set the main file path to `app.py`.
5. Deploy.

Use the nano model (`yolov8n.pt`) for the best fit on free-tier cloud resources.
