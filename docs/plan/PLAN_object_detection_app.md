# 🔍 Real-Time Object Detection App
**Stack:** Python · YOLOv8 · OpenCV · Streamlit · streamlit-webrtc  
**Target Deploy:** [streamlit.app](https://streamlit.app) (free tier)

---

## 📁 Struktur Project

```
object-detection-app/
├── app.py                  # Main Streamlit app
├── processor.py            # Logic WebRTC + YOLOv8
├── requirements.txt        # Dependencies
├── packages.txt            # Minimal system package untuk libGL
└── README.md
```

---

## 📦 Dependencies

### `requirements.txt`
```
--extra-index-url https://download.pytorch.org/whl/cpu

streamlit==1.58.0
streamlit-webrtc==0.72.2
torch==2.2.2+cpu
torchvision==0.17.2+cpu
ultralytics==8.2.0
opencv-python==4.9.0.80
av==16.1.0
numpy==1.26.4
pillow==10.3.0
```

> ⚠️ `ultralytics` sudah include YOLOv8 — tidak perlu install terpisah.  
> ⚠️ `ultralytics` membutuhkan `opencv-python`, jadi pin versinya eksplisit.
> ⚠️ Tambahkan `--extra-index-url https://download.pytorch.org/whl/cpu` di `requirements.txt` supaya PyTorch yang terinstall adalah build CPU-only.
> ⚠️ Jika perlu ganti versi Python di Streamlit Cloud, atur dari Advanced settings saat deploy/redeploy.

### System packages

`packages.txt` cukup berisi:

```
libgl1
```

Jangan tambahkan `ffmpeg` atau `libglib2.0-0` karena bisa memicu konflik apt di
Streamlit Cloud.

---

## 🎨 Warna Bounding Box per Kategori

YOLOv8 COCO model bisa deteksi **80 objek**. Warna dibedakan otomatis per class ID:

```python
# Contoh kategori utama (COCO dataset)
person       → pink/magenta
car          → yellow/lime
truck        → green
bus          → cyan
traffic light→ blue
handbag      → orange
# ... dan 74 objek lainnya
```

---

## 🧠 Logic Flow

```
Browser Webcam
     │
     ▼
streamlit-webrtc (WebRTC)
     │  (frame per frame via aiortc)
     ▼
VideoProcessorBase (processor.py)
     │  recv(frame) → numpy array
     ▼
YOLOv8.predict(frame)
     │  → boxes, class_ids, confidence scores
     ▼
cv2.rectangle() + cv2.putText()
     │  → bounding box + label + confidence
     ▼
Return frame ke browser
```

---

## 📝 File by File

### `processor.py`
```python
import av
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import VideoProcessorBase

# Load model sekali saat startup (bukan per-frame)
model = YOLO(os.getenv("YOLO_MODEL", "yolov8n.pt"))  # cloud-safe default

# Generate warna unik per class
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(80, 3), dtype=np.uint8)

class ObjectDetectionProcessor(VideoProcessorBase):
    def __init__(self):
        self.confidence_threshold = 0.5

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        results = model.predict(
            img,
            conf=self.confidence_threshold,
            verbose=False
        )

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = model.names[class_id]
                color = COLORS[class_id % len(COLORS)].tolist()

                # Bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                # Label + confidence
                text = f"{label} {confidence:.0%}"
                (tw, th), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
                )
                cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(
                    img, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1
                )

        return av.VideoFrame.from_ndarray(img, format="bgr24")
```

---

### `app.py`
```python
import os

import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
from processor import ObjectDetectionProcessor

def get_rtc_configuration():
    stun_enabled = os.getenv("WEBRTC_USE_STUN", "").lower() in {"1", "true", "yes"}

    if not stun_enabled:
        return None

    stun_url = os.getenv("WEBRTC_STUN_URL", "stun:stun.l.google.com:19302")
    return RTCConfiguration({"iceServers": [{"urls": [stun_url]}]})

st.set_page_config(
    page_title="Object Detection - YOLOv8",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 Real-Time Object Detection")
st.caption("Powered by YOLOv8 + WebRTC")

# Sidebar: confidence threshold
st.sidebar.header("⚙️ Settings")
confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** YOLOv8n default / YOLOv8s optional")
st.sidebar.markdown("**Objects:** 80 categories")
st.sidebar.markdown("""
**Contoh objek:**  
🧍 person · 🚗 car · 🚌 bus · 🚛 truck  
🚦 traffic light · 🎒 backpack · 📱 phone  
🐕 dog · 🐈 cat · 🪑 chair · ...
""")

st.divider()

ctx = webrtc_streamer(
    key="object-detection",
    video_processor_factory=ObjectDetectionProcessor,
    rtc_configuration=get_rtc_configuration(),
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Update confidence threshold secara dinamis
if ctx.video_processor:
    ctx.video_processor.confidence_threshold = confidence

st.divider()
st.markdown(
    "<small>Made with ❤️ using Streamlit · YOLOv8 · OpenCV</small>",
    unsafe_allow_html=True
)
```

---

## 🚀 Cara Run Lokal

```bash
# 1. Buat virtual environment
python3 -m venv venv
source venv/bin/activate     # Mac/Linux
# venv\Scripts\activate      # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

> Pertama kali run, YOLOv8 otomatis download model weights `yolov8n.pt` (~6MB).  
> Jauh lebih kecil dari DeepFace, jadi lebih cepat siap.

---

## ☁️ Deploy ke Streamlit Cloud

1. Push semua file ke **GitHub repo**
2. Buka [streamlit.app](https://streamlit.app) → **New app**
3. Pilih repo → **Main file path:** `app.py`
4. Klik **Deploy**
5. Tunggu build ~5 menit (install ultralytics cukup besar)

---

## 🔄 Pilihan Model YOLO (Trade-off)

| Model | Size | Speed | Accuracy | Cocok untuk |
|---|---|---|---|---|
| `yolov8n.pt` | ~6MB | ⚡⚡⚡ | ⭐⭐ | Cloud / low-end |
| `yolov8s.pt` | ~22MB | ⚡⚡ | ⭐⭐⭐ | Balance |
| `yolov8m.pt` | ~50MB | ⚡ | ⭐⭐⭐⭐ | High-end local |

> App default memakai `yolov8n.pt` untuk Streamlit Cloud. Untuk akurasi lebih tinggi di mesin yang cukup kuat, set `YOLO_MODEL=yolov8s.pt`.

---

## ⚠️ Catatan Penting

| Issue | Solusi |
|---|---|
| Webcam tidak jalan di cloud | Pakai `streamlit-webrtc`, bukan `cv2.VideoCapture` |
| Model lambat load | Load model di luar class (global), bukan di dalam `recv()` |
| Segmentation fault saat load YOLO | Pakai default `yolov8n`; `yolov8s` hanya via `YOLO_MODEL` di resource lebih besar |
| `ImportError: libGL.so.1` di cloud | Isi `packages.txt` hanya dengan `libgl1` |
| Error apt package di cloud | Jangan tambahkan `ffmpeg` atau `libglib2.0-0` |
| Confidence terlalu banyak false positive | Naikkan slider threshold ke 0.6–0.7 |

---

## 🔧 Upgrade Ideas (Opsional)

- [x] Tambah **counter objek** per kategori (sidebar stats)
- [x] Tambah **screenshot** / capture frame
- [x] Mode **upload image** selain webcam
- [ ] Highlight warna khusus untuk objek tertentu (misal: hanya `person`)
- [ ] Filter tampilkan **class tertentu** saja via multiselect
- [x] Ganti model ke **YOLOv8s** untuk akurasi lebih tinggi via `YOLO_MODEL`

---

*Ikuti urutan: buat file → install deps → run lokal → push ke GitHub → deploy ke streamlit.app*
