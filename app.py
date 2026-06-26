import os

import streamlit as st
from streamlit_webrtc import RTCConfiguration, webrtc_streamer

from processor import ObjectDetectionProcessor


def get_rtc_configuration() -> RTCConfiguration | None:
    stun_enabled = os.getenv("WEBRTC_USE_STUN", "").lower() in {"1", "true", "yes"}

    if not stun_enabled:
        return None

    stun_url = os.getenv("WEBRTC_STUN_URL", "stun:stun.l.google.com:19302")
    return RTCConfiguration({"iceServers": [{"urls": [stun_url]}]})


st.set_page_config(
    page_title="Object Detection - YOLOv8",
    page_icon="🔍",
    layout="centered",
)

st.title("🔍 Real-Time Object Detection")
st.caption("Powered by YOLOv8 nano, OpenCV, and WebRTC")

with st.sidebar:
    st.header("⚙️ Settings")
    confidence = st.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=1.00,
        value=0.50,
        step=0.05,
    )

    st.divider()
    st.markdown("**Model:** YOLOv8n (COCO)")
    st.markdown("**Objects:** 80 categories")
    st.markdown(
        """
        **Examples:** person · car · bus · truck · traffic light · backpack ·
        phone · dog · cat · chair
        """
    )

ctx = webrtc_streamer(
    key="object-detection",
    video_processor_factory=ObjectDetectionProcessor,
    rtc_configuration=get_rtc_configuration(),
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if ctx.video_processor:
    ctx.video_processor.confidence_threshold = confidence

st.divider()
st.caption("Made with Streamlit, streamlit-webrtc, YOLOv8, and OpenCV.")
st.markdown(
    "Created by [Akbar Fadilah](https://muhamadakbarfadilah.my.id/) · "
    "Founder & Co-Founder at "
    "[Afda Technology Solutions](https://afdatech.com/)"
)
