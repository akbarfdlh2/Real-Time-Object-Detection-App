import os
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import RTCConfiguration, webrtc_streamer

from processor import (
    CLASS_COUNT,
    MODEL_DISPLAY_NAME,
    ObjectDetectionProcessor,
    detect_objects,
    encode_png,
)


APP_URL = "https://real-time-object-detection-app-createdbyakbarfdlh.streamlit.app/"

WEBRTC_TRANSLATIONS = {
    "start": "Mulai",
    "stop": "Berhenti",
    "select_device": "Pilih kamera",
    "media_api_not_available": "Media API tidak tersedia",
    "device_ask_permission": "Izinkan akses kamera dari browser.",
    "device_not_available": "Kamera tidak tersedia",
    "device_access_denied": "Akses kamera ditolak",
}


def get_rtc_configuration() -> RTCConfiguration | None:
    stun_enabled = os.getenv("WEBRTC_USE_STUN", "").lower() in {"1", "true", "yes"}

    if not stun_enabled:
        return None

    stun_url = os.getenv("WEBRTC_STUN_URL", "stun:stun.l.google.com:19302")
    return RTCConfiguration({"iceServers": [{"urls": [stun_url]}]})


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            :root {
                --app-bg: #f4f6f8;
                --surface: #ffffff;
                --surface-muted: #eef2f6;
                --surface-soft: #f8fafc;
                --border: #d7dee8;
                --border-strong: #c5cfdc;
                --text: #111827;
                --muted: #667085;
                --accent: #0e9384;
                --accent-strong: #0b6f66;
                --warm: #c76f1d;
                --camera-bg: #0b111b;
                --shadow: 0 14px 36px rgba(15, 23, 42, 0.07);
            }

            .stApp {
                background: var(--app-bg);
                color: var(--text);
            }

            .block-container,
            [data-testid="stMainBlockContainer"] {
                max-width: 1160px;
                padding: 2rem 2rem 1.25rem;
            }

            h1, h2, h3, p, label {
                letter-spacing: 0;
            }

            div[data-testid="stSidebar"] {
                background: var(--surface);
                border-right: 1px solid var(--border);
            }

            div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
            div[data-testid="stSidebar"] label {
                color: var(--text);
            }

            div[data-testid="stSidebar"] hr {
                margin: 1.25rem 0;
            }

            div[data-testid="stSidebar"] h3 {
                color: var(--text);
                font-size: 0.88rem;
                letter-spacing: 0;
                margin-bottom: 0.45rem;
            }

            div[data-testid="stSidebar"] p {
                color: var(--muted);
                line-height: 1.55;
                margin-bottom: 0.35rem;
            }

            div[data-testid="stSidebar"] .stMarkdown {
                margin-bottom: 0.25rem;
            }

            div[data-testid="stSlider"] > div {
                padding-top: 0.2rem;
            }

            div[data-testid="stSlider"] {
                max-width: 100%;
            }

            div[data-testid="stSlider"] [role="slider"] {
                border: 2px solid var(--surface);
                box-shadow: 0 0 0 1px rgba(14, 147, 132, 0.25);
            }

            .app-header {
                align-items: flex-start;
                border-bottom: 1px solid var(--border);
                display: flex;
                gap: 1rem;
                justify-content: space-between;
                margin-bottom: 1.15rem;
                padding-bottom: 1rem;
            }

            .header-copy {
                max-width: 760px;
            }

            .app-kicker {
                color: var(--accent-strong);
                font-size: 0.78rem;
                font-weight: 800;
                margin: 0 0 0.35rem;
                text-transform: uppercase;
            }

            .hero-title,
            .app-title {
                color: var(--text);
                font-size: clamp(1.9rem, 4vw, 2.7rem);
                font-weight: 800;
                line-height: 1.08;
                margin: 0;
            }

            .hero-copy {
                color: var(--muted);
                font-size: 0.98rem;
                line-height: 1.6;
                margin: 0.65rem 0 0;
                max-width: 680px;
            }

            .header-action {
                align-items: flex-end;
                display: flex;
                min-width: max-content;
                padding-top: 0.25rem;
            }

            .public-link {
                align-items: center;
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 8px;
                color: var(--accent-strong) !important;
                display: inline-flex;
                font-size: 0.9rem;
                font-weight: 750;
                min-height: 2.4rem;
                padding: 0.55rem 0.8rem;
                text-decoration: none;
            }

            .public-link:hover {
                border-color: var(--accent);
                color: var(--accent-strong) !important;
            }

            .meta-row {
                align-items: center;
                color: var(--muted);
                display: flex;
                flex-wrap: wrap;
                font-size: 0.86rem;
                gap: 0.5rem;
                margin-top: 0.85rem;
            }

            .meta-pill {
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 999px;
                display: inline-block;
                padding: 0.36rem 0.62rem;
            }

            .section-label {
                color: var(--text);
                font-size: 0.9rem;
                font-weight: 800;
                margin: 0 0 0.55rem;
            }

            .camera-title {
                color: var(--text);
                font-size: 1rem;
                font-weight: 800;
                margin: 0;
            }

            .camera-copy {
                color: var(--muted);
                font-size: 0.9rem;
                line-height: 1.55;
                margin: 0.25rem 0 0;
            }

            .camera-card-header {
                align-items: flex-start;
                display: flex;
                gap: 1rem;
                justify-content: space-between;
                margin-bottom: 0.85rem;
            }

            .camera-badge {
                background: #eef8f6;
                border: 1px solid #bfe7df;
                border-radius: 999px;
                color: var(--accent-strong);
                font-size: 0.78rem;
                font-weight: 800;
                line-height: 1;
                min-width: max-content;
                padding: 0.45rem 0.62rem;
            }

            .control-panel {
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 8px;
                box-shadow: var(--shadow);
                margin-bottom: 1rem;
                padding: 1rem 1.1rem 0.65rem;
            }

            .control-header {
                align-items: flex-start;
                display: flex;
                gap: 1rem;
                justify-content: space-between;
                margin-bottom: 0.3rem;
            }

            .control-copy,
            .control-value {
                color: var(--muted);
                font-size: 0.92rem;
                line-height: 1.55;
                margin: 0;
                max-width: 720px;
            }

            .control-value {
                background: var(--surface-muted);
                border: 1px solid var(--border);
                border-radius: 999px;
                color: var(--text);
                font-weight: 800;
                min-width: max-content;
                padding: 0.35rem 0.6rem;
            }

            .panel {
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 8px;
                box-shadow: var(--shadow);
                padding: 1rem;
            }

            div[data-testid="stVerticalBlockBorderWrapper"] {
                background: var(--surface);
                border-color: var(--border);
                border-radius: 8px;
                box-shadow: var(--shadow);
                padding: 1rem;
            }

            div[data-testid="stCustomComponentV1"] {
                background: var(--camera-bg);
                border-radius: 8px;
                margin-top: 0.25rem;
                overflow: hidden;
            }

            div[data-testid="stCustomComponentV1"] iframe {
                background: transparent;
                border: 0;
                border-radius: 8px;
                display: block;
                min-height: 0 !important;
                width: 100% !important;
            }

            .stat-grid {
                display: grid;
                gap: 0.7rem;
                grid-template-columns: 1fr;
            }

            .stat {
                background: var(--surface-soft);
                border: 1px solid var(--border);
                border-radius: 8px;
                padding: 0.85rem;
            }

            .stat-label {
                color: var(--muted);
                font-size: 0.78rem;
                font-weight: 800;
                margin-bottom: 0.35rem;
                text-transform: uppercase;
            }

            .stat-value {
                color: var(--text);
                font-size: 1.05rem;
                font-weight: 750;
                line-height: 1.25;
            }

            .stat-value.accent {
                color: var(--accent-strong);
            }

            .counter-list {
                display: grid;
                gap: 0.45rem;
                margin-top: 0.6rem;
            }

            .counter-row {
                align-items: center;
                background: var(--surface-soft);
                border: 1px solid var(--border);
                border-radius: 8px;
                display: flex;
                gap: 0.7rem;
                justify-content: space-between;
                padding: 0.55rem 0.65rem;
            }

            .counter-label {
                color: var(--text);
                font-size: 0.9rem;
                font-weight: 650;
                min-width: 0;
                overflow-wrap: anywhere;
            }

            .counter-value {
                color: var(--accent-strong);
                font-size: 0.92rem;
                font-weight: 800;
                min-width: max-content;
            }

            .tips {
                color: var(--muted);
                font-size: 0.88rem;
                line-height: 1.6;
                margin: 0.85rem 0 0;
            }

            .footer {
                border-top: 1px solid var(--border);
                color: var(--muted);
                font-size: 0.86rem;
                line-height: 1.7;
                margin-top: 1.5rem;
                padding-top: 1rem;
            }

            .footer a {
                color: var(--accent-strong) !important;
                font-weight: 650;
                text-decoration: none;
            }

            iframe {
                border: 0;
                border-radius: 8px;
                max-width: 100%;
                width: 100%;
            }

            div[data-testid="stIFrame"] {
                max-width: 100%;
                overflow: hidden;
            }

            div[data-testid="stIFrame"] iframe {
                display: block;
                min-width: 0;
                width: 100% !important;
            }

            video,
            canvas {
                max-width: 100% !important;
            }

            video {
                background: var(--camera-bg);
                border-radius: 8px;
                display: block;
            }

            @media (max-width: 900px) {
                .block-container,
                [data-testid="stMainBlockContainer"] {
                    padding: 1rem 0.9rem 1.1rem;
                }

                div[data-testid="stHorizontalBlock"] {
                    flex-direction: column;
                    gap: 1rem;
                }

                div[data-testid="stHorizontalBlock"] > div {
                    flex: 1 1 100% !important;
                    min-width: 0 !important;
                    width: 100% !important;
                }

                .app-header {
                    flex-direction: column;
                    gap: 0.75rem;
                    margin-bottom: 1rem;
                    padding-bottom: 0.95rem;
                }

                .header-action {
                    align-items: flex-start;
                    width: 100%;
                }

                .public-link {
                    justify-content: center;
                    width: 100%;
                }

                .hero-copy {
                    font-size: 0.95rem;
                }

                .control-panel {
                    padding: 0.95rem 0.95rem 0.55rem;
                }

                .panel {
                    padding: 0.85rem;
                }

                div[data-testid="stVerticalBlockBorderWrapper"] {
                    padding: 0.95rem;
                }

                .stat-grid {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }
            }

            @media (max-width: 520px) {
                .block-container,
                [data-testid="stMainBlockContainer"] {
                    padding: 0.85rem 0.7rem 1rem;
                }

                .hero-title,
                .app-title {
                    font-size: 1.85rem;
                }

                .hero-copy,
                .camera-copy,
                .control-copy,
                .footer {
                    font-size: 0.9rem;
                }

                .meta-row {
                    align-items: center;
                    font-size: 0.84rem;
                    gap: 0.45rem;
                }

                .control-header,
                .camera-card-header {
                    flex-direction: column;
                    gap: 0.55rem;
                }

                .control-value,
                .camera-badge {
                    align-self: flex-start;
                }

                .stat-grid {
                    gap: 0.65rem;
                    grid-template-columns: 1fr;
                }

                .stat {
                    padding: 0.8rem;
                }

                .stat-value {
                    font-size: 1rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        f"""
        <header class="app-header">
            <div class="header-copy">
                <p class="app-kicker">YOLOv8 realtime detection</p>
                <h1 class="app-title">Real-Time Object Detection</h1>
                <p class="hero-copy">
                    Deteksi objek dari kamera browser atau gambar upload dengan
                    tampilan yang ringan, fokus, dan responsif.
                </p>
                <div class="meta-row">
                    <span class="meta-pill">{MODEL_DISPLAY_NAME}</span>
                    <span class="meta-pill">{CLASS_COUNT} COCO classes</span>
                    <span class="meta-pill">CPU-ready</span>
                </div>
            </div>
            <div class="header-action">
                <a class="public-link" href="{APP_URL}" target="_blank" rel="noreferrer">
                    Public app
                </a>
            </div>
        </header>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(detection_counts: dict[str, int], source_label: str) -> None:
    total_objects = sum(detection_counts.values())

    with st.sidebar:
        st.markdown("### Model")
        st.markdown(MODEL_DISPLAY_NAME)
        st.markdown(f"COCO / {CLASS_COUNT} kelas")

        st.divider()
        st.markdown("### Runtime")
        stun_status = "Enabled" if get_rtc_configuration() else "Local"
        st.markdown(f"WebRTC / {stun_status}")

        st.divider()
        st.markdown("### Object Counter")
        st.markdown(f"Sumber: {source_label}")
        st.metric("Total objek", total_objects)

        if detection_counts:
            rows = "".join(
                f"""
                <div class="counter-row">
                    <span class="counter-label">{label}</span>
                    <span class="counter-value">{count}</span>
                </div>
                """
                for label, count in detection_counts.items()
            )
            st.markdown(
                f'<div class="counter-list">{rows}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown("Belum ada objek terdeteksi.")


@st.fragment(run_every="1s")
def render_live_sidebar(
    processor: ObjectDetectionProcessor | None,
    source_label: str,
) -> None:
    detection_counts: dict[str, int] = {}

    if processor:
        detection_counts = processor.get_detection_counts()
    elif st.session_state.get("captured_frame_counts"):
        detection_counts = st.session_state["captured_frame_counts"]

    render_sidebar(detection_counts=detection_counts, source_label=source_label)


def render_mode_selector() -> str:
    mode_options = ("Webcam", "Upload image")

    st.markdown(
        """
        <div class="control-panel">
            <div class="control-header">
                <div>
                    <p class="section-label">Input mode</p>
                    <p class="control-copy">
                        Pilih sumber gambar untuk deteksi objek.
                    </p>
                </div>
                <div class="control-value">Source</div>
            </div>
        """,
        unsafe_allow_html=True,
    )

    segmented_control = getattr(st, "segmented_control", None)
    if segmented_control:
        selected_mode = segmented_control(
            "Input mode",
            options=mode_options,
            default=mode_options[0],
            label_visibility="collapsed",
        )
    else:
        selected_mode = st.radio(
            "Input mode",
            options=mode_options,
            horizontal=True,
            label_visibility="collapsed",
        )

    st.markdown("</div>", unsafe_allow_html=True)
    return selected_mode or mode_options[0]


def render_controls() -> tuple[float, str]:
    default_confidence = 0.50
    input_mode = render_mode_selector()

    st.markdown(
        f"""
        <div class="control-panel">
            <div class="control-header">
                <div>
                    <p class="section-label">Confidence</p>
                    <p class="control-copy">
                        Ambang deteksi untuk menyeimbangkan sensitivitas dan
                        akurasi prediksi.
                    </p>
                </div>
                <div class="control-value">Threshold</div>
            </div>
        """,
        unsafe_allow_html=True,
    )
    confidence = st.slider(
        "Ambang confidence",
        min_value=0.10,
        max_value=1.00,
        value=default_confidence,
        step=0.05,
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    return confidence, input_mode


def render_camera_header() -> None:
    st.markdown(
        """
        <div class="camera-card-header">
            <div>
                <p class="camera-title">Camera Preview</p>
                <p class="camera-copy">
                    Area kamera akan menampilkan bounding box dan confidence
                    secara realtime setelah stream berjalan.
                </p>
            </div>
            <span class="camera-badge">WebRTC</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_upload_header() -> None:
    st.markdown(
        """
        <div class="camera-card-header">
            <div>
                <p class="camera-title">Upload Image</p>
                <p class="camera-copy">
                    Gambar upload akan diproses dengan bounding box, confidence,
                    dan counter objek yang sama.
                </p>
            </div>
            <span class="camera-badge">Image</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_summary(
    confidence: float,
    status: str,
    source_label: str,
    detection_counts: dict[str, int],
) -> None:
    confidence_label = f"{confidence:.0%}"
    total_objects = sum(detection_counts.values())

    st.markdown(
        f"""
        <div class="panel">
            <p class="section-label">Session</p>
            <div class="stat-grid">
                <div class="stat">
                    <div class="stat-label">Status</div>
                    <div class="stat-value accent">{status}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Confidence</div>
                    <div class="stat-value">{confidence_label}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Model</div>
                    <div class="stat-value">{MODEL_DISPLAY_NAME}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Objek</div>
                    <div class="stat-value">{total_objects}</div>
                </div>
            </div>
            <p class="tips">
                Source aktif: {source_label}.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _download_name(prefix: str, original_name: str | None = None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    if not original_name:
        return f"{prefix}-{timestamp}.png"

    stem = original_name.rsplit(".", maxsplit=1)[0]
    safe_stem = "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in stem
    ).strip("-")

    if not safe_stem:
        safe_stem = "image"

    return f"{prefix}-{safe_stem}-{timestamp}.png"


def render_capture_controls(processor: ObjectDetectionProcessor | None) -> None:
    st.divider()
    action_column, download_column = st.columns(2, gap="small")

    if action_column.button(
        "Capture frame",
        disabled=processor is None,
        use_container_width=True,
    ):
        latest_frame = processor.get_latest_frame() if processor else None

        if latest_frame is None:
            st.warning("Belum ada frame yang bisa dicapture.")
        else:
            screenshot_png = encode_png(latest_frame)

            if screenshot_png is None:
                st.error("Screenshot gagal dibuat.")
            else:
                st.session_state["captured_frame_png"] = screenshot_png
                st.session_state["captured_frame_name"] = _download_name(
                    "webcam-capture"
                )
                st.session_state["captured_frame_counts"] = (
                    processor.get_detection_counts() if processor else {}
                )
                st.success("Frame berhasil dicapture.")

    screenshot_png = st.session_state.get("captured_frame_png")
    screenshot_name = st.session_state.get(
        "captured_frame_name",
        _download_name("webcam-capture"),
    )

    if screenshot_png:
        download_column.download_button(
            "Download screenshot",
            data=screenshot_png,
            file_name=screenshot_name,
            mime="image/png",
            use_container_width=True,
        )

        with st.expander("Preview screenshot", expanded=False):
            st.image(screenshot_png, use_container_width=True)


def render_webcam_mode(
    confidence: float,
) -> tuple[dict[str, int], str, ObjectDetectionProcessor | None]:
    detection_counts: dict[str, int] = {}
    processor: ObjectDetectionProcessor | None = None
    status = "Ready"

    with st.container(border=True):
        render_camera_header()
        ctx = webrtc_streamer(
            key="object-detection",
            video_processor_factory=ObjectDetectionProcessor,
            rtc_configuration=get_rtc_configuration(),
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 1280},
                    "height": {"ideal": 720},
                    "frameRate": {"ideal": 24, "max": 30},
                },
                "audio": False,
            },
            async_processing=True,
            video_html_attrs={
                "autoPlay": True,
                "controls": False,
                "playsInline": True,
                "style": {
                    "width": "100%",
                    "height": "auto",
                    "borderRadius": "8px",
                    "objectFit": "cover",
                },
            },
            translations=WEBRTC_TRANSLATIONS,
        )

        processor = ctx.video_processor
        if processor:
            processor.confidence_threshold = confidence
            detection_counts = processor.get_detection_counts()
        elif st.session_state.get("captured_frame_counts"):
            detection_counts = st.session_state["captured_frame_counts"]

        status = "Live" if ctx.state.playing else "Ready"
        render_capture_controls(processor)

    return detection_counts, status, processor


def render_upload_mode(confidence: float) -> tuple[dict[str, int], str]:
    detection_counts: dict[str, int] = {}
    status = "Ready"

    with st.container(border=True):
        render_upload_header()
        uploaded_file = st.file_uploader(
            "Upload image",
            type=("jpg", "jpeg", "png", "webp"),
            label_visibility="collapsed",
        )

        if uploaded_file is None:
            st.info("Upload file JPG, PNG, atau WEBP untuk mulai deteksi.")
            return detection_counts, status

        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception:
            st.error("File gambar tidak bisa dibaca.")
            return detection_counts, status

        rgb_image = np.array(image)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        annotated_bgr, detection_counts = detect_objects(
            image=bgr_image,
            confidence_threshold=confidence,
        )
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        result_png = encode_png(annotated_bgr)

        st.image(
            annotated_rgb,
            caption=uploaded_file.name,
            use_container_width=True,
        )

        if result_png:
            st.download_button(
                "Download hasil deteksi",
                data=result_png,
                file_name=_download_name("upload-detection", uploaded_file.name),
                mime="image/png",
                use_container_width=True,
            )

        status = "Processed"

    return detection_counts, status


def render_footer() -> None:
    st.markdown(
        """
        <div class="footer">
            Dibuat dengan Streamlit, streamlit-webrtc, YOLOv8, OpenCV, dan PyTorch.
            Created by
            <a href="https://muhamadakbarfadilah.my.id/" target="_blank" rel="noreferrer">Akbar Fadilah</a>
            for
            <a href="https://afdatech.com/" target="_blank" rel="noreferrer">Afda Technology Solutions</a>.
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Object Detection - YOLOv8",
        layout="wide",
    )
    inject_styles()

    render_header()
    confidence, input_mode = render_controls()

    camera_column, summary_column = st.columns([1.65, 0.85], gap="large")
    source_label = "Webcam" if input_mode == "Webcam" else "Upload image"
    live_processor: ObjectDetectionProcessor | None = None

    with camera_column:
        if input_mode == "Webcam":
            detection_counts, status, live_processor = render_webcam_mode(confidence)
        else:
            detection_counts, status = render_upload_mode(confidence)

    with summary_column:
        render_summary(
            confidence=confidence,
            status=status,
            source_label=source_label,
            detection_counts=detection_counts,
        )

    if input_mode == "Webcam":
        render_live_sidebar(processor=live_processor, source_label=source_label)
    else:
        render_sidebar(detection_counts=detection_counts, source_label=source_label)

    render_footer()


if __name__ == "__main__":
    main()
