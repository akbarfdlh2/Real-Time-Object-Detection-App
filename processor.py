import av
import cv2
import numpy as np
from streamlit_webrtc import VideoProcessorBase
from ultralytics import YOLO


MODEL = YOLO("yolov8n.pt")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(80, 3), dtype=np.uint8)


def _draw_detection(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    label: str,
    confidence: float,
    color: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    text = f"{label} {confidence:.0%}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )

    label_top = max(y1 - text_height - baseline - 8, 0)
    label_bottom = max(y1, text_height + baseline + 8)
    label_right = min(x1 + text_width + 8, image.shape[1] - 1)

    cv2.rectangle(
        image,
        (x1, label_top),
        (label_right, label_bottom),
        color,
        -1,
    )
    cv2.putText(
        image,
        text,
        (x1 + 4, label_bottom - baseline - 4),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


class ObjectDetectionProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.confidence_threshold = 0.5

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")

        results = MODEL.predict(
            image,
            conf=self.confidence_threshold,
            verbose=False,
        )

        for result in results:
            for detected_box in result.boxes:
                x1, y1, x2, y2 = map(int, detected_box.xyxy[0])
                class_id = int(detected_box.cls[0])
                confidence = float(detected_box.conf[0])
                label = MODEL.names[class_id]
                color = tuple(int(channel) for channel in COLORS[class_id % len(COLORS)])

                _draw_detection(
                    image=image,
                    box=(x1, y1, x2, y2),
                    label=label,
                    confidence=confidence,
                    color=color,
                )

        return av.VideoFrame.from_ndarray(image, format="bgr24")
