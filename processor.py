from collections import Counter
from threading import Lock

import av
import cv2
import numpy as np
from streamlit_webrtc import VideoProcessorBase
from ultralytics import YOLO


MODEL_NAME = "yolov8s.pt"
MODEL_DISPLAY_NAME = "YOLOv8s"
MODEL = YOLO(MODEL_NAME)
CLASS_COUNT = len(MODEL.names)

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(CLASS_COUNT, 3), dtype=np.uint8)


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


def detect_objects(
    image: np.ndarray,
    confidence_threshold: float,
) -> tuple[np.ndarray, dict[str, int]]:
    annotated_image = image.copy()
    object_counts: Counter[str] = Counter()

    results = MODEL.predict(
        annotated_image,
        conf=confidence_threshold,
        verbose=False,
    )

    for result in results:
        for detected_box in result.boxes:
            x1, y1, x2, y2 = map(int, detected_box.xyxy[0])
            class_id = int(detected_box.cls[0])
            confidence = float(detected_box.conf[0])
            label = MODEL.names[class_id]
            color = tuple(int(channel) for channel in COLORS[class_id % len(COLORS)])

            object_counts[label] += 1
            _draw_detection(
                image=annotated_image,
                box=(x1, y1, x2, y2),
                label=label,
                confidence=confidence,
                color=color,
            )

    sorted_counts = dict(
        sorted(object_counts.items(), key=lambda item: (-item[1], item[0]))
    )
    return annotated_image, sorted_counts


def encode_png(image: np.ndarray) -> bytes | None:
    success, buffer = cv2.imencode(".png", image)
    if not success:
        return None

    return buffer.tobytes()


class ObjectDetectionProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.confidence_threshold = 0.5
        self._lock = Lock()
        self._detection_counts: dict[str, int] = {}
        self._latest_frame: np.ndarray | None = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        annotated_image, detection_counts = detect_objects(
            image=image,
            confidence_threshold=self.confidence_threshold,
        )

        with self._lock:
            self._detection_counts = detection_counts
            self._latest_frame = annotated_image.copy()

        return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

    def get_detection_counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._detection_counts)

    def get_latest_frame(self) -> np.ndarray | None:
        with self._lock:
            if self._latest_frame is None:
                return None

            return self._latest_frame.copy()
