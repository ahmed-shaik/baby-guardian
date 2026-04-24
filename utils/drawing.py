"""
Drawing utilities for annotating frames with pose, risk, and object detection info.

All draw_* functions operate IN-PLACE on the frame (no copies) for performance.
The caller is responsible for copying the frame first if needed.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from config.settings import CombinedAnalyzerConfig
from utils.schemas import PoseResult, RiskAssessment, ObjectDetection

# MediaPipe Pose skeleton connections (pairs of landmark indices)
SKELETON = [
    # Head
    (0, 1), (1, 2), (2, 3), (3, 7),    # nose → left ear
    (0, 4), (4, 5), (5, 6), (6, 8),    # nose → right ear
    (9, 10),                             # mouth
    # Torso
    (11, 12),                            # shoulders
    (11, 23), (12, 24),                  # shoulders → hips
    (23, 24),                            # hips
    # Left arm
    (11, 13), (13, 15),                  # shoulder → elbow → wrist
    (15, 17), (15, 19), (15, 21),        # wrist → fingers
    # Right arm
    (12, 14), (14, 16),
    (16, 18), (16, 20), (16, 22),
    # Left leg
    (23, 25), (25, 27),                  # hip → knee → ankle
    (27, 29), (27, 31),                  # ankle → heel/foot
    # Right leg
    (24, 26), (26, 28),
    (28, 30), (28, 32),
]

RISK_COLORS = {
    "safe": (0, 200, 0),        # green
    "uncertain": (0, 200, 255), # orange
    "dangerous": (0, 0, 255),   # red
}

# Detection colors (cycle for multiple classes)
_DETECTION_COLORS = [
    (255, 165, 0),   # orange
    (255, 255, 0),   # yellow
    (0, 255, 255),   # cyan
    (255, 0, 255),   # magenta
    (128, 0, 255),   # purple
]


def draw_pose_annotation(
    frame: np.ndarray,
    pose: PoseResult,
    risk: RiskAssessment,
    kp_conf_threshold: float = 0.5,
) -> None:
    """
    Draw landmarks, skeleton, bounding box, and risk label IN-PLACE.
    Does NOT copy the frame — draws directly on the provided array.
    """
    color = RISK_COLORS.get(risk.label, (200, 200, 200))

    # Bounding box
    x1, y1, x2, y2 = [int(v) for v in pose.bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Risk label
    label_text = f"{risk.label.upper()} ({risk.score:.2f})"
    cv2.putText(
        frame, label_text, (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
    )

    # Landmark dots
    kp_coords: list[tuple[int, int] | None] = []
    for kp in pose.keypoints:
        if kp.confidence >= kp_conf_threshold:
            cx, cy = int(kp.x), int(kp.y)
            kp_coords.append((cx, cy))
            cv2.circle(frame, (cx, cy), 3, color, -1)
        else:
            kp_coords.append(None)

    # Skeleton lines
    for i, j in SKELETON:
        if i < len(kp_coords) and j < len(kp_coords):
            pt1, pt2 = kp_coords[i], kp_coords[j]
            if pt1 is not None and pt2 is not None:
                cv2.line(frame, pt1, pt2, color, 2)

    # Risk reasons (small text below bbox)
    for idx, reason in enumerate(risk.reasons[:3]):
        y_offset = y2 + 18 + idx * 16
        if y_offset < frame.shape[0] - 10:
            cv2.putText(
                frame, reason[:90], (x1, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1,
            )


def _classify_detection_risk(
    class_name: str, cfg: Optional[CombinedAnalyzerConfig],
) -> str:
    """Return 'danger', 'hazard', or 'safe' for a detected object class."""
    if cfg is None:
        return "safe"
    name = class_name.lower()
    if name in {c.lower() for c in cfg.face_danger_classes}:
        return "danger"
    if name in {c.lower() for c in cfg.hazard_classes}:
        return "hazard"
    return "safe"


# Risk-based colors for object detection boxes
_RISK_DETECTION_COLORS = {
    "danger": (0, 0, 255),      # red
    "hazard": (0, 165, 255),    # orange
    "safe":   (0, 200, 0),      # green
}


def draw_object_detections(
    frame: np.ndarray,
    detections: list[ObjectDetection],
    combined_cfg: Optional[CombinedAnalyzerConfig] = None,
) -> None:
    """
    Draw YOLO object detection bounding boxes and labels IN-PLACE.

    Objects are color-coded by risk level:
      - Red + "DANGER" tag for suffocation-risk objects (blanket, pillow, teddy bear, etc.)
      - Orange + "HAZARD" tag for sharp/chokable objects (knife, scissors, etc.)
      - Green for safe/neutral objects (person, chair, etc.)
    """
    for det in detections:
        risk = _classify_detection_risk(det.class_name, combined_cfg)
        color = _RISK_DETECTION_COLORS.get(risk, _DETECTION_COLORS[det.class_id % len(_DETECTION_COLORS)])
        thickness = 3 if risk in ("danger", "hazard") else 2

        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Label with risk tag + optional track ID
        risk_tag = f" [{risk.upper()}]" if risk != "safe" else ""
        label = f"{det.class_name} {det.confidence:.2f}{risk_tag}"
        if det.track_id is not None:
            label = f"[{det.track_id}] {label}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1,
        )
