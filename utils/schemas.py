"""
Data schemas shared across the pose detection pipeline.

These dataclasses define the contract between:
  - PoseDetector  (produces PoseResult)
  - PoseAnalyzer  (consumes PoseResult, produces RiskAssessment)
  - Any future combined pipeline (consumes both)

MediaPipe Pose produces 33 landmarks (BlazePose topology).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional


# ── MediaPipe Pose 33-landmark names ───────────────────────────────────────
LANDMARK_NAMES: list[str] = [
    "nose",                     # 0
    "left_eye_inner",           # 1
    "left_eye",                 # 2
    "left_eye_outer",           # 3
    "right_eye_inner",          # 4
    "right_eye",                # 5
    "right_eye_outer",          # 6
    "left_ear",                 # 7
    "right_ear",                # 8
    "mouth_left",               # 9
    "mouth_right",              # 10
    "left_shoulder",            # 11
    "right_shoulder",           # 12
    "left_elbow",               # 13
    "right_elbow",              # 14
    "left_wrist",               # 15
    "right_wrist",              # 16
    "left_pinky",               # 17
    "right_pinky",              # 18
    "left_index",               # 19
    "right_index",              # 20
    "left_thumb",               # 21
    "right_thumb",              # 22
    "left_hip",                 # 23
    "right_hip",                # 24
    "left_knee",                # 25
    "right_knee",               # 26
    "left_ankle",               # 27
    "right_ankle",              # 28
    "left_heel",                # 29
    "right_heel",               # 30
    "left_foot_index",          # 31
    "right_foot_index",         # 32
]

LANDMARK_INDEX: dict[str, int] = {name: i for i, name in enumerate(LANDMARK_NAMES)}


@dataclass
class Keypoint:
    """A single detected landmark."""

    name: str
    x: float          # pixel x-coordinate
    y: float          # pixel y-coordinate
    z: float          # relative depth (MediaPipe specific)
    confidence: float  # visibility score [0, 1]

    @property
    def visible(self) -> bool:
        return self.confidence > 0.0


@dataclass
class PoseResult:
    """
    Output of the pose detector for one detected person in a frame.

    This is the interface between detection and analysis — keep it stable
    so that swapping the underlying pose model doesn't break downstream code.
    """

    keypoints: list[Keypoint]
    person_confidence: float  # overall detection confidence
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 (computed from landmarks)

    frame_width: int = 0
    frame_height: int = 0

    def keypoint_by_name(self, name: str) -> Optional[Keypoint]:
        idx = LANDMARK_INDEX.get(name)
        if idx is not None and idx < len(self.keypoints):
            return self.keypoints[idx]
        return None

    def visible_count(self, min_conf: float = 0.0) -> int:
        return sum(1 for kp in self.keypoints if kp.confidence > min_conf)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RiskAssessment:
    """
    Output of the pose analyzer — a risk verdict with explanations.
    """

    label: str   # "safe", "dangerous", "uncertain"
    score: float  # aggregated risk score in [0, 1]
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ObjectDetection:
    """A single object detection result from the YOLO model."""

    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    track_id: Optional[int] = None  # persistent ID from YOLO tracker (BoT-SORT)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FrameAnalysis:
    """
    Complete analysis for one frame, combining pose + risk + object detections.
    """

    frame_index: int
    timestamp_ms: float
    persons: list[dict] = field(default_factory=list)
    detections: list[ObjectDetection] = field(default_factory=list)
    annotated_frame_path: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)
