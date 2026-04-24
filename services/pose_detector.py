"""
Pose detection service — wraps MediaPipe Pose Landmarker (Tasks API).

Compatible with mediapipe >= 0.10.13. Uses the modern Tasks API
(mp_vision.PoseLandmarker) — the legacy solutions API is not used here.

On first run, the model file is downloaded automatically to the
models/ directory next to this file.
"""

from __future__ import annotations

import os
import time
import urllib.request
from typing import Optional

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from config.settings import PoseModelConfig
from utils.schemas import Keypoint, PoseResult, LANDMARK_NAMES


# ── Model download URLs (Google-hosted .task bundles) ──────────────────────

_MODEL_URLS = {
    0: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    1: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    2: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}

_MODEL_FILENAMES = {
    0: "pose_landmarker_lite.task",
    1: "pose_landmarker_full.task",
    2: "pose_landmarker_heavy.task",
}

# Store models next to this file
_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")


def _ensure_model(complexity: int) -> str:
    """Download the .task model file if not already present. Returns local path."""
    os.makedirs(_MODELS_DIR, exist_ok=True)
    filename = _MODEL_FILENAMES.get(complexity, _MODEL_FILENAMES[1])
    local_path = os.path.join(_MODELS_DIR, filename)

    if not os.path.exists(local_path):
        url = _MODEL_URLS.get(complexity, _MODEL_URLS[1])
        print(f"[PoseDetector] Downloading model ({filename}) — this is a one-time download...")
        try:
            urllib.request.urlretrieve(url, local_path)
            print(f"[PoseDetector] Model saved to: {local_path}")
        except Exception as exc:
            if os.path.exists(local_path):
                os.remove(local_path)  # remove partial download
            raise RuntimeError(
                f"Failed to download MediaPipe model from {url}.\n"
                "Check your internet connection and try again."
            ) from exc

    return local_path


# ── Running mode helpers ────────────────────────────────────────────────────

VisionRunningMode = mp_vision.RunningMode


def _make_landmarker(model_path: str, config: PoseModelConfig,
                     mode: VisionRunningMode) -> mp_vision.PoseLandmarker:
    """Create a PoseLandmarker with the given running mode."""
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mode,
        num_poses=config.num_poses,
        min_pose_detection_confidence=config.min_detection_confidence,
        min_pose_presence_confidence=config.min_detection_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


# ── Main class ──────────────────────────────────────────────────────────────

class PoseDetector:
    """
    Wrapper around MediaPipe Pose Landmarker (Tasks API) for pose detection.

    Maintains two internal landmarker instances:
      - _landmarker_image  : IMAGE mode  — used by detect_image()
      - _landmarker_video  : VIDEO mode  — used by detect() (live / video frames)

    VIDEO mode enables temporal tracking and gives smoother results on streams.
    """

    def __init__(self, config: Optional[PoseModelConfig] = None) -> None:
        self.config = config or PoseModelConfig()
        self._model_path: Optional[str] = None
        self._landmarker_image: Optional[mp_vision.PoseLandmarker] = None
        self._landmarker_video: Optional[mp_vision.PoseLandmarker] = None
        self._video_start_time: float = time.monotonic()
        self._last_video_timestamp_ms: int = -1

    # ── Lazy initialisation ─────────────────────────────────────────────────

    def _get_model_path(self) -> str:
        if self._model_path is None:
            self._model_path = _ensure_model(self.config.model_complexity)
        return self._model_path

    @property
    def landmarker_image(self) -> mp_vision.PoseLandmarker:
        if self._landmarker_image is None:
            self._landmarker_image = _make_landmarker(
                self._get_model_path(), self.config, VisionRunningMode.IMAGE
            )
        return self._landmarker_image

    @property
    def landmarker_video(self) -> mp_vision.PoseLandmarker:
        if self._landmarker_video is None:
            self._landmarker_video = _make_landmarker(
                self._get_model_path(), self.config, VisionRunningMode.VIDEO
            )
            # Reset clock only when the landmarker is first created,
            # so timestamp_ms always starts from 0 for this landmarker instance.
            self._video_start_time = time.monotonic()
            self._last_video_timestamp_ms = -1
        return self._landmarker_video

    # ── Public API ──────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[PoseResult]:
        """
        Run pose estimation on a single BGR frame from a video or live stream.
        Uses VIDEO mode internally for temporal landmark tracking.
        """
        h, w = frame.shape[:2]
        rgb_frame = frame[:, :, ::-1].copy()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # *** Access the landmarker FIRST. ***
        # The property resets _video_start_time on its very first call, so we
        # must resolve it before computing timestamp_ms. Without this, the clock
        # resets AFTER the first timestamp is computed and sent to MediaPipe,
        # causing the next frame's timestamp to be smaller — triggering the
        # "Input timestamp must be monotonically increasing" error.
        landmarker = self.landmarker_video

        # VIDEO mode requires strictly monotonically increasing timestamps (ms).
        # Clamp so we never repeat or go backwards even if frames arrive faster
        # than 1 ms apart (common at startup).
        timestamp_ms = int((time.monotonic() - self._video_start_time) * 1000)
        if timestamp_ms <= self._last_video_timestamp_ms:
            timestamp_ms = self._last_video_timestamp_ms + 1
        self._last_video_timestamp_ms = timestamp_ms

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        return self._parse_result(result, w, h)

    def detect_image(self, frame: np.ndarray) -> list[PoseResult]:
        """
        Run pose on a single static image (no temporal tracking).
        Use this for standalone image analysis.
        """
        h, w = frame.shape[:2]
        rgb_frame = frame[:, :, ::-1].copy()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = self.landmarker_image.detect(mp_image)
        return self._parse_result(result, w, h)

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._landmarker_image is not None:
            self._landmarker_image.close()
            self._landmarker_image = None
        if self._landmarker_video is not None:
            self._landmarker_video.close()
            self._landmarker_video = None

    # ── Internal helpers ────────────────────────────────────────────────────

    def _parse_result(self, result, w: int, h: int) -> list[PoseResult]:
        """Convert a PoseLandmarkerResult into a list of PoseResult objects."""
        if not result.pose_landmarks:
            return []

        pose_results: list[PoseResult] = []

        for landmarks in result.pose_landmarks:
            keypoints: list[Keypoint] = []
            xs, ys = [], []
            vis_threshold = self.config.landmark_visibility_threshold

            for idx, lm in enumerate(landmarks):
                px = lm.x * w
                py = lm.y * h
                pz = lm.z * w
                vis = getattr(lm, "visibility", 1.0) or 0.0

                name = LANDMARK_NAMES[idx] if idx < len(LANDMARK_NAMES) else f"lm_{idx}"
                keypoints.append(Keypoint(name=name, x=px, y=py, z=pz, confidence=vis))

                if vis >= vis_threshold:
                    xs.append(px)
                    ys.append(py)

            # Bounding box from visible landmarks
            if xs and ys:
                x1, y1 = min(xs), min(ys)
                x2, y2 = max(xs), max(ys)
            else:
                x1, y1, x2, y2 = 0.0, 0.0, float(w), float(h)

            # Overall confidence: average visibility of key body landmarks
            body_indices = [0, 11, 12, 23, 24]  # nose, shoulders, hips
            body_vis = [
                getattr(landmarks[i], "visibility", 1.0) or 0.0
                for i in body_indices if i < len(landmarks)
            ]
            person_conf = sum(body_vis) / len(body_vis) if body_vis else 0.0

            pose_results.append(
                PoseResult(
                    keypoints=keypoints,
                    person_confidence=person_conf,
                    bbox=(x1, y1, x2, y2),
                    frame_width=w,
                    frame_height=h,
                )
            )

        return pose_results