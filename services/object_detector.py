"""
Object detection service — wraps a YOLO model (ultralytics) for detecting
objects in baby monitoring scenes.

Features:
  - Auto-detect GPU (CUDA) or fall back to CPU
  - Half-precision (FP16) on GPU for ~2x speedup
  - Built-in BoT-SORT tracking for consistent object IDs across frames
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

from config.settings import ObjectDetectionConfig
from utils.schemas import ObjectDetection

logger = logging.getLogger(__name__)


def _resolve_device(device_str: str) -> str:
    """
    Resolve 'auto' to the best available device.
    Returns 'cuda' if a CUDA GPU is available, otherwise 'cpu'.
    """
    if device_str != "auto":
        return device_str
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info("CUDA GPU detected: %s — using GPU acceleration", gpu_name)
            return "cuda"
        else:
            logger.info("No CUDA GPU found — running on CPU")
            return "cpu"
    except ImportError:
        logger.info("PyTorch not built with CUDA — running on CPU")
        return "cpu"


class ObjectDetector:
    """Wrapper around ultralytics YOLO for object detection with GPU support."""

    def __init__(self, config: Optional[ObjectDetectionConfig] = None) -> None:
        self.config = config or ObjectDetectionConfig()
        self._model = None
        self._device: Optional[str] = None
        self._use_half: bool = False

    def _ensure_model(self):
        """Lazy-load the model on first use."""
        if self._model is not None:
            return

        from ultralytics import YOLO

        model_path = self.config.model_path

        # Pretrained models like "yolov8m.pt" are auto-downloaded by Ultralytics.
        # Only resolve path for custom models (e.g. "best.pt").
        if not model_path.startswith("yolov8") and not os.path.isabs(model_path):
            project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
            model_path = os.path.join(project_root, model_path)
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"YOLO model not found at: {model_path}\n"
                    "Place your model file in the project root, or use "
                    "'yolov8m.pt' for the pretrained COCO model."
                )
        # else: Ultralytics will download it automatically

        # Resolve device
        self._device = _resolve_device(self.config.device)
        self._use_half = self.config.half_precision and self._device.startswith("cuda")

        logger.info("Loading YOLO model from: %s (device=%s, half=%s)",
                     model_path, self._device, self._use_half)
        self._model = YOLO(model_path)
        self._model.to(self._device)

        if self._use_half:
            self._model.half()
            logger.info("FP16 half-precision enabled")

    @property
    def model(self):
        self._ensure_model()
        return self._model

    @property
    def device(self) -> str:
        self._ensure_model()
        return self._device

    def detect(self, frame: np.ndarray) -> list[ObjectDetection]:
        """
        Run object detection on a single BGR frame (no tracking).
        Use for single-image analysis.
        """
        results = self.model(
            frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.imgsz,
            half=self._use_half,
            verbose=False,
        )
        return self._parse_results(results)

    def track(self, frame: np.ndarray) -> list[ObjectDetection]:
        """
        Run object detection + tracking on a BGR frame.
        Returns detections with persistent track_id across frames.
        Use for live / video streams.
        """
        if not self.config.enable_tracking:
            return self.detect(frame)

        results = self.model.track(
            frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.imgsz,
            half=self._use_half,
            persist=True,
            verbose=False,
        )
        return self._parse_results(results, use_track_id=True)

    def _parse_results(self, results, use_track_id: bool = False) -> list[ObjectDetection]:
        """Convert ultralytics results into ObjectDetection list."""
        detections: list[ObjectDetection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = result.names.get(cls_id, f"class_{cls_id}")

                track_id = None
                if use_track_id and box.id is not None:
                    track_id = int(box.id[0])

                detections.append(ObjectDetection(
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    track_id=track_id,
                ))

        return detections

    def close(self) -> None:
        """Release model resources."""
        self._model = None
        self._device = None
