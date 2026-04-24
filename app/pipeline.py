"""
Combined analysis pipeline: YOLO object detection + MediaPipe pose estimation.

Orchestrates:
  input source → ObjectDetector + PoseDetector → PoseAnalyzer + CombinedAnalyzer
              → TemporalSmoother → AlertManager → annotated output.

Performance optimizations:
  - Frame skipping: YOLO runs every N frames, cached results reused in between.
  - Async inference: YOLO runs in a background thread, never blocking the main loop.
  - Reduced input size: YOLO processes at 320px by default.
  - GPU auto-detect: uses CUDA if available, falls back to CPU.
  - In-place drawing: single frame copy per iteration.
"""

from __future__ import annotations

import logging
import os
import time
import threading
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from config.settings import Settings
from services.pose_detector import PoseDetector
from services.pose_analyzer import PoseAnalyzer
from services.object_detector import ObjectDetector
from services.combined_analyzer import CombinedAnalyzer
from services.temporal_smoother import TemporalSmoother
from services.alert_manager import AlertManager
from utils.schemas import FrameAnalysis, PoseResult, RiskAssessment, ObjectDetection
from utils.drawing import draw_pose_annotation, draw_object_detections

logger = logging.getLogger(__name__)


# ── Frame grabber (eliminates IP cam buffer lag) ───────────────────────────

class _FrameGrabber:
    """
    Reads frames in a dedicated thread so the main loop always gets the
    LATEST frame instead of stale buffered ones. Essential for IP cameras
    where OpenCV buffers 5-30 frames, causing massive lag.
    """

    def __init__(self, cap: cv2.VideoCapture):
        self._cap = cap
        self._lock = threading.Lock()
        self._frame = None
        self._ret = False
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            ret, frame = self._cap.read()
            with self._lock:
                self._ret = ret
                self._frame = frame

    def read(self):
        with self._lock:
            return self._ret, self._frame.copy() if self._frame is not None else None

    def stop(self):
        self._running = False
        self._thread.join(timeout=3.0)


# ── Async YOLO wrapper ──────────────────────────────────────────────────────

class _AsyncObjectDetector:
    """
    Runs YOLO in a background thread with frame skipping.
    The main loop is never blocked — it always gets the latest cached results.
    """

    def __init__(self, detector: ObjectDetector, skip_frames: int = 3,
                 use_tracking: bool = True) -> None:
        self._detector = detector
        self._skip_frames = max(1, skip_frames)
        self._use_tracking = use_tracking
        self._frame_counter = 0

        self._lock = threading.Lock()
        self._cached: list[ObjectDetection] = []
        self._busy = False
        self._thread: Optional[threading.Thread] = None

    def get_detections(self, frame: np.ndarray) -> list[ObjectDetection]:
        """Return detections — dispatches YOLO every N frames, returns cache otherwise."""
        self._frame_counter += 1

        if self._frame_counter % self._skip_frames == 0:
            with self._lock:
                if not self._busy:
                    self._busy = True
                    frame_copy = frame.copy()
                    self._thread = threading.Thread(
                        target=self._run, args=(frame_copy,), daemon=True
                    )
                    self._thread.start()

        with self._lock:
            return list(self._cached)

    def get_detections_sync(self, frame: np.ndarray) -> list[ObjectDetection]:
        """Synchronous detection (single-image mode)."""
        return self._detector.detect(frame)

    def _run(self, frame: np.ndarray) -> None:
        try:
            if self._use_tracking:
                detections = self._detector.track(frame)
            else:
                detections = self._detector.detect(frame)
            with self._lock:
                self._cached = detections
        finally:
            with self._lock:
                self._busy = False

    def close(self) -> None:
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._detector.close()


# ── Main pipeline ────────────────────────────────────────────────────────────

class PosePipeline:
    """End-to-end pose + object detection analysis pipeline."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings()
        self.detector = PoseDetector(self.settings.pose_model)
        self.analyzer = PoseAnalyzer(self.settings.risk)
        self.combined_analyzer = CombinedAnalyzer(self.settings.combined)
        self.smoother = TemporalSmoother(self.settings.risk)
        self.alert_mgr = AlertManager(self.settings.alert, self.settings.whatsapp)

        # Object detector — wrapped for async + skip + tracking
        self._async_obj: Optional[_AsyncObjectDetector] = None
        self.obj_detector: Optional[ObjectDetector] = None
        if self.settings.object_detection.enabled:
            self.obj_detector = ObjectDetector(self.settings.object_detection)
            od_cfg = self.settings.object_detection
            if od_cfg.async_inference:
                self._async_obj = _AsyncObjectDetector(
                    self.obj_detector,
                    skip_frames=od_cfg.skip_frames,
                    use_tracking=od_cfg.enable_tracking,
                )

        os.makedirs(self.settings.output_dir, exist_ok=True)

    # ── Image ───────────────────────────────────────────────────────────────

    def analyze_image(self, image_path: str, save_annotated: bool = True) -> FrameAnalysis:
        """Run full analysis on a single image."""
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        poses = self.detector.detect_image(frame)
        detections = self._run_object_detection_sync(frame)
        analysis, pairs = self._build_analysis(
            poses, frame_index=0, timestamp_ms=0.0,
            apply_smoothing=False, detections=detections,
        )

        if save_annotated:
            annotated = self._annotate(frame, pairs, detections)
            out_path = os.path.join(self.settings.output_dir, f"annotated_{Path(image_path).name}")
            cv2.imwrite(out_path, annotated)
            analysis.annotated_frame_path = out_path

        return analysis

    # ── Video ───────────────────────────────────────────────────────────────

    def analyze_video(
        self,
        video_path: str,
        save_annotated: bool = True,
        max_frames: int = 0,
    ) -> list[FrameAnalysis]:
        """Run analysis on every frame of a video file."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        results = self._process_video_capture(
            cap,
            save_annotated=save_annotated,
            output_name=f"annotated_{Path(video_path).name}",
            max_frames=max_frames,
        )
        cap.release()
        return results

    # ── Webcam / IP Camera / RTSP Stream ────────────────────────────────────

    def run_live(self, source: Union[int, str] = 0) -> None:
        """
        Run real-time pose analysis on a camera source.
        Press 'q' to quit, 'f' to toggle fullscreen.
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open video source: {source}\n"
                "For IP cameras, check the URL format and network connectivity."
            )

        # Minimize buffer — critical for IP cam latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        is_ip_cam = isinstance(source, str)
        source_label = f"camera {source}" if not is_ip_cam else source
        logger.info("Live stream started (%s). Resolution: %dx%d. Press 'q' to quit.",
                     source_label, actual_w, actual_h)

        # For IP cams, use a grabber thread to always get the latest frame
        grabber = None
        if is_ip_cam:
            grabber = _FrameGrabber(cap)

        WIN_NAME = "Baby Monitor — Pose Analysis"
        DISPLAY_W, DISPLAY_H = 1280, 720

        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_NAME, DISPLAY_W, DISPLAY_H)

        fullscreen = False
        frame_idx = 0
        fps_display = 0.0

        while True:
            if grabber:
                ret, frame = grabber.read()
            else:
                ret, frame = cap.read()

            if not ret or frame is None:
                if is_ip_cam:
                    time.sleep(0.05)
                    continue
                logger.info("Stream ended.")
                break

            t0 = time.time()
            timestamp_ms = t0 * 1000
            poses = self.detector.detect(frame)
            detections = self._run_object_detection(frame)
            analysis, pairs = self._build_analysis(
                poses, frame_idx, timestamp_ms, detections=detections,
            )
            annotated = self._annotate(frame, pairs, detections)

            # FPS counter (smoothed)
            elapsed = time.time() - t0
            instant_fps = 1.0 / max(elapsed, 1e-6)
            fps_display = 0.7 * fps_display + 0.3 * instant_fps
            cv2.putText(
                annotated, f"FPS: {fps_display:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
            )

            # Debounced alerts
            for i, person in enumerate(analysis.persons):
                risk = person["risk"]
                self.alert_mgr.check(
                    person_index=i,
                    label=risk["label"],
                    score=risk["score"],
                    reasons=risk["reasons"],
                    frame=annotated,
                )

            # Scale up small frames
            fh, fw = annotated.shape[:2]
            scale = min(DISPLAY_W / fw, DISPLAY_H / fh)
            if scale > 1.0:
                annotated = cv2.resize(
                    annotated,
                    (int(fw * scale), int(fh * scale)),
                    interpolation=cv2.INTER_LINEAR,
                )

            cv2.imshow(WIN_NAME, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("f"):
                fullscreen = not fullscreen
                prop = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, prop)
                if not fullscreen:
                    cv2.resizeWindow(WIN_NAME, DISPLAY_W, DISPLAY_H)

            frame_idx += 1

        if grabber:
            grabber.stop()
        cap.release()
        cv2.destroyAllWindows()
        self._cleanup()

    # ── Internals ───────────────────────────────────────────────────────────

    def _run_object_detection(self, frame: np.ndarray) -> list[ObjectDetection]:
        """Async + frame-skipping detection for live/video."""
        if self._async_obj is not None:
            return self._async_obj.get_detections(frame)
        if self.obj_detector is not None:
            if self.settings.object_detection.enable_tracking:
                return self.obj_detector.track(frame)
            return self.obj_detector.detect(frame)
        return []

    def _run_object_detection_sync(self, frame: np.ndarray) -> list[ObjectDetection]:
        """Synchronous detection for single-image analysis."""
        if self._async_obj is not None:
            return self._async_obj.get_detections_sync(frame)
        if self.obj_detector is not None:
            return self.obj_detector.detect(frame)
        return []

    def _build_analysis(
        self,
        poses: list[PoseResult],
        frame_index: int,
        timestamp_ms: float,
        apply_smoothing: bool = True,
        detections: Optional[list[ObjectDetection]] = None,
    ) -> tuple[FrameAnalysis, list[tuple[PoseResult, RiskAssessment]]]:
        """
        Analyze poses, run cross-model rules, apply smoothing.
        Returns both the serializable FrameAnalysis and typed pairs for drawing.
        """
        dets = detections or []

        # Run cross-model rules (pose + object detection combined)
        cross_model_signals = self.combined_analyzer.analyze(poses, dets)

        persons = []
        raw_pairs: list[tuple[PoseResult, RiskAssessment]] = []

        for idx, pose in enumerate(poses):
            # 1. Pose-only risk
            raw_risk = self.analyzer.analyze(pose)

            # 2. Merge cross-model signals into the risk score
            if cross_model_signals:
                merged_risk = self._merge_cross_model(raw_risk, cross_model_signals)
            else:
                merged_risk = raw_risk

            # 3. Temporal smoothing
            risk = self.smoother.smooth(idx, merged_risk) if apply_smoothing else merged_risk

            persons.append({
                "pose": pose.to_dict(),
                "risk": risk.to_dict(),
                "raw_risk": raw_risk.to_dict(),
            })
            raw_pairs.append((pose, risk))

        # Handle cross-model "person detected but no pose" (no poses to iterate)
        if not poses and cross_model_signals:
            # Create a synthetic risk entry for the no-pose situation
            cm_scores = [s for s, _ in cross_model_signals]
            cm_reasons = [r for _, r in cross_model_signals]
            if cm_scores:
                no_pose_risk = RiskAssessment(
                    label=self._score_to_label(max(cm_scores)),
                    score=round(max(cm_scores), 3),
                    reasons=cm_reasons,
                )
                persons.append({
                    "pose": None,
                    "risk": no_pose_risk.to_dict(),
                    "raw_risk": no_pose_risk.to_dict(),
                })

        # Mark missing person slots for smoother reset
        if apply_smoothing:
            detected_indices = set(range(len(poses)))
            for slot in range(max(len(poses) + 1, 2)):
                if slot not in detected_indices:
                    self.smoother.mark_missing(slot)

        analysis = FrameAnalysis(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            persons=persons,
            detections=dets,
        )
        return analysis, raw_pairs

    @staticmethod
    def _merge_cross_model(
        pose_risk: RiskAssessment,
        cross_signals: list[tuple[float, str]],
    ) -> RiskAssessment:
        """
        Merge cross-model signals into a pose risk assessment.
        Uses the same weighted-max approach: take the highest score.
        """
        all_scores = [pose_risk.score] + [s for s, _ in cross_signals]
        all_reasons = list(pose_risk.reasons) + [r for _, r in cross_signals]

        max_score = max(all_scores)
        label = (
            "dangerous" if max_score >= 0.55
            else "uncertain" if max_score >= 0.30
            else "safe"
        )

        return RiskAssessment(
            label=label,
            score=round(max_score, 3),
            reasons=all_reasons,
        )

    @staticmethod
    def _score_to_label(score: float) -> str:
        if score >= 0.55:
            return "dangerous"
        if score >= 0.30:
            return "uncertain"
        return "safe"

    def _annotate(
        self,
        frame: np.ndarray,
        pairs: list[tuple[PoseResult, RiskAssessment]],
        detections: Optional[list[ObjectDetection]] = None,
    ) -> np.ndarray:
        """
        Draw all detections onto a SINGLE copy of the frame.
        Drawing functions operate in-place — only one copy is made here.
        """
        annotated = frame.copy()

        # Object detection boxes (underneath pose skeleton)
        if detections:
            draw_object_detections(annotated, detections, self.settings.combined)

        vis_threshold = self.settings.pose_model.landmark_visibility_threshold
        for pose, risk in pairs:
            draw_pose_annotation(annotated, pose, risk, vis_threshold)

        return annotated

    def _process_video_capture(
        self,
        cap: cv2.VideoCapture,
        save_annotated: bool,
        output_name: str,
        max_frames: int = 0,
    ) -> list[FrameAnalysis]:
        """Shared logic for processing frames from a VideoCapture."""
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer: Optional[cv2.VideoWriter] = None
        out_path: Optional[str] = None
        if save_annotated:
            out_path = os.path.join(self.settings.output_dir, output_name)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        results: list[FrameAnalysis] = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            poses = self.detector.detect(frame)
            detections = self._run_object_detection(frame)
            analysis, pairs = self._build_analysis(
                poses, frame_idx, timestamp_ms, detections=detections,
            )
            results.append(analysis)

            if writer is not None:
                annotated = self._annotate(frame, pairs, detections)
                writer.write(annotated)

            frame_idx += 1
            if max_frames > 0 and frame_idx >= max_frames:
                break

        if writer is not None:
            writer.release()
            if results:
                results[-1].annotated_frame_path = out_path

        self._cleanup()
        return results

    def _cleanup(self) -> None:
        """Release all resources."""
        self.detector.close()
        if self._async_obj is not None:
            self._async_obj.close()
        elif self.obj_detector is not None:
            self.obj_detector.close()
        self.smoother.reset()
        self.alert_mgr.reset()
