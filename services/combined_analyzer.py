"""
Cross-model analyzer — combines pose estimation and object detection results
to produce risk signals that neither model can produce alone.

Rules:
  1. Face-object overlap: A dangerous object (blanket, pillow, cloth) overlapping
     with the face region derived from pose landmarks.
  2. Object near airway: Any object bbox overlapping the nose/mouth region.
  3. Person-no-pose: YOLO detects a person/baby but MediaPipe finds no pose,
     meaning the baby is fully occluded or covered.
  4. Hazard proximity: A hazard object (knife, scissors, fork, etc.) detected
     near the baby's body — triggers an alert even if not touching the face.
"""

from __future__ import annotations

import logging
from typing import Optional

from config.settings import CombinedAnalyzerConfig
from utils.schemas import PoseResult, ObjectDetection, RiskAssessment

logger = logging.getLogger(__name__)


def _bbox_iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Compute Intersection over Union between two (x1, y1, x2, y2) bboxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0:
        return 0.0

    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def _bbox_overlap_ratio(inner: tuple[float, ...], outer: tuple[float, ...]) -> float:
    """
    Compute what fraction of 'inner' bbox is covered by 'outer' bbox.
    This is NOT symmetric — it measures how much of the smaller region is covered.
    """
    x1 = max(inner[0], outer[0])
    y1 = max(inner[1], outer[1])
    x2 = min(inner[2], outer[2])
    y2 = min(inner[3], outer[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    inner_area = max(0.0, inner[2] - inner[0]) * max(0.0, inner[3] - inner[1])

    return inter / inner_area if inner_area > 0 else 0.0


class CombinedAnalyzer:
    """
    Cross-model risk analyzer that reasons over both pose and object detections.

    This does NOT replace PoseAnalyzer — it produces *additional* risk signals
    that are merged into the final risk score by the pipeline.
    """

    def __init__(self, config: Optional[CombinedAnalyzerConfig] = None) -> None:
        self.config = config or CombinedAnalyzerConfig()
        # Lowercase for case-insensitive matching
        self._face_danger_classes = {c.lower() for c in self.config.face_danger_classes}
        self._hazard_classes = {c.lower() for c in self.config.hazard_classes}
        self._person_classes = {c.lower() for c in self.config.person_classes}

    def analyze(
        self,
        poses: list[PoseResult],
        detections: list[ObjectDetection],
    ) -> list[tuple[float, str]]:
        """
        Run cross-model rules and return a list of (score, reason) tuples.

        These are additional signals to be merged with the PoseAnalyzer output.
        Returns an empty list if no cross-model risks are found.
        """
        results: list[tuple[float, str]] = []

        # Rule 1: Object overlapping face region
        for pose in poses:
            face_bbox = self._get_face_bbox(pose)
            if face_bbox is None:
                continue

            for det in detections:
                if det.class_name.lower() not in self._face_danger_classes:
                    continue
                # Skip the person's own detection (their bbox contains their face)
                if self._is_same_person(pose, det):
                    continue

                overlap = _bbox_overlap_ratio(face_bbox, det.bbox)
                if overlap >= self.config.face_object_overlap_iou:
                    track_info = f" [track {det.track_id}]" if det.track_id is not None else ""
                    results.append((
                        self.config.face_object_overlap_score,
                        f"DANGER: {det.class_name}{track_info} overlapping face region "
                        f"({overlap:.0%} coverage). Possible suffocation risk.",
                    ))

        # Rule 2: Object near airway (nose/mouth area — tighter region)
        for pose in poses:
            airway_bbox = self._get_airway_bbox(pose)
            if airway_bbox is None:
                continue

            for det in detections:
                # Skip person-class detections — a person's own bbox always
                # overlaps their own nose/mouth. This rule is for *other* objects.
                if det.class_name.lower() in self._person_classes:
                    continue
                # Also skip the person's own detection by bbox overlap
                if self._is_same_person(pose, det):
                    continue
                # Skip if we already flagged this object in rule 1
                if det.class_name.lower() in self._face_danger_classes:
                    continue

                overlap = _bbox_overlap_ratio(airway_bbox, det.bbox)
                if overlap >= 0.20:
                    track_info = f" [track {det.track_id}]" if det.track_id is not None else ""
                    results.append((
                        0.65,
                        f"Object near airway: {det.class_name}{track_info} "
                        f"overlapping nose/mouth region ({overlap:.0%}).",
                    ))

        # Rule 4: Hazard object near baby's body (knife, scissors, etc.)
        for pose in poses:
            expanded_bbox = self._expand_bbox(
                pose.bbox, self.config.hazard_proximity_expansion,
            )
            for det in detections:
                if det.class_name.lower() not in self._hazard_classes:
                    continue
                if self._is_same_person(pose, det):
                    continue
                # Already handled by airway rule above
                overlap = _bbox_overlap_ratio(det.bbox, expanded_bbox)
                if overlap >= 0.15:
                    track_info = f" [track {det.track_id}]" if det.track_id is not None else ""
                    results.append((
                        self.config.hazard_proximity_score,
                        f"HAZARD: {det.class_name}{track_info} detected near baby "
                        f"({overlap:.0%} overlap with body region). "
                        "Potential injury risk.",
                    ))

        # Rule 3: YOLO sees a person but MediaPipe found no pose
        if not poses and detections:
            person_dets = [
                d for d in detections
                if d.class_name.lower() in self._person_classes
            ]
            if person_dets:
                best = max(person_dets, key=lambda d: d.confidence)
                track_info = f" [track {best.track_id}]" if best.track_id is not None else ""
                results.append((
                    self.config.person_no_pose_score,
                    f"Person detected by YOLO{track_info} (conf={best.confidence:.2f}) "
                    "but no pose landmarks found. Baby may be fully covered.",
                ))

        return results

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _is_same_person(self, pose: PoseResult, det: ObjectDetection) -> bool:
        """
        Check if a YOLO detection is the same person as the pose.

        True if:
          - The detection is a person-class AND its bbox heavily overlaps the pose bbox
          - OR any detection bbox heavily contains the pose bbox (same entity)

        This prevents a person's own YOLO box from triggering face/airway rules
        on their own landmarks.
        """
        # If it's a person-class detection, check overlap with pose bbox
        if det.class_name.lower() in self._person_classes:
            overlap = _bbox_overlap_ratio(pose.bbox, det.bbox)
            if overlap >= 0.40:
                return True

        # For any class: if the detection bbox largely contains the pose bbox,
        # it's likely the same entity (e.g. "baby" class wrapping the pose)
        overlap = _bbox_overlap_ratio(pose.bbox, det.bbox)
        if overlap >= 0.70:
            return True

        return False

    @staticmethod
    def _expand_bbox(
        bbox: tuple[float, ...], expansion: float,
    ) -> tuple[float, float, float, float]:
        """Expand a bbox by a fraction in each direction (e.g. 0.3 = 30%)."""
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        dx = w * expansion
        dy = h * expansion
        return (x1 - dx, y1 - dy, x2 + dx, y2 + dy)

    @staticmethod
    def _get_face_bbox(pose: PoseResult) -> Optional[tuple[float, float, float, float]]:
        """
        Compute a face bounding box from pose landmarks.
        Uses nose, eyes, ears, and mouth landmarks with some padding.
        """
        face_indices = list(range(11))  # landmarks 0-10
        face_kps = [
            pose.keypoints[i] for i in face_indices
            if i < len(pose.keypoints) and pose.keypoints[i].confidence > 0.2
        ]
        if len(face_kps) < 2:
            return None

        xs = [kp.x for kp in face_kps]
        ys = [kp.y for kp in face_kps]

        # Add 20% padding around the face
        w = max(xs) - min(xs)
        h = max(ys) - min(ys)
        pad_x = max(w * 0.2, 10)
        pad_y = max(h * 0.2, 10)

        return (
            min(xs) - pad_x,
            min(ys) - pad_y,
            max(xs) + pad_x,
            max(ys) + pad_y,
        )

    @staticmethod
    def _get_airway_bbox(pose: PoseResult) -> Optional[tuple[float, float, float, float]]:
        """
        Compute a tight bounding box around the nose and mouth region.
        """
        # nose=0, mouth_left=9, mouth_right=10
        airway_indices = [0, 9, 10]
        kps = [
            pose.keypoints[i] for i in airway_indices
            if i < len(pose.keypoints) and pose.keypoints[i].confidence > 0.3
        ]
        if len(kps) < 1:
            return None

        xs = [kp.x for kp in kps]
        ys = [kp.y for kp in kps]

        # Small tight box
        pad = 15
        return (
            min(xs) - pad,
            min(ys) - pad,
            max(xs) + pad,
            max(ys) + pad,
        )
