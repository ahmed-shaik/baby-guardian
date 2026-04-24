"""
Unit tests for CombinedAnalyzer cross-model rules.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from config.settings import CombinedAnalyzerConfig
from services.combined_analyzer import CombinedAnalyzer, _bbox_iou, _bbox_overlap_ratio
from utils.schemas import Keypoint, PoseResult, ObjectDetection, LANDMARK_NAMES


def _make_simple_pose(face_x=320, face_y=50, conf=0.9) -> PoseResult:
    """Create a minimal pose with visible face and body landmarks."""
    defaults = {
        "nose":           {"x": face_x,      "y": face_y},
        "left_eye":       {"x": face_x - 10, "y": face_y - 5},
        "right_eye":      {"x": face_x + 10, "y": face_y - 5},
        "left_ear":       {"x": face_x - 25, "y": face_y},
        "right_ear":      {"x": face_x + 25, "y": face_y},
        "mouth_left":     {"x": face_x - 5,  "y": face_y + 10},
        "mouth_right":    {"x": face_x + 5,  "y": face_y + 10},
        "left_shoulder":  {"x": face_x - 40, "y": face_y + 70},
        "right_shoulder": {"x": face_x + 40, "y": face_y + 70},
        "left_hip":       {"x": face_x - 20, "y": face_y + 180},
        "right_hip":      {"x": face_x + 20, "y": face_y + 180},
    }

    keypoints = []
    for name in LANDMARK_NAMES:
        d = defaults.get(name, {"x": face_x, "y": face_y + 100})
        keypoints.append(Keypoint(
            name=name, x=d["x"], y=d["y"], z=0.0, confidence=conf,
        ))

    return PoseResult(
        keypoints=keypoints,
        person_confidence=conf,
        bbox=(200, 20, 440, 400),
        frame_width=640,
        frame_height=480,
    )


@pytest.fixture
def analyzer():
    return CombinedAnalyzer(CombinedAnalyzerConfig())


class TestBboxHelpers:
    def test_iou_identical(self):
        assert _bbox_iou((0, 0, 100, 100), (0, 0, 100, 100)) == pytest.approx(1.0)

    def test_iou_no_overlap(self):
        assert _bbox_iou((0, 0, 50, 50), (100, 100, 200, 200)) == 0.0

    def test_overlap_ratio_full_coverage(self):
        # Inner completely inside outer
        assert _bbox_overlap_ratio((10, 10, 20, 20), (0, 0, 100, 100)) == pytest.approx(1.0)

    def test_overlap_ratio_no_overlap(self):
        assert _bbox_overlap_ratio((0, 0, 10, 10), (50, 50, 100, 100)) == 0.0


class TestFaceObjectOverlap:
    def test_teddy_bear_over_face(self, analyzer):
        """Teddy bear bbox overlapping face region → high risk signal."""
        pose = _make_simple_pose(face_x=320, face_y=50)
        teddy = ObjectDetection(
            class_id=0, class_name="teddy bear", confidence=0.8,
            bbox=(290, 20, 370, 100),  # covers face area
        )
        signals = analyzer.analyze([pose], [teddy])
        assert len(signals) >= 1
        assert any("teddy bear" in r.lower() for _, r in signals)
        assert any(s >= 0.85 for s, _ in signals)

    def test_book_over_face(self, analyzer):
        """Book over face → suffocation risk."""
        pose = _make_simple_pose(face_x=320, face_y=50)
        book = ObjectDetection(
            class_id=1, class_name="book", confidence=0.7,
            bbox=(290, 20, 370, 100),
        )
        signals = analyzer.analyze([pose], [book])
        assert any("suffocation" in r.lower() for _, r in signals)

    def test_danger_object_far_from_face(self, analyzer):
        """Danger object far from face → no signal."""
        pose = _make_simple_pose(face_x=320, face_y=50)
        teddy = ObjectDetection(
            class_id=0, class_name="teddy bear", confidence=0.8,
            bbox=(0, 400, 200, 480),  # far away from face
        )
        signals = analyzer.analyze([pose], [teddy])
        face_signals = [r for _, r in signals if "teddy bear" in r.lower()]
        assert len(face_signals) == 0

    def test_non_danger_class_ignored(self, analyzer):
        """Non-danger class (e.g. 'cup') does not trigger face-object overlap."""
        pose = _make_simple_pose()
        cup = ObjectDetection(
            class_id=5, class_name="cup", confidence=0.9,
            bbox=(290, 20, 370, 100),
        )
        signals = analyzer.analyze([pose], [cup])
        face_danger = [r for _, r in signals if "suffocation" in r.lower()]
        assert len(face_danger) == 0


class TestSamePersonFiltering:
    def test_own_person_bbox_does_not_trigger_airway(self, analyzer):
        """A person's own YOLO 'person' detection should NOT trigger airway alert."""
        pose = _make_simple_pose(face_x=320, face_y=50)
        # Person detection that wraps the entire pose — this is the person themselves
        person_det = ObjectDetection(
            class_id=0, class_name="person", confidence=0.9,
            bbox=(200, 20, 440, 400),  # matches pose.bbox
            track_id=1,
        )
        signals = analyzer.analyze([pose], [person_det])
        airway_signals = [r for _, r in signals if "airway" in r.lower()]
        assert len(airway_signals) == 0, (
            "Person's own YOLO bbox should not trigger airway rule"
        )

    def test_own_person_bbox_does_not_trigger_face_overlap(self, analyzer):
        """A person's own 'baby' detection should NOT trigger face-danger rule."""
        config = CombinedAnalyzerConfig(
            face_danger_classes=["baby"],
            person_classes=["person", "baby"],
        )
        a = CombinedAnalyzer(config)
        pose = _make_simple_pose(face_x=320, face_y=50)
        baby_det = ObjectDetection(
            class_id=0, class_name="baby", confidence=0.9,
            bbox=(200, 20, 440, 400),
        )
        signals = a.analyze([pose], [baby_det])
        danger_signals = [r for _, r in signals if "suffocation" in r.lower()]
        assert len(danger_signals) == 0

    def test_separate_teddy_still_triggers(self, analyzer):
        """A teddy bear over the face (NOT the person's own box) should still fire."""
        pose = _make_simple_pose(face_x=320, face_y=50)
        teddy = ObjectDetection(
            class_id=1, class_name="teddy bear", confidence=0.8,
            bbox=(290, 20, 370, 100),  # over face, NOT wrapping entire body
        )
        signals = analyzer.analyze([pose], [teddy])
        assert any("teddy bear" in r.lower() for _, r in signals)


class TestPersonNoPose:
    def test_yolo_person_no_mediapipe(self, analyzer):
        """YOLO detects person but no pose → fully covered alert."""
        person_det = ObjectDetection(
            class_id=0, class_name="person", confidence=0.75,
            bbox=(100, 100, 400, 400),
        )
        signals = analyzer.analyze([], [person_det])
        assert len(signals) >= 1
        assert any("no pose" in r.lower() for _, r in signals)

    def test_yolo_object_no_pose_no_alert(self, analyzer):
        """YOLO detects non-person object, no pose → no person-no-pose alert."""
        obj_det = ObjectDetection(
            class_id=5, class_name="blanket", confidence=0.9,
            bbox=(100, 100, 400, 400),
        )
        signals = analyzer.analyze([], [obj_det])
        person_signals = [r for _, r in signals if "no pose" in r.lower()]
        assert len(person_signals) == 0
