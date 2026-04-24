"""
Unit tests for PoseAnalyzer risk rules.

Each test creates a synthetic PoseResult with specific keypoint configurations
and verifies that the correct rules fire (or don't fire).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from config.settings import RiskThresholds
from services.pose_analyzer import PoseAnalyzer
from utils.schemas import Keypoint, PoseResult, LANDMARK_NAMES


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_keypoint(name: str, x: float, y: float, z: float = 0.0,
                   confidence: float = 0.9) -> Keypoint:
    return Keypoint(name=name, x=x, y=y, z=z, confidence=confidence)


def _make_pose(
    overrides: dict[str, dict] | None = None,
    default_conf: float = 0.9,
    frame_w: int = 640,
    frame_h: int = 480,
) -> PoseResult:
    """
    Create a PoseResult with all 33 landmarks at default positions.
    Override specific landmarks via the overrides dict:
        overrides={"nose": {"y": 300, "confidence": 0.1}}
    """
    # Default "safe" layout: standing upright, roughly centered
    defaults = {
        "nose":             {"x": 320, "y": 50,  "z": 0.0},
        "left_eye_inner":   {"x": 315, "y": 45,  "z": 0.0},
        "left_eye":         {"x": 310, "y": 44,  "z": 0.0},
        "left_eye_outer":   {"x": 305, "y": 45,  "z": 0.0},
        "right_eye_inner":  {"x": 325, "y": 45,  "z": 0.0},
        "right_eye":        {"x": 330, "y": 44,  "z": 0.0},
        "right_eye_outer":  {"x": 335, "y": 45,  "z": 0.0},
        "left_ear":         {"x": 295, "y": 50,  "z": 0.0},
        "right_ear":        {"x": 345, "y": 50,  "z": 0.0},
        "mouth_left":       {"x": 315, "y": 60,  "z": 0.0},
        "mouth_right":      {"x": 325, "y": 60,  "z": 0.0},
        "left_shoulder":    {"x": 280, "y": 120, "z": 0.0},
        "right_shoulder":   {"x": 360, "y": 120, "z": 0.0},
        "left_elbow":       {"x": 260, "y": 200, "z": 0.0},
        "right_elbow":      {"x": 380, "y": 200, "z": 0.0},
        "left_wrist":       {"x": 250, "y": 280, "z": 0.0},
        "right_wrist":      {"x": 390, "y": 280, "z": 0.0},
        "left_pinky":       {"x": 245, "y": 290, "z": 0.0},
        "right_pinky":      {"x": 395, "y": 290, "z": 0.0},
        "left_index":       {"x": 248, "y": 288, "z": 0.0},
        "right_index":      {"x": 392, "y": 288, "z": 0.0},
        "left_thumb":       {"x": 252, "y": 275, "z": 0.0},
        "right_thumb":      {"x": 388, "y": 275, "z": 0.0},
        "left_hip":         {"x": 300, "y": 300, "z": 0.0},
        "right_hip":        {"x": 340, "y": 300, "z": 0.0},
        "left_knee":        {"x": 295, "y": 380, "z": 0.0},
        "right_knee":       {"x": 345, "y": 380, "z": 0.0},
        "left_ankle":       {"x": 290, "y": 450, "z": 0.0},
        "right_ankle":      {"x": 350, "y": 450, "z": 0.0},
        "left_heel":        {"x": 288, "y": 460, "z": 0.0},
        "right_heel":       {"x": 352, "y": 460, "z": 0.0},
        "left_foot_index":  {"x": 285, "y": 465, "z": 0.0},
        "right_foot_index": {"x": 355, "y": 465, "z": 0.0},
    }

    overrides = overrides or {}
    keypoints: list[Keypoint] = []
    xs, ys = [], []

    for name in LANDMARK_NAMES:
        d = defaults.get(name, {"x": 320, "y": 240, "z": 0.0})
        if name in overrides:
            d = {**d, **overrides[name]}
        conf = d.get("confidence", default_conf)
        kp = Keypoint(name=name, x=d["x"], y=d["y"], z=d["z"], confidence=conf)
        keypoints.append(kp)
        if conf > 0.3:
            xs.append(d["x"])
            ys.append(d["y"])

    bbox = (min(xs), min(ys), max(xs), max(ys)) if xs else (0, 0, frame_w, frame_h)

    return PoseResult(
        keypoints=keypoints,
        person_confidence=default_conf,
        bbox=bbox,
        frame_width=frame_w,
        frame_height=frame_h,
    )


# ── Tests ────────────────────────────────────────────────────────────────────

@pytest.fixture
def analyzer():
    return PoseAnalyzer(RiskThresholds())


class TestSafePose:
    def test_default_pose_is_safe(self, analyzer):
        pose = _make_pose()
        result = analyzer.analyze(pose)
        assert result.label == "safe"
        assert result.score < 0.30


class TestFaceOcclusion:
    def test_face_hidden_body_visible(self, analyzer):
        """All face landmarks low confidence, body landmarks visible → face occluded."""
        overrides = {}
        for i, name in enumerate(LANDMARK_NAMES[:11]):
            overrides[name] = {"confidence": 0.1}
        pose = _make_pose(overrides=overrides)
        result = analyzer.analyze(pose)
        assert any("Face occluded" in r for r in result.reasons)
        assert result.score >= 0.55

    def test_face_visible_body_visible(self, analyzer):
        """All landmarks visible → no face occlusion."""
        pose = _make_pose()
        result = analyzer.analyze(pose)
        assert not any("Face occluded" in r for r in result.reasons)


class TestFaceDown:
    def test_nose_below_shoulders_flat_torso(self, analyzer):
        """Nose below shoulders + flat torso → prone detected."""
        pose = _make_pose(overrides={
            "nose": {"y": 200},
            "left_shoulder": {"x": 250, "y": 120},
            "right_shoulder": {"x": 390, "y": 120},
            "left_hip": {"x": 260, "y": 150},
            "right_hip": {"x": 380, "y": 150},
        })
        result = analyzer.analyze(pose)
        assert any("Prone" in r or "face-down" in r for r in result.reasons)

    def test_normal_upright(self, analyzer):
        """Nose above shoulders → no face-down."""
        pose = _make_pose()
        result = analyzer.analyze(pose)
        assert not any("Prone" in r for r in result.reasons)


class TestZDepth:
    def test_nose_far_from_camera(self, analyzer):
        """Nose z much larger than shoulder z → face pointing away."""
        pose = _make_pose(overrides={
            "nose": {"z": 80},
            "left_shoulder": {"x": 280, "y": 120, "z": 0},
            "right_shoulder": {"x": 360, "y": 120, "z": 0},
        })
        result = analyzer.analyze(pose)
        assert any("Z-depth" in r for r in result.reasons)

    def test_nose_normal_depth(self, analyzer):
        """Nose z close to shoulder z → no z-depth alert."""
        pose = _make_pose()
        result = analyzer.analyze(pose)
        assert not any("Z-depth" in r for r in result.reasons)


class TestHeadTurn:
    def test_one_ear_hidden_nose_low(self, analyzer):
        """One ear hidden + low nose confidence → head turn detected."""
        pose = _make_pose(overrides={
            "left_ear": {"confidence": 0.9},
            "right_ear": {"confidence": 0.1},
            "nose": {"confidence": 0.2},
        })
        result = analyzer.analyze(pose)
        assert any("Head turned" in r for r in result.reasons)

    def test_both_ears_visible(self, analyzer):
        """Both ears visible → no head turn."""
        pose = _make_pose()
        result = analyzer.analyze(pose)
        assert not any("Head turned" in r for r in result.reasons)


class TestBodyInversion:
    def test_hips_above_shoulders(self, analyzer):
        """Hips above shoulders (inverted) → body inversion alert."""
        pose = _make_pose(overrides={
            "left_shoulder": {"y": 300},
            "right_shoulder": {"y": 300},
            "left_hip": {"y": 100},
            "right_hip": {"y": 100},
        })
        result = analyzer.analyze(pose)
        assert any("inverted" in r for r in result.reasons)


class TestAllVisibleFix:
    def test_all_visible_uses_configurable_threshold(self):
        """_all_visible should use min_keypoint_confidence, not hardcoded 0.3."""
        thresholds = RiskThresholds(min_keypoint_confidence=0.6)
        analyzer = PoseAnalyzer(thresholds)

        # Keypoint at 0.5 should fail the 0.6 gate
        kp = Keypoint(name="test", x=0, y=0, z=0, confidence=0.5)
        assert not analyzer._all_visible(kp)

        # Keypoint at 0.7 should pass
        kp2 = Keypoint(name="test", x=0, y=0, z=0, confidence=0.7)
        assert analyzer._all_visible(kp2)


class TestLimbCrossing:
    def test_arms_crossed(self, analyzer):
        """Both wrists far on opposite side → limb crossing."""
        pose = _make_pose(overrides={
            "left_wrist": {"x": 450},   # far right of center
            "right_wrist": {"x": 190},  # far left of center
        })
        result = analyzer.analyze(pose)
        assert any("crossed" in r.lower() for r in result.reasons)


class TestCorroborationBonus:
    def test_multiple_rules_increase_score(self, analyzer):
        """When multiple rules fire, corroboration bonus should increase the score."""
        # Trigger face occlusion (face hidden) + low visibility
        # Keep shoulders + hips visible (4 body landmarks) so face-occluded fires,
        # but hide everything else so low-visibility also fires.
        overrides = {}
        # Hide all face landmarks (0-10)
        for name in LANDMARK_NAMES[:11]:
            overrides[name] = {"confidence": 0.1}
        # Hide limbs (13-22, 25-32) but keep shoulders (11,12) and hips (23,24)
        for i in list(range(13, 23)) + list(range(25, 33)):
            overrides[LANDMARK_NAMES[i]] = {"confidence": 0.1}

        pose = _make_pose(overrides=overrides)
        result = analyzer.analyze(pose)
        # Both face-occluded (0.80) and low-visibility (0.7) should fire
        # With corroboration bonus the score should exceed 0.80
        assert result.score > 0.80
        assert len(result.reasons) >= 2


# ── Orientation-aware tests ─────────────────────────────────────────────────

def _make_lying_on_side_pose(
    neck_deviation: float = 0.0,
    side: str = "right",
    frame_w: int = 640,
    frame_h: int = 480,
) -> PoseResult:
    """
    Create a person lying on their side.  Torso axis is roughly horizontal.
    neck_deviation: degrees the neck deviates from the torso axis (0 = aligned).
    side: which side they are lying on ("left" or "right").
    """
    import math

    # Person lying horizontally — shoulders and hips at similar y but spread on x
    if side == "right":
        # Head on the right, feet on the left
        hip_cx, hip_cy = 200, 240
        sh_cx, sh_cy = 400, 240
    else:
        hip_cx, hip_cy = 440, 240
        sh_cx, sh_cy = 240, 240

    # Torso axis direction (hip→shoulder)
    tax = sh_cx - hip_cx
    tay = sh_cy - hip_cy
    mag = math.hypot(tax, tay)
    tax_n, tay_n = tax / mag, tay / mag

    # Perpendicular (for shoulder/hip spread)
    perp_x, perp_y = -tay_n, tax_n

    shoulder_spread = 35
    hip_spread = 25

    # Neck direction: torso axis rotated by neck_deviation degrees
    neck_angle = math.radians(neck_deviation)
    neck_len = 70
    ndx = tax_n * math.cos(neck_angle) - tay_n * math.sin(neck_angle)
    ndy = tax_n * math.sin(neck_angle) + tay_n * math.cos(neck_angle)

    ear_x = sh_cx + ndx * neck_len
    ear_y = sh_cy + ndy * neck_len

    overrides = {
        "nose":            {"x": ear_x + ndx * 15, "y": ear_y + ndy * 15},
        "left_ear":        {"x": ear_x + perp_x * 10, "y": ear_y + perp_y * 10},
        "right_ear":       {"x": ear_x - perp_x * 10, "y": ear_y - perp_y * 10},
        "left_eye":        {"x": ear_x + perp_x * 5, "y": ear_y + perp_y * 5},
        "right_eye":       {"x": ear_x - perp_x * 5, "y": ear_y - perp_y * 5},
        "left_eye_inner":  {"x": ear_x + perp_x * 3, "y": ear_y + perp_y * 3},
        "right_eye_inner": {"x": ear_x - perp_x * 3, "y": ear_y - perp_y * 3},
        "left_eye_outer":  {"x": ear_x + perp_x * 7, "y": ear_y + perp_y * 7},
        "right_eye_outer": {"x": ear_x - perp_x * 7, "y": ear_y - perp_y * 7},
        "mouth_left":      {"x": ear_x + perp_x * 5 + ndx * 10, "y": ear_y + perp_y * 5 + ndy * 10},
        "mouth_right":     {"x": ear_x - perp_x * 5 + ndx * 10, "y": ear_y - perp_y * 5 + ndy * 10},
        "left_shoulder":   {"x": sh_cx + perp_x * shoulder_spread, "y": sh_cy + perp_y * shoulder_spread},
        "right_shoulder":  {"x": sh_cx - perp_x * shoulder_spread, "y": sh_cy - perp_y * shoulder_spread},
        "left_elbow":      {"x": sh_cx + perp_x * 50 - tax_n * 30, "y": sh_cy + perp_y * 50 - tay_n * 30},
        "right_elbow":     {"x": sh_cx - perp_x * 50 - tax_n * 30, "y": sh_cy - perp_y * 50 - tay_n * 30},
        "left_wrist":      {"x": sh_cx + perp_x * 60 - tax_n * 50, "y": sh_cy + perp_y * 60 - tay_n * 50},
        "right_wrist":     {"x": sh_cx - perp_x * 60 - tax_n * 50, "y": sh_cy - perp_y * 60 - tay_n * 50},
        "left_hip":        {"x": hip_cx + perp_x * hip_spread, "y": hip_cy + perp_y * hip_spread},
        "right_hip":       {"x": hip_cx - perp_x * hip_spread, "y": hip_cy - perp_y * hip_spread},
        "left_knee":       {"x": hip_cx - tax_n * 80 + perp_x * 20, "y": hip_cy - tay_n * 80 + perp_y * 20},
        "right_knee":      {"x": hip_cx - tax_n * 80 - perp_x * 20, "y": hip_cy - tay_n * 80 - perp_y * 20},
        "left_ankle":      {"x": hip_cx - tax_n * 160 + perp_x * 15, "y": hip_cy - tay_n * 160 + perp_y * 15},
        "right_ankle":     {"x": hip_cx - tax_n * 160 - perp_x * 15, "y": hip_cy - tay_n * 160 - perp_y * 15},
    }
    return _make_pose(overrides=overrides, frame_w=frame_w, frame_h=frame_h)


class TestNeckAngleOrientationAware:
    """Verify neck angle detection works correctly for all body orientations."""

    def test_lying_on_side_aligned_neck_is_safe(self, analyzer):
        """Person lying on side with neck aligned to torso → safe (no neck alert)."""
        pose = _make_lying_on_side_pose(neck_deviation=0)
        result = analyzer.analyze(pose)
        assert not any("neck angle" in r.lower() for r in result.reasons), \
            f"Should not flag aligned neck when lying down, got: {result.reasons}"

    def test_lying_on_side_slight_tilt_is_safe(self, analyzer):
        """Person lying on side with 30° neck tilt → still safe (within threshold)."""
        pose = _make_lying_on_side_pose(neck_deviation=30)
        result = analyzer.analyze(pose)
        assert not any("neck angle" in r.lower() for r in result.reasons), \
            f"30° deviation from torso should be within threshold, got: {result.reasons}"

    def test_lying_on_side_severe_tilt_is_flagged(self, analyzer):
        """Person lying on side with 75° neck bend → flagged as awkward."""
        pose = _make_lying_on_side_pose(neck_deviation=75)
        result = analyzer.analyze(pose)
        assert any("neck angle" in r.lower() for r in result.reasons), \
            "75° neck deviation from torso axis should be flagged"

    def test_lying_on_left_side_aligned_is_safe(self, analyzer):
        """Same test lying on the left side."""
        pose = _make_lying_on_side_pose(neck_deviation=0, side="left")
        result = analyzer.analyze(pose)
        assert not any("neck angle" in r.lower() for r in result.reasons)

    def test_upright_normal_neck_is_safe(self, analyzer):
        """Default upright pose with normal neck → safe."""
        pose = _make_pose()
        result = analyzer.analyze(pose)
        assert not any("neck angle" in r.lower() for r in result.reasons)

    def test_upright_tilted_neck_is_flagged(self, analyzer):
        """Upright pose with ear far to the side → flagged."""
        # Move the right ear very far to the right relative to the shoulder
        pose = _make_pose(overrides={
            "right_ear": {"x": 500, "y": 120},  # almost level with shoulder
        })
        result = analyzer.analyze(pose)
        assert any("neck angle" in r.lower() for r in result.reasons), \
            "Extreme lateral neck tilt when upright should be flagged"

    def test_reason_says_torso_axis(self, analyzer):
        """Alert reason should mention 'torso axis' not 'vertical'."""
        pose = _make_pose(overrides={
            "right_ear": {"x": 500, "y": 120},
        })
        result = analyzer.analyze(pose)
        neck_reasons = [r for r in result.reasons if "neck angle" in r.lower()]
        assert any("torso axis" in r for r in neck_reasons), \
            f"Reason should reference torso axis, got: {neck_reasons}"


class TestBodyInversionOrientationAware:
    """Body inversion check should not fire when lying down."""

    def test_lying_down_not_flagged_as_inverted(self, analyzer):
        """When lying on side, hips at same y as shoulders is normal."""
        pose = _make_lying_on_side_pose(neck_deviation=0)
        result = analyzer.analyze(pose)
        assert not any("inverted" in r.lower() for r in result.reasons), \
            "Lying-down position should not trigger body inversion"

    def test_upright_inverted_still_flagged(self, analyzer):
        """Truly inverted upright pose should still be caught."""
        pose = _make_pose(overrides={
            "left_shoulder": {"y": 300},
            "right_shoulder": {"y": 300},
            "left_hip": {"y": 100},
            "right_hip": {"y": 100},
        })
        result = analyzer.analyze(pose)
        assert any("inverted" in r.lower() for r in result.reasons)


class TestLimbCrossingOrientationAware:
    """Limb crossing check should use body-relative coordinates."""

    def test_lying_down_normal_arms_not_flagged(self, analyzer):
        """Arms in normal position while lying on side → no limb crossing."""
        pose = _make_lying_on_side_pose(neck_deviation=0)
        result = analyzer.analyze(pose)
        assert not any("crossed" in r.lower() for r in result.reasons)

    def test_upright_crossed_arms_still_flagged(self, analyzer):
        """Arms crossed while upright should still be detected."""
        pose = _make_pose(overrides={
            "left_wrist": {"x": 450},
            "right_wrist": {"x": 190},
        })
        result = analyzer.analyze(pose)
        assert any("crossed" in r.lower() for r in result.reasons)


class TestTorsoTiltDetection:
    """Direct tests for the body orientation helpers."""

    def test_upright_tilt_near_zero(self, analyzer):
        """Default upright pose should have low torso tilt."""
        pose = _make_pose()
        tilt = analyzer._torso_tilt_from_vertical(pose)
        assert tilt < 20, f"Upright pose should have low tilt, got {tilt:.1f}°"

    def test_lying_down_tilt_near_90(self, analyzer):
        """Lying-on-side pose should have high torso tilt."""
        pose = _make_lying_on_side_pose()
        tilt = analyzer._torso_tilt_from_vertical(pose)
        assert tilt > 70, f"Lying-down pose should have high tilt, got {tilt:.1f}°"

    def test_is_lying_down_true_for_horizontal(self, analyzer):
        pose = _make_lying_on_side_pose()
        assert analyzer._is_lying_down(pose)

    def test_is_lying_down_false_for_upright(self, analyzer):
        pose = _make_pose()
        assert not analyzer._is_lying_down(pose)
