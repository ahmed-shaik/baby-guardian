"""
Heuristic-based risk analyzer for baby pose estimation.

Takes a PoseResult (MediaPipe 33-landmark format) and produces a RiskAssessment
by running a battery of rule-based checks.

── Scoring logic ──
Each rule returns a score in [0, 1] and a human-readable reason.

The final score uses a WEIGHTED MAX approach:
    final_score = max_score + corroboration_bonus

Rules included:
  - Low visibility        : too few landmarks detected
  - Face occlusion        : face landmarks missing while body is visible
  - Face down / prone     : nose below shoulders + flat torso
  - Z-depth face away     : nose z further from camera than shoulders (face pressed down)
  - Head turn             : one ear visible, other missing, nose low-conf
  - Neck angle            : ear-to-shoulder angle too steep
  - Collapsed posture     : landmarks bunched in a small area
  - Body inversion        : hips above shoulders
  - Limb crossing         : arms crossed behind body (tangling)
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from config.settings import RiskThresholds
from utils.schemas import PoseResult, RiskAssessment, Keypoint

logger = logging.getLogger(__name__)


class PoseAnalyzer:
    """Rule-based risk classifier for baby poses."""

    def __init__(self, thresholds: Optional[RiskThresholds] = None) -> None:
        self.thresholds = thresholds or RiskThresholds()

    def analyze(self, pose: PoseResult) -> RiskAssessment:
        """Run all risk rules and return a combined assessment."""
        rule_results: list[tuple[float, str]] = []

        rules = [
            self._check_low_visibility,
            self._check_face_occluded,
            self._check_face_down,
            self._check_z_depth_face_away,
            self._check_head_turn,
            self._check_neck_angle,
            self._check_collapsed_posture,
            self._check_body_inversion,
            self._check_limb_crossing,
        ]

        for rule_fn in rules:
            result = rule_fn(pose)
            if result is not None:
                rule_results.append(result)

        if not rule_results:
            return RiskAssessment(label="safe", score=0.0, reasons=["No risk signals detected."])

        scores = [s for s, _ in rule_results]
        reasons = [reason for _, reason in rule_results if reason]

        max_score = max(scores)
        extra_rules = len(scores) - 1
        corroboration_bonus = min(
            self.thresholds.max_corroboration_bonus,
            extra_rules * self.thresholds.corroboration_bonus_per_rule,
        )

        final_score = min(1.0, max_score + corroboration_bonus)
        label = self._score_to_label(final_score)

        return RiskAssessment(label=label, score=round(final_score, 3), reasons=reasons)

    # ── Risk rules ──────────────────────────────────────────────────────────

    def _check_low_visibility(self, pose: PoseResult) -> Optional[tuple[float, str]]:
        """Flag when too few landmarks are visible."""
        total = len(pose.keypoints)
        if total == 0:
            return (0.8, "No landmarks detected — pose cannot be assessed.")

        visible = pose.visible_count(min_conf=self.thresholds.min_keypoint_confidence)
        ratio = visible / total

        if ratio < self.thresholds.min_visible_keypoint_ratio:
            return (
                0.7,
                f"Low landmark visibility ({visible}/{total} = {ratio:.0%}). "
                "Baby may be occluded or in an unusual position.",
            )
        return None

    def _check_face_occluded(self, pose: PoseResult) -> Optional[tuple[float, str]]:
        """
        Detect when the face is not visible but the body is.

        If body landmarks (shoulders, hips) are visible but face landmarks
        (nose, eyes, ears, mouth) are mostly missing, the face is likely
        covered or pressed into a surface — a strong suffocation signal.
        """
        conf_gate = self.thresholds.min_keypoint_confidence

        # Face landmark indices: 0=nose, 1-6=eyes, 7-8=ears, 9-10=mouth
        face_indices = list(range(11))
        face_visible = sum(
            1 for i in face_indices
            if i < len(pose.keypoints) and pose.keypoints[i].confidence >= conf_gate
        )

        # Body landmark indices: 11-12=shoulders, 23-24=hips
        body_indices = [11, 12, 23, 24]
        body_visible = sum(
            1 for i in body_indices
            if i < len(pose.keypoints) and pose.keypoints[i].confidence >= conf_gate
        )

        # Only fire if we can see the body but not the face
        if body_visible >= 3 and face_visible < self.thresholds.min_visible_face_landmarks:
            return (
                0.80,
                f"Face occluded — only {face_visible}/11 face landmarks visible "
                f"while {body_visible}/4 body landmarks detected. "
                "Face may be covered or pressed into a surface.",
            )
        return None

    def _check_face_down(self, pose: PoseResult) -> Optional[tuple[float, str]]:
        """Detect prone / face-down posture."""
        nose = pose.keypoint_by_name("nose")
        l_shoulder = pose.keypoint_by_name("left_shoulder")
        r_shoulder = pose.keypoint_by_name("right_shoulder")
        l_hip = pose.keypoint_by_name("left_hip")
        r_hip = pose.keypoint_by_name("right_hip")

        if not self._all_visible(nose, l_shoulder, r_shoulder):
            return None

        shoulder_mid_y = (l_shoulder.y + r_shoulder.y) / 2
        margin = self.thresholds.prone_nose_below_shoulder_margin * pose.frame_height

        if nose.y <= (shoulder_mid_y + margin):
            return None  # nose is above shoulders — normal

        # Nose is below shoulders. Check if torso is flat (prone indicator).
        if self._all_visible(l_hip, r_hip):
            hip_mid_y = (l_hip.y + r_hip.y) / 2
            torso_height = abs(hip_mid_y - shoulder_mid_y)
            torso_width = abs(l_shoulder.x - r_shoulder.x)
            if torso_width > 0 and torso_height / torso_width < 1.0:
                return (0.9, "Prone/face-down posture detected — nose below shoulders with flat torso.")

        return (0.6, "Nose positioned below shoulders — possible face-down posture.")

    def _check_z_depth_face_away(self, pose: PoseResult) -> Optional[tuple[float, str]]:
        """
        Use the Z (depth) coordinate to detect face pointing away from camera.

        MediaPipe z is relative depth: more positive = further from camera.
        If nose z is significantly further than shoulder z, the face is likely
        pointing away (pressed into mattress / surface).
        """
        nose = pose.keypoint_by_name("nose")
        l_shoulder = pose.keypoint_by_name("left_shoulder")
        r_shoulder = pose.keypoint_by_name("right_shoulder")

        if not self._all_visible(nose, l_shoulder, r_shoulder):
            return None

        shoulder_mid_z = (l_shoulder.z + r_shoulder.z) / 2
        shoulder_width = abs(l_shoulder.x - r_shoulder.x)

        if shoulder_width < 10:  # too narrow to judge
            return None

        z_diff = nose.z - shoulder_mid_z  # positive = nose further from camera
        threshold = self.thresholds.z_depth_face_away_ratio * shoulder_width

        if z_diff > threshold:
            score = min(1.0, 0.5 + (z_diff / shoulder_width))
            return (
                score,
                f"Z-depth: nose is {z_diff:.1f}px further from camera than shoulders "
                f"(threshold: {threshold:.1f}px). Face may be pointing away.",
            )
        return None

    def _check_head_turn(self, pose: PoseResult) -> Optional[tuple[float, str]]:
        """
        Detect head turned into a surface.

        If one ear is visible but the other is not, and nose confidence is low,
        the baby's head is likely turned sideways into a pillow or mattress.
        """
        conf_gate = self.thresholds.min_keypoint_confidence
        nose = pose.keypoint_by_name("nose")
        l_ear = pose.keypoint_by_name("left_ear")
        r_ear = pose.keypoint_by_name("right_ear")

        if nose is None or l_ear is None or r_ear is None:
            return None

        l_ear_vis = l_ear.confidence >= conf_gate
        r_ear_vis = r_ear.confidence >= conf_gate
        nose_low = nose.confidence < self.thresholds.head_turn_nose_conf_threshold

        # One ear visible, other not, and nose confidence is poor
        if l_ear_vis != r_ear_vis and nose_low:
            visible_side = "left" if l_ear_vis else "right"
            hidden_side = "right" if l_ear_vis else "left"
            return (
                0.55,
                f"Head turned: {visible_side} ear visible but {hidden_side} ear hidden, "
                f"nose confidence low ({nose.confidence:.2f}). "
                "Head may be turned into a surface.",
            )
        return None

    def _check_neck_angle(self, pose: PoseResult) -> Optional[tuple[float, str]]:
        """
        Flag awkward neck angles measured relative to the torso axis.

        Instead of measuring ear-to-shoulder angle from absolute vertical (which
        false-positives when lying down), we measure the deviation of the neck
        (shoulder→ear vector) from the torso (hip→shoulder vector).  This gives an
        orientation-independent measurement that works whether the person is
        upright, lying on their side, or in any other position.

        When hips are not visible (no torso axis available), we fall back to
        measuring from vertical but apply a lying-down discount: if the shoulders
        are roughly horizontal we increase the effective threshold, since a large
        angle from vertical is expected and normal when lying down.
        """
        torso_axis = self._torso_axis_vector(pose)
        worst: Optional[tuple[float, str]] = None

        for side in ("left", "right"):
            ear = pose.keypoint_by_name(f"{side}_ear")
            shoulder = pose.keypoint_by_name(f"{side}_shoulder")
            if not self._all_visible(ear, shoulder):
                continue

            neck_dx = ear.x - shoulder.x
            neck_dy = ear.y - shoulder.y
            if abs(neck_dx) < 1e-6 and abs(neck_dy) < 1e-6:
                continue

            if torso_axis is not None:
                # Measure angle between neck vector and torso axis
                angle = self._angle_between(
                    (neck_dx, neck_dy), torso_axis
                )
                threshold = self.thresholds.max_neck_angle_deg
            else:
                # Fallback: measure from vertical, but adjust threshold
                # for lying-down orientation detected via shoulder positions
                angle = math.degrees(math.atan2(abs(neck_dx), abs(shoulder.y - ear.y)))
                threshold = self._effective_neck_threshold(pose)

            if angle > threshold:
                score = min(1.0, angle / 90.0)
                reason = (
                    f"Awkward neck angle ({side}): {angle:.0f}° from torso axis "
                    f"(threshold: {threshold:.0f}°)."
                )
                if worst is None or score > worst[0]:
                    worst = (score, reason)

        return worst

    def _check_collapsed_posture(self, pose: PoseResult) -> Optional[tuple[float, str]]:
        """Detect curled-up or collapsed posture."""
        x1, y1, x2, y2 = pose.bbox
        box_area = (x2 - x1) * (y2 - y1)
        if box_area <= 0:
            return None

        vis_threshold = self.thresholds.min_keypoint_confidence
        visible_kps = [kp for kp in pose.keypoints if kp.confidence >= vis_threshold]
        if len(visible_kps) < 4:
            return None

        xs = [kp.x for kp in visible_kps]
        ys = [kp.y for kp in visible_kps]
        kp_area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        spread_ratio = kp_area / box_area

        if spread_ratio < self.thresholds.min_keypoint_spread_ratio:
            return (
                0.6,
                f"Collapsed/curled posture — landmark spread is {spread_ratio:.0%} "
                f"of bounding box (threshold: {self.thresholds.min_keypoint_spread_ratio:.0%}).",
            )
        return None

    def _check_body_inversion(self, pose: PoseResult) -> Optional[tuple[float, str]]:
        """
        Check if the body is inverted (hips above shoulders).

        When the person is lying down, hips and shoulders are at similar y-levels
        which is normal.  We only flag inversion when the torso is reasonably
        upright (< 55° from vertical) AND hips are significantly above shoulders.
        """
        l_shoulder = pose.keypoint_by_name("left_shoulder")
        r_shoulder = pose.keypoint_by_name("right_shoulder")
        l_hip = pose.keypoint_by_name("left_hip")
        r_hip = pose.keypoint_by_name("right_hip")

        if not self._all_visible(l_shoulder, r_shoulder, l_hip, r_hip):
            return None

        # Skip inversion check when lying down — y-ordering is meaningless
        if self._is_lying_down(pose):
            return None

        shoulder_mid_y = (l_shoulder.y + r_shoulder.y) / 2
        hip_mid_y = (l_hip.y + r_hip.y) / 2

        if hip_mid_y < shoulder_mid_y:
            diff = shoulder_mid_y - hip_mid_y
            if diff > 0.03 * pose.frame_height:
                return (0.85, "Body appears inverted — hips are above shoulders.")

        return None

    def _check_limb_crossing(self, pose: PoseResult) -> Optional[tuple[float, str]]:
        """
        Detect unusual limb positions that may indicate tangling.

        Uses the body's own coordinate frame (perpendicular to the torso axis)
        to project wrist positions.  This makes the check work regardless of
        whether the person is upright or lying down.

        When the person is lying on their side, the torso axis is roughly
        horizontal, so "left vs right of body center" is measured along the
        y-axis (perpendicular to torso) rather than the x-axis.
        """
        l_shoulder = pose.keypoint_by_name("left_shoulder")
        r_shoulder = pose.keypoint_by_name("right_shoulder")
        l_wrist = pose.keypoint_by_name("left_wrist")
        r_wrist = pose.keypoint_by_name("right_wrist")

        if not self._all_visible(l_shoulder, r_shoulder, l_wrist, r_wrist):
            return None

        torso_axis = self._torso_axis_vector(pose)
        body_center_x = (l_shoulder.x + r_shoulder.x) / 2
        body_center_y = (l_shoulder.y + r_shoulder.y) / 2

        if torso_axis is not None:
            # Perpendicular axis to torso (the "lateral" direction of the body)
            tx, ty = torso_axis
            mag = math.hypot(tx, ty)
            if mag < 1e-6:
                return None
            # Perpendicular: rotate 90° → (-ty, tx), normalized
            perp_x, perp_y = -ty / mag, tx / mag

            # Project wrist and shoulder positions onto the perpendicular axis
            l_sh_proj = (l_shoulder.x - body_center_x) * perp_x + (l_shoulder.y - body_center_y) * perp_y
            r_sh_proj = (r_shoulder.x - body_center_x) * perp_x + (r_shoulder.y - body_center_y) * perp_y
            l_wr_proj = (l_wrist.x - body_center_x) * perp_x + (l_wrist.y - body_center_y) * perp_y
            r_wr_proj = (r_wrist.x - body_center_x) * perp_x + (r_wrist.y - body_center_y) * perp_y

            body_width = abs(l_sh_proj - r_sh_proj)
            if body_width < 10:
                return None

            center_proj = (l_sh_proj + r_sh_proj) / 2
            l_crossed = l_wr_proj > center_proj + body_width * 0.5 if l_sh_proj < r_sh_proj else l_wr_proj < center_proj - body_width * 0.5
            r_crossed = r_wr_proj < center_proj - body_width * 0.5 if l_sh_proj < r_sh_proj else r_wr_proj > center_proj + body_width * 0.5
        else:
            # Fallback: original x-axis logic for upright positions
            body_width = abs(l_shoulder.x - r_shoulder.x)
            if body_width < 10:
                return None
            l_crossed = l_wrist.x > body_center_x + body_width * 0.5
            r_crossed = r_wrist.x < body_center_x - body_width * 0.5

        if l_crossed and r_crossed:
            return (0.5, "Both arms appear crossed/twisted behind the body.")

        return None

    # ── Body-orientation helpers ───────────────────────────────────────────

    def _torso_axis_vector(self, pose: PoseResult) -> Optional[tuple[float, float]]:
        """
        Return the torso axis as a (dx, dy) vector pointing from hip midpoint
        toward shoulder midpoint.  Returns None if shoulders or hips are not
        sufficiently visible.
        """
        l_shoulder = pose.keypoint_by_name("left_shoulder")
        r_shoulder = pose.keypoint_by_name("right_shoulder")
        l_hip = pose.keypoint_by_name("left_hip")
        r_hip = pose.keypoint_by_name("right_hip")

        if not self._all_visible(l_shoulder, r_shoulder):
            return None

        shoulder_mx = (l_shoulder.x + r_shoulder.x) / 2
        shoulder_my = (l_shoulder.y + r_shoulder.y) / 2

        # Try both hips, then one hip
        if self._all_visible(l_hip, r_hip):
            hip_mx = (l_hip.x + r_hip.x) / 2
            hip_my = (l_hip.y + r_hip.y) / 2
        elif self._all_visible(l_hip):
            hip_mx, hip_my = l_hip.x, l_hip.y
        elif self._all_visible(r_hip):
            hip_mx, hip_my = r_hip.x, r_hip.y
        else:
            return None

        dx = shoulder_mx - hip_mx
        dy = shoulder_my - hip_my
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return None
        return (dx, dy)

    def _torso_tilt_from_vertical(self, pose: PoseResult) -> float:
        """
        Return the angle (degrees) the torso axis makes with vertical.
        0° = perfectly upright, 90° = lying flat.
        Returns 0.0 if torso axis cannot be computed.
        """
        axis = self._torso_axis_vector(pose)
        if axis is None:
            # Fallback: estimate from shoulder y-spread vs x-spread
            l_sh = pose.keypoint_by_name("left_shoulder")
            r_sh = pose.keypoint_by_name("right_shoulder")
            if self._all_visible(l_sh, r_sh):
                sx = abs(l_sh.x - r_sh.x)
                sy = abs(l_sh.y - r_sh.y)
                # If shoulders are side-by-side (large x-diff, small y-diff)
                # person is more likely upright.  If stacked (small x-diff,
                # large y-diff) person is likely lying on their side.
                # This is a rough heuristic when hips are missing.
                if sx + sy > 0:
                    return math.degrees(math.atan2(sy, sx))
            return 0.0

        dx, dy = axis
        # Angle from vertical: vertical = (0, -1) in image coords (y increases downward)
        # Torso vector points hip→shoulder, so for upright person dy < 0.
        return math.degrees(math.atan2(abs(dx), abs(dy)))

    def _is_lying_down(self, pose: PoseResult) -> bool:
        """Return True if the person appears to be lying down (torso > 55° from vertical)."""
        return self._torso_tilt_from_vertical(pose) > 55.0

    def _effective_neck_threshold(self, pose: PoseResult) -> float:
        """
        Return the neck angle threshold, adjusted for body orientation.

        When the body is lying down, a large ear-to-shoulder angle from vertical
        is normal.  We progressively relax the threshold as the torso tilts away
        from vertical, up to 85° (essentially disabling the check for fully
        reclined positions).
        """
        base = self.thresholds.max_neck_angle_deg
        tilt = self._torso_tilt_from_vertical(pose)

        if tilt < 30.0:
            return base  # upright — use normal threshold
        # Linear ramp from base at 30° tilt to 85° at 90° tilt
        t = min(1.0, (tilt - 30.0) / 60.0)
        return base + t * (85.0 - base)

    @staticmethod
    def _angle_between(v1: tuple[float, float], v2: tuple[float, float]) -> float:
        """Return the unsigned angle in degrees between two 2D vectors."""
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.hypot(*v1)
        mag2 = math.hypot(*v2)
        if mag1 < 1e-9 or mag2 < 1e-9:
            return 0.0
        cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
        return math.degrees(math.acos(cos_angle))

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _all_visible(self, *keypoints: Optional[Keypoint]) -> bool:
        """Check all keypoints are present and above the configurable confidence gate."""
        gate = self.thresholds.min_keypoint_confidence
        return all(kp is not None and kp.confidence >= gate for kp in keypoints)

    def _score_to_label(self, score: float) -> str:
        if score >= self.thresholds.dangerous_score_threshold:
            return "dangerous"
        if score >= self.thresholds.uncertain_score_threshold:
            return "uncertain"
        return "safe"
