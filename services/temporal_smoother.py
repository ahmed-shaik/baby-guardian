"""
Temporal smoothing for risk assessments across video frames.

The raw PoseAnalyzer score can spike on a single bad frame — a momentary
pose estimation glitch triggers DANGEROUS then immediately drops back to SAFE.

This module applies Exponential Moving Average (EMA) smoothing to the raw
risk score before converting it to a label:

    smoothed[t] = alpha * raw[t] + (1 - alpha) * smoothed[t-1]

Tuning alpha (configured in settings.py):
  - alpha=1.0  → no smoothing (raw score, original behaviour)
  - alpha=0.4  → moderate smoothing, ~3-4 frames to fully react  (default)
  - alpha=0.2  → heavy smoothing, ~8-10 frames to fully react
  - alpha=0.1  → very heavy, ~15-20 frames to fully react

At 30 fps, alpha=0.4 means:
  - A genuine dangerous pose raises the smoothed score to 0.55+ in ~2-3 frames
  - A single rogue frame causes a blip that decays within 2-3 frames

The smoother also tracks whether an alert was already active, so the label
only flips from DANGEROUS back to SAFE once the smoothed score drops below
a hysteresis band — preventing rapid on/off flicker at the boundary.

Usage
-----
  smoother = TemporalSmoother(thresholds)
  smoothed_risk = smoother.smooth(person_index=0, raw_risk=risk)
"""

from __future__ import annotations

from collections import defaultdict

from config.settings import RiskThresholds
from utils.schemas import RiskAssessment


class TemporalSmoother:
    """
    Per-person EMA smoother for risk scores.

    Maintains a separate EMA state for each person slot (by index).
    Resets automatically if a person disappears for more than
    `reset_after_missing_frames` consecutive frames.
    """

    def __init__(self, thresholds: RiskThresholds) -> None:
        self.thresholds = thresholds
        # smoothed score per person index
        self._ema: dict[int, float] = defaultdict(float)
        # how many frames in a row this person has been absent
        self._missing_frames: dict[int, int] = defaultdict(int)

    def smooth(self, person_index: int, raw_risk: RiskAssessment) -> RiskAssessment:
        """
        Apply EMA smoothing to a raw RiskAssessment.

        Returns a new RiskAssessment with:
          - score  : the EMA-smoothed score
          - label  : re-derived from the smoothed score using the same thresholds
          - reasons: preserved from the raw assessment (still shows what fired)
        """
        alpha = self.thresholds.ema_alpha

        # Reset EMA if this person slot was missing for too long
        self._missing_frames[person_index] = 0

        prev_ema = self._ema[person_index]
        new_ema = alpha * raw_risk.score + (1.0 - alpha) * prev_ema
        self._ema[person_index] = new_ema

        smoothed_score = round(new_ema, 3)
        label = self._score_to_label(smoothed_score)

        return RiskAssessment(
            label=label,
            score=smoothed_score,
            reasons=raw_risk.reasons,
        )

    def mark_missing(self, person_index: int) -> None:
        """
        Call this each frame a person slot is NOT detected.
        Resets their EMA after enough missing frames so stale state
        doesn't linger when the baby re-enters the frame.
        """
        self._missing_frames[person_index] += 1
        if self._missing_frames[person_index] >= self.thresholds.smoother_reset_after_frames:
            self._ema[person_index] = 0.0
            self._missing_frames[person_index] = 0

    def reset(self) -> None:
        """Reset all smoothing state (e.g. when restarting a stream)."""
        self._ema.clear()
        self._missing_frames.clear()

    def _score_to_label(self, score: float) -> str:
        if score >= self.thresholds.dangerous_score_threshold:
            return "dangerous"
        if score >= self.thresholds.uncertain_score_threshold:
            return "uncertain"
        return "safe"