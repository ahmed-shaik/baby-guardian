"""
Alert cooldown / debounce system.

Prevents the same alert from spamming the console every frame.
Tracks per-person alert state and only emits:
  - On first detection of non-safe risk
  - When severity changes (e.g. uncertain → dangerous)
  - After the cooldown period expires (re-alert for ongoing danger)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config.settings import AlertConfig, WhatsAppConfig
from services.whatsapp_notifier import WhatsAppNotifier

logger = logging.getLogger(__name__)


@dataclass
class _PersonAlertState:
    """Tracks alert state for a single detected person."""
    last_alert_time: float = 0.0
    last_alert_label: str = "safe"
    alert_count: int = 0


class AlertManager:
    """Debounced alert system for the baby monitor pipeline."""

    def __init__(
        self,
        config: AlertConfig | None = None,
        whatsapp_config: WhatsAppConfig | None = None,
    ) -> None:
        self.config = config or AlertConfig()
        self._states: dict[int, _PersonAlertState] = {}
        self._whatsapp = WhatsAppNotifier(whatsapp_config)

    def check(
        self,
        person_index: int,
        label: str,
        score: float,
        reasons: list[str],
        frame: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Check if an alert should be emitted for this person.

        Returns True if the alert should be shown/logged, False if suppressed.
        When True, also logs the alert via the logging framework.
        """
        if label == "safe":
            # Clear state when person returns to safe
            if person_index in self._states:
                prev = self._states[person_index]
                if prev.last_alert_label != "safe":
                    logger.info(
                        "Person %d: returned to SAFE (was %s for %d alerts)",
                        person_index, prev.last_alert_label.upper(), prev.alert_count,
                    )
                self._states[person_index] = _PersonAlertState()
            return False

        now = time.time()
        state = self._states.get(person_index, _PersonAlertState())

        should_alert = False

        # First alert for this person
        if state.last_alert_label == "safe":
            should_alert = True

        # Severity changed
        elif (self.config.realert_on_severity_change and
              label != state.last_alert_label):
            should_alert = True

        # Cooldown expired
        elif (now - state.last_alert_time) >= self.config.cooldown_seconds:
            should_alert = True

        if should_alert:
            state.last_alert_time = now
            state.last_alert_label = label
            state.alert_count += 1
            self._states[person_index] = state

            logger.warning(
                "ALERT Person %d: %s (score=%.3f) — %s",
                person_index, label.upper(), score, "; ".join(reasons[:3]),
            )

            if self.config.enable_sound:
                print("\a", end="", flush=True)  # terminal bell

            # WhatsApp notification
            self._whatsapp.send_alert(
                label=label,
                score=score,
                reasons=reasons,
                person_index=person_index,
                frame=frame,
            )

            return True

        return False

    def reset(self) -> None:
        """Clear all alert states (e.g. when restarting a stream)."""
        self._states.clear()
