"""
WhatsApp alert notifier via Twilio.

Sends WhatsApp messages (+ optional snapshot image) when the baby monitor
detects a dangerous situation.

Setup (one-time — ~5 minutes):
  1. Create a free Twilio account at https://twilio.com
  2. Go to: Messaging → Try it out → Send a WhatsApp message
  3. From your phone, send the join code to the Twilio sandbox number
     (e.g. "join <word>-<word>" to whatsapp:+14155238886)
  4. Set your credentials via environment variables:

       set TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
       set TWILIO_AUTH_TOKEN=your_auth_token
       set WHATSAPP_FROM=whatsapp:+14155238886
       set WHATSAPP_TO=whatsapp:+92xxxxxxxxxx

     OR pass them directly as CLI flags:
       --wa-sid ... --wa-token ... --wa-to ...

Dependency:
    pip install twilio
"""

from __future__ import annotations

import base64
import logging
import os
import time
from typing import Optional

import cv2
import numpy as np

from config.settings import WhatsAppConfig

logger = logging.getLogger(__name__)


class WhatsAppNotifier:
    """
    Sends WhatsApp alerts via the Twilio API.

    Responsibilities:
      - Load Twilio credentials from config or environment variables
      - Enforce its own cooldown (separate from the console alert cooldown)
      - Optionally save a frame snapshot and attach it to the message
      - Gracefully degrade if Twilio is not installed or credentials are missing
    """

    def __init__(self, config: Optional[WhatsAppConfig] = None) -> None:
        self.config = config or WhatsAppConfig()
        self._client = None
        self._last_sent: float = 0.0
        self._ready: bool = False

        if not self.config.enabled:
            return

        self._init_client()

    # ── Setup ────────────────────────────────────────────────────────────────

    def _init_client(self) -> None:
        """Initialise Twilio client. Logs clearly if anything is missing."""
        try:
            from twilio.rest import Client  # type: ignore
        except ImportError:
            logger.error(
                "Twilio is not installed. Run: pip install twilio\n"
                "WhatsApp alerts will be disabled."
            )
            return

        # Credentials — env vars take priority over config values
        sid = os.environ.get("TWILIO_ACCOUNT_SID") or self.config.account_sid
        token = os.environ.get("TWILIO_AUTH_TOKEN") or self.config.auth_token
        self._from = os.environ.get("WHATSAPP_FROM") or self.config.from_number
        self._to = os.environ.get("WHATSAPP_TO") or self.config.to_number

        if not sid or not token:
            logger.error(
                "Twilio credentials missing. Set TWILIO_ACCOUNT_SID and "
                "TWILIO_AUTH_TOKEN environment variables (or use --wa-sid / --wa-token).\n"
                "WhatsApp alerts will be disabled."
            )
            return

        if not self._to:
            logger.error(
                "WhatsApp recipient not set. Use --wa-to whatsapp:+<number> "
                "or set WHATSAPP_TO environment variable.\n"
                "WhatsApp alerts will be disabled."
            )
            return

        self._client = Client(sid, token)
        self._ready = True
        logger.info(
            "WhatsApp notifier ready. Alerts → %s (cooldown: %ds)",
            self._to, int(self.config.whatsapp_cooldown_seconds),
        )

    # ── Public API ───────────────────────────────────────────────────────────

    def send_alert(
        self,
        label: str,
        score: float,
        reasons: list[str],
        person_index: int,
        frame: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Send a WhatsApp alert message.

        Args:
            label:        Risk label ("dangerous" or "uncertain")
            score:        Risk score in [0, 1]
            reasons:      List of reason strings from the analyzer
            person_index: Which detected person triggered the alert
            frame:        Current annotated video frame (for snapshot)

        Returns:
            True if the message was sent, False if skipped/failed.
        """
        if not self._ready:
            return False

        if self.config.only_on_dangerous and label != "dangerous":
            return False

        # Enforce WhatsApp-specific cooldown
        now = time.time()
        if (now - self._last_sent) < self.config.whatsapp_cooldown_seconds:
            remaining = self.config.whatsapp_cooldown_seconds - (now - self._last_sent)
            logger.debug("WhatsApp cooldown active (%.0fs remaining)", remaining)
            return False

        body = self._build_message(label, score, reasons, person_index)
        snapshot_url = None

        if self.config.send_snapshot and frame is not None:
            # Save the snapshot locally (for /api/snapshot endpoint to serve)
            local_path = self._save_snapshot(frame)
            if local_path:
                # Use the configured public URL so Twilio can fetch the image
                public_url = (
                    os.environ.get("WHATSAPP_SNAPSHOT_URL")
                    or self.config.snapshot_url
                )
                if public_url:
                    snapshot_url = public_url

        return self._send(body, snapshot_url)

    # ── Internals ────────────────────────────────────────────────────────────

    def _build_message(
        self,
        label: str,
        score: float,
        reasons: list[str],
        person_index: int,
    ) -> str:
        emoji = "🚨" if label == "dangerous" else "⚠️"
        lines = [
            f"{emoji} *Baby Monitor Alert*",
            f"Status: *{label.upper()}* (score: {score:.2f})",
            f"Person: #{person_index}",
            "",
            "*Reasons:*",
        ]
        for r in reasons[:5]:
            lines.append(f"• {r}")

        import datetime
        lines.append("")
        lines.append(f"🕐 {datetime.datetime.now().strftime('%H:%M:%S')}")
        return "\n".join(lines)

    def _save_snapshot(self, frame: np.ndarray) -> Optional[str]:
        """Save frame to disk and return the local file path."""
        try:
            os.makedirs(os.path.dirname(self.config.snapshot_path) or ".", exist_ok=True)
            cv2.imwrite(self.config.snapshot_path, frame)
            logger.debug("Snapshot saved: %s", self.config.snapshot_path)
            return self.config.snapshot_path
        except Exception as exc:
            logger.warning("Failed to save snapshot: %s", exc)
            return None

    def _send(self, body: str, snapshot_path: Optional[str]) -> bool:
        """
        Send WhatsApp message via Twilio.

        For snapshot attachment, Twilio requires a publicly accessible URL.
        If a public snapshot URL is configured, the image is attached.
        Otherwise, the snapshot is sent as a text-only message.
        To enable image attachments in local development, run ngrok and set
        the snapshot URL accordingly.
        """
        try:
            kwargs: dict = {
                "from_": self._from,
                "to": self._to,
                "body": body,
            }

            if snapshot_path and snapshot_path.startswith("http"):
                # Public URL — Twilio can fetch this directly
                kwargs["media_url"] = [snapshot_path]
            elif snapshot_path:
                # Local file — try to serve it via our API server
                # User needs to set up ngrok or similar for internet access
                logger.debug(
                    "Snapshot saved locally at %s. "
                    "To attach images to WhatsApp, expose your server via ngrok "
                    "and set snapshot_path to the public URL.",
                    snapshot_path,
                )

            self._client.messages.create(**kwargs)
            self._last_sent = time.time()
            logger.info("WhatsApp alert sent to %s", self._to)
            return True

        except Exception as exc:
            # Still update cooldown on failure to prevent spam-retrying
            self._last_sent = time.time()
            logger.error("WhatsApp send failed: %s", exc)
            return False
