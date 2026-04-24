"""
Unit tests for the AlertManager cooldown / debounce system.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import pytest
from config.settings import AlertConfig
from services.alert_manager import AlertManager


@pytest.fixture
def mgr():
    return AlertManager(AlertConfig(cooldown_seconds=1.0))


class TestAlertCooldown:
    def test_safe_never_alerts(self, mgr):
        assert mgr.check(0, "safe", 0.0, []) is False

    def test_first_dangerous_alerts(self, mgr):
        assert mgr.check(0, "dangerous", 0.8, ["face down"]) is True

    def test_repeated_dangerous_suppressed(self, mgr):
        mgr.check(0, "dangerous", 0.8, ["face down"])
        # Immediately again — should be suppressed (within cooldown)
        assert mgr.check(0, "dangerous", 0.8, ["face down"]) is False

    def test_severity_change_alerts(self):
        mgr = AlertManager(AlertConfig(
            cooldown_seconds=999,  # long cooldown
            realert_on_severity_change=True,
        ))
        mgr.check(0, "uncertain", 0.4, ["neck angle"])
        # Severity change from uncertain → dangerous should alert
        assert mgr.check(0, "dangerous", 0.8, ["face down"]) is True

    def test_cooldown_expires(self):
        mgr = AlertManager(AlertConfig(cooldown_seconds=0.1))
        mgr.check(0, "dangerous", 0.8, ["face down"])
        time.sleep(0.15)
        # After cooldown expires, should alert again
        assert mgr.check(0, "dangerous", 0.8, ["face down"]) is True

    def test_different_persons_independent(self, mgr):
        mgr.check(0, "dangerous", 0.8, ["face down"])
        # Person 1 should still get their first alert
        assert mgr.check(1, "dangerous", 0.7, ["neck angle"]) is True

    def test_return_to_safe_clears_state(self, mgr):
        mgr.check(0, "dangerous", 0.8, ["face down"])
        mgr.check(0, "safe", 0.0, [])
        # New dangerous after safe → should alert again
        assert mgr.check(0, "dangerous", 0.8, ["face down"]) is True

    def test_reset_clears_all(self, mgr):
        mgr.check(0, "dangerous", 0.8, ["face down"])
        mgr.reset()
        assert mgr.check(0, "dangerous", 0.8, ["face down"]) is True
