"""Tests for the decision rules and cooldowns."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from chewwatch.config import AlertsConfig, ClassifierConfig
from chewwatch.decision import Cooldowns, evaluate
from chewwatch.models import Verdict


def _verdict(**overrides) -> Verdict:
    defaults = dict(
        dog_present=True,
        chewing=True,
        object="couch",
        confidence=0.9,
        reason="Mouth working the cushion",
        best_frame=2,
    )
    defaults.update(overrides)
    return Verdict(**defaults)


def _cfg(**overrides) -> ClassifierConfig:
    defaults = dict(api_key="k", threshold=0.7, ignore_toys=True)
    defaults.update(overrides)
    return ClassifierConfig(**defaults)


def test_alert_when_chewing():
    decision = evaluate(_verdict(), _cfg())
    assert decision.alert is True
    assert decision.outcome == "alert"
    assert "couch" in decision.detail


def test_no_dog():
    decision = evaluate(_verdict(dog_present=False, chewing=False, object="none"), _cfg())
    assert decision.alert is False
    assert decision.outcome == "no_dog"


def test_no_chewing():
    decision = evaluate(_verdict(chewing=False, object="none"), _cfg())
    assert decision.alert is False
    assert decision.outcome == "no_chewing"


def test_toy_suppressed_when_ignoring():
    decision = evaluate(_verdict(object="toy"), _cfg())
    assert decision.alert is False
    assert decision.outcome == "toy_or_bone"


def test_bone_suppressed_when_ignoring():
    decision = evaluate(_verdict(object="Bone"), _cfg())
    assert decision.alert is False
    assert decision.outcome == "toy_or_bone"


def test_toy_alerts_when_not_ignoring():
    decision = evaluate(_verdict(object="toy", confidence=0.95), _cfg(ignore_toys=False))
    assert decision.alert is True


def test_low_confidence():
    decision = evaluate(_verdict(confidence=0.69), _cfg(threshold=0.7))
    assert decision.alert is False
    assert decision.outcome == "low_confidence"
    assert "0.69" in decision.detail


def test_confidence_at_threshold_alerts():
    decision = evaluate(_verdict(confidence=0.7), _cfg(threshold=0.7))
    assert decision.alert is True


def test_stick_chew_is_not_a_toy():
    # "stick" is not in the acceptable set — a stick chewer still alerts
    decision = evaluate(_verdict(object="stick"), _cfg())
    assert decision.alert is True


def test_counter_only_alert():
    decision = evaluate(
        _verdict(
            chewing=False, object="none", on_counter=True, counter_confidence=0.9
        ),
        _cfg(),
    )
    assert decision.alert is True
    assert decision.outcome == "alert"
    assert "counter" in decision.detail


def test_chewing_and_counter_combined():
    decision = evaluate(_verdict(on_counter=True, counter_confidence=0.8), _cfg())
    assert decision.alert is True
    assert "chewing detected" in decision.detail.lower()
    assert "counter" in decision.detail.lower()


def test_toy_on_counter_still_alerts():
    # ignore_toys suppresses the *chewing* alert only — being on the counter
    # is a violation in its own right
    decision = evaluate(
        _verdict(object="toy", on_counter=True, counter_confidence=0.85), _cfg()
    )
    assert decision.alert is True
    assert "counter" in decision.detail


def test_counter_low_confidence():
    decision = evaluate(
        _verdict(
            chewing=False, object="none", on_counter=True, counter_confidence=0.4
        ),
        _cfg(threshold=0.7),
    )
    assert decision.alert is False
    assert decision.outcome == "low_confidence"
    assert "counter" in decision.detail.lower()


def test_counter_off_is_no_chewing():
    decision = evaluate(_verdict(chewing=False, object="none"), _cfg())
    assert decision.outcome == "no_chewing"


class TestCooldowns:
    def test_first_alert_allowed(self):
        cd = Cooldowns(8)
        now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        assert cd.allow("cam-1", now) is True

    def test_blocked_within_window(self):
        cd = Cooldowns(8)
        now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        cd.mark("cam-1", now)
        assert cd.allow("cam-1", now + timedelta(minutes=7)) is False
        assert cd.allow("cam-1", now + timedelta(minutes=8, seconds=1)) is True

    def test_per_camera(self):
        cd = Cooldowns(8)
        now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        cd.mark("cam-1", now)
        assert cd.allow("cam-2", now) is True

    def test_seconds_remaining(self):
        cd = Cooldowns(10)
        now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        assert cd.seconds_remaining("cam-1", now) == 0.0
        cd.mark("cam-1", now)
        assert cd.seconds_remaining("cam-1", now + timedelta(minutes=4)) == 6 * 60
        assert cd.seconds_remaining("cam-1", now + timedelta(minutes=12)) == 0.0

    def test_zero_cooldown(self):
        cd = Cooldowns(0)
        now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        cd.mark("cam-1", now)
        assert cd.allow("cam-1", now) is True


def test_cooldown_from_config():
    from chewwatch.decision import cooldown_from_config

    cd = cooldown_from_config(AlertsConfig(cooldown_minutes=3))
    assert isinstance(cd, Cooldowns)
