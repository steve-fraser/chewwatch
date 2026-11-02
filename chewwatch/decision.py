"""Decision logic: thresholds, toy suppression and per-camera cooldowns."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from .config import AlertsConfig, ClassifierConfig
from .models import Verdict


@dataclass(frozen=True)
class Decision:
    """Outcome of evaluating a verdict against the configured rules."""

    alert: bool
    outcome: str  # short machine reason for the log
    detail: str  # human explanation


def evaluate(verdict: Verdict, classifier: ClassifierConfig) -> Decision:
    """Apply the alert rules to a classifier verdict (cooldown is separate).

    A verdict alerts when the dog is chewing something unacceptable OR — on
    counter cameras — when it is up on a counter or other elevated surface.
    The toy rule suppresses chewing alerts only: a dog chewing its toy on the
    counter still alerts.
    """
    if not verdict.dog_present:
        return Decision(False, "no_dog", "No dog visible in the sampled frames")

    chewing_alert = (
        verdict.chewing
        and not (classifier.ignore_toys and verdict.is_toy_or_bone)
        and verdict.confidence >= classifier.threshold
    )
    counter_alert = (
        verdict.on_counter and verdict.counter_confidence >= classifier.threshold
    )

    if chewing_alert and counter_alert:
        return Decision(
            True,
            "alert",
            f"Chewing detected: {verdict.object} (confidence "
            f"{verdict.confidence:.2f}) — and it is on the counter "
            f"(confidence {verdict.counter_confidence:.2f})",
        )
    if chewing_alert:
        return Decision(
            True,
            "alert",
            f"Chewing detected: {verdict.object} "
            f"(confidence {verdict.confidence:.2f})",
        )
    if counter_alert:
        return Decision(
            True,
            "alert",
            f"Dog is on the counter (confidence {verdict.counter_confidence:.2f})",
        )

    # No alert — explain why, keeping the documented outcome vocabulary.
    if not verdict.chewing:
        if verdict.on_counter and verdict.counter_confidence < classifier.threshold:
            return Decision(
                False,
                "low_confidence",
                f"Possible counter sighting but confidence "
                f"{verdict.counter_confidence:.2f} below threshold "
                f"{classifier.threshold:.2f}",
            )
        return Decision(False, "no_chewing", "Dog present but not chewing anything")
    if classifier.ignore_toys and verdict.is_toy_or_bone:
        return Decision(
            False,
            "toy_or_bone",
            f"Chewing an acceptable item ({verdict.object}) — ignore_toys is on",
        )
    return Decision(
        False,
        "low_confidence",
        f"Confidence {verdict.confidence:.2f} below threshold "
        f"{classifier.threshold:.2f}",
    )


class Cooldowns:
    """Per-camera alert cooldown so one chewing session sends one alert."""

    def __init__(self, cooldown_minutes: float) -> None:
        self._cooldown = timedelta(minutes=cooldown_minutes)
        self._last_alert: dict[str, datetime] = {}

    def allow(self, camera_id: str, now: datetime) -> bool:
        last = self._last_alert.get(camera_id)
        return last is None or now - last >= self._cooldown

    def seconds_remaining(self, camera_id: str, now: datetime) -> float:
        last = self._last_alert.get(camera_id)
        if last is None:
            return 0.0
        remaining = self._cooldown - (now - last)
        return max(0.0, remaining.total_seconds())

    def mark(self, camera_id: str, now: datetime) -> None:
        self._last_alert[camera_id] = now


def cooldown_from_config(alerts: AlertsConfig) -> Cooldowns:
    return Cooldowns(alerts.cooldown_minutes)
