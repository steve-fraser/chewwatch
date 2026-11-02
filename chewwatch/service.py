"""Pipeline orchestrator: sample → classify → decide → notify → log."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime

from .classifier import Classifier, ClassifierError
from .config import ChewWatchConfig
from .decision import Cooldowns, cooldown_from_config, evaluate
from .models import BackfillEvent, DecisionRecord, Detection, Frame, Verdict
from .notify import Notifier, NotifyError
from .protect import ProtectListener
from .store import DecisionStore

_LOGGER = logging.getLogger(__name__)


class ChewWatch:
    """Wires the components together and runs the per-event pipeline."""

    def __init__(self, config: ChewWatchConfig) -> None:
        self._config = config
        self._store = DecisionStore(config.storage.data_dir)
        self._classifier = Classifier(config.classifier)
        self._notifier = Notifier(config.alerts.ntfy)
        self._cooldowns: Cooldowns = cooldown_from_config(config.alerts)
        self._listener = ProtectListener(config, self._handle_detection)
        self._backfill_task: asyncio.Task[dict] | None = None
        self._backfill_result: dict | None = None

    @property
    def store(self) -> DecisionStore:
        """The decision store (read access, e.g. for the viewer UI)."""
        return self._store

    async def run(self) -> None:
        """Run until cancelled/stopped (SIGINT handled by the entrypoint)."""
        self._store.prepare()
        self._store.prune(self._config.storage.retention_days)
        _LOGGER.info(
            "ChewWatch starting — model %s, threshold %.2f, cooldown %.0f min",
            self._config.classifier.model,
            self._config.classifier.threshold,
            self._config.alerts.cooldown_minutes,
        )
        try:
            await self._listener.start()
            await self._listener.run_until_stopped()
        finally:
            # runs even when start() fails (bad auth, console unreachable) so
            # the Protect client's aiohttp session is always closed
            await self._listener.stop()

    def request_stop(self) -> None:
        """Ask the service to shut down (safe from signal handlers)."""
        self._listener.request_stop()

    # -------------------------------------------------------------- backfill

    #: maximum lookback for a backfill pull (hours)
    BACKFILL_MAX_HOURS = 168.0

    def backfill_ready(self) -> bool:
        """Whether a backfill pull is currently possible (hybrid + connected)."""
        return self._listener.can_backfill()

    def start_backfill(self, hours: float) -> bool:
        """Kick off a background backfill; False when one is already running."""
        if self._backfill_task is not None and not self._backfill_task.done():
            return False
        hours = min(max(float(hours), 0.01), self.BACKFILL_MAX_HOURS)
        self._backfill_result = None
        self._backfill_task = asyncio.get_running_loop().create_task(
            self._run_backfill(hours)
        )
        return True

    def backfill_status(self) -> dict:
        running = self._backfill_task is not None and not self._backfill_task.done()
        return {"running": running, "last_result": self._backfill_result}

    async def _run_backfill(self, hours: float) -> dict:
        """Classify missed animal events from Protect history.

        Deliberately silent: backfilled records never alert, never notify and
        never touch the per-camera cooldowns.
        """
        summary = {
            "hours": hours,
            "found": 0,
            "processed": 0,
            "skipped": 0,
            "failed": 0,
        }
        try:
            candidates = await self._listener.fetch_backfill_events(hours)
        except Exception as err:  # noqa: BLE001 - surfaced to the UI
            summary["error"] = str(err)
            self._backfill_result = summary
            _LOGGER.error("Backfill failed: %s", err)
            return summary
        summary["found"] = len(candidates)

        known = {str(r.get("event_id")) for r in self._store.load_records(self._store.log_path)}
        for event in candidates:
            if event.event_id in known:
                summary["skipped"] += 1
                continue
            try:
                await self._backfill_one(event)
                summary["processed"] += 1
            except Exception:  # noqa: BLE001 - keep pulling remaining events
                summary["failed"] += 1
                _LOGGER.exception("Backfill failed for event %s", event.event_id)
        _LOGGER.info(
            "Backfill done (%.1f h): %d found, %d processed, %d skipped, %d failed",
            hours,
            summary["found"],
            summary["processed"],
            summary["skipped"],
            summary["failed"],
        )
        self._backfill_result = summary
        return summary

    async def _backfill_one(self, event: BackfillEvent) -> None:
        """Classify one historical event from its Protect thumbnail and log it."""
        now = datetime.now(UTC)
        record = DecisionRecord(
            id=uuid.uuid4().hex[:12],
            event_id=event.event_id,
            camera_id=event.camera_id,
            camera_name=event.camera_name,
            started_at=event.started_at,
            decided_at=now,
            backfill=True,  # never alerts, never consumes cooldown
        )
        try:
            thumbnail = await self._listener.fetch_event_thumbnail(event.event_id)
            if not thumbnail:
                record.outcome = "no_frames"
                record.detail = "Backfill — Protect thumbnail unavailable"
                _LOGGER.warning("No thumbnail for backfill event %s", event.event_id)
                return

            frames = [Frame(index=1, captured_at=event.started_at, jpeg=thumbnail)]
            record.frames = frames
            frame_dir = self._store.save_frames(event.camera_name, now, frames)
            record.frame_dir = str(frame_dir)

            verdict = await self._classifier.classify(
                frames,
                event.camera_name,
                check_counter=self._listener.is_counter_camera(event.camera_id),
            )
            record.verdict = verdict
            decision = evaluate(verdict, self._config.classifier)
            record.outcome = decision.outcome
            record.detail = f"Backfill — {decision.detail}"
            _LOGGER.info(
                "Backfill %s on %s: %s — %s",
                event.event_id,
                event.camera_name,
                decision.outcome,
                decision.detail,
            )
        except ClassifierError as err:
            record.outcome = "classifier_error"
            record.error = str(err)
            record.detail = "Backfill — classifier failed"
            _LOGGER.error("Classifier failed for backfill event %s: %s", event.event_id, err)
        finally:
            record.decided_at = datetime.now(UTC)
            self._store.save(record)
            _LOGGER.info(
                "Backfill decision %s: %s (%s)",
                record.id,
                record.outcome,
                record.detail or record.error or "-",
            )

    # ------------------------------------------------------------- pipeline

    async def _handle_detection(self, detection: Detection) -> None:
        """Full pipeline for one animal event; every outcome is recorded."""
        now = datetime.now(UTC)
        record = DecisionRecord(
            id=uuid.uuid4().hex[:12],
            event_id=detection.event_id,
            camera_id=detection.camera_id,
            camera_name=detection.camera_name,
            started_at=detection.started_at,
            decided_at=now,
        )
        _LOGGER.info(
            "Animal event on %s (%s) — sampling %d frame(s) over %.0fs",
            detection.camera_name,
            detection.event_id,
            self._config.sampling.count,
            self._config.sampling.window_seconds,
        )
        try:
            frames = await self._listener.sample_frames(detection.camera_id)
            record.frames = frames
            if not frames:
                record.outcome = "no_frames"
                record.detail = "All snapshot requests failed or returned nothing"
                _LOGGER.warning("No frames captured for event %s", detection.event_id)
                return

            frame_dir = self._store.save_frames(detection.camera_name, now, frames)
            record.frame_dir = str(frame_dir)

            verdict = await self._classifier.classify(
                frames, detection.camera_name, check_counter=detection.check_counter
            )
            record.verdict = verdict
            _LOGGER.info(
                "Verdict on %s: dog=%s chewing=%s object=%s on_counter=%s "
                "confidence=%.2f/%.2f — %s",
                detection.camera_name,
                verdict.dog_present,
                verdict.chewing,
                verdict.object,
                verdict.on_counter,
                verdict.confidence,
                verdict.counter_confidence,
                verdict.reason,
            )

            decision = evaluate(verdict, self._config.classifier)
            record.outcome = decision.outcome
            record.detail = decision.detail

            if decision.alert:
                await self._alert(record, detection, verdict, frames)
        except ClassifierError as err:
            record.outcome = "classifier_error"
            record.error = str(err)
            _LOGGER.error("Classifier failed for %s: %s", detection.camera_name, err)
        except NotifyError as err:
            record.outcome = "notify_error"
            record.error = str(err)
            _LOGGER.error("ntfy push failed for %s: %s", detection.camera_name, err)
        except asyncio.CancelledError:
            record.outcome = "cancelled"
            raise
        except Exception:  # noqa: BLE001 - record and survive
            record.outcome = "error"
            record.error = "unexpected pipeline failure"
            _LOGGER.exception("Unexpected pipeline error for %s", detection.camera_name)
        finally:
            record.decided_at = datetime.now(UTC)
            self._store.save(record)
            _LOGGER.info(
                "Decision %s: %s (%s)",
                record.id,
                record.outcome,
                record.detail or record.error or "-",
            )

    async def _alert(
        self,
        record: DecisionRecord,
        detection: Detection,
        verdict: Verdict,
        frames: list[Frame],
    ) -> None:
        now = datetime.now(UTC)
        if not self._cooldowns.allow(detection.camera_id, now):
            remaining = self._cooldowns.seconds_remaining(detection.camera_id, now)
            record.outcome = "cooldown"
            record.detail = (
                f"Chewing detected but camera is in cooldown "
                f"({remaining / 60:.1f} min left) — already alerted this session"
            )
            _LOGGER.info(
                "Suppressing duplicate alert for %s: cooldown has %.1f min left",
                detection.camera_name,
                remaining / 60,
            )
            return

        best = next((f for f in frames if f.index == verdict.best_frame), frames[-1])
        await self._notifier.send_alert(
            camera_name=detection.camera_name,
            verdict=verdict,
            jpeg=best.jpeg,
            at=now,
        )
        # Mark only after a successful push, so a failed push can retry.
        self._cooldowns.mark(detection.camera_id, now)
        record.alerted = True
        _LOGGER.info(
            "ALERT sent for %s: chewing %s (%.0f%%)",
            detection.camera_name,
            verdict.object,
            verdict.confidence * 100,
        )
