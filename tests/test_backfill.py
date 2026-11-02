"""Backfill orchestration tests: deduped, logged, and provably silent."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import aresponses

from chewwatch.config import (
    AlertsConfig,
    ChewWatchConfig,
    ClassifierConfig,
    NtfyConfig,
    ProtectConfig,
    SamplingConfig,
    StorageConfig,
)
from chewwatch.models import BackfillEvent, DecisionRecord
from chewwatch.service import ChewWatch
from chewwatch.store import DecisionStore

KIMCHI_HOST = "llm.kimchi.dev"
JPEG = b"\xff\xd8thumbnail\xff\xd9"

_CHEWING = json.dumps(
    {
        "dog_present": True,
        "chewing": True,
        "object": "shoe",
        "confidence": 0.9,
        "reason": "Gnawing the shoe",
        "best_frame": 1,
    }
)


class FakeBackfillListener:
    """Canned Protect history + thumbnails; live-pipeline methods unused."""

    def __init__(self, events, thumbnails):
        self.events = events
        self.thumbnails = thumbnails

    def is_counter_camera(self, camera_id: str) -> bool:
        return False

    async def fetch_backfill_events(self, hours: float):
        return self.events

    async def fetch_event_thumbnail(self, event_id: str):
        return self.thumbnails.get(event_id)

    async def sample_frames(self, camera_id: str):
        return []

    def camera_name(self, camera_id: str) -> str:
        return "Living Room"


def _event(eid: str) -> BackfillEvent:
    return BackfillEvent(
        event_id=eid,
        camera_id="cam-1",
        camera_name="Living Room",
        started_at=datetime.now(UTC),
        score=80,
    )


def _service(tmp_path, events, thumbnails) -> ChewWatch:
    config = ChewWatchConfig(
        protect=ProtectConfig(
            host="protect.local", api_key="pk", username="u", password="p"
        ),
        classifier=ClassifierConfig(api_key="kk", threshold=0.7, ignore_toys=True),
        sampling=SamplingConfig(count=3, window_seconds=0.03),
        alerts=AlertsConfig(
            cooldown_minutes=8,
            ntfy=NtfyConfig(server="https://ntfy.example.com", topic="t"),
        ),
        storage=StorageConfig(data_dir=str(tmp_path)),
    )
    service = ChewWatch(config)
    service._listener = FakeBackfillListener(events, thumbnails)  # noqa: SLF001
    return service


def _mock_kimchi(aresponses, repeat: int = 1) -> None:
    aresponses.add(
        KIMCHI_HOST,
        "/openai/v1/chat/completions",
        "POST",
        aresponses.Response(
            text=json.dumps({"choices": [{"message": {"content": _CHEWING}}]}),
            content_type="application/json",
        ),
        repeat=repeat,
    )


async def test_backfill_classifies_and_logs_silently(tmp_path, aresponses):
    _mock_kimchi(aresponses)
    service = _service(tmp_path, [_event("new-evt")], {"new-evt": JPEG})
    # no ntfy route registered — any PUT would fail the test

    summary = await service._run_backfill(24)

    assert summary == {
        "hours": 24,
        "found": 1,
        "processed": 1,
        "skipped": 0,
        "failed": 0,
    }
    records = DecisionStore.load_records(tmp_path / "decisions.jsonl")
    assert len(records) == 1
    record = records[0]
    assert record["event_id"] == "new-evt"
    assert record["backfill"] is True
    assert record["alerted"] is False
    assert record["outcome"] == "alert"  # classified, but never notified
    assert record["detail"].startswith("Backfill — ")
    assert record["verdict"]["chewing"] is True


async def test_backfill_is_idempotent(tmp_path, aresponses):
    _mock_kimchi(aresponses, repeat=2)
    service = _service(tmp_path, [_event("new-evt")], {"new-evt": JPEG})

    first = await service._run_backfill(24)
    second = await service._run_backfill(24)

    assert first["processed"] == 1
    assert second["skipped"] == 1
    assert second["processed"] == 0
    records = DecisionStore.load_records(tmp_path / "decisions.jsonl")
    assert len(records) == 1


async def test_backfill_missing_thumbnail_logged(tmp_path):
    service = _service(tmp_path, [_event("no-thumb")], {})

    summary = await service._run_backfill(24)

    assert summary["processed"] == 1  # handled, not failed
    records = DecisionStore.load_records(tmp_path / "decisions.jsonl")
    assert records[0]["outcome"] == "no_frames"
    assert records[0]["backfill"] is True
    assert records[0]["alerted"] is False


async def test_backfill_skips_known_event_ids(tmp_path):
    service = _service(tmp_path, [_event("new-evt")], {"new-evt": JPEG})
    service.store.prepare()
    service.store.save(
        DecisionRecord(
            id="seed",
            event_id="new-evt",
            camera_id="cam-1",
            camera_name="Living Room",
            started_at=datetime.now(UTC),
            decided_at=datetime.now(UTC),
            outcome="no_chewing",
        )
    )

    summary = await service._run_backfill(24)

    assert summary == {
        "hours": 24,
        "found": 1,
        "processed": 0,
        "skipped": 1,
        "failed": 0,
    }


async def test_backfill_status_and_single_run_guard(tmp_path, aresponses):
    _mock_kimchi(aresponses)
    service = _service(tmp_path, [_event("new-evt")], {"new-evt": JPEG})

    assert service.backfill_status() == {"running": False, "last_result": None}
    assert service.start_backfill(24) is True
    assert service.start_backfill(24) is False  # one at a time
    await service._backfill_task
    status = service.backfill_status()
    assert status["running"] is False
    assert status["last_result"]["processed"] == 1


async def test_backfill_classifier_error_recorded(tmp_path, aresponses):
    aresponses.add(
        KIMCHI_HOST,
        "/openai/v1/chat/completions",
        "POST",
        aresponses.Response(text="boom", status=500),
    )
    service = _service(tmp_path, [_event("new-evt")], {"new-evt": JPEG})

    summary = await service._run_backfill(24)

    assert summary["processed"] == 1  # recorded, loop continued
    records = DecisionStore.load_records(tmp_path / "decisions.jsonl")
    assert records[0]["outcome"] == "classifier_error"
    assert records[0]["backfill"] is True
