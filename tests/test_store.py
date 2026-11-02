"""Tests for the decision store: frames, JSONL roundtrip, pruning."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from chewwatch.models import DecisionRecord, Frame, Verdict
from chewwatch.store import DecisionStore


def _frames(n: int = 2) -> list[Frame]:
    return [
        Frame(index=i + 1, captured_at=datetime.now(UTC), jpeg=b"jpg-%d" % i)
        for i in range(n)
    ]


def _record(**overrides) -> DecisionRecord:
    defaults = dict(
        id="abc123",
        event_id="evt-1",
        camera_id="cam-1",
        camera_name="Living Room",
        started_at=datetime(2026, 1, 1, 10, 0, tzinfo=UTC),
        decided_at=datetime(2026, 1, 1, 10, 0, 30, tzinfo=UTC),
        frames=_frames(),
        verdict=Verdict(
            dog_present=True,
            chewing=True,
            object="shoe",
            confidence=0.8,
            reason="Gnawing",
            best_frame=1,
            raw="{}",
        ),
        alerted=True,
        outcome="alert",
        detail="Chewing detected: shoe",
    )
    defaults.update(overrides)
    return DecisionRecord(**defaults)


def test_save_frames(tmp_path):
    store = DecisionStore(tmp_path)
    at = datetime(2026, 1, 1, 10, 15, 30, tzinfo=UTC)
    folder = store.save_frames("Living Room", at, _frames(3))
    assert folder.is_dir()
    assert folder.parent == tmp_path / "frames"
    assert folder.name == "20260101-101530-living-room"
    assert (folder / "frame-1.jpg").read_bytes() == b"jpg-0"
    assert (folder / "frame-2.jpg").read_bytes() == b"jpg-1"
    assert (folder / "frame-3.jpg").read_bytes() == b"jpg-2"


def test_save_frames_slugifies_camera_name(tmp_path):
    store = DecisionStore(tmp_path)
    folder = store.save_frames("Back Yard / Deck!", datetime.now(UTC), _frames(1))
    assert folder.name.startswith(datetime.now(UTC).strftime("%Y%m%d"))
    assert "deck" in folder.name
    assert "/" not in folder.name.split("-", 2)[-1]


def test_jsonl_roundtrip(tmp_path):
    store = DecisionStore(tmp_path)
    record = _record(frame_dir=str(tmp_path / "frames" / "x"))
    store.save(record)
    store.save(_record(id="def456", alerted=False, outcome="no_chewing", verdict=None))

    records = DecisionStore.load_records(store.log_path)
    assert len(records) == 2
    assert records[0]["id"] == "abc123"
    assert records[0]["verdict"]["object"] == "shoe"
    assert records[1]["verdict"] is None
    assert records[1]["alerted"] is False
    assert records[0]["backfill"] is False  # live pipeline default


def test_backfill_flag_roundtrip(tmp_path):
    store = DecisionStore(tmp_path)
    record = _record(alerted=False, outcome="alert", backfill=True,
                     detail="Backfill — Chewing detected: shoe")
    store.save(record)
    records = DecisionStore.load_records(store.log_path)
    assert records[0]["backfill"] is True
    assert records[0]["alerted"] is False
    # every line is valid standalone JSON
    line = store.log_path.read_text().splitlines()[0]
    assert json.loads(line)["camera_name"] == "Living Room"


def test_prune_old_folders(tmp_path):
    store = DecisionStore(tmp_path)
    old = store.save_frames("Yard", datetime.now(UTC) - timedelta(days=40), _frames(1))
    # backdate the folder so pruning sees it as old
    stamp = datetime.now(UTC).timestamp() - 40 * 86400
    import os

    os.utime(old, (stamp, stamp))
    fresh = store.save_frames("Yard", datetime.now(UTC), _frames(1))

    removed = store.prune(30, now=datetime.now(UTC))
    assert removed == 1
    assert not old.exists()
    assert fresh.exists()


def test_prune_zero_retention_keeps_everything(tmp_path):
    store = DecisionStore(tmp_path)
    folder = store.save_frames("Yard", datetime.now(UTC), _frames(1))
    assert store.prune(0) == 0
    assert folder.exists()


def test_load_records_missing_file(tmp_path):
    assert DecisionStore.load_records(tmp_path / "nope.jsonl") == []


def test_load_records_skips_corrupt_lines(tmp_path):
    store = DecisionStore(tmp_path)
    store.save(_record())
    with store.log_path.open("a", encoding="utf-8") as fh:
        fh.write("{corrupt json\n")
    records = DecisionStore.load_records(store.log_path)
    assert len(records) == 1
