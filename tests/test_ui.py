"""Tests for the read-only viewer UI: routes, filtering, safe downloads."""

from __future__ import annotations

import io
import json
import zipfile
from datetime import UTC, datetime, timedelta

from aiohttp.test_utils import TestClient, TestServer

from chewwatch.models import DecisionRecord, Frame, Verdict
from chewwatch.store import DecisionStore
from chewwatch.ui import build_app


def _store(tmp_path) -> DecisionStore:
    store = DecisionStore(tmp_path)
    store.prepare()
    return store


def _seed(
    store: DecisionStore,
    *,
    camera: str = "Living Room",
    outcome: str = "alert",
    at: datetime | None = None,
    frames: int = 2,
) -> DecisionRecord:
    at = at or datetime.now(UTC)
    record = DecisionRecord(
        id=f"rec-{at.strftime('%H%M%S%f')}",
        event_id=f"evt-{at.strftime('%H%M%S%f')}",
        camera_id="cam-1",
        camera_name=camera,
        started_at=at,
        decided_at=at + timedelta(seconds=11),
        outcome=outcome,
        detail="Chewing detected: couch (confidence 0.90)",
        verdict=Verdict(
            dog_present=True,
            chewing=outcome == "alert",
            object="couch" if outcome == "alert" else "none",
            confidence=0.9,
            reason="Mouth working the cushion",
            best_frame=1,
        ),
    )
    if frames:
        sampled = [
            Frame(index=i + 1, captured_at=at, jpeg=b"jpeg-%d" % i) for i in range(frames)
        ]
        record.frames = sampled
        record.frame_dir = str(store.save_frames(camera, at, sampled))
    store.save(record)
    return record


async def _request(store: DecisionStore, path: str):
    app = build_app(store)
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    try:
        resp = await client.get(path)
        body = await resp.read()
        return resp.status, body, dict(resp.headers)
    finally:
        await client.close()


async def test_index_renders(tmp_path):
    status, body, headers = await _request(_store(tmp_path), "/")
    assert status == 200
    assert headers["Content-Type"].startswith("text/html")
    assert b"ChewWatch" in body
    assert b"/api/events" in body


async def test_api_window_default_is_three_hours(tmp_path):
    store = _store(tmp_path)
    _seed(store)  # now
    _seed(store, at=datetime.now(UTC) - timedelta(hours=4))  # outside default window

    status, body, _ = await _request(store, "/api/events")
    assert status == 200
    data = __import__("json").loads(body)
    assert data["window_hours"] == 3
    assert len(data["events"]) == 1
    event = data["events"][0]
    assert event["camera_name"] == "Living Room"
    assert event["folder"].startswith("2")
    assert event["frames"] == ["frame-1.jpg", "frame-2.jpg"]
    assert event["verdict"]["on_counter"] is False
    assert data["cameras"] == ["Living Room"]


async def test_api_camera_and_outcome_filters(tmp_path):
    store = _store(tmp_path)
    _seed(store, camera="Living Room", outcome="alert")
    _seed(store, camera="Kitchen", outcome="no_chewing")

    _, body, _ = await _request(store, "/api/events?camera=Kitchen")
    data = __import__("json").loads(body)
    assert [e["camera_name"] for e in data["events"]] == ["Kitchen"]

    _, body, _ = await _request(store, "/api/events?outcome=alert")
    data = __import__("json").loads(body)
    assert [e["outcome"] for e in data["events"]] == ["alert"]


async def test_api_wide_window_includes_pruned_frames(tmp_path):
    store = _store(tmp_path)
    record = _seed(store, at=datetime.now(UTC) - timedelta(days=5))
    import shutil

    shutil.rmtree(tmp_path / "frames" / record.frame_dir.split("/")[-1].split("\\")[-1])

    _, body, _ = await _request(store, "/api/events?hours=168")
    data = __import__("json").loads(body)
    assert len(data["events"]) == 1
    assert data["events"][0]["frames"] == []  # folder pruned, event still listed


async def test_frame_serving_and_download(tmp_path):
    store = _store(tmp_path)
    record = _seed(store)
    folder = record.frame_dir.replace("\\", "/").split("/")[-1]

    status, body, headers = await _request(store, f"/frames/{folder}/frame-1.jpg")
    assert status == 200
    assert headers["Content-Type"] == "image/jpeg"
    assert body == b"jpeg-0"

    _, _, headers = await _request(store, f"/frames/{folder}/frame-1.jpg?download=1")
    assert headers["Content-Disposition"] == 'attachment; filename="frame-1.jpg"'


async def test_zip_download(tmp_path):
    store = _store(tmp_path)
    record = _seed(store, frames=3)
    folder = record.frame_dir.replace("\\", "/").split("/")[-1]

    status, body, headers = await _request(store, f"/frames/{folder}/all.zip")
    assert status == 200
    assert headers["Content-Type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(body)) as zf:
        assert zf.namelist() == [
            f"{folder}/frame-1.jpg",
            f"{folder}/frame-2.jpg",
            f"{folder}/frame-3.jpg",
        ]
        assert zf.read(f"{folder}/frame-2.jpg") == b"jpeg-1"


async def test_bad_paths_rejected(tmp_path):
    store = _store(tmp_path)
    record = _seed(store)
    folder = record.frame_dir.replace("\\", "/").split("/")[-1]

    for path in (
        "/frames/not-a-folder/frame-1.jpg",
        f"/frames/{folder}/frame-999.jpg",
        f"/frames/{folder}/../secrets/frame-1.jpg",
        "/frames/..%2F..%2Fetc/frame-1.jpg",
        f"/frames/{folder}/nope.txt",
    ):
        status, _, _ = await _request(store, path)
        assert status == 404, f"expected 404 for {path}"
    # direct unit check on the validator
    from chewwatch.ui import _frames_dir

    assert _frames_dir(store, "../..") is None
    assert _frames_dir(store, f"{folder}/../..") is None
    assert _frames_dir(store, folder) is not None


async def test_empty_store(tmp_path):
    status, body, _ = await _request(_store(tmp_path), "/api/events")
    assert status == 200
    data = json.loads(body)
    assert data["events"] == []
    assert data["cameras"] == []
    assert data["outcomes"] == []


# ------------------------------------------------------------------ backfill


class FakeBackfill:
    """Duck-typed stand-in for the service's backfill surface."""

    def __init__(self, ready: bool = True, busy: bool = False) -> None:
        self.ready = ready
        self.busy = busy
        self.started: list[float] = []

    def backfill_ready(self) -> bool:
        return self.ready

    def start_backfill(self, hours: float) -> bool:
        if self.busy:
            return False
        self.started.append(hours)
        return True

    def backfill_status(self) -> dict:
        return {"running": self.busy, "last_result": None}


async def _post(store, path, payload, backfill=None):
    app = build_app(store, backfill=backfill)
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    try:
        resp = await client.post(path, json=payload)
        body = await resp.read()
        return resp.status, body, dict(resp.headers)
    finally:
        await client.close()


async def test_backfill_post_starts(tmp_path):
    store = _store(tmp_path)
    fake = FakeBackfill()
    status, body, _ = await _post(store, "/api/backfill", {"hours": 24}, backfill=fake)
    assert status == 202
    assert json.loads(body) == {"started": True}
    assert fake.started == [24.0]


async def test_backfill_post_busy(tmp_path):
    store = _store(tmp_path)
    status, _, _ = await _post(
        store, "/api/backfill", {"hours": 3}, backfill=FakeBackfill(busy=True)
    )
    assert status == 409


async def test_backfill_post_requires_hybrid(tmp_path):
    store = _store(tmp_path)
    status, body, _ = await _post(
        store, "/api/backfill", {"hours": 3}, backfill=FakeBackfill(ready=False)
    )
    assert status == 400
    assert b"hybrid auth" in body


async def test_backfill_post_bad_hours(tmp_path):
    store = _store(tmp_path)
    status, _, _ = await _post(
        store, "/api/backfill", {"hours": "soon"}, backfill=FakeBackfill()
    )
    assert status == 400


async def test_backfill_post_without_service(tmp_path):
    store = _store(tmp_path)
    status, _, _ = await _post(store, "/api/backfill", {"hours": 3})
    assert status == 503


async def test_backfill_status_endpoint(tmp_path):
    store = _store(tmp_path)
    status, body, _ = await _request(store, "/api/backfill")
    assert status == 200
    assert json.loads(body) == {"running": False, "last_result": None}
