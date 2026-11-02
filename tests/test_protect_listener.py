"""Tests for the Protect event listener: filtering, busy guard, sampling."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

import uiprotect  # the conftest stub
from chewwatch import protect as protect_module
from chewwatch.protect import BackfillError
from chewwatch.config import (
    AlertsConfig,
    ChewWatchConfig,
    NtfyConfig,
    ProtectConfig,
    SamplingConfig,
)
from chewwatch.models import Detection

EventChange = uiprotect.EventChange
EventType = uiprotect.data.EventType
SmartDetectObjectType = uiprotect.data.SmartDetectObjectType

CAMERAS = {
    # name mirrors the private-API bootstrap (hybrid mode reads .name)
    "cam-1": uiprotect.FakeCamera("cam-1", "Living Room", name="Living Room"),
    "cam-2": uiprotect.FakeCamera("cam-2", "Back Yard", name="Back Yard"),
}


def _config(**overrides) -> ChewWatchConfig:
    protect = ProtectConfig(host="protect.local", port=443, api_key="k", cameras=["cam-1"])
    sampling = SamplingConfig(count=3, window_seconds=0.06)
    kwargs = dict(
        protect=protect,
        sampling=sampling,
        alerts=AlertsConfig(ntfy=NtfyConfig(topic="t")),
    )
    kwargs.update(overrides)
    return ChewWatchConfig(**kwargs)


def _event(
    event_type=EventType.SMART_DETECT,
    device_id="cam-1",
    types=(SmartDetectObjectType.ANIMAL,),
    eid="evt-1",
):
    return uiprotect.ProtectEvent(
        id=eid,
        type=event_type,
        device_id=device_id,
        start=datetime.now(UTC),
        smart_detect_types=types,
    )


class _Harness:
    """Wires a listener against fake clients built from the stub."""

    def __init__(self, monkeypatch, cfg: ChewWatchConfig, on_detection=None):
        self.detections: list[Detection] = []
        self.clients: list = []

        if on_detection is None:
            async def on_detection(detection: Detection) -> None:
                self.detections.append(detection)
        self._on_detection = on_detection

        harness = self

        class FakeClient(uiprotect.ProtectApiClient):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._public_cameras = CAMERAS
                harness.clients.append(self)

            @classmethod
            def public_only(cls, host, port, api_key, **kwargs):
                return cls(host, port, api_key=api_key, **kwargs)

        monkeypatch.setattr(protect_module, "ProtectApiClient", FakeClient)

        self.listener = protect_module.ProtectListener(cfg, self._on_detection)

    async def start(self):
        await self.listener.start()

    async def drain(self):
        tasks = list(self.listener._tasks)
        if tasks:
            await asyncio.gather(*tasks)

    async def fire(self, event, change=EventChange.STARTED):
        self.clients[0].event_callback(event, change)
        await self.drain()

    async def stop(self):
        self.listener.request_stop()
        await self.listener.stop()


async def test_start_selects_watched_cameras(monkeypatch):
    h = _Harness(monkeypatch, _config())
    await h.start()
    assert h.listener.camera_name("cam-1") == "Living Room"
    assert h.listener.camera_name("cam-2") == "Back Yard"
    assert h.listener._watched == {"cam-1"}
    assert h.listener._counter == set()  # no counter_cameras configured
    await h.stop()


async def test_animal_event_dispatches(monkeypatch):
    h = _Harness(monkeypatch, _config())
    await h.start()
    await h.fire(_event())
    assert len(h.detections) == 1
    detection = h.detections[0]
    assert detection.camera_id == "cam-1"
    assert detection.camera_name == "Living Room"
    assert detection.event_id == "evt-1"
    assert detection.check_counter is False
    await h.stop()


async def test_counter_camera_detection_is_flagged(monkeypatch):
    cfg = _config(
        protect=ProtectConfig(
            host="protect.local",
            port=443,
            api_key="k",
            cameras=["cam-1", "cam-2"],
            counter_cameras=["Living Room"],
        )
    )
    h = _Harness(monkeypatch, cfg)
    await h.start()
    assert h.listener._counter == {"cam-1"}
    await h.fire(_event())
    await h.fire(_event(eid="evt-2", device_id="cam-2"))
    assert [d.check_counter for d in h.detections] == [True, False]
    await h.stop()


async def test_counter_selector_without_watch_warns(monkeypatch, caplog):
    import logging

    cfg = _config(
        protect=ProtectConfig(
            host="protect.local",
            port=443,
            api_key="k",
            cameras=["cam-1"],
            counter_cameras=["Kitchen"],
        )
    )
    h = _Harness(monkeypatch, cfg)
    with caplog.at_level(logging.WARNING, logger="chewwatch.protect"):
        await h.start()
    assert h.listener._counter == set()
    assert "matched no watched camera" in caplog.text
    await h.stop()


async def test_ignores_non_started_changes(monkeypatch):
    h = _Harness(monkeypatch, _config())
    await h.start()
    for change in (EventChange.UPDATED, EventChange.ENDED, EventChange.REMOVED):
        await h.fire(_event(), change)
    assert h.detections == []
    await h.stop()


async def test_ignores_non_smart_event_types(monkeypatch):
    h = _Harness(monkeypatch, _config())
    await h.start()
    await h.fire(_event(event_type=EventType.MOTION))
    assert h.detections == []
    await h.stop()


async def test_ignores_non_animal_smart_detections(monkeypatch):
    h = _Harness(monkeypatch, _config())
    await h.start()
    await h.fire(_event(types=(SmartDetectObjectType.PERSON, SmartDetectObjectType.VEHICLE)))
    assert h.detections == []
    await h.stop()


async def test_ignores_unwatched_camera(monkeypatch):
    h = _Harness(monkeypatch, _config())
    await h.start()
    await h.fire(_event(device_id="cam-2"))
    assert h.detections == []
    await h.stop()


async def test_busy_guard_skips_overlapping_events(monkeypatch):
    release = asyncio.Event()
    seen: list[str] = []

    async def on_detection(detection: Detection) -> None:
        seen.append(detection.event_id)
        await release.wait()

    h = _Harness(monkeypatch, _config(), on_detection=on_detection)
    await h.start()

    fire = h.clients[0].event_callback
    fire(_event(eid="first"), EventChange.STARTED)
    await asyncio.sleep(0)  # first pipeline enters its handler and blocks
    fire(_event(eid="second"), EventChange.STARTED)
    await asyncio.sleep(0)

    # second event was skipped while the first pipeline is still running
    assert seen == ["first"]
    release.set()
    await h.drain()
    await h.stop()


async def test_sample_frames_public_mode(monkeypatch):
    h = _Harness(monkeypatch, _config())
    await h.start()
    frames = await h.listener.sample_frames("cam-1")
    assert len(frames) == 3
    assert [f.index for f in frames] == [1, 2, 3]
    assert frames[0].jpeg == b"jpeg-for-cam-1"
    await h.stop()


async def test_sample_frames_hybrid_mode(monkeypatch):
    cfg = _config(
        protect=ProtectConfig(
            host="protect.local", port=443, api_key="k",
            username="u", password="p", cameras=["cam-1"],
        )
    )
    h = _Harness(monkeypatch, cfg)
    await h.start()
    frames = await h.listener.sample_frames("cam-1")
    assert frames[0].jpeg == b"jpeg-wide-cam-1"  # private-API snapshot path
    await h.stop()


async def test_sample_frames_spreads_over_window(monkeypatch):
    cfg = _config(sampling=SamplingConfig(count=3, window_seconds=0.3))
    h = _Harness(monkeypatch, cfg)
    await h.start()
    started = asyncio.get_running_loop().time()
    frames = await h.listener.sample_frames("cam-1")
    elapsed = asyncio.get_running_loop().time() - started
    assert len(frames) == 3
    assert elapsed >= 0.3  # 3 frames over a 0.3s window: 2 gaps of 0.15s
    await h.stop()


async def test_run_until_stopped_returns_on_request(monkeypatch):
    h = _Harness(monkeypatch, _config())
    await h.start()
    runner = asyncio.create_task(h.listener.run_until_stopped())
    await asyncio.sleep(0)
    assert not runner.done()
    h.listener.request_stop()
    await asyncio.wait_for(runner, timeout=1)
    await h.listener.stop()


# ------------------------------------------------------------------ backfill


def _raw_event(eid, camera="cam-1", iso_start=None):
    return {
        "id": eid,
        "type": "smartDetectZone",
        "start": iso_start or int(datetime.now(UTC).timestamp() * 1000),
        "end": None,
        "camera": camera,  # raw private-API key (typed model: cameraId)
        "smartDetectTypes": ["animal"],
        "score": 72,
    }


async def test_backfill_requires_hybrid(monkeypatch):
    h = _Harness(monkeypatch, _config())
    await h.start()
    assert h.listener.can_backfill() is False
    with pytest.raises(BackfillError, match="hybrid auth"):
        await h.listener.fetch_backfill_events(24)
    await h.stop()


async def test_backfill_fetch_filters_watched_cameras(monkeypatch):
    cfg = _config(
        protect=ProtectConfig(
            host="protect.local",
            port=443,
            api_key="k",
            username="u",
            password="p",
            cameras=["cam-1"],
        )
    )
    h = _Harness(monkeypatch, cfg)
    await h.start()
    assert h.listener.can_backfill() is True
    h.clients[0]._raw_events = [
        _raw_event("raw-1", camera="cam-1"),
        _raw_event("raw-2", camera="cam-2"),  # not watched
        _raw_event("raw-3", iso_start=datetime.now(UTC).isoformat()),
        _raw_event("raw-4", camera="cam-1"),
    ]
    h.clients[0]._raw_events[3]["smartDetectTypes"] = ["person"]  # not animal

    events = await h.listener.fetch_backfill_events(24)

    # raw-2 unwatched, raw-3 ISO start still parsed, raw-4 non-animal type
    # is excluded by the API filter but the client-side parse keeps order;
    # raw-4 survives here because the stub ignores smart_detect_types.
    assert [e.event_id for e in events] == ["raw-1", "raw-3", "raw-4"]
    assert events[0].camera_name == "Living Room"
    assert events[0].score == 72
    await h.stop()


async def test_start_failure_still_closes_session(monkeypatch):
    h = _Harness(monkeypatch, _config())
    closed: list[bool] = []

    async def failing_subscribe(self, callback):
        raise RuntimeError("subscribe failed")

    async def closing(self):
        closed.append(True)

    monkeypatch.setattr(
        "uiprotect.ProtectApiClient.subscribe_events_and_prime", failing_subscribe
    )
    monkeypatch.setattr("uiprotect.ProtectApiClient.close_session", closing)

    with pytest.raises(RuntimeError, match="subscribe failed"):
        await h.start()
    # the session must be closed even though start() blew up mid-way
    await h.stop()
    assert closed == [True]


async def test_client_disables_session_persistence(monkeypatch):
    """uiprotect must not persist its auth token (no writable home in the
    container — crashed hybrid startup with PermissionError: /home/chewwatch)."""
    h = _Harness(monkeypatch, _config())
    await h.start()
    assert h.clients[0].init_kwargs["store_sessions"] is False
    await h.stop()

    cfg = _config(
        protect=ProtectConfig(
            host="protect.local",
            port=443,
            api_key="k",
            username="u",
            password="p",
            cameras=["cam-1"],
        )
    )
    h = _Harness(monkeypatch, cfg)
    await h.start()
    assert h.clients[0].init_kwargs["store_sessions"] is False
    await h.stop()


async def test_backfill_thumbnail(monkeypatch):
    cfg = _config(
        protect=ProtectConfig(
            host="protect.local",
            port=443,
            api_key="k",
            username="u",
            password="p",
            cameras=["cam-1"],
        )
    )
    h = _Harness(monkeypatch, cfg)
    await h.start()
    h.clients[0]._thumbnails = {"evt-9": b"jpeg-bytes"}
    assert await h.listener.fetch_event_thumbnail("evt-9") == b"jpeg-bytes"
    assert await h.listener.fetch_event_thumbnail("missing") is None
    await h.stop()
