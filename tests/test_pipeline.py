"""End-to-end pipeline tests with mocked Kimchi Inference and ntfy."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime, timedelta

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
from chewwatch.models import Detection, Frame
from chewwatch.service import ChewWatch
from chewwatch.store import DecisionStore

KIMCHI_HOST = "llm.kimchi.dev"
NTFY_HOST = "ntfy.example.com"
TOPIC = "chewwatch-pipeline-test"

JPEG = b"\xff\xd8\xe0fakejpegdata\xff\xd9"


class FakeListener:
    """Stands in for ProtectListener: hands out canned frames."""

    def __init__(self, frames: list[Frame]) -> None:
        self.frames = frames
        self.sample_calls: list[str] = []

    async def sample_frames(self, camera_id: str) -> list[Frame]:
        await asyncio.sleep(0)  # behave like the real async sampler
        self.sample_calls.append(camera_id)
        return self.frames

    def camera_name(self, camera_id: str) -> str:
        return "Living Room"


def _config(tmp_path) -> ChewWatchConfig:
    return ChewWatchConfig(
        protect=ProtectConfig(host="protect.local", api_key="pk"),
        classifier=ClassifierConfig(
            api_key="kk", model="glm-5.3-flash", threshold=0.7, ignore_toys=True
        ),
        sampling=SamplingConfig(count=3, window_seconds=0.03),
        alerts=AlertsConfig(
            cooldown_minutes=8, ntfy=NtfyConfig(server=f"https://{NTFY_HOST}", topic=TOPIC)
        ),
        storage=StorageConfig(data_dir=str(tmp_path)),
    )


def _service(tmp_path) -> tuple[ChewWatch, FakeListener]:
    service = ChewWatch(_config(tmp_path))
    fake = FakeListener(
        [Frame(index=i + 1, captured_at=datetime.now(UTC), jpeg=JPEG + bytes([i])) for i in range(3)]
    )
    service._listener = fake  # noqa: SLF001 - inject the fake sampler
    return service, fake


def _detection() -> Detection:
    return Detection(
        event_id="evt-42",
        camera_id="cam-1",
        camera_name="Living Room",
        started_at=datetime.now(UTC),
    )


def _kimchi_reply(content: str) -> str:
    return json.dumps({"choices": [{"message": {"content": content}}]})


def _kimchi_content(**fields) -> str:
    return json.dumps(
        {
            "dog_present": True,
            "chewing": True,
            "object": "couch",
            "confidence": 0.88,
            "reason": "Head down, mouth working the cushion",
            "best_frame": 2,
            **fields,
        }
    )


def _mock_kimchi(aresponses, content: str, repeat: int = 1) -> dict:
    captured = {}

    async def handler(request):
        captured["payload"] = json.loads(await request.text())
        return aresponses.Response(
            text=_kimchi_reply(content), content_type="application/json"
        )

    aresponses.add(
        KIMCHI_HOST, "/openai/v1/chat/completions", "POST", handler, repeat=repeat
    )
    return captured


def _mock_ntfy(aresponses) -> dict:
    captured = {}

    async def handler(request):
        captured["body"] = await request.read()
        captured["headers"] = {k.lower(): v for k, v in request.headers.items()}
        return aresponses.Response(text="{}", content_type="application/json")

    aresponses.add(NTFY_HOST, f"/{TOPIC}", "PUT", handler)
    return captured


async def test_full_alert_path(tmp_path, aresponses: aresponses.ResponsesMockServer):
    service, fake = _service(tmp_path)
    kimchi = _mock_kimchi(aresponses, _kimchi_content())
    ntfy = _mock_ntfy(aresponses)

    await service._handle_detection(_detection())

    # the classifier got one request with 3 images and temperature 0
    payload = kimchi["payload"]
    assert payload["model"] == "glm-5.3-flash"
    assert payload["temperature"] == 0.0
    assert len(payload["messages"][1]["content"]) == 4  # text + 3 frames
    assert "Living Room" in payload["messages"][0]["content"]
    assert "on_counter" not in payload["messages"][0]["content"]  # chewing-only prompt

    # ntfy got the *best* frame (frame 2) with the verdict in the title
    assert ntfy["body"] == JPEG + b"\x01"
    assert "chewing couch" in ntfy["headers"]["title"]
    assert ntfy["headers"]["filename"].startswith("chewwatch-")
    assert ntfy["headers"]["priority"] == "high"

    # frames saved to disk and the decision logged
    records = DecisionStore.load_records(tmp_path / "decisions.jsonl")
    assert len(records) == 1
    record = records[0]
    assert record["alerted"] is True
    assert record["outcome"] == "alert"
    assert record["verdict"]["object"] == "couch"
    assert record["verdict"]["on_counter"] is False
    assert record["verdict"]["counter_confidence"] == 0.88  # falls back to confidence
    frames_dir = tmp_path / record["frame_dir"]
    assert (frames_dir / "frame-1.jpg").exists()
    assert (frames_dir / "frame-3.jpg").exists()


async def test_toy_suppression_sends_no_notification(tmp_path, aresponses):
    service, _ = _service(tmp_path)
    _mock_kimchi(aresponses, _kimchi_content(object="toy", best_frame=1))
    # no ntfy route registered — any PUT would fail the test

    await service._handle_detection(_detection())

    records = DecisionStore.load_records(tmp_path / "decisions.jsonl")
    assert len(records) == 1
    assert records[0]["outcome"] == "toy_or_bone"
    assert records[0]["alerted"] is False


async def test_cooldown_suppresses_second_alert(tmp_path, aresponses):
    service, _ = _service(tmp_path)
    _mock_kimchi(aresponses, _kimchi_content(), repeat=2)
    ntfy = _mock_ntfy(aresponses)

    await service._handle_detection(_detection())
    await service._handle_detection(_detection())  # same camera, still hot

    records = DecisionStore.load_records(tmp_path / "decisions.jsonl")
    assert len(records) == 2
    assert records[0]["outcome"] == "alert"
    assert records[1]["outcome"] == "cooldown"
    assert records[1]["alerted"] is False
    assert "cooldown" in records[1]["detail"]
    assert ntfy["body"]  # exactly one push went out


async def test_classifier_error_recorded(tmp_path, aresponses):
    service, _ = _service(tmp_path)
    aresponses.add(
        KIMCHI_HOST,
        "/openai/v1/chat/completions",
        "POST",
        aresponses.Response(text="boom", status=500),
    )

    await service._handle_detection(_detection())

    records = DecisionStore.load_records(tmp_path / "decisions.jsonl")
    assert len(records) == 1
    assert records[0]["outcome"] == "classifier_error"
    assert records[0]["error"]


async def test_no_frames_recorded(tmp_path, aresponses):
    service, fake = _service(tmp_path)
    fake.frames = []

    await service._handle_detection(_detection())

    records = DecisionStore.load_records(tmp_path / "decisions.jsonl")
    assert records[0]["outcome"] == "no_frames"


async def test_counter_alert_path(tmp_path, aresponses: aresponses.ResponsesMockServer):
    service, _ = _service(tmp_path)
    kimchi = _mock_kimchi(
        aresponses,
        _kimchi_content(
            chewing=False,
            object="none",
            on_counter=True,
            confidence=0.2,
            counter_confidence=0.95,
            reason="Up on the counter",
            best_frame=1,
        ),
    )
    ntfy = _mock_ntfy(aresponses)

    detection = Detection(
        event_id="evt-42",
        camera_id="cam-1",
        camera_name="Living Room",
        started_at=datetime.now(UTC),
        check_counter=True,
    )
    await service._handle_detection(detection)

    # the counter prompt variant was used, still one request
    assert "on_counter" in kimchi["payload"]["messages"][0]["content"]
    # ntfy push carries the counter title
    assert "dog on the counter" in ntfy["headers"]["title"]
    records = DecisionStore.load_records(tmp_path / "decisions.jsonl")
    assert len(records) == 1
    assert records[0]["outcome"] == "alert"
    assert records[0]["alerted"] is True
    assert records[0]["verdict"]["on_counter"] is True
    assert records[0]["verdict"]["counter_confidence"] == 0.95


async def test_counter_cooldown_shares_with_chewing(tmp_path, aresponses):
    service, _ = _service(tmp_path)
    _mock_kimchi(
        aresponses,
        _kimchi_content(
            chewing=False,
            object="none",
            on_counter=True,
            confidence=0.1,
            counter_confidence=0.95,
            reason="Up on the counter",
            best_frame=1,
        ),
        repeat=2,
    )
    ntfy = _mock_ntfy(aresponses)

    def _counter_detection() -> Detection:
        return Detection(
            event_id="evt-42",
            camera_id="cam-1",
            camera_name="Living Room",
            started_at=datetime.now(UTC),
            check_counter=True,
        )

    await service._handle_detection(_counter_detection())
    await service._handle_detection(_counter_detection())

    records = DecisionStore.load_records(tmp_path / "decisions.jsonl")
    assert records[0]["outcome"] == "alert"
    assert records[1]["outcome"] == "cooldown"
    assert ntfy["body"]  # exactly one push for both events


async def test_pruning_repeats_while_service_runs(tmp_path, monkeypatch):
    """Startup is not enough: folders that go stale *while the service is up*
    must be pruned too — otherwise a long-lived container fills data/ forever.
    """
    service, _ = _service(tmp_path)
    release = asyncio.Event()

    class IdleListener(FakeListener):
        async def start(self) -> None:
            pass

        async def run_until_stopped(self) -> None:
            await release.wait()

        async def stop(self) -> None:
            pass

    service._listener = IdleListener([])  # noqa: SLF001 - idle lifecycle fake
    monkeypatch.setattr(ChewWatch, "PRUNE_INTERVAL_SECONDS", 0.05)

    store = service.store
    old_stamp = (datetime.now(UTC) - timedelta(days=40)).timestamp()
    frame = Frame(index=1, captured_at=datetime.now(UTC), jpeg=JPEG)

    # already stale before startup — the startup prune handles this one
    stale_at_start = store.save_frames("Yard", datetime.now(UTC), [frame])
    os.utime(stale_at_start, (old_stamp, old_stamp))

    run_task = asyncio.get_running_loop().create_task(service.run())
    await asyncio.sleep(0.1)  # let run() start and the startup prune happen

    # goes stale *after* startup — only the periodic prune can remove it
    stale_later = store.save_frames("Yard", datetime.now(UTC), [frame])
    os.utime(stale_later, (old_stamp, old_stamp))

    for _ in range(20):  # wait up to ~1s for the interval prune
        if not stale_later.exists():
            break
        await asyncio.sleep(0.05)
    release.set()
    await run_task

    assert not stale_at_start.exists(), "startup prune did not run"
    assert not stale_later.exists(), "periodic prune did not run while up"
