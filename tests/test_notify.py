"""Tests for the ntfy notifier."""

from __future__ import annotations

from datetime import UTC, datetime

import aresponses
import pytest

from chewwatch.config import NtfyConfig
from chewwatch.models import Verdict
from chewwatch.notify import NotifyError, Notifier

VERDICT = Verdict(
    dog_present=True,
    chewing=True,
    object="couch",
    confidence=0.88,
    reason="Mouth working the cushion",
    best_frame=2,
    raw="{}",
)


def _cfg(**overrides) -> NtfyConfig:
    defaults = dict(
        server="https://ntfy.example.com",
        topic="chewwatch-test",
        token="",
        priority="high",
    )
    defaults.update(overrides)
    return NtfyConfig(**defaults)


def test_topic_url():
    assert Notifier(_cfg()).topic_url == "https://ntfy.example.com/chewwatch-test"
    assert (
        Notifier(_cfg(server="https://ntfy.example.com/")).topic_url
        == "https://ntfy.example.com/chewwatch-test"
    )


def test_build_title():
    title = Notifier(_cfg()).build_title("Living Room", VERDICT)
    assert title.startswith("🐕 Living Room: chewing couch (88%)")
    assert "Mouth working the cushion" in title


def test_build_title_counter_only():
    verdict = Verdict(
        dog_present=True,
        chewing=False,
        object="none",
        confidence=0.2,
        reason="Up on the worktop",
        best_frame=1,
        on_counter=True,
        counter_confidence=0.93,
    )
    title = Notifier(_cfg()).build_title("Kitchen", verdict)
    assert title.startswith("🚫 Kitchen: dog on the counter (93%)")
    assert "Up on the worktop" in title


def test_build_title_chewing_and_counter():
    verdict = Verdict(
        dog_present=True,
        chewing=True,
        object="couch",
        confidence=0.88,
        reason="Mouth working the cushion",
        best_frame=2,
        on_counter=True,
        counter_confidence=0.9,
    )
    title = Notifier(_cfg()).build_title("Living Room", verdict)
    assert title.startswith("🐕 Living Room: chewing couch (88%)")
    assert "on the counter" in title


async def test_send_alert_puts_jpeg_with_headers(
    aresponses: aresponses.ResponsesMockServer,
):
    captured = {}

    async def handler(request):
        captured["body"] = await request.read()
        captured["headers"] = dict(request.headers)
        return aresponses.Response(text="{}", content_type="application/json")

    aresponses.add("ntfy.example.com", "/chewwatch-test", "PUT", handler)

    notifier = Notifier(_cfg())
    jpeg = b"\xff\xd8fake-jpeg-bytes\xff\xd9"
    at = datetime(2026, 1, 1, 10, 15, 30, tzinfo=UTC)
    await notifier.send_alert(
        camera_name="Living Room", verdict=VERDICT, jpeg=jpeg, at=at
    )

    assert captured["body"] == jpeg
    headers = {k.lower(): v for k, v in captured["headers"].items()}
    assert headers["filename"] == "chewwatch-20260101-101530.jpg"
    assert headers["priority"] == "high"
    assert headers["tags"] == "dog,warning"
    assert "living room" in headers["title"].lower()
    assert "authorization" not in headers


async def test_send_alert_with_token(aresponses: aresponses.ResponsesMockServer):
    captured = {}

    async def handler(request):
        captured["headers"] = dict(request.headers)
        return aresponses.Response(text="{}", content_type="application/json")

    aresponses.add("ntfy.example.com", "/chewwatch-test", "PUT", handler)

    notifier = Notifier(_cfg(token="tk_abcdef"))
    await notifier.send_alert(
        camera_name="Yard", verdict=VERDICT, jpeg=b"jpg", at=datetime.now(UTC)
    )
    assert captured["headers"]["Authorization"] == "Bearer tk_abcdef"


async def test_send_alert_error_status(aresponses: aresponses.ResponsesMockServer):
    aresponses.add(
        "ntfy.example.com",
        "/chewwatch-test",
        "PUT",
        aresponses.Response(text="nope", status=404),
    )
    notifier = Notifier(_cfg())
    with pytest.raises(NotifyError, match="404"):
        await notifier.send_alert(
            camera_name="Yard", verdict=VERDICT, jpeg=b"jpg", at=datetime.now(UTC)
        )


async def test_send_test(aresponses: aresponses.ResponsesMockServer):
    async def handler(request):
        captured["body"] = await request.read()
        return aresponses.Response(text="{}", content_type="application/json")

    captured = {}
    aresponses.add("ntfy.example.com", "/chewwatch-test", "PUT", handler)
    await Notifier(_cfg()).send_test()
    assert b"configured correctly" in captured["body"]
