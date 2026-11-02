"""Tests for the classifier: prompt building, JSON extraction and parsing."""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime

import aresponses
import pytest

from chewwatch.classifier import (
    Classifier,
    ClassifierError,
    build_messages,
    parse_verdict,
)
from chewwatch.config import ClassifierConfig
from chewwatch.models import Frame

# --------------------------------------------------------------------- parsing


def _frames(n: int = 3) -> list[Frame]:
    return [
        Frame(index=i + 1, captured_at=datetime.now(UTC), jpeg=b"jpeg-%d" % i)
        for i in range(n)
    ]


def test_parse_verdict_clean_json():
    verdict = parse_verdict(
        '{"dog_present": true, "chewing": true, "object": "shoe", '
        '"confidence": 0.85, "reason": "Gnawing the laces", "best_frame": 2}',
        3,
    )
    assert verdict.dog_present is True
    assert verdict.chewing is True
    assert verdict.object == "shoe"
    assert verdict.confidence == 0.85
    assert verdict.reason == "Gnawing the laces"
    assert verdict.best_frame == 2


def test_parse_verdict_fenced():
    text = 'Here you go:\n```json\n{"dog_present": true, "chewing": false, "object": "none", "confidence": 0.1, "reason": "no", "best_frame": 1}\n```'
    verdict = parse_verdict(text, 3)
    assert verdict.chewing is False
    assert verdict.object == "none"


def test_parse_verdict_prose_wrapped():
    text = (
        'The dog is asleep on the rug. {"dog_present": true, "chewing": false, '
        '"object": "none", "confidence": 0.4, "reason": "sleeping", "best_frame": 3} '
        "Hope that helps!"
    )
    verdict = parse_verdict(text, 3)
    assert verdict.dog_present is True
    assert verdict.chewing is False


def test_parse_verdict_clamps():
    verdict = parse_verdict(
        '{"dog_present": true, "chewing": true, "object": "SOFA", '
        '"confidence": 7.5, "reason": "x", "best_frame": 99}',
        3,
    )
    assert verdict.confidence == 1.0
    assert verdict.best_frame == 3
    assert verdict.object == "sofa"


def test_parse_verdict_coerces_strings():
    verdict = parse_verdict(
        '{"dog_present": "true", "chewing": "no", "object": "", '
        '"confidence": "0.5", "reason": "", "best_frame": "2"}',
        3,
    )
    assert verdict.dog_present is True
    assert verdict.chewing is False
    assert verdict.object == "none"
    assert verdict.confidence == 0.5
    assert verdict.best_frame == 2


def test_parse_verdict_defaults_on_missing():
    verdict = parse_verdict("{}", 3)
    assert verdict.dog_present is False
    assert verdict.chewing is False
    assert verdict.object == "none"
    assert verdict.confidence == 0.0
    assert verdict.best_frame == 1
    assert verdict.on_counter is False
    assert verdict.counter_confidence == 0.0


def test_parse_verdict_garbage_raises():
    with pytest.raises(ValueError):
        parse_verdict("I am a llama who ate the JSON.", 3)


def test_long_reason_truncated():
    verdict = parse_verdict(
        '{"dog_present": true, "chewing": true, "object": "couch", '
        '"confidence": 0.9, "reason": "' + "x" * 500 + '", "best_frame": 1}',
        3,
    )
    assert len(verdict.reason) == 300
    assert verdict.reason.endswith("...")


def test_parse_verdict_counter_fields():
    verdict = parse_verdict(
        '{"dog_present": true, "chewing": false, "object": "none", '
        '"on_counter": true, "confidence": 0.2, "counter_confidence": 0.93, '
        '"reason": "Up on the worktop", "best_frame": 1}',
        3,
    )
    assert verdict.on_counter is True
    assert verdict.counter_confidence == 0.93
    assert verdict.chewing is False


def test_parse_verdict_counter_fields_missing_fall_back():
    # model omitted the counter keys: on_counter false, confidence falls back
    # to the chewing confidence so the threshold still applies
    verdict = parse_verdict(
        '{"dog_present": true, "chewing": true, "object": "shoe", '
        '"confidence": 0.8, "reason": "x", "best_frame": 1}',
        3,
    )
    assert verdict.on_counter is False
    assert verdict.counter_confidence == 0.8


def test_parse_verdict_counter_confidence_clamped():
    verdict = parse_verdict(
        '{"dog_present": true, "chewing": false, "object": "none", '
        '"on_counter": true, "confidence": 0.1, "counter_confidence": 5, '
        '"reason": "x", "best_frame": 1}',
        3,
    )
    assert verdict.counter_confidence == 1.0


# ----------------------------------------------------------------- message build


def test_build_messages_shape():
    frames = _frames(2)
    messages = build_messages(frames, "Living Room")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "Living Room" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    content = messages[1]["content"]
    assert content[0]["type"] == "text"
    assert len(content) == 3  # text + 2 images
    for part, frame in zip(content[1:], frames):
        assert part["type"] == "image_url"
        url = part["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,")
        assert base64.b64decode(url.split(",", 1)[1]) == frame.jpeg


def test_build_messages_chewing_only_has_no_counter_schema():
    messages = build_messages(_frames(2), "Living Room")
    system = messages[0]["content"]
    assert "chewing" in system
    assert "on_counter" not in system


def test_build_messages_counter_variant():
    messages = build_messages(_frames(2), "Living Room", check_counter=True)
    system = messages[0]["content"]
    assert '"on_counter"' in system
    assert '"counter_confidence"' in system
    assert "Living Room" in system
    user_text = messages[1]["content"][0]["text"]
    assert "counter" in user_text


# ----------------------------------------------------------------------- HTTP


def _openai_body(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}], "usage": {}}


async def test_classify_roundtrip(aresponses: aresponses.ResponsesMockServer):
    aresponses.add(
        "llm.kimchi.dev",
        "/openai/v1/chat/completions",
        "POST",
        aresponses.Response(
            text=json.dumps(
                _openai_body(
                    '{"dog_present": true, "chewing": true, "object": "cable", '
                    '"confidence": 0.91, "reason": "Teething on the charger cable", '
                    '"best_frame": 2}'
                )
            ),
            content_type="application/json",
        ),
    )
    classifier = Classifier(ClassifierConfig(api_key="test-key"))
    verdict = await classifier.classify(_frames(3), "Office")
    assert verdict.chewing is True
    assert verdict.object == "cable"
    assert verdict.confidence == 0.91


async def test_classify_counter_roundtrip(
    aresponses: aresponses.ResponsesMockServer,
):
    captured = {}

    async def handler(request):
        captured["payload"] = json.loads(await request.text())
        return aresponses.Response(
            text=json.dumps(
                _openai_body(
                    '{"dog_present": true, "chewing": false, "object": "none", '
                    '"on_counter": true, "confidence": 0.1, '
                    '"counter_confidence": 0.95, '
                    '"reason": "Up on the counter", "best_frame": 1}'
                )
            ),
            content_type="application/json",
        )

    aresponses.add("llm.kimchi.dev", "/openai/v1/chat/completions", "POST", handler)
    classifier = Classifier(ClassifierConfig(api_key="test-key"))
    verdict = await classifier.classify(_frames(2), "Kitchen", check_counter=True)
    assert '"on_counter"' in captured["payload"]["messages"][0]["content"]
    assert verdict.on_counter is True
    assert verdict.counter_confidence == 0.95


async def test_classify_retries_once_on_bad_json(
    aresponses: aresponses.ResponsesMockServer,
):
    aresponses.add(
        "llm.kimchi.dev",
        "/openai/v1/chat/completions",
        "POST",
        aresponses.Response(
            text=json.dumps(_openai_body("sorry, not json")),
            content_type="application/json",
        ),
    )
    aresponses.add(
        "llm.kimchi.dev",
        "/openai/v1/chat/completions",
        "POST",
        aresponses.Response(
            text=json.dumps(
                _openai_body(
                    '{"dog_present": true, "chewing": false, "object": "none", '
                    '"confidence": 0.2, "reason": "just sitting", "best_frame": 1}'
                )
            ),
            content_type="application/json",
        ),
    )
    classifier = Classifier(ClassifierConfig(api_key="test-key"))
    verdict = await classifier.classify(_frames(2), "Yard")
    assert verdict.chewing is False


async def test_classify_gives_up_after_retry(aresponses: aresponses.ResponsesMockServer):
    for _ in range(2):
        aresponses.add(
            "llm.kimchi.dev",
            "/openai/v1/chat/completions",
            "POST",
            aresponses.Response(
                text=json.dumps(_openai_body("still not json")),
                content_type="application/json",
            ),
        )
    classifier = Classifier(ClassifierConfig(api_key="test-key"))
    with pytest.raises(ClassifierError, match="not valid JSON"):
        await classifier.classify(_frames(2), "Yard")


async def test_classify_http_error(aresponses: aresponses.ResponsesMockServer):
    aresponses.add(
        "llm.kimchi.dev",
        "/openai/v1/chat/completions",
        "POST",
        aresponses.Response(text="unauthorized", status=401),
    )
    classifier = Classifier(ClassifierConfig(api_key="bad"))
    with pytest.raises(ClassifierError, match="401"):
        await classifier.classify(_frames(2), "Yard")


async def test_classify_no_frames():
    classifier = Classifier(ClassifierConfig(api_key="k"))
    with pytest.raises(ClassifierError, match="no frames"):
        await classifier.classify([], "Yard")
