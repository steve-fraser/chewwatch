"""Vision classifier backed by the Kimchi Inference API (OpenAI-compatible)."""

from __future__ import annotations

import base64
import json
import logging
import re
from typing import Any

import aiohttp

from .config import ClassifierConfig
from .models import Frame, Verdict

_LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a strict, literal watchdog for one purpose: deciding whether a dog is \
chewing something it should not.

You will receive {n} still frames sampled a few seconds apart from the security \
camera "{camera}". Judge them as a short sequence, not isolated photos: chewing \
shows up as the dog's head down and mouth working on an object across frames.

Answer with a single JSON object and nothing else — no prose, no markdown fences:
{{
  "dog_present": <bool>,
  "chewing": <bool>,
  "object": "<what is being chewed: e.g. shoe, slipper, couch, cushion, cable, \
rug, furniture, toy, bone, stick, none>",
  "confidence": <number 0.0-1.0>,
  "reason": "<one short sentence explaining your judgement>",
  "best_frame": <integer 1-{n}, the frame that shows the chewing most clearly>
}}

Rules:
- "chewing" means mouth actively biting, gnawing or tearing an object. A dog \
merely holding, carrying, sniffing, lying on or standing near an object is NOT \
chewing. A dog eating from a bowl is NOT chewing.
- If the dog chews its own toy, bone or treat, still report chewing=true and \
set "object" to "toy" or "bone" — the caller decides how to treat it.
- If there is no dog, or no chewing, set chewing=false and object="none".
- confidence is your honest certainty about the chewing call, not about the \
dog being present.
"""

SYSTEM_PROMPT_COUNTER = """\
You are a strict, literal watchdog for two purposes: deciding whether a dog is \
chewing something it should not, and whether the dog has climbed onto a kitchen \
counter or other elevated surface it is not allowed on.

You will receive {n} still frames sampled a few seconds apart from the security \
camera "{camera}". Judge them as a short sequence, not isolated photos: chewing \
shows up as the dog's head down and mouth working on an object across frames; \
"on the counter" means the dog has its paws up on the surface or is fully up on \
it — close enough to reach and grab things.

Answer with a single JSON object and nothing else — no prose, no markdown fences:
{{
  "dog_present": <bool>,
  "chewing": <bool>,
  "object": "<what is being chewed: e.g. shoe, slipper, couch, cushion, cable, \
rug, furniture, toy, bone, stick, none>",
  "on_counter": <bool>,
  "confidence": <number 0.0-1.0>,
  "counter_confidence": <number 0.0-1.0>,
  "reason": "<one short sentence explaining your judgement>",
  "best_frame": <integer 1-{n}, the frame that shows the chewing or the counter \
climb most clearly>
}}

Rules:
- "chewing" means mouth actively biting, gnawing or tearing an object. A dog \
merely holding, carrying, sniffing, lying on or standing near an object is NOT \
chewing. A dog eating from a bowl is NOT chewing.
- If the dog chews its own toy, bone or treat, still report chewing=true and \
set "object" to "toy" or "bone" — the caller decides how to treat it.
- If there is no dog, or no chewing, set chewing=false and object="none".
- "on_counter" is true when the dog has reached UP onto a kitchen counter, \
worktop, dining table or similar elevated food surface — either front paws up \
on the surface (standing on hind legs, leaning/reaching over the edge) OR its \
whole body up on it. The point is whether the dog could grab something off the \
surface. A dog standing on the floor NOT touching the surface, merely sniffing \
the air near it, or lying on a sofa, armchair or dog bed, is NOT on_counter.
- If the dog's paws are on the floor and off the surface, set on_counter=false \
and counter_confidence low.
- "confidence" is your honest certainty about the chewing call; \
"counter_confidence" is your honest certainty about the on_counter call. They \
are independent numbers.
"""

_RETRY_PROMPT = (
    "Your previous reply was not a valid JSON object. Reply again with ONLY "
    "the JSON object, no prose and no markdown fences."
)

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class ClassifierError(Exception):
    """The classifier could not produce a verdict."""


def _extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model reply.

    Tolerates markdown fences and stray prose around the object.
    Raises ``ValueError`` when nothing parseable is found.
    """
    candidates: list[str] = []
    fenced = _JSON_FENCE.search(text)
    if fenced:
        candidates.append(fenced.group(1))
    # Outermost brace span of the raw text.
    start = text.find("{")
    if start != -1:
        end = text.rfind("}")
        if end > start:
            candidates.append(text[start : end + 1])
    candidates.append(text.strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"no JSON object found in reply: {text[:200]!r}")


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def parse_verdict(text: str, frame_count: int) -> Verdict:
    """Turn a raw model reply into a :class:`Verdict` (validates and clamps)."""
    data = _extract_json(text)

    object_name = str(data.get("object") or "none").strip().lower() or "none"
    try:
        confidence = float(data.get("confidence"))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(1.0, max(0.0, confidence))

    try:
        best_frame = int(data.get("best_frame"))
    except (TypeError, ValueError):
        best_frame = 1
    best_frame = min(max(1, best_frame), max(1, frame_count))

    reason = str(data.get("reason") or "").strip()
    if len(reason) > 300:
        reason = reason[:297] + "..."

    on_counter = _coerce_bool(data.get("on_counter"))
    try:
        counter_confidence = float(data.get("counter_confidence"))
    except (TypeError, ValueError):
        # Model skipped the counter keys — reuse the chewing certainty so the
        # threshold still applies sensibly.
        counter_confidence = confidence
    counter_confidence = min(1.0, max(0.0, counter_confidence))

    return Verdict(
        dog_present=_coerce_bool(data.get("dog_present")),
        chewing=_coerce_bool(data.get("chewing")),
        object=object_name,
        confidence=confidence,
        reason=reason,
        best_frame=best_frame,
        on_counter=on_counter,
        counter_confidence=counter_confidence,
        raw=text,
    )


def build_messages(
    frames: list[Frame], camera_name: str, check_counter: bool = False
) -> list[dict[str, Any]]:
    """Chat-completions payload: one text part plus one image part per frame."""
    if check_counter:
        system_prompt = SYSTEM_PROMPT_COUNTER
        question = (
            f"These are {len(frames)} frames sampled a few seconds apart "
            f'from camera "{camera_name}". Is the dog chewing something it '
            "should not, and is the dog up on a counter or other elevated "
            "surface it is not allowed on? Reply with the JSON object only."
        )
    else:
        system_prompt = SYSTEM_PROMPT
        question = (
            f"These are {len(frames)} frames sampled a few seconds apart "
            f'from camera "{camera_name}". Is the dog chewing something it '
            "should not? Reply with the JSON object only."
        )
    content: list[dict[str, Any]] = [{"type": "text", "text": question}]
    for frame in frames:
        b64 = base64.b64encode(frame.jpeg).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            }
        )
    return [
        {
            "role": "system",
            "content": system_prompt.format(n=len(frames), camera=camera_name),
        },
        {"role": "user", "content": content},
    ]


class Classifier:
    """Sends sampled frames to a Kimchi vision model and parses the verdict."""

    def __init__(self, config: ClassifierConfig) -> None:
        self._config = config
        self._timeout = aiohttp.ClientTimeout(total=config.request_timeout_seconds)

    @property
    def model(self) -> str:
        return self._config.model

    async def classify(
        self, frames: list[Frame], camera_name: str, check_counter: bool = False
    ) -> Verdict:
        """Classify a batch of frames; retries once on unparseable output."""
        if not frames:
            raise ClassifierError("no frames to classify")
        messages = build_messages(frames, camera_name, check_counter)

        async with aiohttp.ClientSession(timeout=self._timeout) as session:
            text = await self._request(session, messages)
            try:
                return parse_verdict(text, len(frames))
            except ValueError:
                _LOGGER.warning(
                    "Model reply was not valid JSON, retrying once: %.120s", text
                )
            # One corrective retry, same conversation.
            messages = messages + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": _RETRY_PROMPT},
            ]
            text = await self._request(session, messages)
            try:
                return parse_verdict(text, len(frames))
            except ValueError as err:
                raise ClassifierError(f"model reply was not valid JSON: {text[:200]!r}") from err

    async def _request(
        self, session: aiohttp.ClientSession, messages: list[dict[str, Any]]
    ) -> str:
        url = self._config.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self._config.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 400,
        }
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }
        try:
            async with session.post(url, json=payload, headers=headers) as resp:
                body = await resp.text()
                if resp.status != 200:
                    raise ClassifierError(
                        f"Kimchi Inference returned HTTP {resp.status}: {body[:300]}"
                    )
                try:
                    data = json.loads(body)
                except json.JSONDecodeError as err:
                    raise ClassifierError(
                        f"Kimchi Inference returned non-JSON body: {body[:300]}"
                    ) from err
        except aiohttp.ClientError as err:
            raise ClassifierError(f"Kimchi Inference request failed: {err}") from err

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as err:
            raise ClassifierError(
                f"Unexpected Kimchi Inference response shape: {str(data)[:300]}"
            ) from err
