"""Core value types shared across ChewWatch modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

#: Objects the model may classify as acceptable (non-destructive) chew items.
TOY_OBJECTS = frozenset({"toy", "bone", "chew toy", "chewtoy", "dog toy", "treat"})


@dataclass(frozen=True)
class Frame:
    """One sampled camera snapshot (raw JPEG bytes)."""

    index: int  # 1-based position within the sampling window
    captured_at: datetime
    jpeg: bytes

    @property
    def size_kb(self) -> float:
        return len(self.jpeg) / 1024.0


@dataclass(frozen=True)
class Verdict:
    """Structured answer from the vision classifier."""

    dog_present: bool
    chewing: bool
    object: str  # what is being chewed: "shoe", "couch", "cable", "toy", ...
    confidence: float  # 0.0 .. 1.0
    reason: str  # one-line human-readable justification
    best_frame: int  # 1-based index of the most useful frame
    # counter-camera extras (False/0.0 for chewing-only cameras)
    on_counter: bool = False  # dog up on a counter/table/elevated surface
    counter_confidence: float = 0.0  # 0.0 .. 1.0 certainty of the counter call
    raw: str = ""  # unparsed model output, kept for the decision log

    @property
    def is_toy_or_bone(self) -> bool:
        return self.object.strip().lower() in TOY_OBJECTS


@dataclass(frozen=True)
class Detection:
    """A filtered Protect smart-detect animal event, handed to the pipeline."""

    event_id: str
    camera_id: str
    camera_name: str
    started_at: datetime
    check_counter: bool = False  # also ask "is the dog on the counter?"


@dataclass(frozen=True)
class BackfillEvent:
    """An animal event discovered in Protect's history (not seen live)."""

    event_id: str
    camera_id: str
    camera_name: str
    started_at: datetime
    score: float | None = None


@dataclass
class DecisionRecord:
    """Everything ChewWatch knew at decision time — one JSONL line on disk."""

    id: str
    event_id: str
    camera_id: str
    camera_name: str
    started_at: datetime
    decided_at: datetime
    frames: list[Frame] = field(default_factory=list)
    verdict: Verdict | None = None
    alerted: bool = False
    outcome: str = "error"  # short machine reason: alert / no_dog / ...
    detail: str = ""  # human explanation of the outcome
    error: str | None = None  # pipeline exception, if any
    frame_dir: str | None = None  # relative path of saved frames
    backfill: bool = False  # recovered from Protect history — never alerts

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "event_id": self.event_id,
            "camera_id": self.camera_id,
            "camera_name": self.camera_name,
            "started_at": self.started_at.isoformat(),
            "decided_at": self.decided_at.isoformat(),
            "alerted": self.alerted,
            "outcome": self.outcome,
            "detail": self.detail,
            "error": self.error,
            "frame_dir": self.frame_dir,
            "backfill": self.backfill,
            "verdict": None
            if self.verdict is None
            else {
                "dog_present": self.verdict.dog_present,
                "chewing": self.verdict.chewing,
                "object": self.verdict.object,
                "confidence": self.verdict.confidence,
                "reason": self.verdict.reason,
                "best_frame": self.verdict.best_frame,
                "on_counter": self.verdict.on_counter,
                "counter_confidence": self.verdict.counter_confidence,
                "raw": self.verdict.raw,
            },
        }
