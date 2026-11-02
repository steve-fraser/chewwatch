"""Decision log: JSONL records + per-event frame folders + retention pruning."""

from __future__ import annotations

import json
import logging
import re
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .models import DecisionRecord, Frame

_LOGGER = logging.getLogger(__name__)

_SLUG_RE = re.compile(r"[^a-z0-9]+")

_FRAMES_SUBDIR = "frames"
_LOG_FILENAME = "decisions.jsonl"


def _slugify(name: str) -> str:
    slug = _SLUG_RE.sub("-", name.strip().lower()).strip("-")
    return slug or "camera"


class DecisionStore:
    """Persists every decision with its frames under ``data_dir``.

    Layout::

        data_dir/
        ├── decisions.jsonl                      # one JSON object per line
        └── frames/
            └── 20260101-101530-living-room/    # one folder per event
                ├── frame-1.jpg ... frame-N.jpg
    """

    def __init__(self, data_dir: Path | str) -> None:
        self._root = Path(data_dir)
        self._frames_root = self._root / _FRAMES_SUBDIR
        self._log_path = self._root / _LOG_FILENAME

    @property
    def log_path(self) -> Path:
        return self._log_path

    def prepare(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        self._frames_root.mkdir(parents=True, exist_ok=True)

    def save_frames(self, camera_name: str, at: datetime, frames: list[Frame]) -> Path:
        """Write the sampled frames to a per-event folder; returns the folder."""
        folder = self._frames_root / (
            f"{at.strftime('%Y%m%d-%H%M%S')}-{_slugify(camera_name)}"
        )
        folder.mkdir(parents=True, exist_ok=True)
        for frame in frames:
            (folder / f"frame-{frame.index}.jpg").write_bytes(frame.jpeg)
        return folder

    def save(self, record: DecisionRecord) -> None:
        """Append the decision record as one JSON line."""
        self.prepare()
        with self._log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_json(), ensure_ascii=False) + "\n")

    def prune(self, retention_days: int, now: datetime | None = None) -> int:
        """Delete frame folders older than the retention window.

        Returns the number of folders removed. The JSONL log is kept forever —
        it is small and is the audit trail for tuning.
        """
        if retention_days <= 0:
            return 0
        now = now or datetime.now(UTC)
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        cutoff = now - timedelta(days=retention_days)
        removed = 0
        if not self._frames_root.is_dir():
            return 0
        for folder in self._frames_root.iterdir():
            if not folder.is_dir():
                continue
            try:
                folder_time = datetime.fromtimestamp(folder.stat().st_mtime, tz=UTC)
            except OSError:
                continue
            if folder_time < cutoff:
                shutil.rmtree(folder, ignore_errors=True)
                removed += 1
        if removed:
            _LOGGER.info("Pruned %d frame folder(s) older than %d day(s)", removed, retention_days)
        return removed

    @staticmethod
    def load_records(log_path: Path | str) -> list[dict]:
        """Read back every decision record (for review / tuning)."""
        log_path = Path(log_path)
        if not log_path.is_file():
            return []
        records = []
        for line in log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                _LOGGER.warning("Skipping corrupt JSONL line in %s", log_path)
        return records
