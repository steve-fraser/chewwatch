"""ntfy notifier — one PUT per alert: JPEG body + Title/Priority/Tags headers."""

from __future__ import annotations

import logging
from datetime import datetime

import aiohttp

from .config import NtfyConfig
from .models import Verdict

_LOGGER = logging.getLogger(__name__)


class NotifyError(Exception):
    """The ntfy push failed."""


class Notifier:
    """Sends chewing alerts (with the best frame attached) to an ntfy topic."""

    def __init__(self, config: NtfyConfig) -> None:
        self._config = config
        self._timeout = aiohttp.ClientTimeout(total=30)

    @property
    def topic_url(self) -> str:
        return f"{self._config.server.rstrip('/')}/{self._config.topic}"

    @staticmethod
    def build_title(camera_name: str, verdict: Verdict) -> str:
        """Notification title: camera, trigger (chewing/counter), confidence, why."""
        if verdict.chewing:
            title = (
                f"🐕 {camera_name}: chewing {verdict.object} "
                f"({verdict.confidence:.0%})"
            )
            if verdict.on_counter:
                title += " — on the counter"
        elif verdict.on_counter:
            title = (
                f"🚫 {camera_name}: dog on the counter "
                f"({verdict.counter_confidence:.0%})"
            )
        else:
            title = f"🐕 {camera_name}: all clear ({verdict.confidence:.0%})"
        if verdict.reason:
            title += f" — {verdict.reason}"
        return title

    def _headers(self, title: str, filename: str) -> dict[str, str]:
        headers = {
            "Title": title,
            "Filename": filename,
            "Priority": self._config.priority,
            "Tags": "dog,warning",
        }
        if self._config.token:
            headers["Authorization"] = f"Bearer {self._config.token}"
        return headers

    async def send_alert(
        self, *, camera_name: str, verdict: Verdict, jpeg: bytes, at: datetime
    ) -> None:
        """Push the best frame as an attachment with the verdict in the title."""
        filename = f"chewwatch-{at.strftime('%Y%m%d-%H%M%S')}.jpg"
        await self._put(jpeg, self.build_title(camera_name, verdict), filename)

    async def send_test(self) -> None:
        """Send a plain text notification to verify the topic is reachable."""
        headers = self._headers("ChewWatch test notification", "chewwatch.txt")
        headers["Tags"] = "white_check_mark"
        async with aiohttp.ClientSession(timeout=self._timeout) as session:
            async with session.put(
                self.topic_url, data=b"ChewWatch is configured correctly.", headers=headers
            ) as resp:
                if resp.status not in (200, 201, 202, 204):
                    body = await resp.text()
                    raise NotifyError(
                        f"ntfy test push failed (HTTP {resp.status}): {body[:200]}"
                    )
        _LOGGER.info("Test notification sent to %s", self.topic_url)

    async def _put(self, jpeg: bytes, title: str, filename: str) -> None:
        headers = self._headers(title, filename)
        async with aiohttp.ClientSession(timeout=self._timeout) as session:
            async with session.put(self.topic_url, data=jpeg, headers=headers) as resp:
                if resp.status not in (200, 201, 202, 204):
                    body = await resp.text()
                    raise NotifyError(
                        f"ntfy push failed (HTTP {resp.status}): {body[:200]}"
                    )
        _LOGGER.debug("ntfy notification sent: %s (%d bytes)", title, len(jpeg))
