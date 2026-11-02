"""UniFi Protect event listener and frame sampler (uiprotect).

Auth modes
----------
- **api_key only** (public Integration API): typed events via
  ``subscribe_events_and_prime`` and snapshots via
  ``get_public_api_camera_snapshot``.
- **api_key + username/password** (hybrid): additionally primes the private
  bootstrap and samples via ``get_camera_snapshot`` with a requested width.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from uiprotect import EventChange, ProtectApiClient, ProtectEvent
from uiprotect.data import EventType, SmartDetectObjectType

from .config import ChewWatchConfig, camera_matches
from .models import BackfillEvent, Detection, Frame

_LOGGER = logging.getLogger(__name__)

#: Smart-detect event types that can carry an "animal" detection.
_SMART_EVENT_TYPES = frozenset(
    {
        EventType.SMART_DETECT,
        EventType.SMART_DETECT_LINE,
        EventType.SMART_DETECT_LOITER,
    }
)


class BackfillError(Exception):
    """Backfill is not possible (hybrid auth missing, or Protect not connected)."""


def _parse_event_time(value: Any) -> datetime:
    """Raw Protect event JSON carries millisecond epochs; accept ISO too."""
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value / 1000, tz=UTC)
    ts = datetime.fromisoformat(str(value))
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts


class ProtectListener:
    """Connects to Protect, filters animal events, samples frames on demand."""

    def __init__(
        self,
        config: ChewWatchConfig,
        on_detection: Callable[[Detection], Awaitable[None]],
    ) -> None:
        self._config = config
        self._on_detection = on_detection
        self._protect: ProtectApiClient | None = None
        self._unsubscribe: Callable[[], None] | None = None
        self._stop = asyncio.Event()
        # camera id -> display name
        self._camera_names: dict[str, str] = {}
        # cameras selected for watching (ids)
        self._watched: set[str] = set()
        # subset of watched cameras that also get the counter check
        self._counter: set[str] = set()
        # in-flight pipelines, per camera
        self._in_flight: dict[str, int] = {}
        self._tasks: set[asyncio.Task[None]] = set()

    # ------------------------------------------------------------------ setup

    @property
    def _hybrid(self) -> bool:
        p = self._config.protect
        return bool(p.username and p.password)

    async def start(self) -> None:
        """Connect, prime both bootstraps as configured, and subscribe."""
        p = self._config.protect
        client: ProtectApiClient
        if self._hybrid:
            client = ProtectApiClient(
                host=p.host,
                port=p.port,
                username=p.username,
                password=p.password,
                api_key=p.api_key,
                verify_ssl=p.verify_ssl,
                # ephemeral session: uiprotect would otherwise persist its
                # auth token under Path.home(), which is unwritable for the
                # container's system user (crashed hybrid startup with
                # PermissionError: /home/chewwatch). Re-auth happens per start.
                store_sessions=False,
            )
        else:
            client = ProtectApiClient.public_only(
                host=p.host,
                port=p.port,
                api_key=p.api_key,
                verify_ssl=p.verify_ssl,
                store_sessions=False,
            )
        self._protect = client

        self._subscribe_state_logging(client)

        if self._hybrid:
            bootstrap = await client.update()
            _LOGGER.info(
                "Private bootstrap ready: Protect %s, %d camera(s)",
                bootstrap.nvr.version,
                len(bootstrap.cameras),
            )
            cameras = {
                cam.id: (cam.name or cam.id)
                for cam in bootstrap.cameras.values()
            }
        else:
            await client.update_public()
            cameras = {
                cam.id: (cam.display_name or cam.id)
                for cam in client.public_bootstrap.cameras.values()
            }
        self._camera_names = cameras

        self._watched = {
            cam_id
            for cam_id, name in cameras.items()
            if self._selected(cam_id, name)
        }
        self._counter = self._select_counter_cameras(cameras)
        for cam_id, name in cameras.items():
            state = "watched" if cam_id in self._watched else "ignored"
            _LOGGER.info("Camera %s (%s): %s", name, cam_id, state)
        if not self._watched:
            _LOGGER.warning("No cameras selected — check protect.cameras in the config")
        if self._counter:
            names = ", ".join(sorted(self.camera_name(c) for c in self._counter))
            _LOGGER.info("Counter checks enabled on: %s", names)

        self._unsubscribe = await client.subscribe_events_and_prime(self._on_event)
        _LOGGER.info(
            "Subscribed to Protect events on %s:%s (%d camera(s) watched)",
            p.host,
            p.port,
            len(self._watched),
        )

    def _selected(self, camera_id: str, name: str) -> bool:
        p = self._config.protect
        mac = self._camera_mac(camera_id)
        return camera_matches(
            selectors=p.cameras,
            camera_id=camera_id,
            camera_name=name,
            camera_mac=mac,
        )

    def _select_counter_cameras(self, cameras: dict[str, str]) -> set[str]:
        """Watched cameras that additionally get the counter check.

        ``counter_cameras`` is a subset filter of ``cameras``: a selector that
        matches no watched camera is reported (the camera must be listed in
        ``protect.cameras`` to be watched at all).
        """
        p = self._config.protect
        if not p.counter_cameras:
            return set()
        counter: set[str] = set()
        for selector in p.counter_cameras:
            if not selector.strip():
                continue
            matches = {
                cam_id
                for cam_id in self._watched
                if camera_matches(
                    selectors=[selector],
                    camera_id=cam_id,
                    camera_name=cameras.get(cam_id, cam_id),
                    camera_mac=self._camera_mac(cam_id),
                )
            }
            if matches:
                counter |= matches
            else:
                _LOGGER.warning(
                    "counter_cameras selector %r matched no watched camera — "
                    "counter cameras must also be listed in protect.cameras",
                    selector,
                )
        return counter

    def _camera_mac(self, camera_id: str) -> str | None:
        if self._protect is None:
            return None
        if self._hybrid:
            cam = self._protect.bootstrap.cameras.get(camera_id)
        else:
            cam = self._protect.public_bootstrap.cameras.get(camera_id)
        return getattr(cam, "mac", None)

    def _subscribe_state_logging(self, client: ProtectApiClient) -> None:
        def on_state(state: Any) -> None:
            _LOGGER.info("Protect events websocket: %s", state)

        try:
            client.subscribe_events_websocket_state(on_state)
        except Exception:  # pragma: no cover - depends on library internals
            _LOGGER.debug("Could not subscribe to websocket state", exc_info=True)

    # -------------------------------------------------------------- backfill

    def can_backfill(self) -> bool:
        """Backfill needs the private API: hybrid auth + connected client."""
        return self._hybrid and self._protect is not None

    def is_counter_camera(self, camera_id: str) -> bool:
        """Whether the counter question applies to this camera."""
        return camera_id in self._counter

    async def fetch_backfill_events(self, hours: float) -> list[BackfillEvent]:
        """Animal events in Protect's history over the last ``hours`` hours.

        Private API only (hybrid auth required); returns events on watched
        cameras, newest first.
        """
        if not self._hybrid:
            raise BackfillError(
                "Backfill needs hybrid auth: set protect.username and "
                "protect.password (a local Protect user with camera access)"
            )
        if self._protect is None:
            raise BackfillError("Protect is not connected yet")
        end = datetime.now(UTC)
        raw_events = await self._protect.get_events_raw(
            start=end - timedelta(hours=hours),
            end=end + timedelta(minutes=1),
            types=[
                EventType.SMART_DETECT,
                EventType.SMART_DETECT_LINE,
                EventType.SMART_DETECT_LOITER,
            ],
            smart_detect_types=[SmartDetectObjectType.ANIMAL],
            sorting="desc",
            descriptions=False,
        )
        events: list[BackfillEvent] = []
        for raw in raw_events:
            event_id = str(raw.get("id") or "")
            # raw private-API JSON keys the camera reference "camera" (the
            # typed uiprotect model aliases it to cameraId) — accept both.
            camera_id = str(raw.get("camera") or raw.get("cameraId") or "")
            if not event_id or camera_id not in self._watched:
                continue
            try:
                started = _parse_event_time(raw.get("start"))
            except (TypeError, ValueError):
                _LOGGER.warning("Skipping backfill event %s with bad start", event_id)
                continue
            events.append(
                BackfillEvent(
                    event_id=event_id,
                    camera_id=camera_id,
                    camera_name=self.camera_name(camera_id),
                    started_at=started,
                    score=raw.get("score"),
                )
            )
        return events

    async def fetch_event_thumbnail(self, event_id: str) -> bytes | None:
        """The Protect-generated thumbnail JPEG for a past event."""
        if self._protect is None:
            raise BackfillError("Protect is not connected yet")
        try:
            return await self._protect.get_event_thumbnail(event_id)
        except Exception:  # noqa: BLE001 - transport failures mean "no image"
            _LOGGER.exception("Thumbnail fetch failed for event %s", event_id)
            return None

    # ---------------------------------------------------------------- events

    def _on_event(self, event: ProtectEvent, change: EventChange) -> None:
        """Synchronous websocket callback — must not raise or block."""
        try:
            if change is not EventChange.STARTED:
                return
            if event.type not in _SMART_EVENT_TYPES:
                return
            if SmartDetectObjectType.ANIMAL not in event.smart_detect_types:
                return
            if event.device_id not in self._watched:
                return
            busy = self._in_flight.get(event.device_id, 0)
            if busy >= self._config.sampling.max_concurrent_per_camera:
                _LOGGER.info(
                    "Animal event on %s skipped — pipeline already running",
                    self.camera_name(event.device_id),
                )
                return
            detection = Detection(
                event_id=event.id,
                camera_id=event.device_id,
                camera_name=self.camera_name(event.device_id),
                started_at=event.start,
                check_counter=event.device_id in self._counter,
            )
            task = asyncio.get_running_loop().create_task(self._dispatch(detection))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        except Exception:  # noqa: BLE001 - callback contract: never raise
            _LOGGER.exception("Error dispatching Protect event")

    async def _dispatch(self, detection: Detection) -> None:
        self._in_flight[detection.camera_id] = self._in_flight.get(detection.camera_id, 0) + 1
        try:
            await self._on_detection(detection)
        finally:
            remaining = self._in_flight.get(detection.camera_id, 1) - 1
            if remaining > 0:
                self._in_flight[detection.camera_id] = remaining
            else:
                self._in_flight.pop(detection.camera_id, None)

    # --------------------------------------------------------------- sampling

    def camera_name(self, camera_id: str) -> str:
        return self._camera_names.get(camera_id, camera_id)

    async def sample_frames(self, camera_id: str) -> list[Frame]:
        """Grab ``count`` snapshots spread over ``window_seconds`` (absolute schedule)."""
        sampling = self._config.sampling
        interval = sampling.window_seconds / max(sampling.count - 1, 1)
        started = time.monotonic()
        frames: list[Frame] = []
        for i in range(sampling.count):
            if i > 0:
                delay = started + i * interval - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
            jpeg = await self._get_snapshot(camera_id)
            if not jpeg:
                _LOGGER.warning(
                    "Snapshot %d/%d for %s returned nothing",
                    i + 1,
                    sampling.count,
                    self.camera_name(camera_id),
                )
                continue
            frames.append(
                Frame(index=len(frames) + 1, captured_at=datetime.now(UTC), jpeg=jpeg)
            )
            _LOGGER.debug(
                "Captured frame %d/%d from %s (%.0f KiB)",
                len(frames),
                sampling.count,
                self.camera_name(camera_id),
                len(jpeg) / 1024,
            )
        return frames

    async def _get_snapshot(self, camera_id: str) -> bytes | None:
        assert self._protect is not None
        sampling = self._config.sampling
        try:
            if self._hybrid:
                return await self._protect.get_camera_snapshot(
                    camera_id, width=sampling.snapshot_width
                )
            return await self._protect.get_public_api_camera_snapshot(
                camera_id, high_quality=sampling.snapshot_high_quality
            )
        except Exception:
            _LOGGER.exception("Snapshot request failed for camera %s", camera_id)
            return None

    # ------------------------------------------------------------- lifecycle

    async def run_until_stopped(self) -> None:
        """Block until :meth:`request_stop` is called."""
        await self._stop.wait()

    def request_stop(self) -> None:
        """Ask :meth:`run_until_stopped` to return; cleanup happens in :meth:`stop`."""
        self._stop.set()

    async def stop(self) -> None:
        self._stop.set()
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        if self._protect is not None:
            try:
                await self._protect.async_disconnect_ws()
            except Exception:  # noqa: BLE001 - never prevent closing the session
                _LOGGER.debug("Disconnecting websocket failed", exc_info=True)
            await self._protect.close_session()
            self._protect = None
        _LOGGER.info("Protect connection closed")
