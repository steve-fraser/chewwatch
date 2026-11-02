"""Shared fixtures and a minimal uiprotect stub.

uiprotect v25 may not be installable in every environment (it needs the real
PyPI), so conftest installs a faithful stub of the tiny surface
``chewwatch.protect`` imports — enough to unit-test filtering and to drive the
full pipeline end-to-end against a fake Protect client.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum


def _install_uiprotect_stub() -> None:
    if "uiprotect" in sys.modules:
        return

    class EventChange(StrEnum):
        STARTED = "started"
        UPDATED = "updated"
        ENDED = "ended"
        REMOVED = "removed"

    class EventType(StrEnum):
        MOTION = "motion"
        SMART_DETECT = "smartDetectZone"
        SMART_DETECT_LINE = "smartDetectLine"
        SMART_DETECT_LOITER = "smartDetectLoiterZone"

    class SmartDetectObjectType(StrEnum):
        PERSON = "person"
        VEHICLE = "vehicle"
        ANIMAL = "animal"

    class _Cameras:
        def __init__(self, cameras: dict[str, "FakeCamera"]) -> None:
            self._cameras = cameras

        def values(self):
            return self._cameras.values()

        def get(self, key):
            return self._cameras.get(key)

        def __len__(self) -> int:
            return len(self._cameras)

    class _Bootstrap:
        def __init__(self, cameras: dict[str, "FakeCamera"]) -> None:
            self.cameras = _Cameras(cameras)
            self.nvr = types.SimpleNamespace(version="test")

    @dataclass
    class FakeCamera:
        id: str
        display_name: str
        mac: str = "AA:BB:CC:DD:EE:FF"
        name: str | None = None

    @dataclass
    class FakeProtectEvent:
        id: str
        type: EventType
        device_id: str
        start: datetime = field(default_factory=lambda: datetime.now(UTC))
        smart_detect_types: tuple[SmartDetectObjectType, ...] = ()

    class ProtectApiClient:
        """Replaced/extended by tests; mirrors the real constructor surface."""

        def __init__(self, host, port, username=None, password=None, api_key=None,
                     verify_ssl=True, **kwargs):
            self.host = host
            self.port = port
            self.api_key = api_key
            self.init_kwargs = dict(kwargs)  # e.g. store_sessions
            self.public_bootstrap = _Bootstrap({})
            self.bootstrap = _Bootstrap({})

        @classmethod
        def public_only(cls, host, port, api_key, **kwargs):
            return cls(host, port, api_key=api_key, **kwargs)

        async def update(self):
            self.bootstrap = _Bootstrap(getattr(self, "_public_cameras", {}))
            return self.bootstrap

        async def update_public(self):
            self.public_bootstrap = _Bootstrap(getattr(self, "_public_cameras", {}))
            return self.public_bootstrap

        async def subscribe_events_and_prime(self, callback):
            self.event_callback = callback
            return lambda: None

        def subscribe_events_websocket_state(self, callback):
            self.state_callback = callback

        async def get_public_api_camera_snapshot(self, camera_id, high_quality=False):
            return f"jpeg-for-{camera_id}".encode()

        async def get_camera_snapshot(self, camera_id, width=None, height=None, dt=None):
            return f"jpeg-wide-{camera_id}".encode()

        async def get_events_raw(self, **kwargs):
            return list(getattr(self, "_raw_events", []))

        async def get_event_thumbnail(self, event_id, **kwargs):
            return getattr(self, "_thumbnails", {}).get(event_id)

        async def async_disconnect_ws(self):
            pass

        async def close_session(self):
            pass

    pkg = types.ModuleType("uiprotect")
    pkg.EventChange = EventChange
    pkg.ProtectApiClient = ProtectApiClient
    pkg.ProtectEvent = FakeProtectEvent
    pkg.FakeCamera = FakeCamera
    data_mod = types.ModuleType("uiprotect.data")
    data_mod.EventType = EventType
    data_mod.SmartDetectObjectType = SmartDetectObjectType
    pkg.data = data_mod

    sys.modules["uiprotect"] = pkg
    sys.modules["uiprotect.data"] = data_mod


_install_uiprotect_stub()
