"""Configuration loading: YAML with ${ENV} / ${ENV:-default} substitution."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(
    r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}"
)


class ConfigError(Exception):
    """Raised when the configuration file is missing or invalid."""


def _substitute_env(value: Any) -> Any:
    """Recursively replace ``${VAR}`` / ``${VAR:-default}`` in string values."""
    if isinstance(value, str):
        return _ENV_PATTERN.sub(
            lambda m: os.environ.get(m.group("name"), m.group("default") or ""),
            value,
        )
    if isinstance(value, dict):
        return {k: _substitute_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute_env(v) for v in value]
    return value


@dataclass(frozen=True)
class ProtectConfig:
    host: str = ""
    port: int = 443
    username: str = ""
    password: str = ""
    api_key: str = ""
    verify_ssl: bool = False
    cameras: list[str] = field(default_factory=list)  # names / ids / MACs; empty = all
    # cameras that additionally get the "is the dog on the counter?" check
    counter_cameras: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ClassifierConfig:
    base_url: str = "https://llm.kimchi.dev/openai/v1"
    model: str = "glm-5.3-flash"
    api_key: str = ""
    threshold: float = 0.7
    ignore_toys: bool = True
    request_timeout_seconds: float = 120.0


@dataclass(frozen=True)
class SamplingConfig:
    count: int = 5
    window_seconds: float = 10.0
    snapshot_width: int = 1600  # private-API snapshots (username/password set)
    snapshot_high_quality: bool = False  # public-API snapshots (api key only)
    max_concurrent_per_camera: int = 1


@dataclass(frozen=True)
class NtfyConfig:
    server: str = "https://ntfy.sh"
    topic: str = ""
    token: str = ""
    priority: str = "high"


@dataclass(frozen=True)
class AlertsConfig:
    cooldown_minutes: float = 8.0
    ntfy: NtfyConfig = field(default_factory=NtfyConfig)


@dataclass(frozen=True)
class StorageConfig:
    data_dir: str = "./data"
    retention_days: int = 30


@dataclass(frozen=True)
class UiConfig:
    """Read-only viewer UI (recent events + frames); no alerting, no auth."""

    enabled: bool = True
    port: int = 8080


@dataclass(frozen=True)
class ChewWatchConfig:
    protect: ProtectConfig = field(default_factory=ProtectConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    alerts: AlertsConfig = field(default_factory=AlertsConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    ui: UiConfig = field(default_factory=UiConfig)


def _build(root: dict[str, Any]) -> ChewWatchConfig:
    protect_raw = dict(root.get("protect") or {})
    classifier_raw = dict(root.get("classifier") or {})
    sampling_raw = dict(root.get("sampling") or {})
    alerts_raw = dict(root.get("alerts") or {})
    ntfy_raw = dict(alerts_raw.get("ntfy") or {})
    storage_raw = dict(root.get("storage") or {})
    ui_raw = dict(root.get("ui") or {})

    cfg = ChewWatchConfig(
        protect=ProtectConfig(
            host=str(protect_raw.get("host", "")),
            port=int(protect_raw.get("port", 443)),
            username=str(protect_raw.get("username", "")),
            password=str(protect_raw.get("password", "")),
            api_key=str(protect_raw.get("api_key", "")),
            verify_ssl=bool(protect_raw.get("verify_ssl", False)),
            cameras=[str(c) for c in protect_raw.get("cameras", [])],
            counter_cameras=[str(c) for c in protect_raw.get("counter_cameras", [])],
        ),
        classifier=ClassifierConfig(
            base_url=str(classifier_raw.get("base_url", "https://llm.kimchi.dev/openai/v1")),
            model=str(classifier_raw.get("model", "glm-5.3-flash")),
            api_key=str(classifier_raw.get("api_key", "")),
            threshold=float(classifier_raw.get("threshold", 0.7)),
            ignore_toys=bool(classifier_raw.get("ignore_toys", True)),
            request_timeout_seconds=float(
                classifier_raw.get("request_timeout_seconds", 120.0)
            ),
        ),
        sampling=SamplingConfig(
            count=int(sampling_raw.get("count", 5)),
            window_seconds=float(sampling_raw.get("window_seconds", 10.0)),
            snapshot_width=int(sampling_raw.get("snapshot_width", 1600)),
            snapshot_high_quality=bool(sampling_raw.get("snapshot_high_quality", False)),
            max_concurrent_per_camera=int(
                sampling_raw.get("max_concurrent_per_camera", 1)
            ),
        ),
        alerts=AlertsConfig(
            cooldown_minutes=float(alerts_raw.get("cooldown_minutes", 8.0)),
            ntfy=NtfyConfig(
                server=str(ntfy_raw.get("server", "https://ntfy.sh")),
                topic=str(ntfy_raw.get("topic", "")),
                token=str(ntfy_raw.get("token", "")),
                priority=str(ntfy_raw.get("priority", "high")),
            ),
        ),
        storage=StorageConfig(
            data_dir=str(storage_raw.get("data_dir", "./data")),
            retention_days=int(storage_raw.get("retention_days", 30)),
        ),
        ui=UiConfig(
            enabled=bool(ui_raw.get("enabled", True)),
            port=int(ui_raw.get("port", 8080)),
        ),
    )
    _validate(cfg)
    return cfg


def _validate(cfg: ChewWatchConfig) -> None:
    errors: list[str] = []

    p = cfg.protect
    if not p.host:
        errors.append("protect.host is required")
    if not p.api_key:
        errors.append(
            "protect.api_key is required (create an Integration API key in "
            "UniFi Protect under Settings → Integrations)"
        )
    if bool(p.username) != bool(p.password):
        errors.append("protect: set both username and password, or neither")

    c = cfg.classifier
    if not c.api_key:
        errors.append("classifier.api_key is required (set ${KIMCHI_API_KEY} in the YAML)")
    if not 0.0 <= c.threshold <= 1.0:
        errors.append("classifier.threshold must be within 0.0 .. 1.0")
    if not c.model:
        errors.append("classifier.model is required")

    s = cfg.sampling
    if not 2 <= s.count <= 12:
        errors.append("sampling.count must be within 2 .. 12")
    if s.window_seconds <= 0:
        errors.append("sampling.window_seconds must be > 0")
    if s.snapshot_width < 320:
        errors.append("sampling.snapshot_width must be >= 320")
    if s.max_concurrent_per_camera < 1:
        errors.append("sampling.max_concurrent_per_camera must be >= 1")

    a = cfg.alerts
    if a.cooldown_minutes < 0:
        errors.append("alerts.cooldown_minutes must be >= 0")
    n = a.ntfy
    if not n.topic:
        errors.append("alerts.ntfy.topic is required (use a long random topic name)")
    if n.server and not n.server.startswith(("http://", "https://")):
        errors.append("alerts.ntfy.server must start with http:// or https://")

    if cfg.storage.retention_days < 0:
        errors.append("storage.retention_days must be >= 0")
    if not 1024 <= cfg.ui.port <= 65535:
        errors.append("ui.port must be within 1024 .. 65535")

    if errors:
        raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(errors))


def load_config(path: Path | str) -> ChewWatchConfig:
    """Load, environment-substitute and validate the YAML configuration."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as err:
        raise ConfigError(f"Could not parse YAML {path}: {err}") from err
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"Top level of {path} must be a mapping")
    return _build(_substitute_env(raw))


def camera_matches(
    *,
    selectors: list[str],
    camera_id: str | None,
    camera_name: str | None,
    camera_mac: str | None,
) -> bool:
    """True when a camera is selected. Empty selector list selects every camera.

    Matching is case-insensitive against the camera name, id or MAC address;
    MACs also match with ``:``/``-`` separators stripped.
    """
    if not selectors:
        return True

    def norm(value: str | None) -> str:
        return (value or "").strip().lower()

    mac = norm(camera_mac).replace(":", "").replace("-", "")
    for selector in selectors:
        s = norm(selector)
        if not s:
            continue
        if s == norm(camera_id) or s == norm(camera_name):
            return True
        if mac and s.replace(":", "").replace("-", "") == mac:
            return True
    return False
