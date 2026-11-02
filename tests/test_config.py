"""Tests for config loading, env substitution and camera matching."""

from __future__ import annotations

from pathlib import Path

import pytest

from chewwatch.config import ConfigError, camera_matches, load_config

VALID_YAML = """
protect:
  host: 192.168.1.10
  api_key: "prot-key"
  cameras: ["Living Room", "aa:bb:cc:dd:ee:ff"]
  counter_cameras: ["Kitchen"]
classifier:
  api_key: "kim-key"
  threshold: 0.8
sampling:
  count: 4
alerts:
  cooldown_minutes: 5
  ntfy:
    topic: "chewwatch-test"
    token: "${NTFY_TOKEN}"
storage:
  data_dir: "/tmp/chewwatch-test"
"""

FULL_YAML = """\
protect:
  host: 10.0.0.1
  api_key: pk
  cameras: []
classifier:
  api_key: kk
  threshold: 0.7
sampling:
  count: 5
  window_seconds: 10
alerts:
  cooldown_minutes: 8
  ntfy:
    topic: t
storage:
  data_dir: /tmp/x
"""


def _write(tmp_path, content: str):
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def _mutated(replacements: dict[str, str]) -> str:
    text = FULL_YAML
    for old, new in replacements.items():
        assert old in text, f"anchor not found: {old!r}"
        text = text.replace(old, new)
    return text


def test_load_valid(tmp_path):
    cfg = load_config(_write(tmp_path, VALID_YAML))
    assert cfg.protect.host == "192.168.1.10"
    assert cfg.protect.api_key == "prot-key"
    assert cfg.protect.verify_ssl is False
    assert cfg.protect.cameras == ["Living Room", "aa:bb:cc:dd:ee:ff"]
    assert cfg.protect.counter_cameras == ["Kitchen"]
    assert cfg.classifier.model == "glm-5.3-flash"
    assert cfg.classifier.base_url == "https://llm.kimchi.dev/openai/v1"
    assert cfg.classifier.threshold == 0.8
    assert cfg.classifier.ignore_toys is True
    assert cfg.sampling.count == 4
    assert cfg.sampling.window_seconds == 10.0
    assert cfg.alerts.cooldown_minutes == 5.0
    assert cfg.alerts.ntfy.topic == "chewwatch-test"
    assert cfg.storage.data_dir == "/tmp/chewwatch-test"


def test_env_substitution(tmp_path, monkeypatch):
    monkeypatch.setenv("KIMCHI_API_KEY", "env-secret")
    monkeypatch.setenv("NTFY_TOKEN", "tok_123")
    monkeypatch.delenv("MISSING_VAR", raising=False)
    content = VALID_YAML.replace('"kim-key"', "${KIMCHI_API_KEY}")
    cfg = load_config(_write(tmp_path, content))
    assert cfg.classifier.api_key == "env-secret"
    assert cfg.alerts.ntfy.token == "tok_123"


def test_env_substitution_default_and_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("NO_SUCH_VAR", raising=False)
    content = VALID_YAML.replace(
        'topic: "chewwatch-test"', 'topic: "${NTFY_TOPIC:-fallback-topic}"'
    ).replace('token: "${NTFY_TOKEN}"', 'token: "${NO_SUCH_VAR}"')
    cfg = load_config(_write(tmp_path, content))
    assert cfg.alerts.ntfy.topic == "fallback-topic"
    assert cfg.alerts.ntfy.token == ""


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_empty_file_is_invalid(tmp_path):
    with pytest.raises(ConfigError, match="host"):
        load_config(_write(tmp_path, ""))


def test_non_mapping_top_level(tmp_path):
    with pytest.raises(ConfigError, match="mapping"):
        load_config(_write(tmp_path, "- a\n- b\n"))


def test_unparsable_yaml(tmp_path):
    with pytest.raises(ConfigError, match="parse"):
        load_config(_write(tmp_path, "protect: [unclosed\n"))


@pytest.mark.parametrize(
    "replacements, match",
    [
        ({"  api_key: pk\n": ""}, "protect.api_key"),
        ({"  host: 10.0.0.1\n": ""}, "protect.host"),
        ({"  api_key: kk\n": ""}, "classifier.api_key"),
        ({"  threshold: 0.7\n": "  threshold: 1.5\n"}, "threshold"),
        ({"  count: 5\n": "  count: 1\n"}, "sampling.count"),
        ({"  count: 5\n": "  count: 99\n"}, "sampling.count"),
        ({"  window_seconds: 10\n": "  window_seconds: 0\n"}, "window_seconds"),
        ({"    topic: t\n": ""}, "ntfy.topic"),
        ({"  cooldown_minutes: 8\n": "  cooldown_minutes: -1\n"}, "cooldown"),
        ({"  api_key: pk\n": "  username: lone\n"}, "username and password"),
    ],
)
def test_validation_errors(tmp_path, replacements, match):
    with pytest.raises(ConfigError, match=match):
        load_config(_write(tmp_path, _mutated(replacements)))


def test_camera_matching():
    kw = {
        "camera_id": "cam-1",
        "camera_name": "Living Room",
        "camera_mac": "AA:BB:CC:DD:EE:01",
    }
    assert camera_matches(selectors=[], **kw) is True  # empty = all
    assert camera_matches(selectors=["living room"], **kw) is True  # case-insensitive
    assert camera_matches(selectors=["cam-1"], **kw) is True  # by id
    assert camera_matches(selectors=["aa:bb:cc:dd:ee:01"], **kw) is True  # by mac
    assert camera_matches(selectors=["aabbccddeE01"], **kw) is True  # mac, no separators
    assert camera_matches(selectors=["Kitchen"], **kw) is False
    assert camera_matches(selectors=["", "  "], **kw) is False  # blank selectors
    assert camera_matches(selectors=["Kitchen", "living ROOM"], **kw) is True
    assert camera_matches(selectors=["x"], camera_id=None, camera_name=None, camera_mac=None) is False


def test_counter_cameras_default_empty(tmp_path):
    cfg = load_config(_write(tmp_path, FULL_YAML))
    assert cfg.protect.counter_cameras == []


def test_ui_defaults_and_parse(tmp_path):
    cfg = load_config(_write(tmp_path, FULL_YAML))
    assert cfg.ui.enabled is True
    assert cfg.ui.port == 8080
    content = FULL_YAML + "\nui:\n  enabled: false\n  port: 9090\n"
    cfg2 = load_config(_write(tmp_path, content))
    assert cfg2.ui.enabled is False
    assert cfg2.ui.port == 9090


def test_ui_port_validation(tmp_path):
    with pytest.raises(ConfigError, match="ui.port"):
        load_config(_write(tmp_path, FULL_YAML + "\nui:\n  port: 80\n"))


def test_example_config_parses(monkeypatch):
    monkeypatch.setenv("PROTECT_API_KEY", "pk")
    monkeypatch.setenv("KIMCHI_API_KEY", "kk")
    monkeypatch.setenv("NTFY_TOPIC", "t")
    example = Path(__file__).resolve().parents[1] / "config.example.yaml"
    cfg = load_config(example)
    assert cfg.protect.host == "192.168.1.10"
    assert cfg.protect.counter_cameras == []  # documented but commented out
    assert "counter_cameras:" in example.read_text(encoding="utf-8")
