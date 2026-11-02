"""Command line entrypoint: ``python -m chewwatch``."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

from chewwatch import __version__
from chewwatch.config import ChewWatchConfig, ConfigError, load_config
from chewwatch.notify import Notifier, NotifyError
from chewwatch.ui import start_ui

try:  # the only module that needs uiprotect at import time
    from chewwatch.service import ChewWatch
except ImportError as _err:  # pragma: no cover - depends on environment
    if "uiprotect" in str(_err):
        ChewWatch = None  # type: ignore[assignment]
    else:
        raise

_LOGGER = logging.getLogger("chewwatch")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="chewwatch",
        description="Watch UniFi Protect cameras for a dog chewing things it shouldn't.",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("config.yaml"),
        help="Path to the YAML config file (default: ./config.yaml)",
    )
    parser.add_argument(
        "--test-notify",
        action="store_true",
        help="Send a test notification to the configured ntfy topic and exit",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )
    parser.add_argument("--version", action="version", version=f"chewwatch {__version__}")
    return parser.parse_args(argv)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if not verbose:
        for noisy in ("asyncio", "aiohttp.access", "uiprotect"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


async def _run_test_notify(config: ChewWatchConfig) -> int:
    notifier = Notifier(config.alerts.ntfy)
    try:
        await notifier.send_test()
    except NotifyError as err:
        print(f"FAILED: {err}", file=sys.stderr)
        return 1
    print(f"OK — notification sent to {notifier.topic_url}")
    return 0


async def _run_service(args: argparse.Namespace) -> int:
    config = load_config(args.config)  # validate before anything else
    if ChewWatch is None:
        print(
            "uiprotect is not installed — run: pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1
    service = ChewWatch(config)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, service.request_stop)
        except NotImplementedError:  # pragma: no cover - non-POSIX
            pass
    ui_runner = None
    if config.ui.enabled:
        ui_runner = await start_ui(config.ui, service.store, backfill=service)
    try:
        await service.run()
    finally:
        if ui_runner is not None:
            await ui_runner.cleanup()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _setup_logging(args.verbose)

    try:
        if args.test_notify:
            config = load_config(args.config)
            return asyncio.run(_run_test_notify(config))
        return asyncio.run(_run_service(args))
    except ConfigError as err:
        print(f"Configuration error: {err}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover - interactive use
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
