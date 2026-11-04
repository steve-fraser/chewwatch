"""Read-only viewer UI: browse recent animal events and their saved frames.

Strictly read-only: GET routes only. The UI never calls Protect, the
classifier or ntfy, and never writes to the decision log — it only reads
``data/decisions.jsonl`` and the frame folders under ``data/frames/``.
"""

from __future__ import annotations

import io
import json
import logging
import re
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from aiohttp import web

from .config import UiConfig
from .store import DecisionStore

_LOGGER = logging.getLogger(__name__)

# Store-generated names only — anything else 404s (traversal-safe by design).
_FOLDER_RE = re.compile(r"^[0-9]{8}-[0-9]{6}-[a-z0-9-]+$")
_FRAME_RE = re.compile(r"^frame-[0-9]+\.jpg$")

#: window presets offered in the header (hours)
_PRESETS = (1, 3, 6, 24, 168)
_DEFAULT_HOURS = 3.0
_MAX_HOURS = 24 * 31

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ChewWatch viewer</title>
<style>
  :root { color-scheme: dark; }
  body { font-family: system-ui, sans-serif; margin: 0; background: #14171c; color: #e8eaf0; }
  header { padding: 12px 20px; background: #1b1f27; border-bottom: 1px solid #2a2f3a;
           display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
           position: sticky; top: 0; z-index: 1; }
  header h1 { font-size: 18px; margin: 0 12px 0 0; }
  button, select { background: #262c38; color: #e8eaf0; border: 1px solid #3a4150;
                   border-radius: 8px; padding: 6px 12px; font-size: 14px; cursor: pointer; }
  button.active { background: #3b6ea5; border-color: #3b6ea5; }
  main { padding: 16px 20px; max-width: 1100px; margin: 0 auto; }
  .card { background: #1b1f27; border: 1px solid #2a2f3a; border-radius: 12px;
          padding: 14px 16px; margin-bottom: 14px; }
  .row { display: flex; flex-wrap: wrap; gap: 10px; align-items: baseline; }
  .cam { font-weight: 600; }
  .time { color: #9aa3b2; font-size: 13px; }
  .badge { font-size: 12px; padding: 2px 8px; border-radius: 999px; border: 1px solid; }
  .badge.alert { color: #ff8484; border-color: #ff8484; }
  .badge.ok { color: #7fd49a; border-color: #7fd49a; }
  .badge.muted { color: #9aa3b2; border-color: #9aa3b2; }
  .detail { margin: 6px 0 2px; font-size: 14px; }
  .verdict { font-size: 13px; color: #b9c1cf; margin-bottom: 10px; }
  .frames { display: flex; flex-wrap: wrap; gap: 8px; }
  .frames img { width: 160px; height: 90px; object-fit: cover; border-radius: 8px;
                border: 1px solid #2a2f3a; }
  .frames img:hover { border-color: #3b6ea5; }
  a { color: #7fb0e8; text-decoration: none; }
  .dl { font-size: 13px; margin-top: 8px; }
  .empty { color: #9aa3b2; padding: 48px 0; text-align: center; line-height: 1.7; }
  .missing { color: #9aa3b2; font-size: 13px; font-style: italic; }
</style>
</head>
<body>
<header>
  <h1>🐕 ChewWatch</h1>
  <div id="presets">
    <button data-hours="1">1h</button>
    <button data-hours="3" class="active">3h</button>
    <button data-hours="6">6h</button>
    <button data-hours="24">24h</button>
    <button data-hours="168">7d</button>
  </div>
  <select id="camera"><option value="">All cameras</option></select>
  <select id="outcome"><option value="">All outcomes</option></select>
  <span style="flex: 1"></span>
  <span id="bf-status" class="time"></span>
  <select id="bf-hours">
    <option value="1">last 1h</option>
    <option value="3">last 3h</option>
    <option value="6">last 6h</option>
    <option value="24" selected>last 24h</option>
    <option value="72">last 3d</option>
    <option value="168">last 7d</option>
  </select>
  <button id="backfill-btn" title="Classify animal events Protect recorded while chewwatch was not running (never sends notifications)">&#10227; Pull missed events</button>
</header>
<main id="content"><div class="empty">Loading…</div></main>
<script>
"use strict";
let hours = 3, camera = "", outcome = "";
const $ = (id) => document.getElementById(id);
// injected by the server from ui.refresh_seconds (0 = auto-refresh off)
const REFRESH_MS = __REFRESH_SECONDS__ * 1000;

function esc(s) {
  const d = document.createElement("div");
  d.textContent = s == null ? "" : String(s);
  return d.innerHTML;
}

function fmt(iso) {
  const d = new Date(iso);
  return isNaN(d) ? esc(iso) : d.toLocaleString();
}

function pct(v) { return Math.round((Number(v) || 0) * 100) + "%"; }

function fill(sel, values, current, label) {
  sel.innerHTML = "<option value=''>" + label + "</option>" +
    values.map(v => "<option" + (v === current ? " selected" : "") + ">" +
                   esc(v) + "</option>").join("");
}

function badgeClass(ev) {
  if (ev.alerted || ev.outcome === "alert") return "alert";
  if (["no_dog", "no_chewing", "toy_or_bone"].includes(ev.outcome)) return "ok";
  return "muted";
}

function verdictLine(v) {
  if (!v) return "";
  const parts = ["dog: " + (v.dog_present ? "yes" : "no")];
  parts.push(v.chewing ? "chewing " + esc(v.object || "?") : "not chewing");
  if (v.on_counter) parts.push("ON COUNTER (" + pct(v.counter_confidence) + ")");
  parts.push("conf " + pct(v.confidence));
  return parts.join(" · ") + (v.reason ? " — " + esc(v.reason) : "");
}

function card(ev) {
  const el = document.createElement("div");
  el.className = "card";
  const frames = (ev.frames || []).map(f =>
    "<a href='/frames/" + ev.folder + "/" + f + "' target='_blank' rel='noopener'>" +
    "<img loading='lazy' src='/frames/" + ev.folder + "/" + f + "' alt='" + f + "'></a>"
  ).join("");
  const frameBits = frames
    ? "<div class='frames'>" + frames + "</div>" +
      "<div class='dl'><a href='/frames/" + ev.folder + "/all.zip'>⬇ download all frames (.zip)</a></div>"
    : "<div class='missing'>No frames saved (pruned after retention, or snapshot failure)</div>";
  el.innerHTML =
    "<div class='row'>" +
      "<span class='cam'>" + esc(ev.camera_name) + "</span>" +
      "<span class='time'>" + fmt(ev.started_at) + "</span>" +
      "<span class='badge " + badgeClass(ev) + "'>" + esc(ev.outcome) + "</span>" +
      (ev.backfill ? "<span class='badge muted'>backfill</span>" : "") +
    "</div>" +
    "<div class='detail'>" + esc(ev.detail || ev.error || "") + "</div>" +
    "<div class='verdict'>" + verdictLine(ev.verdict) + "</div>" +
    frameBits;
  return el;
}

async function load() {
  const params = new URLSearchParams({ hours, camera, outcome });
  let data;
  try {
    const res = await fetch("/api/events?" + params);
    data = await res.json();
  } catch (err) {
    $("content").innerHTML = "<div class='empty'>Could not load events: " + esc(err) + "</div>";
    return;
  }
  fill($("camera"), data.cameras, camera, "All cameras");
  fill($("outcome"), data.outcomes, outcome, "All outcomes");
  const box = $("content");
  if (!data.events.length) {
    box.innerHTML = "<div class='empty'>No animal events in the last " +
      (hours >= 24 ? (hours / 24) + " days" : hours + " h") + ".<br>" +
      "Frames of older events are pruned after storage.retention_days; " +
      "try a wider window or check that the camera raised animal detections.</div>";
    return;
  }
  box.innerHTML = "";
  for (const ev of data.events) box.appendChild(card(ev));
}

document.getElementById("presets").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-hours]");
  if (!b) return;
  hours = Number(b.dataset.hours);
  document.querySelectorAll("#presets button").forEach(
    x => x.classList.toggle("active", x === b));
  load();
});
$("camera").addEventListener("change", () => { camera = $("camera").value; load(); });
$("outcome").addEventListener("change", () => { outcome = $("outcome").value; load(); });

const bfBtn = $("backfill-btn"), bfHours = $("bf-hours"), bfStatus = $("bf-status");
bfBtn.addEventListener("click", async () => {
  bfBtn.disabled = true;
  bfStatus.textContent = "";
  try {
    const res = await fetch("/api/backfill", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({hours: Number(bfHours.value)}),
    });
    const data = await res.json();
    if (res.status === 202) {
      bfStatus.textContent = "Pulling missed events… (this can take a while)";
      pollBackfill();
    } else {
      bfStatus.textContent = data.error || "Could not start the pull";
      bfBtn.disabled = false;
    }
  } catch (err) {
    bfStatus.textContent = "Backfill request failed: " + err;
    bfBtn.disabled = false;
  }
});

let bfPolling = false;
function pollBackfill() {
  if (bfPolling) return;
  bfPolling = true;
  const timer = setInterval(async () => {
    let st;
    try { st = await (await fetch("/api/backfill")).json(); }
    catch (e) { return; }  // transient error — keep polling
    if (st.running) return;
    clearInterval(timer);
    bfPolling = false;
    bfBtn.disabled = false;
    const r = st.last_result;
    if (r && !r.error) {
      bfStatus.textContent = "Pulled last " + r.hours + " h — " + r.processed +
        " new, " + r.skipped + " already known" +
        (r.failed ? ", " + r.failed + " failed" : "");
    } else if (r) {
      bfStatus.textContent = "Backfill failed: " + (r.error || "see container logs");
    }
    load();
  }, 2000);
}

load();

// Auto-refresh: serial ticks (no overlapping loads), paused in hidden tabs.
async function refreshTick() {
  try {
    if (!document.hidden) await load();
  } catch (e) {
    console.warn("auto-refresh failed, retrying next interval", e);
  } finally {
    setTimeout(refreshTick, REFRESH_MS);
  }
}
if (REFRESH_MS > 0) setTimeout(refreshTick, REFRESH_MS);
</script>
</body>
</html>
"""


def _parse_ts(value: object) -> datetime | None:
    try:
        ts = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts


def _parse_hours(raw: str | None) -> float:
    try:
        hours = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return _DEFAULT_HOURS
    return min(max(hours, 0.01), _MAX_HOURS)


def _filter_records(
    records: list[dict],
    *,
    hours: float,
    camera: str,
    outcome: str,
    now: datetime,
) -> list[dict]:
    """Newest-first events within the window, matching the optional filters."""
    cutoff = now - timedelta(hours=hours)
    picked: list[tuple[datetime, dict]] = []
    for rec in records:
        ts = _parse_ts(rec.get("started_at") or rec.get("decided_at"))
        if ts is None or ts < cutoff or ts > now + timedelta(minutes=5):
            continue
        if camera and str(rec.get("camera_name") or "").lower() != camera.lower():
            continue
        if outcome and rec.get("outcome") != outcome:
            continue
        picked.append((ts, rec))
    picked.sort(key=lambda item: item[0], reverse=True)
    return [rec for _, rec in picked]


def _frames_dir(store: DecisionStore, folder: str) -> Path | None:
    """Validated on-disk path for a store-generated folder name, else None."""
    if not _FOLDER_RE.fullmatch(folder):
        return None
    frames_root = (store.log_path.parent / "frames").resolve()
    path = (frames_root / folder).resolve()
    if frames_root not in path.parents:
        return None
    return path


def _list_frames(store: DecisionStore, record: dict) -> list[str]:
    folder = Path(str(record.get("frame_dir") or "")).name
    if not record.get("frame_dir"):
        return []
    path = _frames_dir(store, folder)
    if path is None or not path.is_dir():
        return []
    try:
        names = [p.name for p in path.iterdir() if _FRAME_RE.fullmatch(p.name)]
    except OSError:
        return []
    return sorted(names, key=lambda n: int(n[6:-4]))


class UiApp:
    """Handlers for the read-only viewer."""

    def __init__(
        self, store: DecisionStore, backfill=None, refresh_seconds: float = 30.0
    ) -> None:
        self._store = store
        self._backfill = backfill  # the running service, when wired
        self._refresh_seconds = refresh_seconds

    def build(self) -> web.Application:
        app = web.Application()
        # /all.zip must be registered before the generic /{name} route
        app.router.add_get("/frames/{folder}/all.zip", self.zip_frames)
        app.router.add_get("/frames/{folder}/{name}", self.frame)
        app.router.add_get("/api/backfill", self.backfill_status)
        app.router.add_post("/api/backfill", self.start_backfill)
        app.router.add_get("/api/events", self.api_events)
        app.router.add_get("/", self.index)
        return app

    async def index(self, request: web.Request) -> web.Response:
        page = _PAGE.replace("__REFRESH_SECONDS__", repr(float(self._refresh_seconds)))
        return web.Response(text=page, content_type="text/html")

    async def api_events(self, request: web.Request) -> web.Response:
        records = self._store.load_records(self._store.log_path)
        now = datetime.now(UTC)
        hours = _parse_hours(request.query.get("hours"))
        camera = (request.query.get("camera") or "").strip()
        outcome = (request.query.get("outcome") or "").strip()

        events = []
        for rec in _filter_records(records, hours=hours, camera=camera, outcome=outcome, now=now):
            folder = Path(str(rec.get("frame_dir") or "")).name
            events.append(
                {
                    **rec,
                    "folder": folder if _FOLDER_RE.fullmatch(folder) else "",
                    "frames": _list_frames(self._store, rec),
                }
            )
        return web.json_response(
            {
                "window_hours": hours,
                "cameras": sorted(
                    {str(r.get("camera_name") or "") for r in records} - {""}
                ),
                "outcomes": sorted(
                    {str(r.get("outcome") or "") for r in records} - {""}
                ),
                "events": events,
                "generated_at": now.isoformat(),
            },
            dumps=lambda obj: json.dumps(obj, ensure_ascii=False),
        )

    async def backfill_status(self, request: web.Request) -> web.Response:
        if self._backfill is None:
            return web.json_response({"running": False, "last_result": None})
        return web.json_response(self._backfill.backfill_status())

    async def start_backfill(self, request: web.Request) -> web.Response:
        if self._backfill is None:
            return web.json_response(
                {"error": "backfill requires the running chewwatch service"}, status=503
            )
        if not self._backfill.backfill_ready():
            return web.json_response(
                {
                    "error": "Backfill needs hybrid auth: set protect.username and "
                    "protect.password (a local Protect user with camera access), "
                    "then restart"
                },
                status=400,
            )
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001 - malformed body
            return web.json_response({"error": "expected a JSON body"}, status=400)
        try:
            hours = float(body.get("hours"))
        except (TypeError, ValueError, AttributeError):
            return web.json_response({"error": "hours must be a number"}, status=400)
        if not self._backfill.start_backfill(hours):
            return web.json_response({"error": "a backfill is already running"}, status=409)
        return web.json_response({"started": True}, status=202)

    async def frame(self, request: web.Request) -> web.Response:
        folder = request.match_info["folder"]
        name = request.match_info["name"]
        base = _frames_dir(self._store, folder)
        if base is None or not _FRAME_RE.fullmatch(name):
            raise web.HTTPNotFound(text="no such frame")
        path = base / name
        if not path.is_file():
            raise web.HTTPNotFound(text="no such frame")
        headers = {}
        if request.query.get("download"):
            headers["Content-Disposition"] = f'attachment; filename="{name}"'
        return web.Response(body=path.read_bytes(), content_type="image/jpeg", headers=headers)

    async def zip_frames(self, request: web.Request) -> web.Response:
        folder = request.match_info["folder"]
        base = _frames_dir(self._store, folder)
        if base is None or not base.is_dir():
            raise web.HTTPNotFound(text="no such event folder")
        names = [p.name for p in base.iterdir() if _FRAME_RE.fullmatch(p.name)]
        if not names:
            raise web.HTTPNotFound(text="no frames in folder")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
            for name in sorted(names, key=lambda n: int(n[6:-4])):
                zf.write(base / name, arcname=f"{folder}/{name}")
        return web.Response(
            body=buf.getvalue(),
            content_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{folder}.zip"'},
        )


def build_app(
    store: DecisionStore, backfill=None, refresh_seconds: float = 30.0
) -> web.Application:
    """The read-only viewer application (``backfill`` = the running service)."""
    return UiApp(store, backfill, refresh_seconds).build()


async def start_ui(
    config: UiConfig, store: DecisionStore, backfill=None
) -> web.AppRunner:
    """Serve the viewer on 0.0.0.0:<port>; caller owns the runner's cleanup."""
    runner = web.AppRunner(
        build_app(store, backfill, refresh_seconds=config.refresh_seconds)
    )
    await runner.setup()
    site = web.TCPSite(runner, host="0.0.0.0", port=config.port)
    await site.start()
    _LOGGER.info("Viewer UI on http://0.0.0.0:%d (read-only)", config.port)
    return runner
