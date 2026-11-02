# ChewWatch 🐕

**ChewWatch** watches your UniFi Protect cameras for a dog chewing things it
shouldn't, and pushes a snapshot to your phone.

```
Protect "animal" smart-detect ──▶ sample 5 frames over 10s ──▶ Kimchi vision model
                                                                      │
                       decisions.jsonl + frames/ ◀─────────────────────┤
                                │                                     ▼
                                └────────────────── ntfy alert with the best frame
```

- **Event listener** — connects to Protect's event stream with the unofficial
  [`uiprotect`](https://github.com/uilibs/uiprotect) library (typed Public
  Integration API over a websocket) and filters for smart-detect **animal**
  events on the cameras you choose.
- **Frame sampler** — one event ≠ one chewing pose, so 4–6 snapshots are taken
  across ~10 seconds and judged as a sequence.
- **Classifier** — all frames go in one request to a Kimchi vision model
  ([Kimchi Inference](https://kimchi.dev/inference), OpenAI-compatible) which
  answers with strict JSON: `dog_present`, `chewing`, `object`, `confidence`,
  `reason`, `best_frame`. On *counter cameras* (`protect.counter_cameras`) the
  same request also answers `on_counter` with its own `counter_confidence`.
- **Decision & dedupe** — alert only when chewing is `true`, confidence clears
  the threshold (default `0.7`), the object isn't a toy/bone (configurable),
  and the camera isn't in its cooldown window (default 8 min).
- **Notifier** — `PUT`s the best frame to your ntfy topic with the verdict in
  the title.
- **Log & review** — every decision (alert or not) is written with its frames
  to `data/decisions.jsonl` + `data/frames/`, so you can tune the prompt,
  threshold and cooldown against real false positives and misses.

## Requirements

- UniFi Protect console with a camera that does AI **animal** smart detection
  (G5/G6 line, or any camera paired with an AI Key).
- A Protect **Integration API key** with camera read access
  (Protect → Settings → Integrations).
- A [Kimchi](https://app.kimchi.dev/) account and API key.
- An [ntfy](https://ntfy.sh) topic (or your self-hosted ntfy server).
- Docker, or Python 3.11+.

## Quick start (Docker)

```bash
cd chewwatch
cp config.example.yaml config.yaml
cp .env.example .env
# edit config.yaml (host, cameras) and .env (keys + topic)

openssl rand -hex 16          # print a long random ntfy topic name

docker compose up -d --build
docker compose logs -f chewwatch
```

`data/` (decision log + frames) is created next to the compose file.

## Quick start (manual)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m chewwatch --config config.yaml
```

## Configuration

See [`config.example.yaml`](config.example.yaml). Secrets use `${VAR}`
substitution from the environment (docker-compose loads `.env` automatically).

| Key | Default | Meaning |
| --- | --- | --- |
| `protect.host` / `port` | — | UniFi OS console address |
| `protect.api_key` | — | Protect Integration API key (required) |
| `protect.username` / `password` | `""` | Optional local user; enables private-API snapshots, and is **required for backfill** |
| `protect.cameras` | `[]` | Names / ids / MACs to watch; empty = all |
| `protect.counter_cameras` | `[]` | Watched cameras that also get the "is the dog on the counter?" check |
| `classifier.model` | `glm-5.3-flash` | Any Kimchi vision model (`kimi-k3`, `minimax-m3`, …) |
| `classifier.threshold` | `0.7` | Minimum confidence to alert |
| `classifier.ignore_toys` | `true` | Toy/bone chewing never alerts (still logged) |
| `sampling.count` / `window_seconds` | `5` / `10` | Frames per event and the spread |
| `alerts.cooldown_minutes` | `8` | Per-camera alert cooldown |
| `alerts.ntfy.topic` | — | **Pick a long random name — topics are public unless your server has auth.** |
| `storage.retention_days` | `30` | Frame folders older than this are pruned — at startup and every 12 h while running (JSONL log is kept) |

### Counter checks

For cameras listed in `protect.counter_cameras` (they must also be in
`protect.cameras`), every animal event is judged for two things in the same
model request: chewing as usual, plus whether the dog is up on a kitchen
counter, worktop or table. "On the counter" counts the dog reaching onto the
surface — front paws up on it (standing on hind legs / leaning over the edge)
or fully up on it, i.e. close enough to grab something; a dog on the floor or
on the sofa does not. Either trigger alerts — same ntfy topic, same
per-camera cooldown — and the ntfy title tells you which one fired. The toy
rule only suppresses chewing alerts, so chewing a toy on the counter still
alerts. `classifier.threshold` applies to both checks.

### Testing your ntfy setup

```bash
python -m chewwatch --config config.yaml --test-notify
```

## Viewer UI

A small read-only web page ships with chewwatch (`ui.enabled: true` by
default): open `http://<host>:8080` to browse the animal events of the last
few hours — window presets 1h/**3h**/6h/24h/7d plus camera and outcome
filters — and view or download each event's saved frames (individual JPEGs,
or one `.zip` per event).

- Strictly read-only: it only reads `data/decisions.jsonl` and
  `data/frames/`. It never calls Protect, the classifier or ntfy, and never
  writes anything — alerts still come only via ntfy.
- No auth: it is meant for your home LAN. Don't expose the port to the
  internet.
- `ui.enabled: false` turns it off; `ui.port` changes the port (compose maps
  `8080:8080`).
- Events older than `storage.retention_days` keep their log lines but their
  frames are pruned — the UI lists them without thumbnails.

## Reviewing decisions

Every event produces one JSON line in `data/decisions.jsonl`:

```json
{"id": "9f2c…", "camera_name": "Living Room", "alerted": true,
 "outcome": "alert", "detail": "Chewing detected: couch (confidence 0.88)",
 "verdict": {"dog_present": true, "chewing": true, "object": "couch",
             "confidence": 0.88, "reason": "Head down, mouth working the cushion across frames 2-4", "best_frame": 3},
 "frame_dir": "data/frames/20260101-101530-living-room"}
```

Outcomes: `alert`, `no_dog`, `no_chewing`, `toy_or_bone`, `low_confidence`,
`cooldown`, `no_frames`, `classifier_error`, `notify_error`. The `detail`
field names the trigger — chewing, the counter, or both.

To review what the model saw and where it was wrong:

```bash
grep '"outcome": "low_confidence"' data/decisions.jsonl | tail -5
ls data/frames/ | tail
```

- **Too many false positives** → raise `threshold` or add objects to the
  ignore list in `chewwatch/classifier.py` (`TOY_OBJECTS`).
- **Missed chewing** → lower `threshold`, try `kimi-k3` as the model, or raise
  `sampling.count`.
- **One session spams alerts** → raise `alerts.cooldown_minutes`.
- **Counter alerts fire on the sofa** → the prompt only counts
  counters/worktops/tables; if the model is still over-eager, raise
  `classifier.threshold` (it applies to both checks) or drop the camera from
  `protect.counter_cameras`.

## Costs

`glm-5.3-flash` is ~$0.15 per 1M input tokens; a 5-frame classification is
roughly 5–10k image+text tokens — well under a cent per event.

## Notes & limitations

- `uiprotect` is unofficial; the public Integration API it uses is official and
  documented, but pin the version (`requirements.txt` does) and expect
  occasional breaking bumps.
- Snapshots are "most recent keyframe at request time" — a few seconds of
  drift is normal; sampling several frames is the mitigation.
- ntfy attachments on ntfy.sh expire after 3 h and are capped at 15 MB; frames
  are also kept locally under `data/frames/`.

## Development

```bash
pip install -r requirements-dev.txt
pytest
```
