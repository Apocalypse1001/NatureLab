---
name: run-naturelab
description: Build, run, and drive NatureLab (physics sandbox - FastAPI/Warp backend + Three.js frontend). Use when asked to start NatureLab, launch the app, take a screenshot of the simulation, run its tests, or interact with the running sim (add objects, set the edge inflow, read the water field, read a GAUGE).
---

NatureLab is a physics sandbox: a Python FastAPI + NVIDIA Warp backend that streams a
GPU shallow-water simulation over a WebSocket to a Three.js frontend it also serves.
Agents drive it with **`.claude/skills/run-naturelab/driver.mjs`** — a stdin REPL that
owns the backend *and* a headless Chrome, so you can place objects, run the sim,
screenshot it, and read the actual water field. For work on the physics itself, skip
the app and use [Direct invocation](#direct-invocation-physics-work).

All paths below are relative to the repo root. Verified on Windows 11 / PowerShell +
Git Bash, Python 3.12.7, Node v24.18.0, NVIDIA Warp 1.17.0, RTX 5090, CUDA driver 13.2.

## Prerequisites

Python, Node, and a Chromium-family browser. The driver auto-detects Chrome or Edge at
the standard install paths (and `/usr/bin/google-chrome` etc. on Linux); override with
`CHROME_PATH` if yours is elsewhere.

## Setup

```bash
python -m pip install -r backend/requirements.txt
cd frontend && npm ci --no-audit --no-fund && cd ..
cd tests && npm ci --no-audit --no-fund && cd ..
```

`tests/node_modules` is **not optional** — the driver resolves `puppeteer-core` from
there rather than carrying its own copy.

## Build

The backend serves `frontend/dist/`, so the frontend must be built before the app shows
anything. This also type-checks (`tsc --noEmit && vite build`):

```bash
cd frontend && npm run build && cd ..
```

**Rebuild after every frontend edit.** The backend serves the built bundle, not your
sources — a stale `dist/` silently shows you the *old* UI while the backend is new.

## Run (agent path)

One command, exits 0/1, proves the whole stack — backend, Warp on CUDA, WebGL,
WebSocket round-trip, and a real advancing water front:

```bash
node .claude/skills/run-naturelab/driver.mjs --smoke
```

```
ok SMOKE PASS  sim=RUNNING t = 5.3s objects=1 triangles=27444 depth 0.5m flat -> west 0.556m / east 0m
ok .../shots/smoke.png bytes=226688 triangles=27444 drawCalls=22
```

For anything else, pipe commands to the REPL. Each command prints exactly one line
starting with `ok ` or `err `, so you can read one line per command sent:

```bash
printf 'place HOUSE 5 0\nwater 1.5\nstart\nwait 20000\ndepth\nss river\nquit\n' \
  | node .claude/skills/run-naturelab/driver.mjs
```

```
ok {"cells":10201,"wet":6329,"min":0,"max":1.5,"mean":0.631,"westMean":1.329,
    "eastMean":0,"visible":true,"drawnTriangles":12309,"idlePreview":false}
```

### Commands

| Command | What it does |
|---|---|
| `status` | ws / sim status / clock / object count+types / edge inflow level / tracer count |
| `place <TYPE> <x> <z>` | add at an exact position — `HOUSE CAR TREE BOX DEBRIS GAUGE` |
| `add <TYPE>` | add via the UI button (auto-places on a 3-wide grid) |
| `start` `pause` `reset` | sim transport, returns the new `status` |
| `water <m>` | **edge inflow level** — the height held at the west source columns |
| `gauge` | GAUGE readings: depth, surface elevation, speed, wave arrival time |
| `tracers <n>` | visible flow-tracer count (`0` hides them) |
| `send <json>` | raw WebSocket op, e.g. `send {"op":"terrain_brush","x":0,"z":0,"radius":6,"strength":0.4}` |
| `eval <js>` | evaluate in the page, JSON-printed; `__NL` is in scope |
| `depth` | water field: min/mean/max depth, wet cells, **westMean vs eastMean** |
| `ss [name]` | screenshot → `.claude/skills/run-naturelab/shots/<name>.png` |
| `wait <ms>`, `backendlog`, `quit` | |

`eval` reaches the app's debug handle `__NL` (`frontend/src/main.ts`), which exposes
`store`, `net`, `sceneManager`, `editor`, `ui`.

`depth` is the load-bearing one for physics. `SceneManager` never keeps a depth array —
`setWaterHeights()` writes the **absolute water-surface elevation** into vertex Z, so the
driver recovers real depth as `max(0, waterMeshZ − terrainMeshZ)`. `westMean > eastMean`
confirms water is moving west→east; `eastMean` legitimately stays `0` for a long while,
because the grid starts dry except for the source columns and a genuine wavefront has to
travel the full 100 m.

## Direct invocation (physics work)

Most changes here are backend physics (`fluid_solver.py`, `rigid_body.py`), and the
solver needs neither the server nor the browser:

```bash
python -c "
import sys; sys.path.insert(0, 'backend')
import numpy as np
from app.world_state import WorldState
from app.compute_engine import create_engine
from app.fluid_solver import create_fluid_solver

world = WorldState()
world.water.level = 1.0
engine = create_engine()
s = create_fluid_solver(engine.device)
s.initialize(world)
for step in range(1, 1801):
    s.set_boundaries(world.terrain, {}, 0, 0)
    s.advance(1/60, 8, 1/120)
    if step % 600 == 0:
        d = s.diagnostics()
        print('t=%4.0fs  wet=%5d  vol=%8.1f m3  vmax=%.3f m/s  substeps=%d'
              % (step/60, d['wet_cells'], d['volume_m3'], d['max_velocity'], d['substeps']))
"
```

```
t=  10s  wet= 3030  vol=  2290.4 m3  vmax=2.605 m/s  substeps=1
t=  20s  wet= 5151  vol=  3484.0 m3  vmax=2.023 m/s  substeps=1
t=  30s  wet= 6868  vol=  4311.3 m3  vmax=1.587 m/s  substeps=1
```

`set_boundaries(terrain, obstacles, terrain_revision, obstacle_revision)` only re-uploads
to the GPU when a revision number actually changes — passing constants (as above) is
correct for a fixed scene and is what keeps the upload counters flat.

## Test

```bash
python tests/test_backend.py          # 21 CUDA/Warp tests, ~8s, needs no server
node tests/e2e.mjs                    # 17 browser checks, spawns its own backend on 8756
```

Do **not** use `tests/run_all.bat` unless you have produced a PyInstaller build first —
its last step runs `tests/test_launcher.ps1`, which needs `NatureLab.exe`.

## Run (human path)

```bash
cd backend && python -m uvicorn app.main:app --host 127.0.0.1 --port 8756
```

Then open <http://127.0.0.1:8756/>. `start.bat` wraps exactly this and also opens the
browser. Ctrl-C to stop — and **actually stop it** (see Gotchas).

## Release

Every version ships as a standalone archive so any past version can be unpacked and run:

```bash
python tools/make_release.py 0.5.1
```

Writes `releases/NatureLab_v<version>.zip` (frontend `dist/` included, `.git`/
`node_modules`/`__pycache__` excluded) and records its SHA-256 in `releases/CHECKSUMS.txt`.

## Gotchas

- **A leftover backend on port 8756 will silently hijack everything.** This is the worst
  trap in the repo. `start.bat` / `launcher.pyw` leave a `python` process holding 8756; a
  later `uvicorn` on that port fails to bind, and the client connects to the *stale*
  process instead — which still has its own world and sim clock. The driver defends
  against this two ways: it defaults to **port 8770**, and it refuses to start if
  something already answers there. Check with:
  ```bash
  netstat -ano | grep ":8756" | grep LISTENING
  ```
- **Water enters from the west edge and grows a wavefront — it does not fill the map.**
  `initialize()` starts the grid dry except for `FLUID_SOURCE_COLUMNS` (2) at the west
  edge, held at `max(0, level − bed)`. Everything after that is real flux physics, so the
  front takes tens of seconds to cross 100 m. `eastMean: 0` early in a run is correct, not
  broken. There is **no other water source** — no rain, no point inflow (a deliberate open
  item, see `docs/06_next_steps.md`).
- **The edge inflow respects terrain.** A hill touching the west edge stays dry if it is
  taller than the level you set — `_apply_source` computes `max(0, level − bed)` per cell.
  This is a fix relative to the pre-0.5.1 solver, which clamped the whole column and
  poured water over hilltops.
- **The water mesh carries absolute surface elevation, not depth.** The `WATER_HEIGHT`
  bulk frame is `bed + depth` for wet cells and `bed − 0.05` for dry ones; dry cells are
  hidden by *dropping their triangles from the index*, not by a Z nudge or an alpha fade.
  There is no render-side exaggeration constant any more — if the water looks wrong,
  suspect the physics, not a display multiplier.
- **Terrain edits are rejected while RUNNING.** `test_running_terrain_edit_is_rejected`
  asserts this. It becomes a live question once erosion lands (erosion mutates terrain
  every tick by definition) — the ban should then apply to the user's brush, not the solver.
- **`terrain_revision` / `obstacle_revision` gate every GPU upload.** A test asserts the
  upload counters stay at 2/2 across 600 unchanged steps. Anything that needs to mutate
  `bed` per tick (erosion) must own the GPU array rather than bumping the revision.
- **Do not pixel-probe the 3D canvas to decide whether it rendered.** `#viewport` is a
  WebGL canvas without `preserveDrawingBuffer`, so its buffer is cleared once composited:
  `drawImage`/`toDataURL` read it back as **pure black even on a perfectly good frame**.
  Puppeteer's `page.screenshot` captures the composited surface and is correct. Use the
  driver's `triangles=` / `drawCalls=` (Three.js' own render stats) as the honest "did it
  draw" signal.
- **Headless Chrome needs software GL forced on.** The driver passes
  `--use-gl=angle --use-angle=swiftshader --enable-unsafe-swiftshader`. Renderer then
  reports SwiftShader even though the box has an RTX 5090 — expected: the GPU is used by
  Warp on the backend, not by headless Chrome.
- **`npm ci` under npm 12 blocks install scripts** — it warns that `esbuild@0.21.5
  (postinstall)` was blocked. The build still succeeds; ignore it.
- **Sim state persists across driver commands, not across driver runs.** Each driver
  process spawns and kills its own backend, so every run starts at `IDLE, t=0.0s, 0
  objects`. `data/` is empty by default — nothing is auto-loaded.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `err port 8770 is already serving a NatureLab backend` | A previous driver run leaked. Kill it (`netstat -ano \| grep ":8770" \| grep LISTENING`, then `taskkill //PID <pid> //F`) or use `NL_PORT=8781 node ... driver.mjs`. |
| `err frontend/dist missing` | `cd frontend && npm ci && npm run build` |
| `err no Chrome/Edge found; set CHROME_PATH` | `CHROME_PATH="/path/to/chrome" node ... driver.mjs` |
| App loads but a control you just added is missing | Stale `frontend/dist` — rebuild. |
| E2E fails at `START idempotent` with `Error: timeout` | Stale backend on 8756 with objects already in its world. Kill it, re-run. |
| `NatureLab.exe not found` | You ran `tests/run_all.bat`. Use `python tests/test_backend.py` + `node tests/e2e.mjs`. |
| Sim runs but the east half stays dry | Expected — the wavefront has not arrived yet. Wait, or raise `water`. |
