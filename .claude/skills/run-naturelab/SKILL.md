---
name: run-naturelab
description: Build, run, and drive NatureLab (physics sandbox - FastAPI/Warp backend + Three.js frontend). Use when asked to start NatureLab, launch the app, take a screenshot of the simulation, run its tests, or interact with the running sim (add objects, toggle river flow, read the water depth field).
---

NatureLab is a physics sandbox: a Python FastAPI + NVIDIA Warp backend that streams
a shallow-water simulation over a WebSocket to a Three.js frontend it also serves.
Agents drive it with **`.claude/skills/run-naturelab/driver.mjs`** — a stdin REPL
that owns the backend *and* a headless Chrome, so you can place objects, run the
sim, screenshot it, and read the actual water field. For work on the physics
itself (most changes here), skip the app entirely and use
[Direct invocation](#direct-invocation-physics-work).

All paths below are relative to the repo root. Verified on Windows 11 /
PowerShell + Git Bash, Python 3.12.7, Node v24.18.0, npm 12.0.2, RTX 5090.

## Prerequisites

Python, Node, and a Chromium-family browser. There is no `apt-get` step — this is
a Windows host, and Chrome was already installed. The driver auto-detects Chrome
or Edge at the standard install paths (and `/usr/bin/google-chrome` etc. on Linux);
override with `CHROME_PATH` if yours is elsewhere.

```bash
python -V   # Python 3.12.7
node -v     # v24.18.0
```

## Setup

```bash
python -m pip install -r backend/requirements.txt
cd frontend && npm ci --no-audit --no-fund && cd ..
cd tests && npm ci --no-audit --no-fund && cd ..
```

`tests/node_modules` is **not optional** — the driver resolves `puppeteer-core`
from there rather than carrying its own copy.

## Build

The backend serves `frontend/dist/`, so the frontend must be built before the app
shows anything. This also type-checks (`tsc --noEmit && vite build`):

```bash
cd frontend && npm run build && cd ..
```

**Rebuild after every frontend edit.** The backend serves the built bundle, not
your sources — a stale `dist/` silently shows you the *old* UI while the backend
is new. (This bit during authoring: a control added in the same session was simply
absent from the page until a rebuild.)

## Run (agent path)

One command, exits 0/1, proves the whole stack — backend, WebGL, WebSocket
round-trip, and moving water:

```bash
node .claude/skills/run-naturelab/driver.mjs --smoke
```

```
ok SMOKE PASS  sim=RUNNING t = 5.2s objects=1 triangles=60260 depth 0.5m flat -> west 1.244m / east 0.3m
ok .../shots/smoke.png bytes=281164 triangles=60260 drawCalls=20
```

For anything else, pipe commands to the REPL. Each command prints exactly one
line starting with `ok ` or `err `, so you can read one line per command sent:

```bash
printf 'place ROCK 0 0\nflow on\nstart\nwait 5000\ndepth\nss river\nquit\n' \
  | node .claude/skills/run-naturelab/driver.mjs
```

```
ok placed ROCK at (0, 0)
ok river flow on
ok {"ws":"connected","sim":"RUNNING","clock":"t = 0.2s","objects":1,...}
ok waited 5000ms
ok {"cells":10201,"wet":10200,"min":0,"max":1.4,"mean":0.775,"westMean":1.244,"eastMean":0.3,"visible":true,"renderExaggeration":4}
ok .../shots/river.png bytes=281645 triangles=60260 drawCalls=20
```

### Commands

| Command | What it does |
|---|---|
| `status` | ws / sim status / clock / object count+types / water level / river-flow checkbox |
| `place <TYPE> <x> <z>` | add at an exact position — `HOUSE CAR TREE BOX DEBRIS ROCK` |
| `add <TYPE>` | add via the UI button (auto-places on a 3-wide grid) |
| `start` `pause` `reset` | sim transport, returns the new `status` |
| `flow on\|off` | `water.flow_enabled` — the continuous west→east river current |
| `water <m>` / `temp <c>` | water level / baseline water temperature |
| `send <json>` | raw WebSocket op, e.g. `send {"op":"terrain_brush","x":0,"z":0,"radius":6,"strength":0.4}` |
| `eval <js>` | evaluate in the page, JSON-printed; `__NL` is in scope |
| `depth` | water depth field: min/mean/max, wet cells, **westMean vs eastMean** |
| `ss [name]` | screenshot → `.claude/skills/run-naturelab/shots/<name>.png` |
| `wait <ms>`, `backendlog`, `quit` | |

`eval` reaches the app's debug handle `__NL` (`frontend/src/main.ts`), which
exposes `store`, `net`, `sceneManager`, `editor`, `ui`:

```bash
printf 'eval __NL.sceneManager.renderer.info.render.triangles\nquit\n' \
  | node .claude/skills/run-naturelab/driver.mjs
```

`depth` is the load-bearing one for physics: `SceneManager` never keeps the depth
array (it folds it straight into vertex Z, `terrainZ + depth * WATER_VISUAL_EXAGGERATION`
— the exaggeration is render-only, added 2026-09-01 so gradients read as a visible
slope instead of a flat-looking plane), so the driver recovers the **real** depth
as `(waterMeshZ - terrainMeshZ) / WATER_VISUAL_EXAGGERATION`, matching the numbers
the backend actually streamed. `westMean > eastMean` is how you confirm a river is
actually flowing.

## Direct invocation (physics work)

Most changes here are backend physics (`fluid_solver.py`, `rigid_body.py`), and
the solver needs neither the server nor the browser. Drive it directly — this is
the fast path, and it is how the erosion/deflection behaviour was measured:

```bash
python -c "
import sys; sys.path.insert(0, 'backend')
import numpy as np
from app.world_state import WorldState
from app.fluid_solver import ShallowWaterFluidSolver

world = WorldState()
w, h = world.terrain.width, world.terrain.height
world.terrain.heights[:, :] = np.tile(np.linspace(2.0, 0.0, w+1, dtype=np.float32), (h+1, 1))
world.water.flow_enabled = True
s = ShallowWaterFluidSolver(); s.initialize(world); s._depth[:, :] = 0.6
for _ in range(300):
    s.set_boundaries(world.terrain, {})
    s.set_bed_obstructions({'positions': [[0.,0.,0.]], 'radii': [3.0], 'heights': [0.8]})
    s.advance(1/60, 8, 1/120)
print('depth west/east: %.3f / %.3f' % (s._depth[:, :25].mean(), s._depth[:, -25:].mean()))
print('lateral deflection max |flow_z|: %.5f' % abs(s._flow_z).max())
"
```

```
depth west/east: 0.851 / 0.455
lateral deflection max |flow_z|: 0.15177
```

Note `set_boundaries` / `set_bed_obstructions` must be called **every step** —
both are stateless by design and recompute from live positions.

## Test

```bash
python tests/test_backend.py          # 47 tests, ~13s, needs no server
cd tests && npm test && cd ..         # browser E2E, spawns its own backend on 8756
```

Do **not** use `tests/run_all.bat` — its second step runs `tests/test_launcher.ps1`,
which throws `NatureLab.exe not found` unless you have first produced a PyInstaller
build via `build_exe.bat`. Run the two commands above instead.

## Run (human path)

```bash
cd backend && python -m uvicorn app.main:app --host 127.0.0.1 --port 8756
```

Then open <http://127.0.0.1:8756/>. `start.bat` wraps exactly this and also opens
the browser for you. Ctrl-C to stop — and **actually stop it** (see Gotchas).

## Gotchas

- **A leftover backend on port 8756 will silently hijack everything.** This is the
  worst trap in the repo. `start.bat` / `launcher.pyw` leave a `python` process
  holding 8756; a later `uvicorn` on that port fails to bind, and the client
  connects to the *stale* process instead — which still has its own world and sim
  clock. Symptom seen during authoring: a "fresh" launch reported `PAUSED, t=88.6s,
  10 objects`, and `tests/npm test` failed at `START idempotent` with a bare
  `Error: timeout`, because it hard-codes 8756 and found objects already there.
  Killing the stale PID made the E2E pass 9/9 unchanged. The driver defends
  against this two ways: it defaults to **port 8770**, and it refuses to start if
  something already answers there. Check with:
  ```bash
  netstat -ano | grep ":8756" | grep LISTENING
  ```
- **Do not pixel-probe the 3D canvas to decide whether it rendered.** `#viewport`
  is a WebGL canvas without `preserveDrawingBuffer`, so its buffer is cleared once
  composited: `drawImage`/`toDataURL` read it back as **pure black even on a
  perfectly good frame**. An early version of this driver reported
  `canvasLuma=0..0` on a screenshot that was completely fine. Puppeteer's
  `page.screenshot` captures the composited surface and is correct. Use the
  driver's `triangles=` / `drawCalls=` (Three.js' own render stats) as the honest
  "did it draw" signal — ~60k triangles is a normal scene.
- **Headless Chrome needs software GL forced on.** The driver passes
  `--use-gl=angle --use-angle=swiftshader --enable-unsafe-swiftshader`. Without
  them there is no WebGL context and the scene really is blank. Renderer then
  reports `ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device ...))` — SwiftShader is
  expected here even though the box has an RTX 5090; the GPU is used by Warp on
  the backend, not by headless Chrome.
- **Water does not move on its own.** `initialize()` fills `depth = water_level -
  terrain`, which is a perfectly flat surface — zero gradient, zero flow. A fresh
  world with `START` pressed shows a still lake forever. Turn on `flow on` (or
  place an obstacle / edit terrain) or you will conclude the solver is broken.
  This is documented behaviour, not a bug — see `docs/04_TZ_v0.3_roadmap.md` v0.4
  "Важная находка". `flow on` itself is now instant (`_seed_river_profile()`
  jumps the depth field straight to the steady-state ramp rather than waiting
  for it to diffuse in from the edges — that used to take real *minutes* for
  the grid's centre to move at all), so this gotcha is now scoped tightly to
  "flow off = genuinely static," not "flow looks static for a while too."
- **Raising terrain above the current water line briefly draped it in a
  visible "wet slope" before this session's fix.** Not a bug in the physics
  (the depth field was always correct — a dry cell here, a filling moat
  there), but at the old `FLUID_FLOW_GAIN` a newly displaced ring of water
  took tens of seconds to redistribute, so a freshly raised hill looked
  permanently flooded on its flanks instead of settling into a dry island
  within a normal viewing window. Fixed together with the flow-gain
  recalibration below.
- **Don't push `SceneManager.WATER_VISUAL_EXAGGERATION` up casually.** A first
  attempt exaggerated the water *surface's* departure from its own mean,
  clamped to never draw below terrain -- that clamp pinned a whole swath of
  shallow cells to *exactly* the terrain's height, and coplanar transparent
  geometry z-fights: a large, ugly, moving checkerboard (screenshot it if you
  don't believe how bad it looked). The fix was to exaggerate `depth` itself
  (always >= 0, so the clamp is never needed) rather than surface height. If
  you see a checkerboard/moiré on the water again, this is almost certainly
  why — check what changed the Z formula in `updateWaterField`, not the physics.
- **The river gradient is deliberately over-drawn, 4x.** `SceneManager.
  WATER_VISUAL_EXAGGERATION` (2026-09-01) multiplies *displayed* depth only —
  a real 1.24m/0.30m west/east split used to read as a flat blue sheet in a
  screenshot (user-reported: "no dynamics visible"), so the mesh now draws
  it at 4x. The `depth` command divides that factor back out (unless the sim
  is IDLE, where the flat preview plane was never exaggerated to begin with),
  so its numbers are always the true backend depth — don't re-multiply them.
- **The Water Level slider does nothing while RUNNING.** The solver reads
  `world.water.level` only in `initialize()`. `flow on/off` *is* live. Not a driver
  bug; a known open item.
- **`npm ci` under npm 12 blocks install scripts** — it warns that
  `esbuild@0.21.5 (postinstall)` was blocked. The build still succeeded, so ignore
  it; only if `npm run build` fails would you need `npm install-scripts approve esbuild`.
- **Sim state persists across driver commands, not across driver runs.** Each
  driver process spawns and kills its own backend, so every run starts at
  `IDLE, t=0.0s, 0 objects`. `data/` is empty by default — nothing is auto-loaded.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `err port 8770 is already serving a NatureLab backend` | A previous driver run leaked, or you started one manually. Kill it (`netstat -ano \| grep ":8770" \| grep LISTENING`, then `taskkill //PID <pid> //F`) or use `NL_PORT=8781 node ... driver.mjs`. |
| `err frontend/dist missing` | `cd frontend && npm ci && npm run build` |
| `err no Chrome/Edge found; set CHROME_PATH` | `CHROME_PATH="/path/to/chrome" node ... driver.mjs` |
| Driver boot fails; backend log mentions bind/port | Something else owns the port — see the first row. The driver dumps the last 20 backend log lines on boot failure. |
| App loads but a control you just added is missing | Stale `frontend/dist` — rebuild. |
| `npm test` fails at `START idempotent` with `Error: timeout` | Stale backend on 8756 with objects already in its world. Kill it, re-run. |
| `NatureLab.exe not found` | You ran `tests/run_all.bat`. Use `python tests/test_backend.py` + `cd tests && npm test`. |
| Sim runs but water never moves | Expected — `flow on`. See Gotchas. |
