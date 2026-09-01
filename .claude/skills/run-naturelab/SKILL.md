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
ok SMOKE PASS  sim=RUNNING t = 5.3s objects=1 triangles=60260 depth 0.5m flat -> west 0.161m / east 0m
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
ok {"cells":10201,"wet":2828,"min":0,"max":0.5,"mean":0.041,"westMean":0.161,"eastMean":0,"visible":true,"renderExaggeration":4}
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
array (it folds it straight into vertex Z, `terrainZ + depth * WATER_VISUAL_EXAGGERATION
+ WATER_DRY_BIAS` — both render-only, added 2026-09-01, see Gotchas), so the driver
recovers the **real** depth as `(waterMeshZ - terrainMeshZ - WATER_DRY_BIAS) /
WATER_VISUAL_EXAGGERATION`, matching the numbers the backend actually streamed.
`westMean > eastMean` confirms a river is flowing -- but note `eastMean` legitimately
stays exactly `0` for a while after enabling flow: the current design (2026-09-01)
resets the grid dry and grows a real wavefront in from the west edge, it does not
fill the whole map at once, so the east edge is genuinely untouched until the front
reaches it (can take upwards of a minute for the full 100-cell width).

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
depth west/east: 0.569 / 0.423
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
  "Важная находка".
- **`flow on` resets the grid dry and grows a real wavefront in from the west
  edge — it does not fill the whole map.** This went through two designs in one
  day, both from testing the live app rather than trusting the numbers: a first
  version instantly ramped the *entire* grid to a source→sink gradient the
  moment flow was enabled (to fix "flow on still looks like a still pool" —
  the old `FLUID_FLOW_GAIN=1.6` needed real *minutes* for the grid's centre to
  move at all). The user then tested that and asked for the opposite: water
  should visibly *enter* from an edge and flow across at the height they set,
  not appear everywhere at once. Current design
  (`ShallowWaterFluidSolver._seed_river_profile`) resets to bone dry on
  enable and lets `FLUID_FLOW_GAIN=10` carry a genuine front in — measured,
  it reaches 15% of a 100-cell grid by 1s, 43% by 10s, 77% by 40s. So `depth`
  reading `eastMean: 0` for a good while after enabling flow is correct, not
  broken — the front just hasn't arrived yet.
- **The river's source/sink height follows `world.water.level`, live — not a
  fixed constant.** `FLUID_RIVER_SOURCE_DEPTH`/`FLUID_RIVER_SINK_DEPTH` were
  removed; the west edge is held at `max(0, world.water.level)` and the east
  at `FLUID_RIVER_SINK_FRACTION` (0.1) of that, both read every tick. This
  also means the Water Level slider now actually does something while
  RUNNING **when River flow is on** — see the next gotcha for the general case.
- **The Water Level slider does nothing while RUNNING, *except* when River
  flow is on.** The lake-fill solver otherwise reads `world.water.level` only
  in `initialize()`. `flow on/off`, and the river's source height while flow
  is on, *are* live. Not a driver bug; a known partially-open item.
- **Don't push `SceneManager.WATER_VISUAL_EXAGGERATION` up casually.** A first
  attempt exaggerated the water *surface's* departure from its own mean,
  clamped to never draw below terrain -- that clamp pinned a whole swath of
  shallow cells to *exactly* the terrain's height, and coplanar transparent
  geometry z-fights: a large, ugly, moving checkerboard (screenshot it if you
  don't believe how bad it looked). The fix was to exaggerate `depth` itself
  (always >= 0, so the clamp is never needed) rather than surface height. If
  you see a checkerboard/moiré on the water again, this is almost certainly
  why — check what changed the Z formula in `updateWaterField`, not the physics.
- **The river gradient is deliberately over-drawn, 4x — and dry cells are
  deliberately faded to fully transparent, separately.** `WATER_VISUAL_EXAGGERATION`
  multiplies *displayed* depth only (a real ~0.15-1.5m depth split used to read
  as a flat blue sheet in a screenshot). Once the design above started resetting
  the grid dry, the checkerboard from the previous gotcha came back *much*
  bigger — half the map, not just a dry hilltop, since a genuinely-untouched
  cell is exactly coplanar with terrain regardless of exaggeration. A constant
  `WATER_DRY_BIAS` Z-offset fixes the z-fighting but, applied everywhere, made
  the *entire* map look uniformly wet — exactly what a wavefront demo must not
  do. The actual fix is a separate per-vertex alpha attribute (`aWet`) that
  fades genuinely-dry cells to fully transparent in the fragment shader,
  independent of the Z bias. Both are needed; neither replaces the other. The
  `depth` command divides the exaggeration and subtracts the bias back out
  (unless the sim is IDLE, where the flat preview plane has neither applied),
  so its numbers are always the true backend depth.
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
