# NatureLab

*[Русская версия](README.ru.md) — including the full version-by-version history.*

**An educational physics sandbox where nothing is animated.** You change the world; a GPU
shallow-water solver works out what happens next. A house floods because the water got
that deep, a car moves because the drag on it exceeded the friction holding it down, and a
bridge's piers speed the flow between them because they are actually in the way.

```text
Three.js  ->  WebSocket  ->  FastAPI  ->  SimulationManager  ->  NVIDIA Warp / CUDA
```

The backend is authoritative. The frontend draws, edits and measures — it never invents
motion. That distinction is the whole project: a prettier picture that hid the physics
would defeat the point, so if the simulation is wrong, the screen shows it.

Version 0.12.2. Backend suite 87/87, browser E2E 19/19, on an RTX 5090.

---

## What it is for

It is aimed at children learning by experiment, and the questions it exists to answer are
the ones a child actually asks:

- What happens if I put the house higher?
- What if I park the car closer to the river?
- What if I build a wall — where does the water go instead?
- Why did that tree stay and this one wash away?
- What if I narrow the channel? What if I make the slope steeper?

Nothing is scripted. There is no "at t = 3 s the car moves". There is a force, a mass, a
drag coefficient and a friction coefficient, and the car moves when the numbers say so.

---

## Scenarios

Two prepared worlds ship with it, in the **Scenarios** panel. Both drop you into the same
small town — ten houses, a street, figures, a wood and a rubbish dump — and both stay
completely editable: brush the terrain, move objects, change the discharge, press PLAY.

| | |
|---|---|
| **River** | The town on the bank of a running river, fed by a prescribed discharge of 12 m³/s, at which the channel runs about a metre deep and the town stays dry. At the discharge slider's maximum of 80 m³/s the channel goes bankfull at 2.1 m and about 13 cm of water stands in the street from roughly 200 s. |
| **Dam** | The same town, below a dam holding a reservoir. At the discharge it ships with, the spillway carries the river and the dam holds. Push Q past about 60 m³/s and the crest is overtopped after roughly six minutes of simulated time — or cut the crest with the terrain brush and watch it go at once. |

The dam is **terrain, not an object**, and that is a physics decision. Solid obstacles are
rasterized as infinitely tall walls, so an object dam could never be overtopped — and
overtopping is the entire lesson. A ridge written into the heightfield spills correctly
and can be breached with the ordinary brush.

Every number above is measured, not chosen. `docs/probe_dam_v0121.py` runs the real solver
on the real generated terrain and prints the reservoir surface against the crest; the river
figures come from the same headless path on the shipped world.

Both worlds are ordinary saved worlds, rebuilt with:

```bat
python tools\make_scenarios.py
```

The rubbish dump sits **upstream** of the town on purpose. Everything in it is light and
draggy, so when the water arrives the rubbish is what moves first — and it moves into the
town. That is a causal chain, not scenery.

---

## Physics

**Fluid.** A real shallow-water solver (`h`, `u`, `v`) in NVIDIA Warp on CUDA, with an
adaptive CFL timestep driven by `max(|u| + sqrt(g·h))`. Reflective/no-flux boundaries with
no frozen edge depth; bed-aware face fluxes, so water cannot pass through terrain standing
above the free surface. The map starts dry — a wavefront has to physically travel across
it.

**Water sources.** Three, and only ever one at a time: an edge inflow held at a level, a
placeable `SOURCE` disc, and a prescribed-discharge river inlet where Q is the control and
the water level is the channel's answer. Prescribing both would over-determine the
boundary.

**Boundaries and bed.** A `DRAIN` removes water through a smooth radial sink and spins the
flow from the circulation it measures — the vortex is the velocity field's, not an
animation. Optional erosion lets the river cut and fill its own bed. Bed friction reads
the local depth, so a thin sheet is slowed more than a deep channel.

**Rigid bodies.** Mass, volume, drag coefficient, contact area and ground friction, with
buoyancy from the sampled water column and collision correction on 2D footprints. Objects
report real states — INTACT, MOVING, FLOATING, SETTLED. A `PERSON` is light, tall for its
footprint and draggy, which is exactly why moving water carries one off so readily.

**Obstacles.** A `HOUSE` is solid to the flow at its exact yaw-oriented footprint. A
`BRIDGE` obstructs with its piers only — water flows under the deck, because a bridge that
dams its own river is not a bridge — and reports the moment the deck goes under. A `ROCK`
is riverbed rather than wall: it raises the bed by a dome of its own visible radius, so
water passes over it. A `ROAD` changes neither, and that is correct: flat asphalt on a
floodplain changes roughness, not bed elevation.

**Measurement.** A `GAUGE` is a staff gauge that measures without disturbing what it
measures — excluded from the obstacle mask, and asserted to be so by its own test. It
reports depth, absolute surface elevation, flow speed and wave arrival time, keeps a
bounded history, and fires `WATER_ENTERED_AREA` exactly once.

**Volume ledger.** Every boundary reports what it added and removed, and the difference
against the measured volume is displayed live. A boundary condition that cannot be audited
is a boundary condition nobody should trust.

---

## Visuals

What is drawn is what the solver uses. The water surface is the real height field, and the
ripples travel along the real velocity field at the speed the water is really moving —
still water is visibly still. Foam appears where the flow is genuinely fast and shallow,
which is where white water actually breaks. Spray is emitted from the same fields.

Objects are grounded with real shadows, whose sun position and shadow frustum are derived
from the world size rather than fixed metres. Every builder is bound by one rule: it may
not change a dimension the solver reads. A house keeps the 2.0 × 2.0 m half-extent its
obstacle mask is stamped from; a rock keeps its dome radius; a bridge keeps its span and
pier spacing. All detail is added inside those envelopes.

---

## Running it

Python 3.12 and a Chromium-family browser.

```bat
python -m pip install -r backend\requirements.txt
start.bat
```

`start.bat` starts the backend and opens the browser at `http://127.0.0.1:8756/`. It binds
to localhost only. `NatureLab.exe` does the same if it has been built (`build_exe.bat`).

An unpacked release archive (`releases/NatureLab_v*.zip`) runs as-is — the built frontend
is inside it, so npm is not needed to run.

Building from source:

```bat
cd frontend && npm ci && npm run build && cd ..
cd backend && python -m uvicorn app.main:app --host 127.0.0.1 --port 8756
```

---

## Tests

Node.js ≥ 22.12 and Chrome/Edge.

```bat
python tests\test_backend.py     # 87 CUDA/Warp physics and regression tests
node tests\e2e.mjs               # 19 browser + WebSocket checks, own backend on 8756
```

The physics suite covers lake-at-rest, 1D symmetry, volume conservation, terrain barriers,
static and moving obstacles, GPU upload revisions, adaptive CFL, heavy-vs-light response,
zero flow, RESET, determinism, the generated channel's geometry, the discharge inlet's
delivered Q, the drain's measured circulation, and the scenarios' layout rules.

`tests\run_all.bat` additionally runs the launcher test and needs a built `NatureLab.exe`.

---

## Releases

```bat
cd frontend && npm run build && cd ..
python tools\make_release.py 0.12.2
```

Writes `releases\NatureLab_v<version>.zip` and records its SHA-256 in
`releases\CHECKSUMS.txt`. The same number is written into `config.VERSION`, so a running
build always reports which one it is.

---

## Documentation

| | |
|---|---|
| [`docs/01_vision.md`](docs/01_vision.md) | What the project is for, and what it refuses to be |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Data flow and coupling |
| [`docs/07_river_plan.md`](docs/07_river_plan.md) | The river, its measurements, and one unresolved solver defect |
| [`docs/08_volcano_plan.md`](docs/08_volcano_plan.md) | **Current entry point** — v0.13.0, lava that flows and solidifies |
| [`docs/09_debris_flow_plan.md`](docs/09_debris_flow_plan.md) | Mountain debris flow: the same solver as the volcano, with concentration in place of temperature |
| [`README.ru.md`](README.ru.md) | Russian, with the full per-version history |

`CONTINUATION.md` is a historical document and describes a different source tree.

---

## Roadmap

- **v0.13.0 — Volcano I.** Lava that flows, cools, thickens and solidifies into terrain
  the next flow runs over. The first task is a measurement, not code: the run-out length
  before solidification has to land in a range that fits on the map.
- **Debris flow.** A mountain valley, a cloudburst, and a mudflow that carries everything
  away. It shares the volcano's shape exactly — a transported scalar that drives viscosity
  and density, and a threshold at which the flow stops and becomes ground. Both need the
  same single change: bed roughness as a field rather than a constant.
- Angular dynamics for rigid bodies, damage, and a per-object waterline.

---

## Limitations

Stated plainly, because a teaching tool that overstates itself teaches the wrong thing.

- First-order solver, sized for educational scenarios rather than engineering analysis.
- Rigid bodies use a translational force model; there is no angular dynamics, so a car
  carried by the flow does not yaw or tumble.
- Collision correction uses 2D footprints and radii, not a full contact solver.
- `damage` exists in the schema and is never computed yet.
- The evolved `h/u/v` field is not serialized by SAVE — a saved world restores its
  terrain, objects and boundary settings, not the water in flight.
- No rain, no destruction, no vegetation physics, no replay.
- Erosion is off by default: the long-run incision feedback is not calibrated yet, and
  `docs/07_river_plan.md` records the measurement that has to come first.

---

## Tested with

Python 3.12.7 · NVIDIA Warp 1.17.0 · RTX 5090 32 GB (driver 596.49) · FastAPI 0.141.1 ·
Uvicorn 0.52.4 · NumPy 2.5.2 · websockets 17.1 · Three.js 0.169.0 · Vite 5.4.21 ·
TypeScript 5.9.3 · Puppeteer Core 25.9.0 · PyInstaller 6.22.2
