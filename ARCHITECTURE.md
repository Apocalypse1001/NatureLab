# NatureLab 0.5.1 - Architecture

Проектная документация, которая объясняет *почему* архитектура такая, живёт в `docs/`:
[`01_vision.md`](docs/01_vision.md) (цели и критерий качества),
[`04_TZ_v0.3_roadmap.md`](docs/04_TZ_v0.3_roadmap.md) (процесс и Definition of Done),
[`05_audit_v0.4_water.md`](docs/05_audit_v0.4_water.md) (почему прежний солвер заменён),
[`06_next_steps.md`](docs/06_next_steps.md) (порядок версий дальше).

## Data Flow

```text
Three.js editor / WATER_HEIGHT visualization
                    <-> WebSocket protocol v2
FastAPI / SimulationManager (authoritative state, fixed global dt=1/60)
                    -> revision sync
WarpShallowWaterSolver h/u/v + bed/solid arrays
                    -> GPU sample depth/velocity/drag
RigidStateBuffer -> Warp force integration -> compact transforms/states
```

## Fluid

Grid 101x101 follows terrain vertices. Velocity and continuity use ping-pong Warp arrays.
Outer faces and solid faces have zero normal flux; tangential edge velocity remains valid.
Face discharge uses water above the maximum neighboring bed elevation.

CFL diagnostics reduce max depth, speed, wave speed, volume and wet count to scalar GPU
buffers. `FLUID_CFL=0.45` selects internal substeps; `FLUID_MAX_SUBSTEPS` is a guardrail.

## Obstacles And Revisions

SimulationManager increments `terrain_revision` and `obstacle_revision` only after edits.
The solver tracks seen revisions and reports `terrain_gpu_uploads` and
`obstacle_gpu_uploads`. Unchanged ticks do not recreate GPU arrays.

HOUSE footprints are yaw-oriented solid cells. On a mask revision, water and momentum from newly solid
cells move deterministically to the nearest unchanged fluid cells. Newly freed cells start
dry, so moving/removing a HOUSE neither stores phantom water nor creates volume.

## GPU Coupling

`_sample_bodies` reads h/u/v directly on device at a rotated 3x3 body footprint,
accounts for body-base elevation, and calculates hydrodynamic drag.
`_integrate_bodies` applies force/mass, buoyancy and friction on device. Only compact body
transforms, states and sample depths return to CPU for WorldState synchronization,
educational events and the current 2D collision correction. Full h/u/v readback occurs only
when producing a frontend/debug field.

## Rigid Model

Dense buffers include mass, sealed buoyancy coefficient, volume, drag coefficient,
contact/cross areas, friction and static flag. Buoyancy is displaced water weight; grounded motion begins when drag exceeds
Coulomb friction. Semi-implicit integration drives INTACT/MOVING/FLOATING/SETTLED states.

## Edge Inflow And Editing

The map starts dry except for two prescribed source columns at the left boundary (`x=-50 m`).
The selected Edge inflow level is enforced only there; every downstream cell becomes wet
through physical flux. Terrain commands are rejected while RUNNING and accepted while
IDLE/PAUSED.

## Protocol

Protocol v2 header is `NL | version:u8 | kind:u8 | count:u32 | time_ms:u64` followed by
little-endian float32 data. `WATER_HEIGHT` contains absolute Y elevations in row-major
`[z,x]`. PARTICLES carries GPU-advected flow tracers derived from the real velocity field.

## Educational Gauges

`GAUGE` reuses the compact per-body GPU sampling path with a zero-size footprint. It never
enters the fluid obstacle mask or collision pairs. SimulationManager records depth, absolute
surface, horizontal speed and first wave-arrival time. Incremental samples are streamed in
`sim_state`; backend and frontend histories are bounded to 600 entries at 10 Hz simulation time.

## Versioning and releases

`backend/app/config.py:VERSION` — единственный источник истины для номера версии. Он
уходит во frontend через `engine_info()` (виден в заголовке приложения и в
`/api/status`), и его же читает `tools/make_release.py`, собирая
`releases/NatureLab_v<version>.zip`.

Архив самодостаточен: внутрь кладётся собранный `frontend/dist/`, поэтому распакованная
версия запускается без `npm`. Не кладутся `.git/`, `node_modules/`, `__pycache__/`,
предыдущие архивы, вывод PyInstaller и скриншоты драйвера. SHA-256 каждого архива
пишется в `releases/CHECKSUMS.txt`.

Правило именования: номер версии должен говорить, какая физика внутри. 0.5.1 — это
слияние двух веток без новой физики; следующая версия с эрозией/осадком будет 0.6.0.
