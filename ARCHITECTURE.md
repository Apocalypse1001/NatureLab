# NatureLab 0.7.0 - Architecture

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

Grid 201x201 follows terrain vertices. Velocity and continuity use ping-pong Warp arrays.
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

## Sediment and bed (RiverLab, v0.6.0)

The bed is deliberately split into three GPU arrays:

```text
_bed_terrain   erodible ground -- OWNED BY THE SOLVER, mutated every tick
_bed_offset    ROCK domes -- rebuilt from live positions, never written to the world
_bed           their sum -- the only one the flow kernels read
```

That split is what makes two otherwise conflicting requirements coexist. Erosion has
to change the ground permanently, so `_bed_terrain` cannot stay a copy of
`world.terrain` re-uploaded on a revision bump; a rock has to change the flow without
leaving a crater when it moves, so its dome must be transient. Keeping them apart also
means the existing, already-verified flow kernels were not touched: they still take a
single `bed` array.

`terrain_revision` therefore now means "the HOST changed the terrain" (brush, load,
reset) and never fires for erosion. The eroded bed reaches the world -- and the screen
-- through a throttled device-to-host readback plus the existing `terrain_patch`
message, once per `TERRAIN_RESYNC_INTERVAL_S`.

Whether a body is a wall or a riverbed is decided from data, not from a type name:
`_is_solid()` treats anything carrying a positive `metadata.bed_height` as bed, and
everything in `SOLID_OBSTACLE_TYPES` as wall. `SimulationManager._affects_fluid_boundary()`
uses the same rule to decide when adding, moving, scaling or deleting an object must
bump `obstacle_revision`.

## World size

`WORLD_SIZE_M` and `TERRAIN_CELLS` are the only two numbers that define the domain;
everything else derives from them. That includes the frontend: `SceneManager` rebuilds
terrain, water, grid helper, camera framing, fog and zoom limits whenever the grid
resolution it is handed differs from the one it currently holds, so a resize is a
config edit rather than a coordinated change across both sides.

Two things that are *not* derived and are worth knowing before the next resize:

- Water mesh indices follow whatever width Three.js picks. 201x201 = 40 401 vertices
  still fits Uint16 (max 65 535); 401x401 would not, and the code mirrors the chosen
  array type rather than assuming either.
- `_build_obstacle_mask` and `_build_bed_offset` each allocate an `np.mgrid` over the
  whole grid per call. That is fine at the current rate (only on an obstacle-revision
  bump) and would not be if anything ever needed them per tick.

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
