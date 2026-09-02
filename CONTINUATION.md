# NatureLab Project Handoff

> **ИСТОРИЧЕСКИЙ ДОКУМЕНТ, 2026-09-01. Не описывает этот репозиторий.**
> «Working source directory» ниже указывает на дерево-донор
> `D:\AI\Simulator Wartp\NatureLab_0.4\NatureLab_0.4` (версия 0.5.0, без git,
> только для чтения). Его движок импортирован сюда в v0.5.1 и развит до v0.11.0;
> перечисленные ниже «Implemented Features» и «Pending User Request» (DRAIN,
> удвоение карты) давно закрыты. Актуальная точка продолжения —
> [`docs/07_river_plan.md`](docs/07_river_plan.md).


Last audited: 2026-09-01

## Current Project Location

Working source directory:

```text
D:\AI\Simulator Wartp\NatureLab_0.4\NatureLab_0.4
```

Important: the directory name still ends in `0.4`, although the application,
source, and release metadata are currently `NatureLab 0.5.0`.

## Current Release

- Release: NatureLab 0.5.0 Educational FloodLab
- Executable: `NatureLab.exe`
- Latest archive: `D:\AI\Simulator Wartp\NatureLab_0.5.0.zip`
- Latest archive SHA-256: `D34FF155021A299D8C674BCFA07F6957D43DA4E1AB0A8BEBD0A1709787DD08AC`
- Archive contains the tested GAUGE implementation; regenerate it after the DRAIN/map-expansion work.

## Architecture

```text
Three.js -> WebSocket -> FastAPI -> SimulationManager -> NVIDIA Warp/CUDA
```

Backend is authoritative. Frontend is visualization, UI, camera, and editing.
Default tested device is `cuda:0`, NVIDIA RTX 5090, Warp `1.17.0`.

## Implemented Features

- Reflective/no-flux outer and HOUSE-solid boundaries.
- Real shallow-water `h/u/v` Warp solver with adaptive CFL.
- Initial wave enters from the left map edge, not from a pre-filled reservoir.
- `FLUID_SOURCE_COLUMNS = 2`; map starts dry except for the first two source columns.
- Edge inflow level is enforced at the left boundary and propagates physically downstream.
- Exact yaw-oriented HOUSE mask; default mask is 25 grid points at zero yaw and 13 at 45 degrees.
- Conservative obstacle MOVE/REMOVE remapping without phantom water.
- Terrain and obstacle revision-driven GPU uploads.
- GPU body fluid sampling and rigid force integration.
- Body sampling uses a rotated 3x3 footprint and body-base elevation.
- Buoyancy coefficient has semantics `0..1`; old world version 1 data migrates to `1.0`.
- BOX physical defaults match the visible 1.2 m cube.
- GPU-advected flow tracers; default source count is 8,000.
- Frontend tracer visibility and display-count controls.
- Dry and HOUSE-solid water triangles are removed from the water geometry.
- Added `GAUGE` object with depth, surface elevation, speed, wave arrival time,
  bounded history, sparkline UI, and `WATER_ENTERED_AREA` event.

## Verification Status

Latest completed verification before this handoff:

- 21 backend CUDA/Warp tests: PASS.
- Browser/WebSocket E2E: PASS.
- Launcher lifecycle: PASS.
- 30-minute simulation-time soak: PASS.
- GAUGE tests cover schema, point sampling, no obstacle/collision influence,
  arrival event once, bounded history, stream schema, UI, and RESET.

Commands:

```bat
python tests\test_backend.py
tests\run_all.bat
```

Frontend build:

```bat
cd frontend
npm run build
```

## Pending User Request

The next requested feature is a physical bathtub-style drain:

- Drain is placeable as an ordinary scene object.
- Drain has a position on the terrain and should be saved with the world.
- Drain removes water through a localized sink boundary.
- Sink must create a rotating vortex from the actual velocity field.
- Objects entering the drain region should be advected and sucked toward it,
  subject to drag, buoyancy, collision, and mass.
- Rotation trajectory must be verified with physics tests, not decorative animation.
- Drain should have configurable radius, discharge strength, and optional outlet behavior.

Recommended design for the drain:

1. Add `DRAIN` to backend/frontend object types and defaults.
2. Do not treat DRAIN as a HOUSE solid obstacle.
3. Add a localized GPU sink/vortex kernel after depth/velocity update, using
   radial/tangential velocity around the drain center and conservative depth removal.
4. Use a smooth radial sink profile and cap removal by available local water.
5. Expose drain center/radius/strength to the fluid solver through the existing
   revision/snapshot path.
6. Sample drain-induced velocity through the existing GPU body sampling path.
7. Add tests for mass/volume loss, radial convergence, tangential rotation sign,
   zero drain effect outside radius, and object trajectories.
8. Add a browser visual test that checks tracer angular motion around the drain.

Do not fake the vortex with frontend-only rotation. The frontend must only render
the authoritative tracer and water fields.

## Pending Map Expansion

User also requested doubling the map size. Current map is:

- `WORLD_SIZE_M = 100.0`
- `TERRAIN_CELLS = 100`
- Grid: 101 x 101 vertices
- Cell size: 1 m
- Left edge: x = -50 m

Preferred expansion is to keep the 1 m cell size and change the logical grid to
200 x 200 cells, producing a 201 x 201 vertex grid and map bounds -100..100 m.
This requires checking:

- GPU memory and WATER_HEIGHT bandwidth.
- Three.js geometry/index capacity and Uint32 index path.
- Flow tracer spawn distribution.
- Source edge coordinates and drain placement.
- Terrain editing and persistence validation.
- E2E timeout/performance thresholds.

Do not change map size and drain behavior in the same untested patch. First add
the drain on the current 100 x 100 grid, then expand the grid with dedicated
performance and geometry tests.

## Files To Continue From

- `backend/app/config.py`: simulation constants, source columns, gauge history.
- `backend/app/world_state.py`: object types/defaults/world schema.
- `backend/app/fluid_solver.py`: Warp shallow-water kernels, source, tracers,
  obstacle mask, body sampling.
- `backend/app/rigid_body.py`: GPU rigid integration, buoyancy, collisions.
- `backend/app/simulation.py`: lifecycle, fixed timestep, streaming, gauge runtime.
- `frontend/src/world/types.ts`: frontend wire/object types.
- `frontend/src/world/WorldStore.ts`: mirrored world and gauge history.
- `frontend/src/world/ObjectFactory.ts`: visual object builders.
- `frontend/src/scene/SceneManager.ts`: water topology, tracer rendering.
- `frontend/src/ui/UI.ts`: controls and GAUGE readout.
- `frontend/src/main.ts`: WebSocket/store/UI wiring.
- `tests/test_backend.py`: backend regression suite.
- `tests/e2e.mjs`: browser and WebSocket E2E suite.
- `README.md`: user-facing feature documentation.
- `ARCHITECTURE.md`: data flow and coupling documentation.
- `TEST_REPORT.md`: latest test record.

## Safe Continuation Procedure

1. Check whether port `8756` is occupied before running tests.
2. Read this file, `README.md`, `ARCHITECTURE.md`, and `TEST_REPORT.md`.
3. Run backend tests before changing the solver.
4. Implement DRAIN with GPU tests before changing the map dimensions.
5. Stop any running EXE before rebuilding or running launcher E2E.
6. Run `frontend\npm run build`, `tests\run_all.bat`, and a live EXE test.
7. Rebuild `NatureLab.exe`.
8. Rename the release directory to `NatureLab_0.5.0` before creating the next archive.
9. Create a new archive and record its SHA-256 here and in `TEST_REPORT.md`.
