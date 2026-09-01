# NatureLab Foundation 0.2 — Architecture

```text
Three.js editor/visualization
          ↕ JSON + versioned bulk WebSocket frames
FastAPI / SimulationManager (authoritative WorldState, fixed dt=1/60)
          ↕ boundaries / batched samples
FluidSolver  ↔  RigidBodySystem (dense arrays)
          ↘       ↙
        ComputeEngine → NVIDIA Warp → CUDA/CPU
```

## State ownership

Backend является источником истины. Frontend делает optimistic editor update только
для реально отправленной команды и принимает authoritative ответ. RESET восстанавливает
единственный InitialWorldState, снятый при IDLE→RUNNING. PLAY при RUNNING ничего не меняет;
PLAY при PAUSED только возобновляет часы.

## Dynamic body lifecycle

`RigidStateBuffer` содержит contiguous NumPy arrays:

```text
ids[], index{id→slot}, positions[N,3], velocities[N,3], rotations[N,3],
masses[N], buoyancies[N], states[N]
```

```text
+ frictions[N], drags[N], foundation_heights[N], footprint_radii[N]
```

ADD регистрирует slot, UPDATE синхронизирует массивы, REMOVE делает swap-delete.
v0.3: `ForceRigidBodySystem` считает gravity/buoyancy/drag/friction векторно по всем телам
разом (см. `docs/04_TZ_v0.3_roadmap.md`); будущий Warp solver сможет загрузить эти же массивы
без изменения контракта.

## Fluid ↔ Rigid contract

На каждом global fixed tick:

1. Rigid предоставляет object obstacle snapshot (positions, rotations, masses, **radii**).
2. Fluid получает terrain + obstacle boundaries; каждый obstacle вырезает дыру радиусом
   `radii[i]` (не единый фиксированный радиус — дом и коробка вытесняют разный объём).
3. `FluidSolver.advance(global_dt, max_substeps, stability_dt)` выбирает внутренние substeps.
4. Fluid batch-sample возвращает depths, velocities и forces для body positions — усреднением
   по всем открытым (не-obstacle) клеткам в окрестности каждого тела, а не по фиксированному
   направленному кольцу (направленное кольцо ловило дыру *соседнего* объекта в multi-object
   сценах — см. `ShallowWaterFluidSolver.sample_for_bodies` docstring).
5. Rigid применяет samples к плотным массивам через явную модель сил (gravity/buoyancy/drag/
   friction), не через порог.

v0.3: `ShallowWaterFluidSolver` реализует этот контракт (height field на сетке terrain,
outflow-limited обмен, obstacle mask из `obstacle_snapshot()`). Полноценный
CFD/Navier-Stokes-уровень намеренно не реализован — см. `docs/04_TZ_v0.3_roadmap.md`, раздел 3
(причинный реализм — цель, инженерный CFD-реализм — не цель проекта).

## Water rendering (frontend)

`SimulationManager._stream()` шлёт `WATER_HEIGHT` bulk-фрейм (полное поле depth) каждый tick
пока RUNNING/PAUSED. `SceneManager.waterMesh` имеет ту же сетку вершин, что и terrain;
`updateWaterField()` деформирует её по клеткам (`y = terrain_height + depth`), так что вода
физически дренирует до уровня terrain там, где depth≈0 — включая внутри/вокруг препятствий.
До первого `START` — плоский editor-preview на уровне слайдера (`setWater()`).

## Terrain consistency

Brush sample имеет одну последовательность: frontend применяет только отправленные samples;
backend применяет sample к float32 TerrainGrid и отвечает `terrain_patch` с полным массивом
и SHA-256 над little-endian float32 bytes. Frontend заменяет локальные heights. Автотест
сравнивает checksum числовых массивов.

## Bulk protocol v2

Header (16 bytes): `NL | version:u8 | kind:u8 | count:u32 | time_ms:u64`.
Далее строго проверяемый little-endian float32 payload. Kinds:

- `PARTICLES`
- `WATER_HEIGHT`
- `VELOCITY_FIELD`
- `OBJECT_TRANSFORMS`
- `TERRAIN_PATCH`
- `EVENTS`

Simulation particles не определяют visualization. Сейчас backend deterministic-downsample
до `VISUALIZATION_PARTICLE_LIMIT=25000`; frontend buffer динамически растёт. Будущий CUDA
gather kernel устранит полный device readback, не меняя протокол.

## Validation

WebSocket root обязан быть JSON object. Vector transforms имеют ровно 3 finite numbers,
scale strictly positive. Terrain length/dimensions, environment wind, states, duplicate IDs,
speed/radius/strength и binary frame length/version/kind валидируются до мутации state.

## Launcher

Frozen launcher разрешает install root через `sys.executable.parent`, но никогда не запускает
backend через frozen `sys.executable`. Он ищет внешний/portable Python, хранит `Popen`,
останавливает только собственный backend и ждёт завершения. Test mode проверяет отсутствие
рекурсии и orphan processes.
