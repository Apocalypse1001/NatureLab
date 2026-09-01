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

ADD регистрирует slot, UPDATE синхронизирует массивы, REMOVE делает swap-delete.
Placeholder использует batched masks; будущий Warp solver сможет загрузить эти массивы.

## Fluid ↔ Rigid contract

На каждом global fixed tick:

1. Rigid предоставляет object obstacle snapshot.
2. Fluid получает terrain + obstacle boundaries.
3. `FluidSolver.advance(global_dt, max_substeps, stability_dt)` выбирает внутренние substeps.
4. Fluid batch-sample возвращает depths, velocities и forces для body positions.
5. Rigid применяет samples к плотным массивам.

Это только контракт; FloodSolver намеренно не реализован.

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
