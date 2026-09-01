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

v0.4: `terrain.heights` может также меняться физически (erosion/deposition, см. Sediment ниже),
не только через ручной brush. `SimulationManager._stream()` шлёт тот же JSON `terrain_patch`
проактивно во время RUNNING (раз в `config.TERRAIN_RESYNC_INTERVAL_S`), не только как ответ на
`terrain_brush` op — frontend-обработчик идентичен для обоих случаев (диспетчеризация по типу
сообщения, не по тому, был ли это ответ на конкретный запрос).

## Sediment transport (RiverLab, v0.4)

`ShallowWaterFluidSolver._sediment` — то же поле, что depth, на той же сетке. Каждый substep,
после расчёта потока: `capacity = SEDIMENT_CAPACITY_SCALE * speed * depth`; где capacity выше
текущего sediment — эрозия (`terrain -= amount; sediment += amount`, ограничено bedrock floor и
только на мокрых клетках); где ниже — осаждение (наоборот). Перенос осадка переиспользует уже
посчитанные и уже clamped flow-значения (не отдельный adverction solver) — концентрация
(`sediment/depth`) переносится пропорционально объёму воды, ушедшему в каждом направлении, что
автоматически гарантирует: клетка не может отдать больше осадка, чем в ней есть (то же
рассуждение о сохранении, что уже используется для depth).

`terrain.heights` мутируется по ссылке (тот же объект, что `world.terrain`) — значит erosion
автоматически меняет дальнейший flow тем же путём, что и ручной `terrain.brush()`. Намеренно
медленно (см. docstring `config.SEDIMENT_*_RATE`) — реальная эрозия медленна относительно
одного паводка; должна быть заметна на длинных RiverLab-сравнениях (River A vs River B), не
искажать короткие FloodLab-сценарии.

## Water temperature / shade (RiverLab, v0.4, Schauberger hypothesis)

`ShallowWaterFluidSolver.set_environment(base_temperature, shade)` пересчитывает
`_temperature_factor` **каждый tick заново** из `RigidBodySystem.shade_snapshot()` (позиции тел
с `shade_cooling > 0`, по умолчанию только TREE) — намеренно **не** персистентное/диффундирующее
поле (решение пользователя 2026-09-01, после разбора первоисточников Шаубергера про тень/русло —
см. `docs/04_TZ_v0.3_roadmap.md` v0.4). Множитель применяется и к `FLUID_FLOW_GAIN`, и к
`SEDIMENT_CAPACITY_SCALE`, clamp `[TEMP_FACTOR_MIN, TEMP_FACTOR_MAX]`. `environment.temperature`
— ручная база (UI-слайдер + op `environment_temperature`), тень — модулирует её локально
автоматически из позиций деревьев, без ручной покраски.

## River flow (RiverLab, v0.4)

`water.flow_enabled` (чекбокс River flow) → `ShallowWaterFluidSolver.set_river_flow()`,
читается **живьём каждый tick**, поэтому переключается во время RUNNING — в отличие от
`water.level`, который солвер читает только в `initialize()`. Реализовано граничными условиями,
не новым солвером: западная кромка удерживается как исток (`FLUID_RIVER_SOURCE_DEPTH`),
восточная — как устье (`FLUID_RIVER_SINK_DEPTH`); постоянный перепад между ними уже существующая
консервативная flux-схема несёт как непрерывное течение. Масса намеренно не сохраняется **только**
на этих двух кромочных столбцах. Без этого плоская заливка `depth = level - terrain` даёт нулевой
градиент и вода не двигается вообще.

## Riverbed rocks (RiverLab, v0.4)

ROCK — часть русла, не стена. `obstacle_snapshot()` отдаёт солверу только тела с
`bed_height == 0` (их бинарная obstacle-mask делает бесконечно высокими — верно для дома);
тела с `bed_height > 0` идут отдельным каналом `bed_snapshot()` →
`ShallowWaterFluidSolver.set_bed_obstructions()` и поднимают **эффективное дно** куполом
(`_bed_offset`, добавляется к `terrain.heights` в `_step`). Отсюда сразу: обтекание, зависимость
от размера (радиус — горизонтальный `scale`, высота — вертикальный) и от положения (на сухой
отмели вытеснять нечего), а вместе с sediment-механикой — размыв боков и осаждение в тени, то
есть meandering.

Как и `_temperature_factor`, поле **stateless**: пересчитывается каждый tick из текущих позиций
и никогда не пишется в terrain мира, поэтому передвинутый камень не оставляет кратера. Terrain
под камнем защищён от эрозии (`config.BED_EROSION_SHIELD`) — валун это скальная порода, река
обтекает его, а не выкапывает под ним яму.

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
