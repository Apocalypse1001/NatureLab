# NatureLab Foundation 0.2

Стабилизированный фундамент интерактивного образовательного симулятора.
Архитектура сохранена: **Three.js → WebSocket → Python/FastAPI → NVIDIA Warp**.

## Главное в 0.2

- `NatureLab.exe` больше не вызывает себя через `sys.executable`: frozen launcher
  находит внешний Python, проверяет NatureLab `/api/status`, владеет созданным
  backend-процессом и завершает его кнопкой **Stop NatureLab**.
- Повторный PLAY во время RUNNING idempotent: время и InitialWorldState не сбрасываются.
- Объекты можно добавлять, перемещать и удалять во время RUNNING/PAUSED.
- Rigid state хранится в плотных массивах positions/velocities/rotations/masses/
  buoyancies/states + stable IDs и ID→index map (swap-delete).
- Terrain-команды применяются локально только когда реально отправлены. Backend
  возвращает authoritative heightmap + SHA-256 checksum; frontend заменяет свой массив.
- Position/rotation/scale строго валидируются как 3 конечных числа; scale > 0.
- Rotation по WebSocket всегда ровно `[x, y, z]`, без Euler order.
- Fluid API принимает terrain/object boundaries, поддерживает внутренние substeps и
  возвращает batched water depths/velocities/forces для rigid objects.
- Simulation particle count отделён от visualization count. Frontend particle buffer
  растёт динамически, лимита 120 000 больше нет.
- Bulk protocol v2 резервирует PARTICLES, WATER_HEIGHT, VELOCITY_FIELD,
  OBJECT_TRANSFORMS, TERRAIN_PATCH и EVENTS с проверкой kind/версии/точной длины.
- В архив включены воспроизводимые backend + browser E2E-тесты.

## Запуск

1. Установить Python 3.12 и зависимости:

```bat
pip install -r backend\requirements.txt
```

2. Запустить:

```bat
start.bat
```

или двойным кликом `NatureLab.exe`. Приложение откроется на
`http://127.0.0.1:8756/`.

`NatureLab.exe` является тонким launcher и ожидает `backend/` и `frontend/` рядом.
Если используется portable Python, положите его в `runtime/python/python.exe`.

## Сборка frontend

Требуется Node.js 22.14.0 (тестовый runtime):

```bat
cd frontend
npm ci
npm run build
```

Готовый `frontend/dist` уже включён.

## Тесты из чистого архива

Требования: Python dependencies, Node.js >=22.12, Chrome или Edge.

```bat
tests\run_all.bat
```

Или отдельно:

```bat
python tests\test_backend.py
cd tests
npm ci
npm test
```

Путь браузера можно задать переменной `CHROME_PATH`, Python — `PYTHON`.

## Структура

```text
NatureLab/
├── NatureLab.exe / launcher.pyw / start.bat
├── backend/app/
│   ├── main.py, simulation.py, world_state.py
│   ├── compute_engine.py, fluid_solver.py, rigid_body.py
│   └── protocol.py, events.py, persistence.py
├── frontend/src/
│   ├── scene/, world/, editor/, net/, ui/
│   └── main.ts
├── tests/
│   ├── test_backend.py, e2e.mjs, run_all.bat
│   └── package.json, package-lock.json
├── data/
├── ARCHITECTURE.md
└── TEST_REPORT.md
```

## Точные tested versions

- Python 3.12.10
- NVIDIA Warp `1.17.0`
- FastAPI `0.141.1`
- Uvicorn `0.52.4`
- NumPy `2.5.2`
- websockets `17.1`
- Three.js `0.169.0`
- Vite `5.4.21`, TypeScript `5.9.3`
- Puppeteer Core `25.9.0`
- PyInstaller `6.22.2`

На тестовой машине CUDA-драйвер отсутствует, поэтому настоящий Warp kernel выполнялся
на устройстве `cpu`. На RTX 5090 автоматически выбирается `cuda:0`.

## v0.3: реальный FluidSolver

`ShallowWaterFluidSolver` (`backend/app/fluid_solver.py`) заменил плоский water-level
placeholder: height-field воды на сетке terrain, outflow-limited обмен с соседями (conservation
гарантирован клампингом, а не допущением), препятствия реально исключают воду из своих клеток и
меняют направление потока. Регрессионные тесты — `tests/test_backend.py::ShallowWaterSolverTests`:
conservation, отсутствие phantom water внутри объектов, обязательное изменение потока при
перемещении препятствия (причинный критерий из `docs/01_vision.md`).

Важно: этот solver работает на NumPy/CPU **всегда**, независимо от того, выбран `cuda:0` или
`cpu` в `ComputeEngine` (тот отвечает только за визуализационные частицы). GPU/Warp-порт этого
же алгоритма для больших сеток — отдельный будущий пункт (`REQUIRES GPU VERIFICATION` для
производительности на RTX 5090), не сделан в этом коммите.

## v0.3: реальная физика rigid body

`ForceRigidBodySystem` (`backend/app/rigid_body.py`) заменил бинарный buoyancy-порог на честную
интеграцию сил: gravity, buoyancy (растёт с глубиной, capped только собственным весом — не
искусственным потолком), hydrodynamic drag, Coulomb ground friction (уменьшается вместе с
buoyancy). `metadata.foundation_height` — ранее сохранялось, но нигде не использовалось — теперь
реально влияет на исход (Experiment A/B из `docs/01_vision.md`: высота фундамента дома меняет,
затопит ли его). У каждого типа объекта теперь есть `drag` и `footprint_radius` (совпадает по
масштабу с примитивами в `frontend/src/world/ObjectFactory.ts`), поэтому дом и коробка вытесняют
разный объём воды, а не один и тот же фиксированный радиус.

## v0.3: вода реально стримится и рендерится по клеткам

Раньше backend вычислял честное поле глубины, но никуда его не отправлял — а frontend рисовал
воду как плоскость на уровне `Water Level` слайдера, никак не связанную с реальной физикой. Это
означало, что вода **визуально** проходила сквозь препятствия, даже когда backend уже корректно
исключал их из потока. Исправлено:

- backend: `SimulationManager._stream()` отправляет `WATER_HEIGHT` bulk-фрейм (`FrameKind.WATER_HEIGHT`,
  протокол уже был зарезервирован под это в 0.2) с полным полем глубины каждый tick, пока RUNNING/PAUSED.
- frontend: `BackendClient` диспатчит эти фреймы (`waterHeightHandler`), `SceneManager.waterMesh`
  теперь имеет ту же сетку вершин, что и terrain (не плоскость 1×1), и `updateWaterField()`
  деформирует её по клеткам: `y = terrain_height + depth`. До первого `START` (IDLE, редактор)
  показывается плоский preview на уровне слайдера — `setWater()`, как раньше.

Проверено реальным headless-браузером в этой сессии (Chrome for Testing + puppeteer-core,
скачан отдельно, не входит в архив): вода видимо дренирует до уровня terrain ровно в точке
объекта (глубина 0.000 м) и остаётся на реальной глубине (1.2 м) в стороне от него — то есть
вода реально огибает/исключает объект, а не течёт сквозь него. Этот прогон не автоматизирован
в `tests/run_all.bat` (требует Chrome/Chromium и скачивание), но сама логика (`updateWaterField`,
`waterHeightHandler`) покрыта TypeScript-компиляцией (`npm run build` зелёный).

## v0.3: столкновения объектов (интерим)

`ForceRigidBodySystem._resolve_collisions()` — импульсное столкновение дисков в плоскости XZ
(радиус = `footprint_radius`, тот же, что вырезает дыру в воде), mass-weighted позиционная
коррекция + импульс по нормали (Ньютон третий закон выполняется по построению, `restitution=0.2`).
Тяжёлый объект почти не сдвигается от лёгкого; событие `OBJECT_COLLISION` пишется один раз на
начало контакта, а не каждый tick, пока тела соприкасаются.

Это **не** полноценный 3D rigid body (нет вращения/торка/сложных форм) — эквивалент такого
движка есть в `warp.sim` (NVIDIA Warp), и RTX 5090 с большим запасом тянет нужный масштаб; это
осознанно отдельный будущий milestone, а не архитектурное ограничение текущего стека. См.
`docs/04_TZ_v0.3_roadmap.md`.

## v0.3: дерево закреплено к земле (root_strength) и умеет ломаться

`root_strength` (Ньютоны, метаданные объекта, дефолт 0 для всех типов кроме TREE=15000) —
дополнительное сопротивление сверх обычного Coulomb friction, пока объект «закреплён»
(`RigidStateBuffer.rooted`, флаг только True→False, необратимо). Общий механизм, не хардкод
под TREE — любой тип может получить root_strength.

Дерево ломается (`OBJECT_BROKEN`, `obj.state = BROKEN`) двумя путями:
- вода: drag превысил `friction_max + root_strength` (обычный паводок — не превышает; экстремальный
  поток — превышает);
- удар другим телом: эквивалентная сила удара (`impulse / dt` из `_resolve_collisions`) превысила
  root_strength — машина, влетевшая в дерево на большой скорости, может оторвать его, даже без воды.

После разрыва `rooted=False` навсегда — дерево дальше ведёт себя как обычное тело (MOVING/FLOATING
по той же физике), и остаётся зарегистрированным rigid body — то есть автоматически продолжает
вырезать дыру в воде на новом месте: «дерево упало → стало препятствием → поток изменился»
из `01_vision.md` получается бесплатно, без отдельного кода.

Упрощение: `root_strength` объединяет отдельные TREE-свойства из `01_vision.md` (root strength
И break strength) в один порог — различать «вырвало с корнем» и «переломило ствол» как разные
исходы сейчас не нужно, оба дают BROKEN.

Найден и исправлен реальный баг при добавлении ROCK (ниже): `root_strength` защищал закреплённое
тело только от накопления скорости (drag/friction), но НЕ от позиционной коррекции при
столкновении — быстрая машина физически сдвигала с места объект с `root_strength=1e8`, потому
что `_resolve_collisions` считал позиционный push по реальной массе, не по `rooted`. Исправлено:
закреплённое тело теперь имеет эффективную `inv_mass=0` в коллизии (бесконечно тяжёлое), пока не
оторвано, — так и позиция, и скорость теперь одинаково уважают root_strength.

## v0.4 RiverLab: ROCK — камни на дне реки

Новый `ObjectType.ROCK` — не отдельная система, а `root_strength=1e8` (практически бесконечное)
поверх уже готового механизма якорения. `footprint_radius` уже масштабируется через `obj.scale`
(редактируется в UI Properties), поэтому «эффект зависит от размера камня» получилось без новых
полей — крупный камень (`scale=3`) вырезает пропорционально большую дыру в воде и сильнее
отклоняет поток, чем мелкий (`scale=1`), проверено тестом `test_bigger_rock_disrupts_flow_more_than_smaller_rock`.
Фронтенд: палитра/геометрия (`DodecahedronGeometry`, приплюснут и наполовину погружён — не
перекатывается, как DEBRIS).

## v0.4 RiverLab: эрозия, перенос осадка, обратная связь в русло

`ShallowWaterFluidSolver` теперь считает `sediment` (то же поле, что depth/velocity):
capacity = скорость×глубина (упрощённая real-time hydraulic erosion, в духе того же класса
техники, что и сам shallow-water solver — Mei et al.); где capacity>текущий sediment —
эрозия (terrain понижается, sediment растёт), где меньше — осаждение (наоборот). Перенос
осадка использует **те же** flux-значения, что уже двигают воду — не отдельный adverction-солвер.
`terrain.heights` мутируется по ссылке, поэтому эрозия реально меняет русло и это меняет
дальнейший поток — тот же механизм, что уже доказан для ручного brush/препятствий.

Terrain теперь **стримится во время RUNNING** (не только как ответ на ручной brush) —
переиспользует существующий JSON `terrain_patch` канал и уже готовый frontend-обработчик,
раз в `config.TERRAIN_RESYNC_INTERVAL_S` (1с). Без этого эрозия была бы реальной физикой,
невидимой на экране — тот же класс бага, что уже был найден и исправлен для воды.

**Два честных нюанса, найденных при тестировании (не решены в этом коммите):**
- `ShallowWaterFluidSolver.initialize()` заполняет глубину как `water_level - terrain`, что
  даёт абсолютно плоскую водную поверхность с первого тика — **нулевой поток**, пока что-то
  (объект, ручное редактирование terrain) не создаст неровность. Пользователь, поставивший
  только наклон terrain и уровень воды без единого объекта, ничего текущего не увидит.
- Слайдер Water Level во время RUNNING не влияет на `ShallowWaterFluidSolver` — тот читает
  `world.water.level` только один раз при `initialize()`.

## Placeholder (намеренно)

- `OBJECT_TRANSFORMS`, `EVENTS` (бинарный) bulk frame kinds подготовлены как типизированный
  контракт, но ещё не стримятся (PARTICLES, WATER_HEIGHT стримятся; TERRAIN — через JSON,
  не через зарезервированный бинарный `TERRAIN_PATCH` kind).
- RiverLab пока без деревьев-у-берега/тени (см. `docs/04_TZ_v0.3_roadmap.md`), VolcanoLab,
  разрушение зданий/машин (DAMAGED health-модель) и replay не реализованы.
