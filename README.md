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

## Placeholder (намеренно)

- Object-object collision не реализовано: два плавающих объекта могут пересекаться, это не
  проверяется и не разрешается.
- Дерево не умеет ломаться (DAMAGED/BROKEN, break_strength) и становиться новым препятствием —
  отдельный будущий пункт, требует своей физики разрушения, не часть v0.3 force model.
- `OBJECT_TRANSFORMS`, `TERRAIN_PATCH` (потоковый), `EVENTS` (бинарный) bulk frame kinds
  подготовлены как типизированный контракт, но ещё не стримятся (PARTICLES и WATER_HEIGHT — стримятся).
- RiverLab, VolcanoLab, эрозия, разрушение и replay не реализованы.
