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

## Placeholder (намеренно)

- Вода — плоский уровень, не FloodSolver/CFD.
- Rigid body — batched deterministic placeholder, не полноценная динамика.
- Bulk frame kinds кроме particles подготовлены как типизированный контракт.
- Shallow-water, эрозия, разрушение, вулканы и replay не реализованы.
