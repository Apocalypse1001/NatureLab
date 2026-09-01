# NatureLab Foundation 0.2 / v0.3 — TEST REPORT

Дата: 2026-09-01 (v0.3 addendum поверх 0.2, дата 0.2-раздела ниже: 2026-08-31). Все тесты
находятся в `tests/` и запускаются из чисто распакованного проекта командой `tests\run_all.bat`.

## v0.3 — ShallowWaterFluidSolver (проверено на этой машине, CPU, без GPU)

Реально выполнено в этой сессии (`python3 tests/test_backend.py`), не только заявлено:

| Проверка | Статус |
|---|---|
| Conservation: суммарный объём воды не меняется без источника/стока (closed boundary) | PASS |
| Нет отрицательной глубины; нет phantom water внутри препятствий (клетки под объектом = 0) | PASS |
| Перемещение препятствия меняет итоговое распределение потока (причинный тест) | PASS (после исправления фикстуры теста — препятствие изначально ставилось в заведомо сухую зону, solver был прав, тест — нет) |
| Объект видит глубину воды вокруг своего футпринта и может всплыть | PASS (после исправления бага: `sample_for_bodies` сэмплил ровно в центре объекта, а объект сам исключает воду в своей точке — объект физически не мог всплыть; найдено integration smoke-тестом через `SimulationManager`, не unit-тестом) |
| Полный набор из 4 тестов Foundation 0.2 остаётся зелёным без изменений | PASS |
| Integration smoke test: BOX на плоском мокром terrain реально переходит в FLOATING | PASS (ручной прогон через SimulationManager, не входит в run_all.bat) |

`REQUIRES GPU VERIFICATION`: нет для этого пункта — solver работает на чистом NumPy, не через
Warp/CUDA, поэтому численно идентичен на любой машине. GPU-порт для производительности на
больших сетках — отдельный будущий пункт, не покрыт этим прогоном.

## Foundation 0.2 (дата 2026-08-31)

| Проверка | Статус |
|---|---|
| Python compile + Warp selftest 100 000 points | PASS |
| START idempotent, InitialWorldState не заменяется | PASS |
| RUNNING: ADD CAR → MOVE → REMOVE → ADD TREE → PAUSE → RESUME → RESET | PASS |
| Dynamic rigid register/update/unregister без KeyError | PASS |
| RESET удаляет runtime edits и восстанавливает baseline | PASS |
| Strict transform/world validation (length, finite, scale) | PASS |
| Rotation backend round-trip ровно xyz | PASS |
| Terrain frontend/backend SHA-256 checksum | PASS |
| Bulk protocol version/kind/exact length validation | PASS |
| Dynamic frontend particle buffer, frame 150 001 | PASS |
| Visualization count отделён от simulation count | PASS |
| Fluid boundaries + adaptive internal substeps + batched samples | PASS |
| Browser WebSocket validation и отсутствие console errors | PASS |
| Frozen NatureLab.exe: root, no recursion, owned backend shutdown | PASS |

## Команды

```bat
python tests\test_backend.py
cd tests
npm ci
npm test
```

Backend tests: 4 suites PASS. Portable browser E2E: PASS.

Launcher lifecycle test (`NATURELAB_TEST_MODE=1`):

```text
launcher root=<archive root>
backend pid=<pid> python=<external python.exe>
backend ready; stopping owned process
launcher done
orphan backends=0
```

## Окружение

Python 3.12.10, Warp 1.17.0, CPU device (CUDA driver unavailable), Node 22.14.0,
Chrome/Edge headless, versions dependencies зафиксированы в lock/requirements files.

CUDA на RTX 5090 требует отдельного прогона на целевой машине; device selection `cuda:0`
реализован, но в данном окружении физически проверить CUDA невозможно.
