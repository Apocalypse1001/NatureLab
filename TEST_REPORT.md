# NatureLab Foundation 0.2 / v0.3 / v0.4 — TEST REPORT

Дата: 2026-09-01 (v0.4 addendum поверх v0.3/0.2, даты разделов ниже: 2026-08-31 для 0.2). Все
тесты находятся в `tests/` и запускаются из чисто распакованного проекта командой `tests\run_all.bat`.

## v0.4 — RiverLab: ROCK, эрозия/перенос осадка, terrain-стриминг (проверено на этой машине, CPU)

| Проверка | Статус |
|---|---|
| ROCK неподвижен под экстремальным drag и ударом (root_strength=1e8) | PASS |
| Footprint ROCK масштабируется через obj.scale | PASS |
| Больший камень сильнее нарушает поток, чем меньший | PASS |
| Быстрый поток эрозирует terrain и переносит sediment | PASS (после исправления фикстуры теста — `solver.initialize()` даёт гидростатически плоскую поверхность = нулевой поток с первого тика; понадобилось явно задать равномерную глубину поверх наклона, как настоящее русло) |
| Erosion+deposition сохраняют суммарное количество материала (terrain+sediment) | PASS |
| Изменение terrain от эрозии реально меняет дальнейший flow (не игнорируется) | PASS |
| Камень меняет локальный паттерн эрозии (не просто статичная дыра) | PASS |
| Полный набор из 24 предыдущих тестов остаётся зелёным без изменений | PASS |
| Интеграционный тест: `terrain_patch` при RUNNING реально соответствует live-состоянию backend в момент отправки (не рассинхронизирован) | PASS (ручной прогон через SimulationManager._stream, не в run_all.bat) |

**Найдено при тестировании, не является багом физики, но честно задокументировано** (см.
`docs/04_TZ_v0.3_roadmap.md` и README): `ShallowWaterFluidSolver.initialize()` создаёт абсолютно
плоскую водную поверхность (нулевой поток) с первого тика на закрытой системе без объектов;
слайдер Water Level не влияет на solver во время RUNNING (читается только при `initialize()`).

## v0.3 — дерево закреплено к земле, root_strength (проверено на этой машине, CPU)

| Проверка | Статус |
|---|---|
| Дерево остаётся rooted/INTACT под обычным паводковым потоком (depth=1.0, flow=2 м/с) | PASS |
| Дерево отрывается (BROKEN) от экстремального потока (depth=2.0, flow=12 м/с) | PASS |
| Дерево отрывается от удара машиной на большой скорости, даже без воды | PASS |
| Сломанное дерево остаётся зарегистрированным rigid body и продолжает вырезать препятствие в воде | PASS |
| Интеграционный прогон через SimulationManager: машина на 15 м/с врезается в дерево → BROKEN, cause=body_impact_exceeded_root_strength, impact_force≈458717 | PASS (ручной прогон, не в run_all.bat) |

## v0.3 — импульсное столкновение объектов (проверено на этой машине, CPU)

| Проверка | Статус |
|---|---|
| Два тела не проникают друг в друга дальше суммы footprint_radius | PASS |
| Момент импульса сохраняется при столкновении (Ньютон 3); тяжёлое тело почти не двигается от лёгкого | PASS |
| Событие OBJECT_COLLISION пишется один раз на контакт, не каждый tick | PASS |
| Прямая трассировка через SimulationManager: коробка на 6 м/с останавливается ровно на границе радиусов (3.16м при сумме радиусов 3.1м), с отскоком и одним событием (impact_speed≈4.12 м/с) | PASS (ручной прогон, не в run_all.bat) |

Диагностика NaN-warning в позиционной коррекции (0×-inf на диагонали матрицы столкновений)
найдена и убрана до коммита — не влияла на результат (маскировалась), но была неаккуратной.

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

## v0.3 — ForceRigidBodySystem (проверено на этой машине, CPU, без GPU)

| Проверка | Статус |
|---|---|
| Лёгкая коробка начинает двигаться раньше тяжёлой машины при одинаковом потоке | PASS |
| Дом остаётся INTACT на мелкой воде (низкая buoyancy, огромная масса) | PASS |
| Объект с положительной buoyancy рано или поздно всплывает на достаточной глубине | PASS (после исправления бага: buoyant force капалась на уровне `buoyancy_coeff*weight`, поэтому объект с buoyancy<0.85 не мог всплыть вообще ни при какой глубине) |
| `foundation_height` реально меняет исход при одинаковой воде (Experiment A/B из `01_vision.md`) | PASS |
| Полный набор из 14 предыдущих тестов остаётся зелёным без изменений | PASS |

## v0.3 — связка Fluid↔Rigid при нескольких объектах (найдено и исправлено в этой сессии)

Integration smoke test через `SimulationManager` (HOUSE+CAR+BOX на одном поле) вскрыл то, что
изолированные unit-тесты не ловили:

| Проверка | Статус |
|---|---|
| Кольцо сэмплирования не блендится билинейно с dry-клеткой на границе своего же препятствия | PASS (было: ring на +0.5 клетки за краем диска всё равно интерполировался с сухой граничной клеткой, глубина читалась вдвое заниженной даже на полностью открытой воде) |
| Сэмплинг одного объекта не искажается дырой *соседнего* объекта в многообъектной сцене | PASS (было: HOUSE в 3.6 м от BOX занижал прочитанную у BOX глубину с 1.2 до 0.9 м — направленное кольцо задевало чужой obstacle mask; исправлено маскированным усреднением по всем открытым клеткам в окрестности) |
| BOX+HOUSE+CAR на одном поле дают физически согласованные состояния | PASS (ручной прогон через SimulationManager) |

## v0.3 — вода реально видна на фронтенде (реальный headless-браузер, эта сессия)

Backend вычислял честное поле глубины с первого коммита v0.3, но не отправлял его, а frontend
рисовал воду как не связанную с физикой плоскость — то есть вода **визуально** проходила сквозь
препятствия. Исправлено (`WATER_HEIGHT` bulk-стрим + `SceneManager.updateWaterField`) и
проверено настоящим Chrome (Chrome for Testing 152, скачан через `@puppeteer/browsers`,
не входит в архив) с реальным backend и реальной сборкой frontend:

| Проверка | Статус |
|---|---|
| `npm run build` (tsc --noEmit + vite build) зелёный после изменений | PASS |
| Реальный браузер: water mesh получает WATER_HEIGHT-фреймы и становится visible | PASS |
| Глубина воды в точке объекта = 0.000 м (терраса, не вода) | PASS |
| Глубина воды в стороне от объекта = 1.200 м (реальная физика, не 0 и не искажена) | PASS |

Этот прогон не автоматизирован в `tests/run_all.bat` (требует скачивание Chrome/Chromium),
воспроизводился вручную в сессии. `REQUIRES GPU VERIFICATION`: нет — чистый CPU/browser путь.

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
