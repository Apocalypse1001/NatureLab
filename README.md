# NatureLab 0.5.1 - Educational FloodLab

Стабилизированная физическая и GPU-архитектура FloodLab.

```text
Three.js -> WebSocket -> FastAPI -> SimulationManager -> NVIDIA Warp/CUDA
```

## Что нового в 0.5.1

Собственной физики эта версия не добавляет — она сводит воедино две ветки, которые
разошлись: GPU-движок 0.5.0 и слой документации/истории этого репозитория. Разбор
расхождения — [`docs/06_next_steps.md`](docs/06_next_steps.md).

- Движок 0.5.0 (Warp `h/u/v` shallow water, GAUGE, tracers, e2e) перенесён в
  репозиторий под git; вся история и цепочка `docs/01…06` сохранены.
- Прежний NumPy-солвер (диффузия высот) удалён; его аудит и причины замены —
  [`docs/05_audit_v0.4_water.md`](docs/05_audit_v0.4_water.md). Состояние до замены
  доступно по тегу `v0.4-numpy-riverlab`.
- Появился единый источник истины для версии (`config.VERSION`) — она видна в
  заголовке приложения и в `/api/status`.
- `tools/make_release.py` собирает самодостаточный архив в `releases/`: распаковал
  и запустил, без пересборки фронтенда.
- Агентский драйвер `.claude/skills/run-naturelab/` приведён в соответствие с новым
  протоколом (кадр `WATER_HEIGHT` теперь несёт абсолютную отметку поверхности, а не
  глубину).

**Временно отсутствует относительно ветки v0.4:** эрозия, перенос осадка, тип `ROCK`
с поднятием дна и температурная гипотеза Шаубергера. Переносятся на Warp-солвер
следующей версией (v0.6.0) — см. `docs/06_next_steps.md`, раздел 2.

## Physics Foundation

- Reflective/no-flux границы без frozen edge depth и поперечного drift.
- Bed-aware face flux: вода не проходит через terrain выше free surface.
- HOUSE footprint является solid, depth/u/v внутри solid всегда равны нулю.
- Rotated HOUSE использует точный yaw-oriented footprint без solid AABB corners.
- Dry и solid water triangles удаляются из frontend geometry.
- MOVE/REMOVE HOUSE консервативно remap существующую воду без phantom volume.
- Terrain и obstacle GPU buffers обновляются только по revision.
- CFL timestep зависит от `max(|velocity| + sqrt(g*depth))`.
- Fluid sampling по ориентированному 3x3 footprint, drag force и rigid integration выполняются Warp kernels на `cuda:0`.
- Высота основания объекта учитывается при расчёте погружения.
- Object motion использует mass, volume, drag, contact area и ground friction.
- Поддерживаются реальные состояния INTACT, MOVING, FLOATING, SETTLED.
- Edge inflow level задаёт высоту воды на левой границе `x=-50 м`.
- При старте вся карта сухая, кроме двух крайних source columns; волна распространяется физически от края.
- Terrain editing заблокирован во время RUNNING.
- Старые demo particles заменены 8 000 tracers реального velocity field с UI-контролем.

## Educational Measurements

- `GAUGE` является не влияющей на поток измерительной точкой.
- Live measurements: water depth, absolute surface elevation и flow speed.
- Wave arrival фиксируется один раз событием `WATER_ENTERED_AREA`.
- История хранит 600 samples с шагом 0.1 s simulation time и отображается sparkline.
- Runtime measurements очищаются при MOVE/RESET и не загрязняют сохранённый WorldState.

## Запуск

Требуется Python 3.12:

```bat
python -m pip install -r backend\requirements.txt
start.bat
```

`start.bat` поднимает backend и открывает браузер; `NatureLab.exe` делает то же самое,
если он собран (`build_exe.bat`). Приложение доступно только на `http://127.0.0.1:8756/`.

Из распакованного архива (`releases/NatureLab_v*.zip`) всё работает сразу: собранный
фронтенд лежит внутри, `npm` для запуска не нужен.

## Сборка релиза

```bat
cd frontend && npm run build && cd ..
python tools\make_release.py 0.5.1
```

Создаёт `releases\NatureLab_v0.5.1.zip` и записывает его SHA-256 в
`releases\CHECKSUMS.txt`. Тот же номер проставляется в `config.VERSION`, поэтому
запущенное приложение всегда сообщает, какая это версия.

## Тесты

Требования: Node.js >=22.12 и Chrome/Edge.

```bat
python tests\test_backend.py
node tests\e2e.mjs
```

`tests\run_all.bat` дополнительно запускает launcher-тест и требует собранного
`NatureLab.exe`.

Physics suite проверяет lake-at-rest, 1D symmetry, conservation, terrain barrier,
static obstacle, obstacle move/remove, upload revisions, adaptive CFL, heavy-vs-light,
zero flow, RESET и determinism.

## Tested versions

- Python 3.12.7
- NVIDIA Warp `1.17.0` (stable PyPI release)
- NVIDIA GeForce RTX 5090 32 GB, driver 596.49
- FastAPI `0.141.1`, Uvicorn `0.52.4`, NumPy `2.5.2`, websockets `17.1`
- Three.js `0.169.0`, Vite `5.4.21`, TypeScript `5.9.3`
- Puppeteer Core `25.9.0`, PyInstaller `6.22.2`

## Ограничения

- Solver первого порядка предназначен для образовательных сценариев, не инженерных расчётов.
- Rigid bodies имеют translational force model без angular dynamics.
- Collision correction использует 2D footprint/radius, не полноценный rigid contact solver.
- Evolved h/u/v field пока не сериализуется через SAVE.
- Erosion, sediment, destruction, vegetation physics и replay не реализованы
  (эрозия/осадок/`ROCK` — план v0.6.0, см. `docs/06_next_steps.md`).
- Единственный источник воды — приток с западной кромки карты. Дождя, точечного
  притока и подъёма уровня во времени нет (осознанно отложено).
