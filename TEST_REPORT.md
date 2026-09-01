# NatureLab Foundation 0.2 — TEST REPORT

Дата: 2026-08-31. Все тесты находятся в `tests/` и запускаются из чисто распакованного
проекта командой `tests\run_all.bat`.

## Результат

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
