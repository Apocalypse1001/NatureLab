Проведён внешний аудит NatureLab Foundation.

Архитектуру проекта НЕ переписывать. Сохранить Three.js → WebSocket → Python → Warp и существующее разделение WorldState / FluidSolver / RigidBodySystem / ComputeEngine / EventLog.

Подготовить NatureLab Foundation 0.2.

Обязательные исправления:

1. Исправить NatureLab.exe. Текущий PyInstaller launcher не должен запускать backend через `sys.executable -m uvicorn`, поскольку в frozen-сборке sys.executable указывает на сам NatureLab.exe. Исключить рекурсивный запуск. Сделать надёжный launcher и корректное завершение backend.

2. Поддержать добавление и удаление объектов во время RUNNING. Сейчас добавление объекта после initialize вызывает KeyError в PlaceholderRigidBodySystem. RigidBodySystem должен динамически регистрировать/удалять тела.

3. START во время RUNNING сделать idempotent. Повторное нажатие PLAY не должно сбрасывать sim_time и не должно заменять InitialWorldState.

4. Исправить синхронизацию terrain frontend/backend. Сейчас frontend применяет brush на каждом pointermove, а backend получает throttled-команды, из-за чего heightmaps расходятся. После последовательности редактирования числовой terrain frontend и backend должен быть идентичен. Добавить автоматический тест checksum.

5. Исправить передачу rotation. В backend всегда передавать ровно `[x, y, z]`, без Euler order.

6. Добавить строгую валидацию входящих World/Object данных. Position/rotation/scale должны иметь ожидаемую размерность и numeric значения.

7. Включить реальные E2E-тесты в архив. TEST_REPORT не должен ссылаться на отсутствующие `nl_tests`. Все заявленные тесты должны запускаться из чисто распакованного проекта.

8. Зафиксировать воспроизводимые версии runtime dependencies. Использовать стабильную версию NVIDIA Warp и зафиксировать версию, на которой реально выполнялись тесты.

9. Подготовить SimulationManager к двусторонней связи Fluid ↔ RigidBody. Вода должна получать obstacle/boundary information от terrain и объектов, а objects — water depth/velocity/forces.

10. Не строить будущий rigid/debris solver вокруг Python-loop по каждому объекту. Подготовить GPU-friendly representation: массивы positions, velocities, rotations, masses, states и object IDs.

11. Общий tick оставить fixed, например 1/60, но FluidSolver должен поддерживать внутренние substeps/adaptive timestep для будущего shallow-water solver.

12. Убрать архитектурную зависимость от передачи всех Warp particles в Three.js каждый stream tick. Предусмотреть bulk frames для WATER_HEIGHT, VELOCITY_FIELD, OBJECT_TRANSFORMS, TERRAIN_PATCH и EVENTS.

13. Убрать фиксированный frontend limit 120 000 particles либо сделать динамическое выделение buffer. Visualization particle count не должен определять simulation particle count.

14. Добавить тест редактирования мира во время RUNNING:
ADD CAR → MOVE → REMOVE → ADD TREE → PAUSE → RESUME → RESET.
Не должно быть exceptions, state corruption или изменения InitialWorldState.

После исправлений предоставить новый архив `NatureLab_Foundation_0.2.zip`, обновлённые README.md, ARCHITECTURE.md и воспроизводимый TEST_REPORT.md.

НЕ начинать реализацию полноценного FloodSolver на этом этапе.

Цель 0.2 — окончательно стабилизировать фундамент перед GPU shallow-water simulation.