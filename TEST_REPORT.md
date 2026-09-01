# NatureLab 0.5.1 — TEST REPORT

Дата: 2026-09-02. Windows 11 Pro 10.0.26200, RTX 5090 32 GB (sm_120, mempool enabled),
CUDA Toolkit 12.9 / driver 13.2, Python 3.12.7, NVIDIA Warp 1.17.0, Node v24.18.0.

Все результаты ниже — фактический вывод команд на этой машине в этой сессии, а не
перенос отчёта донорской сборки. Версия 0.5.1 не добавляет своей физики: это перенос
GPU-движка 0.5.0 в git-репозиторий, поэтому основной смысл прогона — доказать, что
движок работает **здесь** ровно так же, как работал там.

## 1. Backend physics (CUDA / Warp)

```bat
python tests\test_backend.py
```

```text
Ran 21 tests in 8.894s

OK
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.2
   Devices:
     "cpu"      : "Intel64 Family 6 Model 198 Stepping 2, GenuineIntel"
     "cuda:0"   : "NVIDIA GeForce RTX 5090" (32 GiB, sm_120, mempool enabled)
```

| Physics regression | Result |
|---|---|
| Warp selftest 100 000 points на `cuda:0` | PASS |
| Lake at rest: h постоянна, max velocity 0 | PASS |
| Closed edge slug: ошибка объёма за 10 с | PASS |
| Ridge 3 м блокирует 0.5 м воды | PASS |
| HOUSE footprint отклоняет поток | PASS |
| Rotated HOUSE — точная yaw-OBB маска | PASS |
| MOVE/REMOVE HOUSE без phantom water | PASS |
| Adaptive CFL увеличивает подшаги на глубоком/быстром поле | PASS |
| Стартовая вода занимает только два столбца у западной кромки | PASS |
| Runtime edge inflow создаёт физический поток вниз по течению | PASS |
| BOX реагирует раньше CAR при равной глубине | PASS |
| 3x3 footprint: частичное смачивание, приподнятый объект сухой | PASS |
| Миграция buoyancy world v1 и валидация 0..1 | PASS |
| Нулевая вода/поток оставляет объект строго INTACT | PASS |
| Плавающие объекты разрешают столкновения | PASS |
| GAUGE: точечный сэмпл, отсутствие влияния на поток, история, RESET | PASS |
| Стриминг GPU tracers | PASS |
| RESET и детерминированный replay | PASS |
| Terrain edit во время RUNNING отклоняется | PASS |
| START идемпотентен, RUNNING edit sequence + RESET | PASS |
| Строгая валидация трансформаций и протокола | PASS |

**21 / 21 PASS.**

## 2. Browser + WebSocket E2E

```bat
node tests\e2e.mjs
```

```text
PASS  frontend + WebSocket
PASS  strict WebSocket root validation
PASS  Warp selftest (cuda:0)
PASS  dynamic particle buffer >120k
PASS  terrain frontend/backend checksum
PASS  Warp shallow-water heightfield
PASS  dry and HOUSE water triangles masked
PASS  initial wave starts at left map edge
PASS  flow tracer display controls
PASS  GPU flow tracers follow fluid velocity
PASS  GAUGE depth, speed, arrival and history
PASS  runtime Water level updates fluid field
PASS  START idempotent
PASS  floating objects resolve collisions
PASS  RUNNING edit sequence + RESET integrity
PASS  rotation xyz round-trip
PASS  no browser errors
NatureLab 0.5.1 E2E: PASS
```

**17 / 17 PASS.** Строка версии читается с работающего backend (`/api/status`), а не
захардкожена — раньше она осталась бы `0.5.0` после бампа.

## 3. Агентский драйвер (живое приложение)

```bat
node .claude\skills\run-naturelab\driver.mjs --smoke
```

```text
ok SMOKE PASS  sim=RUNNING t = 5.3s objects=1 triangles=27444 depth 1m flat -> west 0.556m / east 0m
```

Сценарий с тремя объектами, edge inflow 1.5 м, 20 с симуляции:

```text
t= 6s   {"cells":10201,"wet":2525,"max":1.5,"mean":0.296,"westMean":1.148,"eastMean":0,"drawnTriangles":4800}
t=20s   {"cells":10201,"wet":6329,"max":1.5,"mean":0.631,"westMean":1.329,"eastMean":0,"drawnTriangles":12309}
```

Смоченных клеток 2525 → 6329 за 14 с — фронт физически движется, а не заливает карту
мгновенно. Скриншот `t20.png` показывает воду с чёткой неровной кромкой фронта на
западной половине и сухой рельеф на восточной; событие `Car_001 OBJECT_FLOATING
(gpu_buoyancy_supports_weight)` в логе. FPS 57, Sim FPS 58.5.

## 4. Прямой прогон солвера (без сервера и браузера)

```text
t=  10s  wet= 3030  vol=  2290.4 m3  vmax=2.605 m/s  substeps=1
t=  20s  wet= 5151  vol=  3484.0 m3  vmax=2.023 m/s  substeps=1
t=  30s  wet= 6868  vol=  4311.3 m3  vmax=1.587 m/s  substeps=1
```

Объём растёт монотонно (вода поступает только через кромку-источник), максимальная
скорость физически правдоподобна (1.5–2.6 м/с), CFL держит один подшаг.

## 5. Проверка релизного архива

```bat
python tools\make_release.py
```

```text
releases\NatureLab_v0.5.1.zip  57 files  0.3 MB
```

SHA-256 записан в `releases/CHECKSUMS.txt`. Внутрь архива этот файл намеренно не
кладётся: контрольная сумма архива не может лежать в нём самом — она бы
самоинвалидировалась при каждой пересборке.

Архив распакован во временный каталог и запущен оттуда **без пересборки фронтенда**:

```text
GET /  -> 200
{"app":"NatureLab","backend":"online","engine":{"version":"0.5.1","engine":"warp",
 "warp_available":true,"cuda":true,"device":"cuda:0","gpu_name":"NVIDIA GeForce RTX 5090", ...}}
```

**PASS** — требование «распаковал и запустил любую версию» выполняется.

## Не проверено

- `tests\run_all.bat` целиком и `tests\test_launcher.ps1` — требуют собранного
  `NatureLab.exe` (PyInstaller). `build_exe.bat` в этой сессии не запускался.
- 30-минутный soak — донорская сборка проходила его (108 000 шагов, 849.3 шага/с);
  здесь код тот же, но повторно на этой машине не прогонялся.
