# Источники и происхождение данных

Дата получения пакета: 2026-09-22.

## Кадастровая граница

- Поставщик: Kadaster / PDOK
- Набор: BRK Kadastrale Kaart, коллекция `perceel`
- API: <https://api.pdok.nl/kadaster/brk-kadastrale-kaart/ogc/v1/>
- Объект: `AMR04`, секция `C`, номер `2426`
- Файлы: `parcel_rd_epsg28992.geojson`, `parcel_wgs84.geojson`
- Примечание: открытая кадастровая карта показывает ориентировочное положение границ и не заменяет вынос границ геодезистом.

## Рельеф

- Поставщик: Rijkswaterstaat / AHN / PDOK
- Набор: Actueel Hoogtebestand Nederland
- WCS: <https://service.pdok.nl/rws/ahn/wcs/v1_0>
- Покрытия: `dtm_05m`, `dsm_05m`
- Разрешение: 0,5 м
- CRS: `EPSG:28992`
- BBOX: `150340,482880,150410,482970`
- Файлы: `terrain_AHN_DTM_0.5m.tif`, `terrain_AHN_DSM_0.5m.tif`

DTM используется как физический baseline поверхности земли. DSM содержит верхушки объектов и растительности и предназначен для сравнения/визуального анализа.

## Ортофотоснимки

- Поставщик: Beeldmateriaal / PDOK
- WMS: <https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0>
- Слои: `2022_ortho25`, `2023_orthoHR`, `2024_orthoHR`, `2025_orthoHR`, `2026_orthoHR`, `Actueel_orthoHR`
- Лицензия сервиса: CC BY 4.0
- Файлы: `orthophoto_*.png` и соответствующие `.pgw`

## Наземные объекты BGT

- Поставщик: Kadaster / PDOK
- API: <https://api.pdok.nl/lv/bgt/ogc/v1/>
- Лицензия API: CC0 1.0
- Контекстная область: небольшой буфер вокруг участка
- Слои: здания, вода, вспомогательная вода, дороги, вспомогательные дороги, растительность и непокрытая растительностью поверхность
- Файлы: `bgt_*_context.geojson`

## Почва

- Поставщик: BRO / TNO / PDOK
- Набор: BRO Bodemkaart (SGM)
- WMS: <https://service.pdok.nl/tno/bro-bodemkaart/wms/v1_0>
- Точка выборки: `RD (150375, 482925)`
- Полученный код: `Mn35Av`
- Описание: `Kalkrijke poldervaaggronden; lichte klei, profielverloop 5` — известковая польдерная лёгкая глина.
- Файлы: `soil_point_raw.json`, `soil_and_groundwater_summary.json`

Масштаб карты около 1:50 000. Значение нельзя использовать вместо бурения или лабораторного определения фильтрации.

## Грунтовые воды

- Поставщик: BRO / TNO / PDOK
- Набор: Model Grondwaterspiegeldiepte (WDM)
- WMS: <https://service.pdok.nl/tno/bro-model-grondwaterspiegeldiepte/wms/v2_0>
- Разрешение модели: 50 × 50 м
- Точка выборки: `RD (150375, 482925)`
- GHG, средняя высокая глубина: 67 см ниже поверхности
- GLG, средняя низкая глубина: 122 см ниже поверхности
- GVG: значение в точке сервисом не возвращено
- GT: сохранено исходное значение класса `14` без самостоятельной расшифровки

## Погода

- Поставщик: KNMI
- Станция: 269, Lelystad Airport
- Координаты станции: `5.520, 52.458`, высота `−3,70 м`
- Период: 2025-01-01 — 2026-09-22
- Исходник: почасовые наблюдения KNMI
- Файлы: `KNMI_269_Lelystad_hourly_2025-2026.txt`, `weather_KNMI_269_hourly.csv`
- Лицензия KNMI Open Data: CC BY 4.0

KNMI предупреждает, что почасовые ряды могут быть неоднородны из-за переноса станций и изменения методик; этот файл пригоден для сценариев и погодного контекста, но не для самостоятельного вывода о климатическом тренде.

