/**
 * Russian interface text, keyed by the English it replaces (see i18n.ts).
 *
 * Glossary -- one word per idea across labels, hints and readouts, and the
 * same words README.ru.md uses: discharge = расход, gauge = датчик, gauging
 * line = мерный створ, tracers = трассеры, vent = жерло, crest = гребень,
 * spillway = водосброс, outfall = выпуск, storm inlet = дождеприёмник,
 * manhole = колодец, Froude number = число Фруда, calm / shooting flow =
 * спокойное / бурное течение. A hint that names a control quotes the control's
 * Russian label exactly ("Проложить трубу", "Окрашивать воду по числу Фруда",
 * "Расход Q").
 */
export const RU: Record<string, string> = {
  // ---------------------------------------------------------------- top bar
  'PLAY': 'ПУСК',
  'PAUSE': 'ПАУЗА',
  'RESET': 'СБРОС',
  'SAVE': 'СОХРАНИТЬ',
  'LOAD': 'ЗАГРУЗИТЬ',
  'IDLE': 'ОЖИДАНИЕ',
  'RUNNING': 'ИДЁТ',
  'PAUSED': 'ПАУЗА',
  't = {v}s': 't = {v} с',

  // ------------------------------------------------------------- scenarios
  'SCENARIOS': 'СЦЕНАРИИ',
  'River': 'Река',
  'Sewer': 'Ливнёвка',
  'Bridge': 'Мост',
  'Dam': 'Плотина',
  'Volcano': 'Вулкан',
  'Tsunami': 'Цунами',
  'Plot NL01': 'Участок NL01',
  'Scenario: {name}': 'Сценарий: {name}',
  'Loads a prepared world. Everything stays editable: brush the terrain, move objects, change the discharge.':
    'Загружает готовый мир. Всё остаётся редактируемым: правьте рельеф кистью, '
    + 'двигайте объекты, меняйте расход.',
  'A town on the bank of a running river, already flowing when it opens. At the discharge it ships with, the channel runs about 0.7 m deep and the town stays dry. Take Q to the top of the slider and the river leaves its bed: water reaches the street in about a minute and a half and stands about 8 cm deep there after three minutes':
    'Городок на берегу реки; когда мир открывается, река уже течёт. При исходном '
    + 'расходе глубина в русле около 0,7 м, и город остаётся сухим. Поднимите Q '
    + 'до конца ползунка, и река выйдет из берегов: вода дойдёт до улицы примерно '
    + 'за полторы минуты и через три минуты будет стоять там слоем около 8 см',
  'The river town in a downpour, with a storm sewer: three street inlets on 150 mm pipes down to the river. Water runs to each grate, goes down it and pours out of the outfall into the river. Click a pipe to see its flow against what it can carry. The roofs shed their rain onto the ground around them, so the houses feed the street too. At about six minutes the pipe behind the north row is carrying every one of the 11 l/s it can, and a pond starts to grow over its grate -- a centimetre by seven minutes, a hand deep by ten (a faster speed gets there sooner); lay a wider one with "Lay pipe"':
    'Тот же речной городок под ливнем, с ливневой канализацией: три уличных '
    + 'дождеприёмника на трубах 150 мм, ведущих к реке. Вода стекает к каждой '
    + 'решётке, уходит в неё и выливается из выпуска в реку. Щёлкните по трубе, '
    + 'чтобы сравнить её расход с тем, что она способна пропустить. Крыши сбрасывают '
    + 'дождь на землю вокруг домов, так что дома тоже питают улицу. Примерно к '
    + 'шестой минуте труба за северным рядом домов несёт все 11 л/с, на которые '
    + 'способна, и над её решёткой начинает расти лужа: сантиметр к седьмой минуте, '
    + 'в ладонь глубиной к десятой (на большей скорости быстрее); проложите трубу '
    + 'пошире кнопкой «Проложить трубу»',
  'The river town with a bridge on five piers, and a gauging line above and below it; the river is already flowing. Click a line: the same discharge passes both, but the level drops faster across the bridge -- the piers back the water up by about 2 cm and it shoots between them (tick "Colour water by Froude number" and look for white and red). Honest limit: a real bridge narrowing a river this much would raise it by tens of centimetres; this water model has no loss where a flow narrows and widens, so it shows the effect, not its size. Change the piers in the bridge’s properties and watch the numbers':
    'Речной городок с мостом на пяти опорах и мерными створами выше и ниже моста; '
    + 'река уже течёт. Щёлкните по створу: через оба проходит один и тот же расход, '
    + 'но у моста уровень падает быстрее: опоры подпирают воду примерно на 2 см, и '
    + 'между ними она идёт бурным потоком (включите «Окрашивать воду по числу Фруда» '
    + 'и ищите белое и красное). Честное ограничение: настоящий мост, так сильно '
    + 'сужающий реку, поднял бы её на десятки сантиметров; в этой модели воды нет '
    + 'потерь там, где поток сужается и расширяется, поэтому она показывает сам '
    + 'эффект, но не его величину. Меняйте опоры в свойствах моста и следите за '
    + 'цифрами',
  'The same town below a dam. At the discharge it ships with, the spillway carries the river and the dam holds. Push Q past about 60 and the crest goes under in roughly six minutes of simulated time -- or cut the crest with the terrain brush and watch it go at once':
    'Тот же городок ниже плотины. При исходном расходе водосброс пропускает реку, '
    + 'и плотина держит. Поднимите Q выше примерно 60, и гребень уйдёт под воду '
    + 'примерно за шесть минут модельного времени; или прорежьте гребень кистью '
    + 'рельефа и смотрите, как плотину прорывает сразу',
  'A settlement on the flank of a generated cone. The homestead inside the measured 48 m lava run-out is expected to burn; the village just past it is safe at the vent’s shipped discharge -- raise "Discharge Q" in the vent’s own properties, or just wait, to push the flow further':
    'Посёлок на склоне сгенерированного конуса. Хутор в пределах измеренной '
    + 'дальности растекания лавы, 48 м, должен сгореть; деревня сразу за ней при '
    + 'исходном расходе жерла в безопасности. Поднимите «Расход Q» в свойствах '
    + 'самого жерла или просто подождите, чтобы поток ушёл дальше',
  'The only kilometre-scale world here: 2 km across at 10 m cells, because how far the sea goes out is the drawdown divided by the beach slope, and a real gentle shore does not fit in 200 m. The wave is not seeded on the map -- it arrives through the seaward edge, so the sea first withdraws and bares the sea bed, then the wave runs in over the coastal plain. Two things worth knowing: at 10 m cells a house is smaller than one cell, so the buildings are landmarks for scale and do not split the flow the way they do in the River scenario; and water damages nothing in this build (only lava does), so what you see is things carried, not broken':
    'Единственный здесь мир километрового масштаба: 2 км в поперечнике при ячейке '
    + '10 м, потому что то, насколько далеко отступает море, равно понижению уровня, '
    + 'делённому на уклон пляжа, а настоящий пологий берег не помещается в 200 м. '
    + 'Волна не задана на карте: она приходит через морскую границу, поэтому море '
    + 'сначала отступает и обнажает дно, а потом волна накатывает на прибрежную '
    + 'равнину. Стоит знать две вещи: при ячейке 10 м дом меньше одной ячейки, '
    + 'поэтому здания служат ориентирами масштаба и не разделяют поток так, как в '
    + 'сценарии «Река»; и в этой версии вода ничего не разрушает (разрушает только '
    + 'лава), так что вы видите унесённые предметы, а не разрушения',

  // ------------------------------------------------------------ left panel
  'OBJECTS': 'ОБЪЕКТЫ',
  'House': 'Дом',
  'Building': 'Здание',
  'Car': 'Машина',
  'Tree': 'Дерево',
  'Box': 'Ящик',
  'Debris': 'Обломки',
  'Rock': 'Камень',
  'Person': 'Человек',
  'Road': 'Дорога',
  'Source': 'Источник',
  'Drain': 'Сток',
  'Gauge': 'Датчик',
  'Section': 'Створ',
  'Vent': 'Жерло',
  'Storm inlet': 'Дождеприёмник',
  'Manhole': 'Колодец',
  'Outfall': 'Выпуск',
  'Storm sewer': 'Ливневая канализация',
  'Lay pipe': 'Проложить трубу',
  'Pipe diameter': 'Диаметр трубы',
  'Press "Lay pipe"; the steps appear here while you lay it.':
    'Нажмите «Проложить трубу»; пока вы её прокладываете, здесь будут шаги.',
  'Click a storm inlet or a manhole, or bare ground to place an inlet; click along the route; click a manhole or an outfall, or press Enter (or right-click) to place an outfall at the last point. Pipes from several inlets can meet in one manhole and run on in one pipe -- the narrowest pipe on the way sets what gets through. Esc cancels, Backspace takes back a point. A pipe only carries water downhill.':
    'Щёлкните по дождеприёмнику или колодцу либо по пустой земле, чтобы поставить '
    + 'дождеприёмник; щёлкайте вдоль трассы; щёлкните по колодцу или выпуску либо '
    + 'нажмите Enter (или правую кнопку мыши), чтобы поставить выпуск в последней '
    + 'точке. Трубы от нескольких дождеприёмников могут сойтись в одном колодце и '
    + 'дальше идти одной трубой: пропускную способность задаёт самая узкая труба на '
    + 'пути. Esc отменяет, Backspace убирает последнюю точку. Труба несёт воду '
    + 'только под уклон.',
  'Scene': 'Сцена',
  'empty': 'пусто',

  // ----------------------------------------------------------- right panel
  'PROPERTIES': 'СВОЙСТВА',
  'select an object (or add one from the left)': 'выберите объект (или добавьте его слева)',
  'Water': 'Вода',
  'Edge inflow level': 'Уровень притока с края',
  'Open downstream edge (water leaves the map)': 'Открытый нижний край (вода уходит с карты)',
  'Edge inflow (west edge holds the level above)':
    'Приток с края (западный край держит уровень, заданный выше)',
  'River inlet (prescribed discharge)': 'Приток реки (заданный расход)',
  'Discharge Q': 'Расход Q',
  'Settle river': 'Разогнать реку',
  'Start dry': 'Начать с сухого',
  'Starts dry': 'Начинается с сухого русла',
  'Starts with the river flowing': 'Начинается с текущей реки',
  'settling, the river is filling (about a minute)...': 'идёт разгон, русло наполняется (около минуты)...',
  'Flood wave (hydrograph on top of Q)': 'Паводок (гидрограф поверх Q)',
  'Peak Q': 'Пик Q',
  'Flood starts at': 'Паводок начинается на',
  'Rises over': 'Подъём за',
  'Falls over': 'Спад за',
  'Inlet discharge over time': 'Расход притока во времени',
  'Inlet now {q} m³/s': 'Приток сейчас {q} м³/с',
  '(no flood wave)': '(без паводка)',
  'Inlet width': 'Ширина притока',
  'Outlet width': 'Ширина выхода',
  'm (0 = whole edge)': 'м (0 = весь край)',
  'Beyond the downstream edge': 'За нижним краем',
  'sea or pool (free overfall)': 'море или водоём (свободный перелив)',
  'the river runs on': 'река течёт дальше',
  'Colour water by Froude number': 'Окрашивать воду по числу Фруда',
  'Blue: calm, slower than a surface wave travels (Fr < 1). White: critical (Fr = 1). Red: shooting, faster than a wave can go upstream (Fr > 1) -- what you see between bridge piers and over a weir.':
    'Синий: спокойное течение, медленнее поверхностной волны (Fr < 1). Белый: '
    + 'критическое (Fr = 1). Красный: бурное, быстрее, чем волна может идти против '
    + 'течения (Fr > 1), — так бывает между опорами моста и на водосливе.',
  'Erosion (river reshapes the bed)': 'Эрозия (река меняет русло)',
  'Rain': 'Дождь',
  'Intensity': 'Интенсивность',
  'mm/h (= L/m² per hour)': 'мм/ч (= л/м² в час)',
  'Off': 'Нет',
  'No rain': 'Без дождя',
  'Light': 'Слабый',
  'Moderate': 'Умеренный',
  'Heavy': 'Сильный',
  'Downpour': 'Ливень',
  'Extreme': 'Экстремальный',
  '{v} mm/h': '{v} мм/ч',
  'applied: none': 'применено: нет',
  'applied: {mm} mm/h ≈ {q} m³/s over the map': 'применено: {mm} мм/ч ≈ {q} м³/с на всю карту',
  'Every drop becomes runoff: no infiltration, no evaporation. Rain on a roof drips off its eaves onto the ground around the building (no downpipes yet). On flat ground light rain never gets deep enough to see -- the water shows where the terrain gathers it.':
    'Каждая капля становится стоком: впитывания и испарения нет. Дождь с крыши '
    + 'стекает с карниза на землю вокруг здания (водосточных труб пока нет). На '
    + 'ровной земле слабый дождь не набирает видимой глубины: вода видна там, где её '
    + 'собирает рельеф.',
  'Flow visualization': 'Визуализация течения',
  'Show physical tracers': 'Показывать трассеры',
  'Visible tracers': 'Видимых трассеров',
  'Terrain': 'Рельеф',
  'Select / Move': 'Выбор / перемещение',
  'Raise': 'Поднять',
  'Lower': 'Опустить',
  'Brush size': 'Размер кисти',
  'Brush strength': 'Сила кисти',
  'Show cell grid (every 10th line stronger)': 'Показывать сетку ячеек (каждая 10-я линия ярче)',
  'River valley': 'Речная долина',
  'Slope': 'Уклон',
  'Channel width': 'Ширина русла',
  'Incision': 'Врез русла',
  'Meander swing': 'Размах меандров',
  'm (0 = straight)': 'м (0 = прямое)',
  'Meander length': 'Длина меандра',
  'Generate river valley': 'Создать речную долину',

  // ------------------------------------------------------------ properties
  'Position X': 'Положение X',
  'Position Y': 'Положение Y',
  'Position Z': 'Положение Z',
  'Rotation X°': 'Поворот X°',
  'Rotation Y°': 'Поворот Y°',
  'Rotation Z°': 'Поворот Z°',
  'Scale X': 'Масштаб X',
  'Scale Y': 'Масштаб Y',
  'Scale Z': 'Масштаб Z',
  'Mass (kg)': 'Масса (кг)',
  'Friction': 'Трение',
  'Sealed buoyancy (0–1)': 'Герметичность для плавучести (0–1)',
  'Volume (m³)': 'Объём (м³)',
  'Drag coefficient': 'Коэффициент сопротивления',
  'Ground area (m²)': 'Площадь опоры (м²)',
  'Cross area (m²)': 'Площадь сечения (м²)',
  'Foundation height': 'Высота фундамента',
  'Damage resistance': 'Прочность',
  'Diameter (mm)': 'Диаметр (мм)',
  'Line length (m)': 'Длина створа (м)',
  'Pipe depth below ground (m)': 'Глубина трубы под землёй (м)',
  'Discharge Q (m³/s)': 'Расход Q (м³/с)',
  'Vent radius (m)': 'Радиус жерла (м)',
  'Eruption temp (°C)': 'Температура извержения (°C)',
  'Piers': 'Опоры',
  'Pier radius (m)': 'Радиус опоры (м)',
  'Floors': 'Этажи',
  'Height (m)': 'Высота (м)',
  'Continue the pipe': 'Продолжить трубу',

  // -------------------------------------------------------------- readouts
  'Live measurements': 'Текущие измерения',
  'Depth': 'Глубина',
  'Surface': 'Уровень воды',
  'Speed': 'Скорость',
  'Wave arrival': 'Приход волны',
  'dry': 'сухо',
  'not arrived': 'не дошла',
  '0.000 m/s': '0.000 м/с',
  'Gauge depth history': 'История глубины на датчике',
  'Gauging line': 'Мерный створ',
  'Discharge': 'Расход',
  'Water width': 'Ширина потока',
  'Mean depth': 'Средняя глубина',
  'Mean speed': 'Средняя скорость',
  'Froude (section)': 'Число Фруда (створ)',
  'Water level': 'Уровень воды',
  'Discharge history': 'История расхода',
  '(calm)': '(спокойное)',
  '(shooting)': '(бурное)',
  'Not joined to a pipe yet.': 'Пока не соединён с трубой.',
  'Flow <strong>{q} L/s</strong> of {cap} L/s ({load}%)':
    'Расход <strong>{q} л/с</strong> из {cap} л/с ({load}%)',
  'Fall {fall} m over {len} m': 'Перепад {fall} м на {len} м',
  'Dig {id} to {d} m deep for a 0.5% grade': 'Заглубите {id} до {d} м для уклона 0,5%',
  'Runs uphill: a gravity pipe carries nothing.': 'Идёт в гору: самотёчная труба ничего не несёт.',
  'Not joined at both ends.': 'Не соединена с обоих концов.',
  'A pipe already runs on from its start; only one may.':
    'От её начала уже идёт труба; дальше может идти только одна.',
  'Carries nothing: a pipe on the way carries nothing': 'Ничего не несёт: труба на пути ничего не несёт',
  'Carries nothing: the chain ends at a manhole with no pipe out.':
    'Ничего не несёт: цепочка кончается колодцем без выходящей трубы.',
  'Carries nothing: the chain runs back into itself.': 'Ничего не несёт: цепочка замыкается сама на себя.',
  'Close': 'Закрыть',
  'Close the scenario description': 'Закрыть описание сценария',

  // --------------------------------------------------------- units, values
  'm': 'м',
  'mm': 'мм',
  's': 'с',
  'm³/s': 'м³/с',
  '{v} m': '{v} м',
  '{v} m/s': '{v} м/с',
  '{v} m³/s': '{v} м³/с',
  '{v} m³': '{v} м³',
  '{v} s': '{v} с',

  // ----------------------------------------------------------- status bar
  'Sim FPS': 'FPS симуляции',
  'Objects': 'Объекты',
  'Tracers source': 'Трассеры',
  'Wet cells': 'Мокрые ячейки',
  'Q in/out': 'Q вход/выход',
  'Substeps': 'Подшаги',
  'offline': 'нет связи',
  'connecting': 'подключение',
  'connected': 'подключено',
  'disconnected': 'отключено',
  'ready': 'готов',
  'missing': 'нет',
  'yes': 'да',
  'no (CPU mode)': 'нет (режим CPU)',
  'CFL-LIMITED': 'ОГРАНИЧЕНО CFL',
  '{a} in / {r} out m³': 'вошло {a} / вышло {r} м³',
  '{q} m³/s | {a} in / {r} out m³': '{q} м³/с | вошло {a} / вышло {r} м³',
  'events: –': 'события: –',
  'events: {time}s {id} {type} ({cause})': 'события: {time} с {id} {type} ({cause})',

  // ------------------------------------------- event types and causes (backend)
  'SIM_STARTED': 'запуск',
  'SIM_PAUSED': 'пауза',
  'SIM_RESET': 'сброс',
  'WORLD_LOADED': 'мир загружен',
  'WORLD_SAVED': 'мир сохранён',
  'OBJECT_STARTED_MOVING': 'сдвинулся',
  'OBJECT_COLLISION': 'столкновение',
  'OBJECT_DAMAGED': 'повреждён',
  'OBJECT_BROKEN': 'разрушен',
  'OBJECT_FLOATING': 'всплыл',
  'OBJECT_SETTLED': 'остановился',
  'WATER_ENTERED_AREA': 'вода дошла',
  'BRIDGE_DECK_FLOODED': 'настил моста залит',
  'user': 'пользователь',
  'resume': 'продолжение',
  'gauge_depth_threshold_crossed': 'глубина превысила порог',
  'buoyancy_supports_weight': 'выталкивающая сила держит вес',
  'gpu_buoyancy_supports_weight': 'выталкивающая сила держит вес',
  'drag_exceeds_friction': 'напор воды сильнее трения',
  'gpu_drag_exceeds_friction': 'напор воды сильнее трения',
  'force_below_friction': 'сила меньше трения',
  'gpu_force_below_friction': 'сила меньше трения',
  'lava_contact': 'контакт с лавой',
  'water_reached_deck': 'вода дошла до настила',
  "A real parcel: Buitenplaats Oosterwold 178, Almere (NL), 70 x 90 m at 0.5 m from the AHN terrain model, its lowest point set to 0 (-5.07 m NAP). Surfaces and buildings come from the Dutch BGT map; the yellow line is the parcel boundary. Nothing enters from the edges -- rain is the only water: try 30 mm/h for an hour (R30_60) or 90 mm/h for 15 minutes (R90_15). The four gauges sit in the parcel's hollows. Not modelled yet: soaking into the soil, the beds of the pond and the ditch (they read as flat ground), drains":
    "Реальный участок: Buitenplaats Oosterwold 178, Алмере (Нидерланды), 70 × 90 м с шагом 0,5 м по модели рельефа AHN, нижняя точка принята за 0 (−5,07 м NAP). Покрытия и здания — из нидерландской карты BGT; жёлтая линия — граница участка. С краёв вода не поступает — единственный источник дождь: попробуйте 30 мм/ч в течение часа (R30_60) или 90 мм/ч 15 минут (R90_15). Четыре датчика стоят в понижениях участка. Пока не моделируется: впитывание в почву, дно пруда и канавы (они выглядят ровной землёй), дренаж",
};
