/** HUD: top bar, object palette, properties panel, terrain/water controls,
 *  debug strip. Pure DOM — no framework, easy to extend. */
import { OBJECT_TYPES, type GaugeSample, type GaugeState, type ObjectData,
  type ObjectType, type SimEvent } from '../world/types';

export interface UICallbacks {
  play(): void;
  pause(): void;
  reset(): void;
  save(): void;
  load(): void;
  setSpeed(v: number): void;
  add(type: ObjectType): void;
  select(id: string | null): void;
  removeSelected(): void;
  updateObject(id: string, patch: Partial<ObjectData>): void;
  setWaterLevel(v: number): void;
  setErosion(enabled: boolean): void;
  setOutflow(enabled: boolean): void;
  setTracerVisible(visible: boolean): void;
  setTracerCount(count: number): void;
  setTool(tool: 'select' | 'raise' | 'lower'): void;
  setBrush(radius: number, strength: number): void;
  generateRiver(params: { slope: number; bed_width: number; incision: number }): void;
  getObjects(): ObjectData[];
}

export class UI {
  readonly root: HTMLElement;
  private brandEl!: HTMLElement;
  private clockEl: HTMLSpanElement;
  private statusEl: HTMLSpanElement;
  private fpsEl: HTMLSpanElement;
  private simFpsEl: HTMLSpanElement;
  private objectsEl: HTMLSpanElement;
  private particlesEl: HTMLSpanElement;
  private fluidEl: HTMLSpanElement;
  private wsEl: HTMLSpanElement;
  private gpuEl: HTMLSpanElement;
  private warpEl: HTMLSpanElement;
  private cudaEl: HTMLSpanElement;
  private eventEl: HTMLElement;
  private objectList: HTMLElement;
  private propsHost: HTMLElement;
  private selectedId: string | null = null;

  constructor(root: HTMLElement, private cb: UICallbacks) {
    this.root = root;
    this.root.innerHTML = '';
    this.root.append(this.buildTopBar(), this.buildLeft(), this.buildRight(),
                     this.buildBottom());
    this.clockEl = this.root.querySelector('#clock')!;
    this.statusEl = this.root.querySelector('#sim-status')!;
    this.fpsEl = this.root.querySelector('#dbg-fps')!;
    this.simFpsEl = this.root.querySelector('#dbg-simfps')!;
    this.objectsEl = this.root.querySelector('#dbg-objects')!;
    this.particlesEl = this.root.querySelector('#dbg-particles')!;
    this.fluidEl = this.root.querySelector('#dbg-fluid')!;
    this.wsEl = this.root.querySelector('#dbg-ws')!;
    this.gpuEl = this.root.querySelector('#dbg-gpu')!;
    this.warpEl = this.root.querySelector('#dbg-warp')!;
    this.cudaEl = this.root.querySelector('#dbg-cuda')!;
    this.eventEl = this.root.querySelector('#event-log')!;
    this.objectList = this.root.querySelector('#object-list')!;
    this.propsHost = this.root.querySelector('#properties')!;
    this.showProperties(null);
  }

  // ------------------------------------------------------------------ layout
  private buildTopBar(): HTMLElement {
    const bar = el('div', 'topbar');
    this.brandEl = el('div', 'brand', 'NatureLab');
    bar.append(this.brandEl);
    const controls = el('div', 'controls');
    controls.append(
      btn('PLAY', () => this.cb.play()),
      btn('PAUSE', () => this.cb.pause()),
      btn('RESET', () => this.cb.reset()),
      btn('SAVE', () => this.cb.save()),
      btn('LOAD', () => this.cb.load()),
    );
    const speed = el('select', '') as HTMLSelectElement;
    speed.id = 'speed';
    for (const v of [0.25, 0.5, 1, 2, 4]) {
      const opt = el('option', '', `${v}x`) as HTMLOptionElement;
      opt.value = String(v);
      opt.selected = v === 1;
      speed.append(opt);
    }
    speed.onchange = () => this.cb.setSpeed(parseFloat(speed.value));
    controls.append(speed);
    const clock = el('span', 'clock');
    clock.innerHTML = '<span id="sim-status">IDLE</span> <span id="clock">t = 0.0s</span>';
    bar.append(controls, clock);
    return bar;
  }

  private buildLeft(): HTMLElement {
    const panel = el('div', 'panel left');
    panel.append(el('h2', '', 'OBJECTS'));
    const palette = el('div', 'palette');
    for (const t of OBJECT_TYPES) {
      palette.append(btn(t.charAt(0) + t.slice(1).toLowerCase(),
                         () => this.cb.add(t)));
    }
    panel.append(palette);
    this.objectListPlaceholder();
    panel.append(el('h3', '', 'Scene'));
    const list = el('div', 'object-list');
    list.id = 'object-list';
    panel.append(list);
    return panel;
  }

  private objectListPlaceholder(): void { /* reserved for future grouping UI */ }

  private buildRight(): HTMLElement {
    const panel = el('div', 'panel right');
    panel.append(el('h2', '', 'PROPERTIES'));
    const props = el('div', 'properties');
    props.id = 'properties';
    panel.append(props);

    panel.append(el('h3', '', 'Water'));
    const water = el('div', 'slider-row');
    water.innerHTML = '<label>Edge inflow level <output id="water-out">0.5</output> m</label>';
    const slider = el('input', '') as HTMLInputElement;
    slider.id = 'water-level';
    slider.type = 'range'; slider.min = '-2'; slider.max = '8'; slider.step = '0.1';
    slider.value = '0.5';
    slider.oninput = () => {
      this.root.querySelector<HTMLSpanElement>('#water-out')!.textContent = slider.value;
      this.cb.setWaterLevel(parseFloat(slider.value));
    };
    water.append(slider);
    panel.append(water);

    // RiverLab (v0.6.0): let the river cut and fill its own bed. Off by default
    // -- with it on, the terrain the user built is no longer the terrain they
    // get back, so it has to be a deliberate choice, not a hidden one.
    // v0.8.0: without an open downstream edge the map simply fills up, which
    // is what every version through 0.7.0 did. On by default.
    const outflowRow = el('label', 'slider-row', 'Open downstream edge (water leaves the map)');
    const outflow = el('input', '') as HTMLInputElement;
    outflow.id = 'water-outflow';
    outflow.type = 'checkbox';
    outflow.checked = true;
    outflow.onchange = () => this.cb.setOutflow(outflow.checked);
    outflowRow.append(outflow);
    panel.append(outflowRow);

    const erosionRow = el('label', 'slider-row', 'Erosion (river reshapes the bed)');
    const erosion = el('input', '') as HTMLInputElement;
    erosion.id = 'water-erosion';
    erosion.type = 'checkbox';
    erosion.onchange = () => this.cb.setErosion(erosion.checked);
    erosionRow.append(erosion);
    panel.append(erosionRow);

    panel.append(el('h3', '', 'Flow visualization'));
    const tracerToggle = el('label', 'slider-row', 'Show physical tracers');
    const visible = el('input', '') as HTMLInputElement;
    visible.id = 'tracers-visible'; visible.type = 'checkbox'; visible.checked = true;
    visible.onchange = () => this.cb.setTracerVisible(visible.checked);
    tracerToggle.append(visible);
    panel.append(tracerToggle);
    const tracerCount = el('div', 'slider-row');
    tracerCount.innerHTML =
      '<label>Visible tracers <output id="tracer-count-out">36000</output></label>';
    const count = el('input', '') as HTMLInputElement;
    // Ceiling follows the backend's FLOW_TRACER_COUNT: a slider that maxed out
    // at the old 8 000 would have silently hidden three quarters of the tracers
    // the solver is actually advecting after the v0.7.0 bump to 36 000.
    count.id = 'tracer-count'; count.type = 'range'; count.min = '0'; count.max = '36000';
    count.step = '1000'; count.value = '36000';
    count.oninput = () => {
      this.root.querySelector('#tracer-count-out')!.textContent = count.value;
      this.cb.setTracerCount(parseInt(count.value, 10));
    };
    tracerCount.append(count);
    panel.append(tracerCount);

    panel.append(el('h3', '', 'Terrain'));
    const tools = el('div', 'palette');
    tools.append(
      btn('Select / Move', () => this.cb.setTool('select')),
      btn('Raise', () => this.cb.setTool('raise')),
      btn('Lower', () => this.cb.setTool('lower')),
    );
    panel.append(tools);
    const brush = el('div', 'slider-row');
    brush.innerHTML = '<label>Brush size <output id="brush-size-out">6</output> m</label>';
    const size = el('input', '') as HTMLInputElement;
    size.type = 'range'; size.min = '1'; size.max = '15'; size.step = '0.5'; size.value = '6';
    const strength = el('input', '') as HTMLInputElement;
    strength.type = 'range'; strength.min = '0.05'; strength.max = '1.5';
    strength.step = '0.05'; strength.value = '0.4';
    brush.append(size);
    brush.insertAdjacentHTML('beforeend',
      '<label>Brush strength <output id="brush-str-out">0.4</output> m</label>');
    brush.append(strength);
    size.oninput = () => {
      this.root.querySelector('#brush-size-out')!.textContent = size.value;
      this.cb.setBrush(parseFloat(size.value), parseFloat(strength.value));
    };
    strength.oninput = () => {
      this.root.querySelector('#brush-str-out')!.textContent = strength.value;
      this.cb.setBrush(parseFloat(size.value), parseFloat(strength.value));
    };
    panel.append(brush);

    // v0.12.0: a channel is a precondition for a river, not a decoration -- the
    // flow is driven by the slope of the bed, so on the flat starting map no
    // inflow setting produces anything but a spreading puddle. Drawing 200 m of
    // channel with the brush is hundreds of strokes and leaves lumps that stand
    // waves up, so the bed is generated analytically instead. Replaces the whole
    // terrain, which is why it is a button and not a drag tool.
    panel.append(el('h3', '', 'River valley'));
    const river = el('div', 'slider-row');
    river.innerHTML =
      '<label>Slope <output id="river-slope-out">0.20</output> %</label>';
    const slope = el('input', '') as HTMLInputElement;
    slope.id = 'river-slope';
    slope.type = 'range'; slope.min = '0.05'; slope.max = '1.0';
    slope.step = '0.05'; slope.value = '0.20';
    river.append(slope);
    river.insertAdjacentHTML('beforeend',
      '<label>Channel width <output id="river-width-out">12</output> m</label>');
    const width = el('input', '') as HTMLInputElement;
    width.id = 'river-width';
    width.type = 'range'; width.min = '4'; width.max = '40';
    width.step = '1'; width.value = '12';
    river.append(width);
    river.insertAdjacentHTML('beforeend',
      '<label>Incision <output id="river-depth-out">2.0</output> m</label>');
    const incision = el('input', '') as HTMLInputElement;
    incision.id = 'river-incision';
    incision.type = 'range'; incision.min = '0.5'; incision.max = '6.0';
    incision.step = '0.5'; incision.value = '2.0';
    river.append(incision);
    slope.oninput = () => {
      this.root.querySelector('#river-slope-out')!.textContent = slope.value;
    };
    width.oninput = () => {
      this.root.querySelector('#river-width-out')!.textContent = width.value;
    };
    incision.oninput = () => {
      this.root.querySelector('#river-depth-out')!.textContent = incision.value;
    };
    panel.append(river);
    const generate = btn('Generate river valley', () => this.cb.generateRiver({
      slope: parseFloat(slope.value) / 100,      // the control is in percent
      bed_width: parseFloat(width.value),
      incision: parseFloat(incision.value),
    }));
    generate.id = 'river-generate';
    panel.append(generate);
    return panel;
  }

  private buildBottom(): HTMLElement {
    const bar = el('div', 'bottombar');
    const left = el('div', '');
    left.innerHTML =
      'FPS <span id="dbg-fps">–</span> | ' +
      'Sim FPS <span id="dbg-simfps">–</span> | ' +
      'Objects <span id="dbg-objects">0</span> | ' +
      'Tracers source <span id="dbg-particles">0</span> | ' +
      'Wet cells <span id="dbg-fluid">0</span> | ' +
      'WS <span id="dbg-ws">offline</span> | ' +
      'Warp <span id="dbg-warp">–</span> | ' +
      'CUDA <span id="dbg-cuda">–</span> | ' +
      'GPU <span id="dbg-gpu">–</span>';
    const events = el('div', 'events');
    events.id = 'event-log';
    events.textContent = 'events: –';
    bar.append(left, events);
    return bar;
  }

  // ------------------------------------------------------------------ updates
  setConnection(status: string): void {
    this.wsEl.textContent = status;
    this.wsEl.className = status === 'connected' ? 'ok' : 'bad';
  }

  setEngine(engine: { warp_available: boolean; cuda: boolean; device: string;
                      gpu_name: string; version?: string }): void {
    if (engine.version) this.brandEl.textContent = `NatureLab ${engine.version}`;
    this.warpEl.textContent = engine.warp_available ? 'ready' : 'missing';
    this.cudaEl.textContent = engine.cuda ? 'yes' : 'no (CPU mode)';
    this.gpuEl.textContent = engine.gpu_name;
  }

  setClock(time: number, status: string): void {
    this.clockEl.textContent = `t = ${time.toFixed(1)}s`;
    this.statusEl.textContent = status;
    this.statusEl.className = status === 'RUNNING' ? 'ok' : '';
  }

  setErosionEnabled(enabled: boolean): void {
    const box = this.root.querySelector<HTMLInputElement>('#water-erosion');
    if (box) box.checked = enabled;
  }

  setOutflowEnabled(enabled: boolean): void {
    const box = this.root.querySelector<HTMLInputElement>('#water-outflow');
    if (box) box.checked = enabled;
  }

  setReservoirLevel(level: number): void {
    const slider = this.root.querySelector<HTMLInputElement>('#water-level');
    const output = this.root.querySelector<HTMLOutputElement>('#water-out');
    if (slider) slider.value = String(level);
    if (output) output.textContent = String(level);
  }

  setFps(fps: number): void {
    this.fpsEl.textContent = fps.toFixed(0);
  }

  setSimStats(state: { sim_fps: number; objects: number; particles: number;
                       fluid?: { wet_cells?: number; volume_m3?: number;
                                 erosion?: boolean; outflow_columns?: number;
                                 sources?: number; drains?: number } }): void {
    // Keep the checkbox honest about what the solver is actually doing: the
    // flag is streamed live, so a world loaded with erosion on -- or a state
    // changed by anything other than this checkbox -- still shows correctly.
    if (state.fluid?.erosion !== undefined) this.setErosionEnabled(state.fluid.erosion);
    if (state.fluid?.outflow_columns !== undefined) {
      this.setOutflowEnabled(state.fluid.outflow_columns > 0);
    }
    this.simFpsEl.textContent = state.sim_fps.toFixed(1);
    this.objectsEl.textContent = String(state.objects);
    this.particlesEl.textContent = state.particles.toLocaleString();
    const wet = state.fluid?.wet_cells ?? 0;
    const volume = state.fluid?.volume_m3 ?? 0;
    this.fluidEl.textContent = `${wet.toLocaleString()} / ${volume.toFixed(1)} m³`;
  }

  logEvent(e: SimEvent): void {
    this.eventEl.textContent =
      `events: ${e.time.toFixed(2)}s ${e.object_id ?? ''} ${e.type} (${e.cause})`;
  }

  refreshObjectList(objects: ObjectData[], selected: string | null): void {
    this.objectList.innerHTML = '';
    for (const obj of objects) {
      const row = el('div', `obj-row${obj.id === selected ? ' selected' : ''}`);
      const label = el('span', '', obj.id);
      const del = btn('×', () => {
        this.cb.select(obj.id);
        this.cb.removeSelected();
      });
      row.append(label, del);
      row.onclick = () => this.cb.select(obj.id);
      this.objectList.append(row);
    }
    if (!objects.length) this.objectList.append(el('div', 'hint', 'empty'));
  }

  setSelected(id: string | null): void {
    this.selectedId = id;
    for (const row of this.objectList.children) {
      row.classList.toggle('selected', row.textContent?.startsWith(id ?? '∅') ?? false);
    }
  }

  showProperties(obj: ObjectData | null): void {
    this.selectedId = obj?.id ?? null;
    const host = this.propsHost;
    host.innerHTML = '';
    if (!obj) {
      host.append(el('div', 'hint', 'select an object (or add one from the left)'));
      return;
    }
    const fields: [string, string, number, (v: number) => void][] = [
      ['Position X', 'pos_x', obj.position[0], (v) => this.patchPosition(obj.id, 0, v)],
      ['Position Y', 'pos_y', obj.position[1], (v) => this.patchPosition(obj.id, 1, v)],
      ['Position Z', 'pos_z', obj.position[2], (v) => this.patchPosition(obj.id, 2, v)],
      ['Rotation X°', 'rot_x', deg(obj.rotation[0]), (v) => this.patchRotation(obj.id, 0, rad(v))],
      ['Rotation Y°', 'rot_y', deg(obj.rotation[1]), (v) => this.patchRotation(obj.id, 1, rad(v))],
      ['Rotation Z°', 'rot_z', deg(obj.rotation[2]), (v) => this.patchRotation(obj.id, 2, rad(v))],
      ['Scale X', 'scl_x', obj.scale[0], (v) => this.patchScale(obj.id, 0, v)],
      ['Scale Y', 'scl_y', obj.scale[1], (v) => this.patchScale(obj.id, 1, v)],
      ['Scale Z', 'scl_z', obj.scale[2], (v) => this.patchScale(obj.id, 2, v)],
    ];
    if (obj.type !== 'GAUGE') fields.push(
      ['Mass (kg)', 'mass', obj.mass, (v) => this.cb.updateObject(obj.id, { mass: v })],
      ['Friction', 'friction', obj.friction, (v) => this.cb.updateObject(obj.id, { friction: v })],
      ['Sealed buoyancy (0–1)', 'buoyancy', obj.buoyancy,
       (v) => this.cb.updateObject(obj.id, { buoyancy: v })],
      ['Volume (m³)', 'volume', obj.volume_m3, (v) => this.cb.updateObject(obj.id, { volume_m3: v })],
      ['Drag coefficient', 'drag', obj.drag_coefficient,
       (v) => this.cb.updateObject(obj.id, { drag_coefficient: v })],
      ['Ground area (m²)', 'ground_area', obj.ground_contact_area,
       (v) => this.cb.updateObject(obj.id, { ground_contact_area: v })],
      ['Cross area (m²)', 'cross_area', obj.cross_sectional_area,
       (v) => this.cb.updateObject(obj.id, { cross_sectional_area: v })],
      ['Foundation height', 'foundation', obj.metadata.foundation_height ?? 0,
       (v) => this.cb.updateObject(obj.id, { metadata: { ...obj.metadata, foundation_height: v } })],
      ['Damage resistance', 'resistance', obj.metadata.damage_resistance ?? 0,
       (v) => this.cb.updateObject(obj.id, { metadata: { ...obj.metadata, damage_resistance: v } })],
    );
    for (const [label, key, value, apply] of fields) {
      const row = el('div', 'prop-row');
      row.innerHTML = `<label>${label}</label>`;
      const input = el('input', '') as HTMLInputElement;
      input.type = 'number'; input.step = '0.1'; input.value = String(round2(value));
      input.dataset.key = key;
      input.onchange = () => {
        const v = parseFloat(input.value);
        if (Number.isFinite(v)) apply(v);
      };
      row.append(input);
      host.append(row);
    }
    if (obj.type === 'GAUGE') {
      const readout = el('div', 'gauge-readout');
      readout.id = 'gauge-readout';
      readout.innerHTML =
        '<h3>Live measurements</h3>' +
        '<div>Depth <strong id="gauge-depth">dry</strong></div>' +
        '<div>Surface <strong id="gauge-surface">dry</strong></div>' +
        '<div>Speed <strong id="gauge-speed">0.000 m/s</strong></div>' +
        '<div>Wave arrival <strong id="gauge-arrival">not arrived</strong></div>' +
        '<svg id="gauge-chart" viewBox="0 0 240 70" role="img" aria-label="Gauge depth history">' +
        '<polyline points="" /></svg>';
      host.append(readout);
    }
  }

  updateGaugeReadout(state: GaugeState | undefined, history: GaugeSample[]): void {
    if (!state || state.id !== this.selectedId || !this.root.querySelector('#gauge-readout')) return;
    const latest = state.latest;
    this.root.querySelector('#gauge-depth')!.textContent = latest
      ? `${latest.water_depth_m.toFixed(3)} m` : 'dry';
    this.root.querySelector('#gauge-surface')!.textContent = latest?.surface_elevation_m != null
      ? `${latest.surface_elevation_m.toFixed(3)} m` : 'dry';
    this.root.querySelector('#gauge-speed')!.textContent = latest
      ? `${latest.speed_m_s.toFixed(3)} m/s` : '0.000 m/s';
    this.root.querySelector('#gauge-arrival')!.textContent = state.arrival_time_s != null
      ? `${state.arrival_time_s.toFixed(2)} s` : 'not arrived';
    const points = this.root.querySelector<SVGPolylineElement>('#gauge-chart polyline');
    if (!points || history.length < 2) return;
    const visible = history.slice(-240);
    const maxDepth = Math.max(0.001, ...visible.map((sample) => sample.water_depth_m));
    points.setAttribute('points', visible.map((sample, i) => {
      const x = i * 240 / Math.max(1, visible.length - 1);
      const y = 68 - sample.water_depth_m / maxDepth * 64;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(' '));
  }

  updatePropertyInputs(obj: ObjectData): void {
    if (obj.id !== this.selectedId) return;
    const values: Record<string, number> = {
      pos_x: obj.position[0], pos_y: obj.position[1], pos_z: obj.position[2],
      rot_x: deg(obj.rotation[0]), rot_y: deg(obj.rotation[1]), rot_z: deg(obj.rotation[2]),
      scl_x: obj.scale[0], scl_y: obj.scale[1], scl_z: obj.scale[2],
      mass: obj.mass, friction: obj.friction, buoyancy: obj.buoyancy,
      volume: obj.volume_m3, drag: obj.drag_coefficient,
      ground_area: obj.ground_contact_area, cross_area: obj.cross_sectional_area,
    };
    for (const input of this.propsHost.querySelectorAll<HTMLInputElement>('input')) {
      const key = input.dataset.key;
      if (key && key in values && document.activeElement !== input) {
        input.value = String(round2(values[key]));
      }
    }
  }

  private patchPosition(id: string, axis: number, v: number): void {
    const obj = this.cb.getObjects().find((o) => o.id === id);
    if (!obj) return;
    const p = [...obj.position]; p[axis] = v;
    this.cb.updateObject(id, { position: p });
  }

  private patchRotation(id: string, axis: number, v: number): void {
    const obj = this.cb.getObjects().find((o) => o.id === id);
    if (!obj) return;
    const r = [...obj.rotation]; r[axis] = v;
    this.cb.updateObject(id, { rotation: r });
  }

  private patchScale(id: string, axis: number, v: number): void {
    const obj = this.cb.getObjects().find((o) => o.id === id);
    if (!obj) return;
    const s = [...obj.scale]; s[axis] = v;
    this.cb.updateObject(id, { scale: s });
  }
}

// ---------------------------------------------------------------------- utils
function el(tag: string, cls: string, text = ''): HTMLElement {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text) node.textContent = text;
  return node;
}

function btn(label: string, onclick: () => void): HTMLButtonElement {
  const b = el('button', '', label) as HTMLButtonElement;
  b.onclick = onclick;
  return b;
}

function deg(rad: number): number { return rad * 180 / Math.PI; }
function rad(deg: number): number { return deg * Math.PI / 180; }
function round2(v: number): number { return Math.round(v * 100) / 100; }
