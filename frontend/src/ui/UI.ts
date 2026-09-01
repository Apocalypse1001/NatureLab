/** HUD: top bar, object palette, properties panel, terrain/water controls,
 *  debug strip. Pure DOM — no framework, easy to extend. */
import { OBJECT_TYPES, type ObjectData, type ObjectType, type SimEvent } from '../world/types';

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
  setTool(tool: 'select' | 'raise' | 'lower'): void;
  setBrush(radius: number, strength: number): void;
  getObjects(): ObjectData[];
}

export class UI {
  readonly root: HTMLElement;
  private clockEl: HTMLSpanElement;
  private statusEl: HTMLSpanElement;
  private fpsEl: HTMLSpanElement;
  private simFpsEl: HTMLSpanElement;
  private objectsEl: HTMLSpanElement;
  private particlesEl: HTMLSpanElement;
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
    bar.append(el('div', 'brand', 'NatureLab'));
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
    water.innerHTML = '<label>Water level <output id="water-out">0.5</output> m</label>';
    const slider = el('input', '') as HTMLInputElement;
    slider.type = 'range'; slider.min = '-2'; slider.max = '8'; slider.step = '0.1';
    slider.value = '0.5';
    slider.oninput = () => {
      this.root.querySelector<HTMLSpanElement>('#water-out')!.textContent = slider.value;
      this.cb.setWaterLevel(parseFloat(slider.value));
    };
    water.append(slider);
    panel.append(water);

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
    return panel;
  }

  private buildBottom(): HTMLElement {
    const bar = el('div', 'bottombar');
    const left = el('div', '');
    left.innerHTML =
      'FPS <span id="dbg-fps">–</span> | ' +
      'Sim FPS <span id="dbg-simfps">–</span> | ' +
      'Objects <span id="dbg-objects">0</span> | ' +
      'Particles <span id="dbg-particles">0</span> | ' +
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

  setEngine(engine: { warp_available: boolean; cuda: boolean; device: string; gpu_name: string }): void {
    this.warpEl.textContent = engine.warp_available ? 'ready' : 'missing';
    this.cudaEl.textContent = engine.cuda ? 'yes' : 'no (CPU mode)';
    this.gpuEl.textContent = engine.gpu_name;
  }

  setClock(time: number, status: string): void {
    this.clockEl.textContent = `t = ${time.toFixed(1)}s`;
    this.statusEl.textContent = status;
    this.statusEl.className = status === 'RUNNING' ? 'ok' : '';
  }

  setFps(fps: number): void {
    this.fpsEl.textContent = fps.toFixed(0);
  }

  setSimStats(state: { sim_fps: number; objects: number; particles: number }): void {
    this.simFpsEl.textContent = state.sim_fps.toFixed(1);
    this.objectsEl.textContent = String(state.objects);
    this.particlesEl.textContent = state.particles.toLocaleString();
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
      ['Mass (kg)', 'mass', obj.mass, (v) => this.cb.updateObject(obj.id, { mass: v })],
      ['Friction', 'friction', obj.friction, (v) => this.cb.updateObject(obj.id, { friction: v })],
      ['Buoyancy', 'buoyancy', obj.buoyancy, (v) => this.cb.updateObject(obj.id, { buoyancy: v })],
      ['Foundation height', 'foundation', obj.metadata.foundation_height ?? 0,
       (v) => this.cb.updateObject(obj.id, { metadata: { ...obj.metadata, foundation_height: v } })],
      ['Damage resistance', 'resistance', obj.metadata.damage_resistance ?? 0,
       (v) => this.cb.updateObject(obj.id, { metadata: { ...obj.metadata, damage_resistance: v } })],
    ];
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
  }

  updatePropertyInputs(obj: ObjectData): void {
    if (obj.id !== this.selectedId) return;
    const values: Record<string, number> = {
      pos_x: obj.position[0], pos_y: obj.position[1], pos_z: obj.position[2],
      rot_x: deg(obj.rotation[0]), rot_y: deg(obj.rotation[1]), rot_z: deg(obj.rotation[2]),
      scl_x: obj.scale[0], scl_y: obj.scale[1], scl_z: obj.scale[2],
      mass: obj.mass, friction: obj.friction, buoyancy: obj.buoyancy,
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
