import { SceneManager } from './scene/SceneManager';
import { WorldStore } from './world/WorldStore';
import { EditorController } from './editor/EditorController';
import { BackendClient } from './net/BackendClient';
import { SCENARIOS, UI } from './ui/UI';
import type { ObjectData, OutletKind, WorldData } from './world/types';
import type { EdgeWaterMode } from './scene/EdgeSkirt';
import './style.css';

// Backend runs on the same host/port that serves this page.
const wsUrl = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`;

const store = new WorldStore();
let currentSimStatus = 'IDLE';
// v0.18.0: the loaded world's gravity, for the Froude colouring
let currentGravity = 9.81;
// The tsunami wavemaker closes the outlet during a pulse, but the sea past
// that edge is there all the same; the stream alone cannot say so.
let tsunamiWorld = false;

/**
 * What the drawn-only skirt past the map edge should do with water (v0.16.0).
 * East: only across an edge the solver treats as open; a "river" outlet runs
 * on at its depth, anything else (sea, pool) holds its level. West: a river
 * inlet arrives from upstream at its depth, a held edge inflow is a level.
 * North/south are walls in the solver; only a coast's sea runs on past them.
 */
function edgeWaterModes(eastOpen: boolean, kind: OutletKind | undefined,
                        inlet: boolean, edgeInflow: boolean, lava: boolean) {
  if (lava) return { east: null, west: null, sides: null };
  const east = eastOpen || tsunamiWorld ? (kind === 'river' ? 'depth' : 'level') : null;
  const west = inlet ? 'depth' : edgeInflow ? 'level' : null;
  const sides = tsunamiWorld ? 'level' : null;
  return { east, west, sides } as
    { east: EdgeWaterMode; west: EdgeWaterMode; sides: EdgeWaterMode };
}

const canvas = document.createElement('canvas');
canvas.id = 'viewport';
document.getElementById('app')!.append(canvas);

const uiHost = document.createElement('div');
uiHost.id = 'ui-chrome';
document.getElementById('app')!.append(uiHost);

const sceneManager = new SceneManager(canvas, store.terrain);

// ---------------------------------------------------------------------- net
const net = new BackendClient(wsUrl, {
  onStatus: (s) => ui.setConnection(s),
  onHello: (engine) => {
    ui.setEngine(engine);
    console.log('[NatureLab] backend engine:', engine);
  },
  onWorld: (world, simStatus, op, name) => applyWorld(world, simStatus, op, name),
  onSimState: (state) => {
    currentSimStatus = state.status;
    editor.terrainEditingEnabled = state.status !== 'RUNNING';
    ui.setClock(state.time, state.status);
    ui.setSimStats(state);
    // the streaks draw what the solver applies, never the slider position
    sceneManager.setRain(state.fluid?.rain_mm_h ?? 0, state.status === 'RUNNING');
    const fluid = state.fluid;
    if (fluid && fluid.outflow_columns !== undefined) {
      const modes = edgeWaterModes(fluid.outflow_columns > 0, fluid.outlet_kind,
        fluid.inlet_enabled ?? false, fluid.edge_inflow ?? false, fluid.lava_enabled ?? false);
      sceneManager.setEdgeWater(modes.east, modes.west, modes.sides);
    }
    store.applyGaugeStates(state.gauges ?? [], state.gauge_history_capacity ?? 600);
    ui.updateSections(state.sections ?? []);
    sceneManager.setSewerState(state.sewer ?? [], state.status === 'RUNNING');
    ui.updateSewerReadout(state.sewer ?? []);
    for (const moved of state.moved_objects) {
      store.updateObject(moved.id,
        { position: moved.position, state: moved.state, damage: moved.damage });
      const obj = store.objects.get(moved.id);
      if (obj) {
        sceneManager.setObject(obj);
        ui.updatePropertyInputs(obj);
      }
    }
    for (const e of state.events) ui.logEvent(e);
  },
  onSaved: (name, path) => console.log('[NatureLab] world saved:', name, path),
  onError: (msg) => console.error('[NatureLab] backend error:', msg),
  onObjectAdded: (obj) => {
    store.addObject(obj);
    sceneManager.setObject(obj);
    store.select(obj.id);
  },
  // v0.17.0 storm sewer: a pipe edit returns every object it created or moved
  onPipe: (objects, sewer) => {
    let pipeId: string | null = null;
    for (const obj of objects) {
      if (store.objects.has(obj.id)) store.updateObject(obj.id, obj);
      else store.addObject(obj);
      sceneManager.setObject(obj);
      if (obj.type === 'PIPE') pipeId = obj.id;
    }
    sceneManager.setSewerState(sewer, currentSimStatus === 'RUNNING');
    ui.updateSewerReadout(sewer);
    if (pipeId) store.select(pipeId);
  },
  onTerrainPatch: (heights, checksum) => {
    store.terrain.loadHeights(heights);
    sceneManager.rebuildTerrain(store.terrain);
    sceneManager.redrawPipes(store.objects.values());
    (globalThis as Record<string, unknown>).__terrainChecksum = checksum;
  },
});

net.particleHandler = (positions, count) => sceneManager.setParticles(positions, count);
// v0.10.0: the real velocity field, used as a physics-derived flow map -- see
// SceneManager.buildWaterMaterial for why an off-the-shelf water shader was the
// wrong shape for this project.
net.velocityFieldHandler = (velocities, count) => {
  sceneManager.setVelocityField(velocities, count);
};
net.waterHeightHandler = (heights, count, simTime) => {
  if (sceneManager.setWaterHeights(heights, count)) {
    store.waterFrameCount = count;
    store.waterFrameTime = simTime;
  }
};
net.lavaTemperatureHandler = (temperatures, count) => {
  sceneManager.setLavaTemperature(temperatures, count);
};

// ---------------------------------------------------------------------- ui
const ui = new UI(uiHost, {
  play: () => net.send({ op: 'start' }),
  pause: () => net.send({ op: 'pause' }),
  reset: () => net.send({ op: 'reset' }),
  save: () => net.send({ op: 'save', name: 'default' }),
  load: () => net.send({ op: 'load', name: 'default' }),
  // A scenario is just a saved world under data/, so it rides the existing
  // `load` op and comes back as an ordinary `world` message -- the same path
  // the LOAD button uses. See tools/make_scenarios.py.
  loadScenario: (name) => net.send({ op: 'load', name }),
  setSpeed: (v) => net.send({ op: 'set_speed', value: v }),
  add: (type) => editor.addObject(type),
  select: (id) => store.select(id),
  removeSelected: () => editor.deleteSelected(),
  updateObject: (id, patch) => {
    store.updateObject(id, patch);
    const obj = store.objects.get(id);
    if (obj) sceneManager.setObject(obj);
    net.send({ op: 'object_update', id, fields: patch });
  },
  setWaterLevel: (v) => {
    store.setWaterLevel(v);
    if (currentSimStatus === 'IDLE') sceneManager.setWater(v, store.waterVisible);
    net.send({ op: 'water_level', level: v });
  },
  setErosion: (enabled) => net.send({ op: 'water_erosion', enabled }),
  setOutflow: (enabled) => net.send({ op: 'water_outflow', enabled }),
  setTracerVisible: (visible) => sceneManager.setTracerVisible(visible),
  setGridVisible: (visible) => sceneManager.setGridVisible(visible),
  setTracerCount: (count) => sceneManager.setTracerDisplayLimit(count),
  setTool: (tool) => editor.setTool(tool),
  setBrush: (radius, strength) => {
    editor.brushRadius = radius;
    editor.brushStrength = strength;
  },
  // The reply is an ordinary terrain_patch, so the generated valley reaches the
  // scene through the same path a brush stroke does -- no second sync to keep
  // right. Rejected by the backend while RUNNING, like every terrain edit.
  generateRiver: (params) => net.send({ op: 'terrain_river', params }),
  setRiverInlet: (fields) => net.send({ op: 'river_inlet', fields }),
  setRiverOutlet: (fields) => net.send({ op: 'river_outlet', fields }),
  // RainLab-1: read live by the backend every tick, like the inlet, so the
  // intensity can be changed while RUNNING
  setRain: (fields) => net.send({ op: 'rain', fields }),
  setEdgeInflow: (enabled) => net.send({ op: 'edge_inflow', enabled }),
  setFroudeView: (on) => sceneManager.setFroudeView(on, currentGravity),
  settleRiver: (clear) => net.send({ op: 'water_settle', fields: { clear } }),
  setPipeDiameter: (diameterM) => { editor.pipeDiameter = diameterM; },
  extendPipe: (id) => editor.extendPipe(id),
  updatePipe: (id, diameterM) => net.send({ op: 'pipe_update', pipe: { id, diameter_m: diameterM } }),
  getObjects: () => [...store.objects.values()],
});

// ---------------------------------------------------------------------- editor
const editor = new EditorController(sceneManager, store, net);
editor.onToolChange = (tool) => ui.setActiveTool(tool);

// ---------------------------------------------------------------------- world sync
function applyWorld(world: WorldData, simStatus: string, op?: string,
                    name?: string): void {
  currentSimStatus = simStatus;
  currentGravity = world.environment?.gravity ?? 9.81;
  editor.terrainEditingEnabled = simStatus !== 'RUNNING';
  store.replaceWorld(world);
  sceneManager.rebuildTerrain(store.terrain);
  tsunamiWorld = world.water.tsunami_enabled ?? false;
  const modes = edgeWaterModes(world.water.outflow_enabled ?? true, world.water.outlet_kind,
    world.water.inlet_enabled ?? false, world.water.edge_inflow_enabled ?? true, false);
  sceneManager.setEdgeWater(modes.east, modes.west, modes.sides);
  sceneManager.setWater(store.waterLevel, store.waterVisible);
  sceneManager.clearTracers();
  ui.setReservoirLevel(store.waterLevel);
  ui.setErosionEnabled(store.erosionEnabled);
  ui.setOutflowEnabled(store.outflowEnabled);
  // A scenario carries its own river boundary; the controls have to show it.
  ui.setRiverControls(world.water);
  ui.setRainControls(world.water);
  ui.setSettled(!!world.water.initial_flow);
  sceneManager.clearObjects();
  for (const obj of world.objects) sceneManager.setObject(obj);
  ui.refreshObjectList([...store.objects.values()], null);
  ui.showProperties(null);
  ui.setClock(0, simStatus);
  // Frame the camera on a world that has just ARRIVED -- a scenario, LOAD, the
  // first connection -- but never on RESET or Settle river, which hand back
  // the same world while the user may be looking somewhere on purpose.
  if (op === undefined || op === 'load' || op === 'request_world') {
    ui.showScenario(op === 'load' ? name ?? null : null);
    const frame = op === 'load' ? SCENARIOS.find((s) => s.name === name)?.frame : undefined;
    if (frame === 'map') sceneManager.frameBox(null, unobstructedWidth());
    else if (frame) sceneManager.frameBox(frame, unobstructedWidth());
    else sceneManager.frameObjects(world.objects, unobstructedWidth());
  }
}

/** Share of the viewport's width the side panels leave open, for framing. */
function unobstructedWidth(): number {
  const width = window.innerWidth;
  let covered = 0;
  for (const panel of document.querySelectorAll('.panel')) {
    covered += panel.getBoundingClientRect().width + 8;   // + the panel's margin
  }
  return Math.max(0.35, (width - covered) / width);
}

store.on('objects-changed', () => {
  ui.refreshObjectList([...store.objects.values()], store.selectedId);
});
/**
 * v0.17.0: a pipe's end follows its inlet or outfall when either is dragged.
 * The solver already routes by the fixtures themselves (backend/app/sewer.py);
 * this keeps the drawn route -- and the saved one -- attached to them.
 */
function syncPipeEnds(end: ObjectData): void {
  for (const pipe of store.objects.values()) {
    if (pipe.type !== 'PIPE') continue;
    const points = pipe.metadata.points as number[][] | undefined;
    if (!points || points.length < 2) continue;
    const k = pipe.metadata.from_id === end.id ? 0
      : pipe.metadata.to_id === end.id ? points.length - 1 : -1;
    if (k < 0) continue;
    const p = points[k];
    if (Math.hypot(p[0] - end.position[0], p[2] - end.position[2]) < 1e-3) continue;
    const next = points.map((q) => [...q]);
    next[k] = [...end.position];
    const patch: Partial<ObjectData> = { metadata: { ...pipe.metadata, points: next } };
    if (k === 0) patch.position = [...end.position];
    store.updateObject(pipe.id, patch);
    net.send({ op: 'object_update', id: pipe.id, fields: patch });
  }
}

store.on('object-updated', (id) => {
  const obj = store.objects.get(id as string);
  if (obj && (obj.type === 'STORM_INLET' || obj.type === 'OUTFALL' || obj.type === 'MANHOLE')) {
    syncPipeEnds(obj);
  }
  if (obj) {
    const rebuilt = sceneManager.setObject(obj);
    // A rebuild (BUILDING's floors changing) tore down the THREE.Object3D
    // TransformControls was attached to by identity -- it does not silently
    // follow the id to the new one, it warns every frame that its target has
    // left the scene graph. Re-running selection re-attaches it. A no-op for
    // every non-rebuilding edit (mass, friction, position...), since rebuilt
    // stays false for those.
    if (rebuilt && store.selectedId === id) store.select(id as string);
    ui.updatePropertyInputs(obj);
  }
});
store.on('selection-changed', (id) => {
  const selected = id ? store.objects.get(id as string) ?? null : null;
  ui.refreshObjectList([...store.objects.values()], store.selectedId);
  ui.showProperties(selected);
  if (selected?.type === 'GAUGE') {
    ui.updateGaugeReadout(store.gauges.get(selected.id),
                          store.gaugeHistory.get(selected.id) ?? []);
  }
});
store.on('gauge-updated', (id) => {
  const gaugeId = id as string;
  ui.updateGaugeReadout(store.gauges.get(gaugeId), store.gaugeHistory.get(gaugeId) ?? []);
});

// ---------------------------------------------------------------------- loop
let frames = 0;
let fpsTimer = performance.now();

function loop(): void {
  sceneManager.render();
  frames++;
  const now = performance.now();
  if (now - fpsTimer >= 1000) {
    ui.setFps((frames * 1000) / (now - fpsTimer));
    frames = 0;
    fpsTimer = now;
  }
  requestAnimationFrame(loop);
}

window.addEventListener('resize', () => sceneManager.resize());
window.addEventListener('load', () => sceneManager.resize());

// Debug/test API (used by TEST_REPORT automation and manual inspection).
(globalThis as Record<string, unknown>).__NL = {
  store, net, sceneManager, editor, ui,
};

net.connect();
loop();
