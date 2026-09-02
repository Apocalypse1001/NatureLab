import { SceneManager } from './scene/SceneManager';
import { WorldStore } from './world/WorldStore';
import { EditorController } from './editor/EditorController';
import { BackendClient } from './net/BackendClient';
import { UI } from './ui/UI';
import type { WorldData } from './world/types';
import './style.css';

// Backend runs on the same host/port that serves this page.
const wsUrl = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`;

const store = new WorldStore();
let currentSimStatus = 'IDLE';

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
  onWorld: (world, simStatus) => applyWorld(world, simStatus),
  onSimState: (state) => {
    currentSimStatus = state.status;
    editor.terrainEditingEnabled = state.status !== 'RUNNING';
    ui.setClock(state.time, state.status);
    ui.setSimStats(state);
    store.applyGaugeStates(state.gauges ?? [], state.gauge_history_capacity ?? 600);
    for (const moved of state.moved_objects) {
      store.updateObject(moved.id, { position: moved.position, state: moved.state });
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
  onTerrainPatch: (heights, checksum) => {
    store.terrain.loadHeights(heights);
    sceneManager.rebuildTerrain(store.terrain);
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
  getObjects: () => [...store.objects.values()],
});

// ---------------------------------------------------------------------- editor
const editor = new EditorController(sceneManager, store, net);

// ---------------------------------------------------------------------- world sync
function applyWorld(world: WorldData, simStatus: string): void {
  currentSimStatus = simStatus;
  editor.terrainEditingEnabled = simStatus !== 'RUNNING';
  store.replaceWorld(world);
  sceneManager.rebuildTerrain(store.terrain);
  sceneManager.setWater(store.waterLevel, store.waterVisible);
  sceneManager.clearTracers();
  ui.setReservoirLevel(store.waterLevel);
  ui.setErosionEnabled(store.erosionEnabled);
  ui.setOutflowEnabled(store.outflowEnabled);
  // A scenario carries its own river boundary; the controls have to show it.
  ui.setRiverControls(world.water);
  sceneManager.clearObjects();
  for (const obj of world.objects) sceneManager.setObject(obj);
  ui.refreshObjectList([...store.objects.values()], null);
  ui.showProperties(null);
  ui.setClock(0, simStatus);
}

store.on('objects-changed', () => {
  ui.refreshObjectList([...store.objects.values()], store.selectedId);
});
store.on('object-updated', (id) => {
  const obj = store.objects.get(id as string);
  if (obj) {
    sceneManager.setObject(obj);
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
