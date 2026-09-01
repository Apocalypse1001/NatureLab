import puppeteer from 'puppeteer-core';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const TESTS = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(TESTS, '..');
const URL = 'http://127.0.0.1:8756/';
const PYTHON = process.env.PYTHON || 'python';
const candidates = [process.env.CHROME_PATH,
  'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe'];
const browserPath = candidates.find((value) => value && fs.existsSync(value));
if (!browserPath) throw new Error('Chrome/Edge not found; set CHROME_PATH');

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
async function waitFor(fn, timeout = 30_000) {
  const start = Date.now();
  while (Date.now() - start < timeout) {
    if (await fn()) return;
    await sleep(200);
  }
  throw new Error('timeout');
}
function assert(value, message) { if (!value) throw new Error(message); }
function report(name) { console.log(`PASS  ${name}`); }

const backend = spawn(PYTHON, ['-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', '8756'], {
  cwd: path.join(ROOT, 'backend'), stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true,
});
let browser;
try {
  await waitFor(async () => {
    try { return (await fetch(URL + 'api/status')).ok; } catch { return false; }
  }, 60_000);
  browser = await puppeteer.launch({ executablePath: browserPath, headless: 'new',
    args: ['--window-size=1280,800'] });
  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 800 });
  const errors = [];
  page.on('pageerror', (error) => errors.push(String(error)));
  await page.goto(URL, { waitUntil: 'load' });
  await waitFor(() => page.evaluate(() => window.__NL?.net.status === 'connected'));
  report('frontend + WebSocket');

  const validation = await page.evaluate(() => new Promise((resolve, reject) => {
    const socket = new WebSocket(`ws://${location.host}/ws`);
    const timer = setTimeout(() => reject(new Error('validation response timeout')), 5000);
    socket.onopen = () => socket.send('[]');
    socket.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === 'error') { clearTimeout(timer); socket.close(); resolve(msg.error); }
    };
  }));
  assert(validation.includes('object'), 'non-object JSON was not rejected');
  report('strict WebSocket root validation');

  const engine = await page.evaluate(() => window.__NL.net.engineInfo);
  assert(engine.selftest.points === 100000 && engine.selftest.moved_by === 1, 'Warp selftest failed');
  report(`Warp selftest (${engine.device})`);

  // Dynamic visualization capacity is independent from simulation count.
  const dynamic = await page.evaluate(() => {
    const count = 150001;
    const sm = window.__NL.sceneManager;
    const limit = Number(document.querySelector('#tracer-count').value);
    sm.setParticles(new Float32Array(count * 3), count);
    const attr = sm.points.geometry.attributes.position;
    return { count: sm.points.geometry.drawRange.count, capacity: attr.count, limit };
  });
  // the drawn count follows the UI's display limit, whatever it is set to --
  // asserting a literal 8000 broke the moment the tracer ceiling was raised
  assert(dynamic.count === dynamic.limit && dynamic.capacity >= 150001,
    `particle capacity/display limit are not independent (${JSON.stringify(dynamic)})`);
  report('dynamic particle buffer >120k');

  // Terrain commands are authoritative: patch replaces local float32 values.
  await page.evaluate(() => {
    window.__NL.net.send({ op: 'terrain_brush', x: 0, z: 0, radius: 6, strength: 0.4 });
    window.__NL.net.send({ op: 'terrain_brush', x: 2, z: 1, radius: 4, strength: -0.1 });
  });
  await waitFor(() => page.evaluate(() => typeof window.__terrainChecksum === 'string'));
  const terrain = await page.evaluate(async () => {
    const bytes = new Uint8Array(window.__NL.store.terrain.heights.buffer);
    const digest = await crypto.subtle.digest('SHA-256', bytes);
    const local = [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
    return { local, remote: window.__terrainChecksum };
  });
  assert(terrain.local === terrain.remote, `terrain checksum mismatch ${terrain.local} != ${terrain.remote}`);
  report('terrain frontend/backend checksum');

  // Baseline world and idempotent repeated PLAY.
  // Grid geometry is read from the running app, never hard-coded: this file
  // used to spell out 10201 and 101, which silently became wrong the moment the
  // world was resized (v0.7.0, 100 m -> 200 m).
  const GRID = await page.evaluate(() => window.__NL.store.terrain.width + 1);
  const VERTS = GRID * GRID;
  // Five metres in from the inflow edge, at any map size. Spelled as -45 before
  // v0.7.0, which meant "near the edge" on a 100 m map and "55 m downstream" on
  // a 200 m one -- the wavefront never arrived and the wait simply timed out.
  const NEAR_INLET = await page.evaluate(() => -window.__NL.store.terrain.sizeM / 2 + 5);
  await page.evaluate(() => window.__NL.editor.addObject('HOUSE'));
  await waitFor(() => page.evaluate(() => window.__NL.store.objects.size === 1));
  const baseline = await page.evaluate(() => JSON.stringify([...window.__NL.store.objects.values()]));
  await page.evaluate(() => window.__NL.net.send({ op: 'start' }));
  await waitFor(() => page.evaluate(() => document.querySelector('#sim-status')?.textContent === 'RUNNING'));
  await waitFor(() => page.evaluate((n) => window.__NL.store.waterFrameCount === n, VERTS));
  const water = await page.evaluate(() => {
    const attr = window.__NL.sceneManager.waterMesh.geometry.attributes.position;
    const values = [];
    for (let i = 0; i < attr.count; i++) values.push(attr.getZ(i));
    return { count: attr.count, min: Math.min(...values), max: Math.max(...values),
      time: window.__NL.store.waterFrameTime };
  });
  assert(water.count === VERTS && water.max > water.min && water.time >= 0,
    'dynamic WATER_HEIGHT surface was not rendered');
  report('Warp shallow-water heightfield');
  const maskedWater = await page.evaluate(({ grid, verts }) => {
    const geometry = window.__NL.sceneManager.waterMesh.geometry;
    const positions = geometry.attributes.position;
    const indices = geometry.index.array;
    const terrain = window.__NL.store.terrain.heights;
    const used = geometry.drawRange.count;
    let maxWetColumn = -1;
    // an upper bound proportional to the grid, not a number tuned to one size
    let valid = used > 0 && used < verts * 6;
    for (let vertex = 0; vertex < positions.count; vertex++) {
      if (positions.getZ(vertex) > terrain[vertex] + 1e-4) {
        maxWetColumn = Math.max(maxWetColumn, vertex % grid);
      }
    }
    for (let i = 0; i < used; i++) {
      const vertex = indices[i];
      if (!(positions.getZ(vertex) > terrain[vertex] + 1e-4)) valid = false;
    }
    return { used, valid, maxWetColumn };
  }, { grid: GRID, verts: VERTS });
  assert(maskedWater.valid, 'dry or solid water triangles remain visible');
  assert(maskedWater.maxWetColumn < 10, 'initial water did not start at the map edge');
  report('dry and HOUSE water triangles masked');
  report('initial wave starts at left map edge');
  await waitFor(() => page.evaluate(() =>
    window.__NL.sceneManager.points.geometry.drawRange.count > 0));
  await page.evaluate(() => {
    const slider = document.querySelector('#tracer-count');
    slider.value = '2000'; slider.dispatchEvent(new Event('input'));
    const toggle = document.querySelector('#tracers-visible');
    toggle.checked = false; toggle.dispatchEvent(new Event('change'));
  });
  await sleep(200);
  const hiddenTracers = await page.evaluate(() => ({
    visible: window.__NL.sceneManager.points.visible,
    count: window.__NL.sceneManager.points.geometry.drawRange.count,
  }));
  assert(!hiddenTracers.visible && hiddenTracers.count === 2000,
    'tracer visibility/count controls were overwritten by stream');
  await page.evaluate(() => {
    const slider = document.querySelector('#tracer-count');
    slider.value = '8000'; slider.dispatchEvent(new Event('input'));
    const toggle = document.querySelector('#tracers-visible');
    toggle.checked = true; toggle.dispatchEvent(new Event('change'));
  });
  report('flow tracer display controls');
  const firstTracerFrame = await page.evaluate(() =>
    Array.from(window.__NL.sceneManager.points.geometry.attributes.position.array));
  await sleep(1000);
  const tracersMoved = await page.evaluate((before) => {
    const after = window.__NL.sceneManager.points.geometry.attributes.position.array;
    return before.some((value, index) => Math.abs(value - after[index]) > 1e-4);
  }, firstTracerFrame);
  assert(tracersMoved, 'physical flow tracers did not move');
  report('GPU flow tracers follow fluid velocity');

  await page.evaluate(() => window.__NL.editor.addObject('GAUGE'));
  await waitFor(() => page.evaluate(() =>
    [...window.__NL.store.objects.keys()].some((id) => id.startsWith('Gauge_'))));
  const gaugeId = await page.evaluate(() =>
    [...window.__NL.store.objects.keys()].find((id) => id.startsWith('Gauge_')));
  await page.evaluate(({ id, x }) => {
    const gauge = window.__NL.store.objects.get(id);
    gauge.position = [x, 0, 0];
    window.__NL.sceneManager.setObject(gauge);
    window.__NL.net.send({ op: 'object_update', id, fields: { position: [x, 0, 0] } });
  }, { id: gaugeId, x: NEAR_INLET });
  await waitFor(() => page.evaluate((id) => {
    const gauge = window.__NL.store.gauges.get(id);
    return gauge?.latest?.water_depth_m > 0 && gauge.arrival_time_s != null;
  }, gaugeId));
  const gaugeUi = await page.evaluate((id) => ({
    mesh: Boolean(window.__NL.sceneManager.objectsRoot.getObjectByName(id)),
    history: window.__NL.store.gaugeHistory.get(id)?.length ?? 0,
    depth: document.querySelector('#gauge-depth')?.textContent,
    speed: document.querySelector('#gauge-speed')?.textContent,
    arrival: document.querySelector('#gauge-arrival')?.textContent,
  }), gaugeId);
  assert(gaugeUi.mesh && gaugeUi.history > 0, 'GAUGE mesh/history missing');
  assert(gaugeUi.depth?.includes('m') && gaugeUi.speed?.includes('m/s') &&
    gaugeUi.arrival?.includes('s'), 'GAUGE live readout missing');
  report('GAUGE depth, speed, arrival and history');

  await page.evaluate(() => window.__NL.net.send({ op: 'water_level', level: 1.5 }));
  await waitFor(() => page.evaluate(() => {
    const attr = window.__NL.sceneManager.waterMesh.geometry.attributes.position;
    let max = -Infinity;
    for (let i = 0; i < attr.count; i++) max = Math.max(max, attr.getZ(i));
    return max > 1.4;
  }));
  report('runtime Water level updates fluid field');
  await page.evaluate(() => window.__NL.net.send({ op: 'water_level', level: 0.5 }));
  await sleep(500);
  const t1 = await page.evaluate(() => document.querySelector('#clock').textContent);
  await page.evaluate(() => window.__NL.net.send({ op: 'start' }));
  await sleep(400);
  const t2 = await page.evaluate(() => document.querySelector('#clock').textContent);
  assert(t1 !== 't = 0.0s' && t2 !== 't = 0.0s', 'repeated PLAY reset time');
  report('START idempotent');

  // Required RUNNING sequence: ADD CAR -> MOVE -> REMOVE -> ADD TREE -> PAUSE -> RESUME -> RESET.
  await page.evaluate(() => window.__NL.editor.addObject('CAR'));
  await waitFor(() => page.evaluate(() => [...window.__NL.store.objects.keys()].some((id) => id.startsWith('Car_'))));
  await page.evaluate(() => window.__NL.editor.addObject('CAR'));
  await waitFor(() => page.evaluate(() =>
    [...window.__NL.store.objects.keys()].filter((id) => id.startsWith('Car_')).length === 2));
  const carIds = await page.evaluate(() =>
    [...window.__NL.store.objects.keys()].filter((id) => id.startsWith('Car_')));
  await page.evaluate(({ ids, x }) => ids.forEach((id) => window.__NL.net.send({
    op: 'object_update', id,
    fields: { position: [x, 0, 5], rotation: [0, 0.5, 0] } })),
    { ids: carIds, x: NEAR_INLET });
  await waitFor(() => page.evaluate((ids) => {
    const cars = ids.map((id) => window.__NL.store.objects.get(id));
    if (cars.some((car) => car?.state !== 'FLOATING')) return false;
    return Math.hypot(cars[0].position[0] - cars[1].position[0],
      cars[0].position[2] - cars[1].position[2]) >= 4.5;
  }, carIds));
  report('floating objects resolve collisions');
  await page.evaluate((ids) => ids.forEach((id) => {
    window.__NL.net.send({ op: 'object_remove', id });
    window.__NL.store.removeObject(id); window.__NL.sceneManager.removeObject(id);
  }), carIds);
  await page.evaluate(() => window.__NL.editor.addObject('TREE'));
  await waitFor(() => page.evaluate(() => [...window.__NL.store.objects.keys()].some((id) => id.startsWith('Tree_'))));
  await page.evaluate(() => window.__NL.net.send({ op: 'pause' }));
  await waitFor(() => page.evaluate(() => document.querySelector('#sim-status')?.textContent === 'PAUSED'));
  await page.evaluate(() => window.__NL.net.send({ op: 'start' }));
  await waitFor(() => page.evaluate(() => document.querySelector('#sim-status')?.textContent === 'RUNNING'));
  await page.evaluate(() => window.__NL.net.send({ op: 'reset' }));
  await waitFor(() => page.evaluate(() => document.querySelector('#sim-status')?.textContent === 'IDLE'));
  const resetWorld = await page.evaluate(() => JSON.stringify([...window.__NL.store.objects.values()]));
  assert(resetWorld === baseline, 'RESET or InitialWorldState was corrupted by RUNNING edits');
  assert(await page.evaluate(() => window.__NL.store.gauges.size === 0 &&
    window.__NL.store.gaugeHistory.size === 0), 'RESET did not clear gauge runtime data');
  report('RUNNING edit sequence + RESET integrity');

  // Rotation is exactly xyz on backend round-trip.
  const houseId = await page.evaluate(() => [...window.__NL.store.objects.keys()][0]);
  await page.evaluate((id) => window.__NL.net.send({ op: 'object_update', id,
    fields: { rotation: [0.1, 0.2, 0.3] } }), houseId);
  await page.evaluate(() => window.__NL.net.send({ op: 'request_world' }));
  await sleep(300);
  const rotation = await page.evaluate((id) => window.__NL.store.objects.get(id).rotation, houseId);
  assert(rotation.length === 3 && rotation.every(Number.isFinite), 'rotation is not xyz');
  report('rotation xyz round-trip');

  assert(errors.length === 0, errors.join('\n'));
  report('no browser errors');
  // read the version off the running backend rather than hard-coding it, so
  // this line can never disagree with config.VERSION after a release bump
  const version = await page.evaluate(async () => {
    const r = await fetch('/api/status');
    return (await r.json()).engine?.version ?? 'unknown';
  });
  console.log(`NatureLab ${version} E2E: PASS`);
} finally {
  if (browser) await browser.close();
  backend.kill();
  await sleep(500);
}
