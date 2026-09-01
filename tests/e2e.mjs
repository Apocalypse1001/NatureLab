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
    window.__NL.sceneManager.setParticles(new Float32Array(count * 3), count);
    const attr = window.__NL.sceneManager.points.geometry.attributes.position;
    return { count: window.__NL.sceneManager.points.geometry.drawRange.count, capacity: attr.count };
  });
  assert(dynamic.count === 150001 && dynamic.capacity >= 150001, 'particle buffer did not grow');
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
  await page.evaluate(() => window.__NL.editor.addObject('HOUSE'));
  await waitFor(() => page.evaluate(() => window.__NL.store.objects.size === 1));
  const baseline = await page.evaluate(() => JSON.stringify([...window.__NL.store.objects.values()]));
  await page.evaluate(() => window.__NL.net.send({ op: 'start' }));
  await waitFor(() => page.evaluate(() => document.querySelector('#sim-status')?.textContent === 'RUNNING'));
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
  const carId = await page.evaluate(() => [...window.__NL.store.objects.keys()].find((id) => id.startsWith('Car_')));
  await page.evaluate((id) => window.__NL.net.send({ op: 'object_update', id,
    fields: { position: [7, 0, 5], rotation: [0, 0.5, 0] } }), carId);
  await page.evaluate((id) => { window.__NL.net.send({ op: 'object_remove', id });
    window.__NL.store.removeObject(id); window.__NL.sceneManager.removeObject(id); }, carId);
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
  console.log('Foundation 0.2 E2E: PASS');
} finally {
  if (browser) await browser.close();
  backend.kill();
  await sleep(500);
}
