/**
 * NatureLab agent driver.
 *
 * Owns the whole app: starts the uvicorn backend, opens the real frontend in
 * headless Chrome, and then takes line commands on stdin so an agent can poke
 * the running simulation and screenshot it.
 *
 *   node .claude/skills/run-naturelab/driver.mjs
 *
 * Commands (one per line, on stdin):
 *   status              sim status / clock / object + particle counts
 *   add <TYPE>          HOUSE | CAR | TREE | BOX | DEBRIS | ROCK
 *   start | pause | reset
 *   flow on|off         water.flow_enabled (the continuous river current)
 *   temp <c>            baseline water temperature slider
 *   water <m>           water level slider
 *   send <json>         raw WebSocket op, e.g. send {"op":"start"}
 *   eval <js>           evaluate in the page, JSON-printed. `__NL` is in scope.
 *   depth               water depth field stats (min/mean/max, wet cells)
 *   wait <ms>
 *   ss [name]           screenshot -> shots/<name>.png  (default: shot)
 *   quit
 *
 * Every command prints exactly one line starting with `ok ` or `err `, so a
 * caller can drive this synchronously by reading a line per command sent.
 */
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import readline from 'node:readline';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const SKILL_DIR = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(SKILL_DIR, '..', '..', '..');
const SHOTS = path.join(SKILL_DIR, 'shots');

// puppeteer-core is a dependency of tests/, not of the skill -- resolve it
// from there rather than duplicating a node_modules tree for the driver.
const require = createRequire(path.join(ROOT, 'tests', 'package.json'));
const puppeteer = require('puppeteer-core');

// Deliberately NOT 8756. That is the port start.bat / launcher.pyw use, and a
// backend left running by a previous session keeps holding it: uvicorn then
// fails to bind, the driver connects to the *stale* process anyway, and you
// silently drive someone else's world (10 objects, t=88s, none of your edits).
// Own port + the preflight check below make that failure loud instead.
const PORT = process.env.NL_PORT || '8770';
const URL = `http://127.0.0.1:${PORT}/`;
const PYTHON = process.env.PYTHON || 'python';

const CHROME_CANDIDATES = [
  process.env.CHROME_PATH,
  'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
  'C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe',
  '/usr/bin/google-chrome',
  '/usr/bin/chromium',
  '/usr/bin/chromium-browser',
];

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function waitFor(fn, timeout, what) {
  const start = Date.now();
  while (Date.now() - start < timeout) {
    try { if (await fn()) return; } catch { /* keep waiting */ }
    await sleep(200);
  }
  throw new Error(`timeout waiting for ${what}`);
}

// ---------------------------------------------------------------- boot
fs.mkdirSync(SHOTS, { recursive: true });

const browserPath = CHROME_CANDIDATES.find((p) => p && fs.existsSync(p));
if (!browserPath) {
  console.error('err no Chrome/Edge found; set CHROME_PATH');
  process.exit(1);
}

if (!fs.existsSync(path.join(ROOT, 'frontend', 'dist', 'index.html'))) {
  console.error('err frontend/dist missing -- run `npm ci && npm run build` in frontend/ first');
  process.exit(1);
}

// Preflight: refuse to start on an occupied port instead of silently
// attaching to whatever is already there.
try {
  const r = await fetch(URL + 'api/status', { signal: AbortSignal.timeout(1500) });
  if (r.ok) {
    console.error(`err port ${PORT} is already serving a NatureLab backend. `
      + `Stop it, or run with NL_PORT=<other>. Never assume it is yours: `
      + `a stale backend keeps its own world and sim clock.`);
    process.exit(1);
  }
} catch { /* nothing listening -- this is the good path */ }

console.error(`[driver] backend  ${PYTHON} -m uvicorn app.main:app :${PORT}`);
const backend = spawn(
  PYTHON, ['-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', PORT],
  { cwd: path.join(ROOT, 'backend'), stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true },
);
const backendLog = [];
for (const stream of [backend.stdout, backend.stderr]) {
  stream.on('data', (d) => { backendLog.push(String(d)); if (backendLog.length > 200) backendLog.shift(); });
}

let browser;
let page;

async function boot() {
  await waitFor(async () => (await fetch(URL + 'api/status')).ok, 60_000, 'backend /api/status');
  console.error('[driver] backend up');

  browser = await puppeteer.launch({
    executablePath: browserPath,
    headless: 'new',
    // Three.js needs a GL context. Headless Chrome has no GPU here, so force
    // the ANGLE/SwiftShader software rasteriser -- without these the canvas
    // comes out blank and every screenshot is a lie.
    args: [
      '--window-size=1280,800',
      '--use-gl=angle',
      '--use-angle=swiftshader',
      '--enable-unsafe-swiftshader',
      '--no-sandbox',
    ],
  });
  page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 800 });
  page.on('pageerror', (e) => console.error(`[page-error] ${e}`));
  page.on('console', (m) => { if (m.type() === 'error') console.error(`[console] ${m.text()}`); });

  await page.goto(URL, { waitUntil: 'load' });
  await waitFor(() => page.evaluate(() => globalThis.__NL?.net.status === 'connected'),
                30_000, 'WebSocket connect');
  // one rendered frame before anyone screenshots
  await page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));
  console.error('[driver] page ready -- __NL connected');
}

// ---------------------------------------------------------------- commands
const send = (op) => page.evaluate((o) => globalThis.__NL.net.send(o), op);

async function screenshot(name) {
  const file = path.join(SHOTS, `${(name || 'shot').replace(/[^\w.-]/g, '_')}.png`);
  await page.screenshot({ path: file });
  const { size } = fs.statSync(file);
  // Do NOT pixel-probe the canvas to decide whether the 3D view rendered.
  // #viewport is a WebGL canvas without preserveDrawingBuffer, so its buffer
  // is cleared once composited: drawImage/toDataURL read it back as pure
  // black even on a perfectly good frame. Puppeteer's screenshot captures the
  // composited surface and is fine -- it was the probe that lied.
  // Three.js' own render stats are the honest signal that geometry was drawn.
  const info = await page.evaluate(() => {
    const r = globalThis.__NL?.sceneManager?.renderer;
    return r ? { triangles: r.info.render.triangles, calls: r.info.render.calls } : null;
  });
  return `${file} bytes=${size} `
       + (info ? `triangles=${info.triangles} drawCalls=${info.calls}` : 'renderer=n/a');
}

async function status() {
  return page.evaluate(() => ({
    ws: globalThis.__NL.net.status,
    sim: document.querySelector('#sim-status')?.textContent,
    clock: document.querySelector('#clock')?.textContent,
    objects: globalThis.__NL.store.objects.size,
    types: [...globalThis.__NL.store.objects.values()].map((o) => o.type),
    waterLevel: globalThis.__NL.store.waterLevel,
    riverFlow: document.querySelector('#water-flow')?.checked ?? null,
    fps: document.querySelector('#fps')?.textContent,
  }));
}

async function depthStats() {
  // SceneManager does not keep the depth array -- updateWaterField() folds it
  // straight into vertex Z as (terrainZ + depth * WATER_VISUAL_EXAGGERATION).
  // The exaggeration (2026-09-01, see SceneManager.ts) is a rendering-only
  // multiplier so gradients read as visible slopes instead of a flat-looking
  // plane -- it is NOT applied to physics/sampling, only to what gets drawn.
  // Divide it back out here so these numbers stay the real backend depth
  // (comparable across a run regardless of what the constant is tuned to),
  // not the inflated on-screen one. Only while RUNNING/PAUSED, though: the
  // IDLE editor preview (SceneManager.setWater) paints every vertex at the
  // flat slider level directly, with NO exaggeration applied at all -- correct
  // for that path is EXAG=1, and dividing it by 4 anyway was a real bug caught
  // here (a smoke run reported "0.125m flat" for a 0.5m slider before this).
  return page.evaluate(() => {
    const sm = globalThis.__NL.sceneManager;
    const idle = document.querySelector('#sim-status')?.textContent === 'IDLE';
    const EXAG = idle ? 1 : (sm.constructor.WATER_VISUAL_EXAGGERATION
      ?? Object.getPrototypeOf(sm).constructor.WATER_VISUAL_EXAGGERATION ?? 1);
    const w = sm.waterMesh.geometry.attributes.position;
    const t = sm.terrainMesh.geometry.attributes.position;
    const n = Math.min(w.count, t.count);
    let min = Infinity, max = -Infinity, sum = 0, wet = 0;
    // west/east halves, to show a river current as an actual gradient
    let westSum = 0, westN = 0, eastSum = 0, eastN = 0;
    const side = Math.round(Math.sqrt(n));
    for (let i = 0; i < n; i++) {
      const d = (w.getZ(i) - t.getZ(i)) / EXAG;
      if (d < min) min = d;
      if (d > max) max = d;
      sum += d;
      if (d > 1e-3) wet++;
      const col = i % side;
      if (col < side / 4) { westSum += d; westN++; }
      else if (col > (3 * side) / 4) { eastSum += d; eastN++; }
    }
    const r = (v) => Math.round(v * 1000) / 1000;
    return {
      cells: n, wet, min: r(min), max: r(max), mean: r(sum / n),
      westMean: r(westSum / Math.max(1, westN)),
      eastMean: r(eastSum / Math.max(1, eastN)),
      visible: sm.waterMesh.visible,
      renderExaggeration: EXAG,
    };
  });
}

async function handle(line) {
  const [cmd, ...rest] = line.trim().split(/\s+/);
  const arg = line.trim().slice(cmd.length).trim();
  switch (cmd) {
    case '': return null;
    case 'status': return JSON.stringify(await status());
    case 'add':
      await page.evaluate((t) => globalThis.__NL.editor.addObject(t), rest[0].toUpperCase());
      await waitFor(async () => (await status()).types.includes(rest[0].toUpperCase()), 10_000, 'object added');
      return `added ${rest[0].toUpperCase()}`;
    case 'place': {
      // editor.addObject() auto-places on a 3-wide grid; this puts an object
      // exactly where you want it, which is what physics checks need.
      const [type, x, z] = [rest[0].toUpperCase(), parseFloat(rest[1]), parseFloat(rest[2])];
      const before = (await status()).objects;
      const y = await page.evaluate((a) => globalThis.__NL.store.terrain.heightAt(a[0], a[1]), [x, z]);
      await send({ op: 'object_add', object: { type, position: [x, y, z] } });
      await waitFor(async () => (await status()).objects > before, 10_000, 'object placed');
      return `placed ${type} at (${x}, ${z})`;
    }
    case 'start': case 'pause': case 'reset':
      await send({ op: cmd });
      await sleep(300);
      return JSON.stringify(await status());
    case 'flow':
      await send({ op: 'water_flow', enabled: arg === 'on' });
      await page.evaluate((on) => { const c = document.querySelector('#water-flow'); if (c) c.checked = on; },
                          arg === 'on');
      return `river flow ${arg}`;
    case 'temp':
      await send({ op: 'environment_temperature', temperature: parseFloat(arg) });
      return `temperature ${arg}`;
    case 'water':
      await send({ op: 'water_level', level: parseFloat(arg) });
      return `water level ${arg}`;
    case 'send':
      await send(JSON.parse(arg));
      await sleep(200);
      return `sent ${arg}`;
    case 'eval':
      return JSON.stringify(await page.evaluate((src) => {
        const v = (0, eval)(src);
        return v === undefined ? null : JSON.parse(JSON.stringify(v));
      }, arg));
    case 'depth': return JSON.stringify(await depthStats());
    case 'wait': await sleep(parseInt(arg, 10) || 0); return `waited ${arg}ms`;
    case 'ss': return await screenshot(rest[0]);
    case 'backendlog': return backendLog.join('').split('\n').slice(-15).join(' | ');
    case 'quit': return 'QUIT';
    default: return `ERRunknown command: ${cmd}`;
  }
}

// ---------------------------------------------------------------- main
async function shutdown(code) {
  try { if (browser) await browser.close(); } catch { /* already gone */ }
  backend.kill();
  await sleep(400);
  process.exit(code);
}

try {
  await boot();
} catch (e) {
  console.error(`err boot failed: ${e.message}`);
  console.error(backendLog.join('').split('\n').slice(-20).join('\n'));
  await shutdown(1);
}

// `--smoke`: one command, exits 0/1. Proves the whole stack end to end --
// backend up, WebGL rendering real geometry, an object round-tripping through
// the WebSocket, and the sim clock plus the water field actually advancing.
if (process.argv.includes('--smoke')) {
  const fail = (m) => { console.log(`err SMOKE FAIL: ${m}`); return shutdown(1); };
  try {
    await handle('place ROCK 0 0');
    await handle('flow on');
    const before = JSON.parse(await handle('depth'));
    await handle('start');
    await sleep(5000);
    const after = JSON.parse(await handle('depth'));
    const st = JSON.parse(await handle('status'));
    const shot = await handle('ss smoke');
    const tris = Number(/triangles=(\d+)/.exec(shot)?.[1] ?? 0);

    if (st.sim !== 'RUNNING') await fail(`sim status is ${st.sim}, expected RUNNING`);
    else if (!st.types.includes('ROCK')) await fail('ROCK did not round-trip through the backend');
    else if (tris < 1000) await fail(`only ${tris} triangles drawn -- WebGL is not rendering`);
    else if (!(after.westMean > after.eastMean + 0.05)) {
      await fail(`no west->east gradient (west=${after.westMean} east=${after.eastMean}); `
               + 'river flow is not moving water');
    } else {
      console.log(`ok SMOKE PASS  sim=${st.sim} ${st.clock} objects=${st.objects} `
        + `triangles=${tris} depth ${before.mean}m flat -> west ${after.westMean}m / `
        + `east ${after.eastMean}m`);
      console.log(`ok ${shot}`);
      await shutdown(0);
    }
  } catch (e) { await fail(e.message); }
}

console.log('ok ready');
const rl = readline.createInterface({ input: process.stdin, terminal: false });
for await (const line of rl) {
  if (!line.trim()) continue;
  try {
    const out = await handle(line);
    if (out === 'QUIT') { console.log('ok bye'); await shutdown(0); }
    else if (out !== null) console.log(out.startsWith?.('ERR') ? `err ${out.slice(3)}` : `ok ${out}`);
  } catch (e) {
    console.log(`err ${e.message.replace(/\n/g, ' ')}`);
  }
}
await shutdown(0);
