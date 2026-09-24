/**
 * The world past the edge of the simulated map: terrain, and the water running
 * on past it. Drawn, never simulated.
 *
 * Without it a river that leaves the map runs into a black void, and a child
 * watching reads that as "the river stops at the end of the world" -- which is
 * the opposite of what the open outlet is doing (v0.16.0, asked for by the
 * user after seeing exactly that). The skirt only continues what the map's own
 * edge already says, and invents nothing the map does not show:
 *
 * - terrain: every edge row keeps the slope it has over the last
 *   SLOPE_REACH_M of the map (the same reach the backend fits for a "river"
 *   outlet), so a valley falls on past the east edge and rises on past the
 *   west one, and a flat floodplain stays flat;
 * - water: only past a side given a mode, and only where the edge vertex it
 *   continues is actually wet. "depth" holds the edge depth over the
 *   continuing bed (a river running on, or arriving from upstream); "level"
 *   holds the edge surface elevation (a sea or a held inflow level), and dries
 *   where the continuing bed rises above it. A corner has water only when both
 *   sides beside it do.
 *
 * Terrain past a side with no water mode is still drawn, and would read as dry
 * land even where it continues a sea bed -- which the first version did along
 * the tsunami coast's north and south walls, a straight green shore that does
 * not exist. So the caller gives those sides the sea's level (`setWaterModes`).
 *
 * The skirt is shaded slightly darker toward its rim so it does not read as
 * part of the playable map.
 */
import * as THREE from 'three';
import type { TerrainGrid } from '../world/TerrainGrid';

export type EdgeWaterMode = 'depth' | 'level' | null;

const STEPS = 24;               // rings of vertices outward from each edge
const REACH_FACTOR = 1.5;       // how far past each edge, in map widths
const SPACING_EXPONENT = 1.6;   // fine at the seam, coarse toward the rim
const SLOPE_REACH_M = 20;       // matches backend config.OUTLET_SLOPE_REACH_M
const MAX_SLOPE = 0.05;         // a cone or a cliff at the edge is not continued as one
const SLOPE_CAP_FACTOR = 0.5;   // the slope is continued this many map widths at most
const WET_M = 1e-3;             // matches the water shader's 1 mm fade-in

export class EdgeSkirt {
  readonly group = new THREE.Group();
  private terrainMesh: THREE.Mesh;
  private waterMesh: THREE.Mesh;
  private terrain: TerrainGrid | null = null;
  private offsets = new Float32Array(STEPS + 1);   // offsets[0] = 0, the seam
  private slopeE = new Float32Array(0);
  private slopeW = new Float32Array(0);
  private slopeN = new Float32Array(0);
  private slopeS = new Float32Array(0);
  private eastMode: EdgeWaterMode = null;
  private westMode: EdgeWaterMode = null;
  private sideMode: EdgeWaterMode = null;          // north and south
  private waterVisible = true;
  // per skirt-grid vertex, filled by buildGrid
  private nx = 0;
  private nz = 0;
  private sizeX = 0;
  private sizeZ = 0;
  private ground = new Float32Array(0);
  private mapIndex = new Int32Array(0);     // the map vertex this one continues
  private sideX = new Int8Array(0);         // -1 west, 0 inside, +1 east
  private sideZ = new Int8Array(0);         // -1 south, 0 inside, +1 north
  private outward = new Float32Array(0);    // 0..1 of the reach
  private outerQuads = new Uint32Array(0);  // first vertex of every quad outside the map
  private waterFlow = new Float32Array(0);
  private waterDepth = new Float32Array(0);
  private wet = new Uint8Array(0);

  // groundMaterial: the map's own ground (SceneManager.buildTerrainMaterial,
  // vertex colours on, grid off), so slopes keep their colour past the edge
  constructor(groundMaterial: THREE.Material, waterMaterial: THREE.Material) {
    this.terrainMesh = new THREE.Mesh(new THREE.BufferGeometry(), groundMaterial);
    this.terrainMesh.receiveShadow = true;
    this.terrainMesh.frustumCulled = false;
    this.waterMesh = new THREE.Mesh(new THREE.BufferGeometry(), waterMaterial);
    this.waterMesh.frustumCulled = false;
    this.group.add(this.terrainMesh, this.waterMesh);
  }

  /** Water past the east (outlet), west (inflow) and north/south edges. */
  setWaterModes(east: EdgeWaterMode, west: EdgeWaterMode, sides: EdgeWaterMode): void {
    this.eastMode = east;
    this.westMode = west;
    this.sideMode = sides;
  }

  setWaterVisible(visible: boolean): void {
    this.waterVisible = visible;
    this.waterMesh.visible = visible;
  }

  /** No water frame yet (IDLE preview, a fresh load): draw no skirt water. */
  clearWater(): void {
    this.waterMesh.geometry.setDrawRange(0, 0);
  }

  rebuild(terrain: TerrainGrid): void {
    this.terrain = terrain;
    const reach = terrain.sizeM * REACH_FACTOR;
    for (let k = 0; k <= STEPS; k++) {
      this.offsets[k] = reach * Math.pow(k / STEPS, SPACING_EXPONENT);
    }
    this.measureSlopes(terrain);
    this.buildGrid(terrain);
  }

  /**
   * Continue the streamed water surface past the edges. `heights` is the
   * WATER_HEIGHT frame (absolute surface, bed - 0.05 where dry) and `flow` the
   * map water mesh's (u, v) per vertex, both in terrain-vertex order.
   */
  updateWater(heights: Float32Array, flow: Float32Array): void {
    const terrain = this.terrain;
    if (!terrain || !this.ground.length) return;
    const geometry = this.waterMesh.geometry;
    const pos = geometry.attributes.position as THREE.BufferAttribute;
    const index = geometry.index!;
    const indices = index.array as Uint32Array;
    const count = this.nx * this.nz;
    const W = terrain.width, H = terrain.height;

    for (let v = 0; v < count; v++) {
      const sx = this.sideX[v], sz = this.sideZ[v];
      const ground = this.ground[v];
      this.wet[v] = 0;
      this.waterDepth[v] = 0;
      if (sx === 0 && sz === 0) {
        // Inside the map nothing is drawn here -- except the map's own boundary
        // vertices, which are the inner corners of the first ring of skirt
        // quads. Left dry, that whole ring never drew: measured as a strip of
        // grass along every seam of the tsunami sea, and a gap across the river.
        const m = this.mapIndex[v];
        const i = m % (W + 1), j = (m - i) / (W + 1);
        if (i !== 0 && i !== W && j !== 0 && j !== H) continue;
        const depth = heights[m] - terrain.heights[m];
        pos.setY(v, depth > WET_M ? heights[m] : ground - 0.05);
        if (depth > WET_M) {
          this.waterDepth[v] = depth;
          this.wet[v] = 1;
        }
        this.waterFlow[v * 2] = flow[m * 2] ?? 0;
        this.waterFlow[v * 2 + 1] = flow[m * 2 + 1] ?? 0;
        continue;
      }
      const along = sx > 0 ? this.eastMode : sx < 0 ? this.westMode : null;
      const across = sz !== 0 ? this.sideMode : null;
      let mode: EdgeWaterMode;
      if (sx !== 0 && sz !== 0) {
        mode = along && across ? (along === 'level' || across === 'level' ? 'level' : 'depth')
          : null;
      } else {
        mode = sx !== 0 ? along : across;
      }
      const m = this.mapIndex[v];
      const eta = heights[m];
      const depth = eta - terrain.heights[m];
      let surface = ground - 0.05;
      if (mode && depth > WET_M) {
        const s = mode === 'depth' ? ground + depth : eta;
        if (s - ground > WET_M) {
          surface = s;
          this.waterDepth[v] = s - ground;
          this.wet[v] = 1;
        }
      }
      pos.setY(v, surface);
      // a river keeps its current; a held level goes still away from the edge
      const carry = mode === 'depth' ? 1 : 1 - this.outward[v];
      this.waterFlow[v * 2] = (flow[m * 2] ?? 0) * carry;
      this.waterFlow[v * 2 + 1] = (flow[m * 2 + 1] ?? 0) * carry;
    }

    let used = 0;
    const nx = this.nx;
    for (let q = 0; q < this.outerQuads.length; q++) {
      const a = this.outerQuads[q], b = a + 1, c = a + nx, d = c + 1;
      if (!(this.wet[a] && this.wet[b] && this.wet[c] && this.wet[d])) continue;
      indices[used++] = a; indices[used++] = c; indices[used++] = b;
      indices[used++] = b; indices[used++] = c; indices[used++] = d;
    }
    pos.needsUpdate = true;
    (geometry.attributes.aFlow as THREE.BufferAttribute).needsUpdate = true;
    (geometry.attributes.aDepth as THREE.BufferAttribute).needsUpdate = true;
    index.needsUpdate = true;
    geometry.setDrawRange(0, used);
    if (used > 0) geometry.computeVertexNormals();
    this.waterMesh.visible = this.waterVisible && used > 0;
  }

  /** Outward rise per metre along each edge, over SLOPE_REACH_M. */
  private measureSlopes(terrain: TerrainGrid): void {
    const W = terrain.width, H = terrain.height, cell = terrain.cellSize;
    const kx = Math.max(1, Math.min(W - 1, Math.round(SLOPE_REACH_M / cell)));
    const kz = Math.max(1, Math.min(H - 1, Math.round(SLOPE_REACH_M / cell)));
    const clamp = (s: number) => Math.max(-MAX_SLOPE, Math.min(MAX_SLOPE, s));
    this.slopeE = new Float32Array(H + 1);
    this.slopeW = new Float32Array(H + 1);
    for (let j = 0; j <= H; j++) {
      this.slopeE[j] = clamp((terrain.at(W, j) - terrain.at(W - kx, j)) / (kx * cell));
      this.slopeW[j] = clamp((terrain.at(0, j) - terrain.at(kx, j)) / (kx * cell));
    }
    this.slopeN = new Float32Array(W + 1);
    this.slopeS = new Float32Array(W + 1);
    for (let i = 0; i <= W; i++) {
      this.slopeN[i] = clamp((terrain.at(i, H) - terrain.at(i, H - kz)) / (kz * cell));
      this.slopeS[i] = clamp((terrain.at(i, 0) - terrain.at(i, kz)) / (kz * cell));
    }
  }

  /**
   * One tensor grid over map + skirt, shared by the terrain and the water: the
   * map's own vertex lines inside, STEPS outward rings outside. Sharing the
   * map's lines keeps the seam watertight -- every skirt vertex on the edge IS
   * a map vertex -- and the quads inside the map are left out of both indices.
   */
  private buildGrid(terrain: TerrainGrid): void {
    const W = terrain.width, H = terrain.height, cell = terrain.cellSize;
    const size = terrain.sizeM;
    const sizeX = terrain.sizeX, sizeZ = terrain.sizeZ;
    const reach = size * REACH_FACTOR;
    const cap = size * SLOPE_CAP_FACTOR;
    const n = STEPS;
    const NX = W + 1 + 2 * n, NZ = H + 1 + 2 * n;
    const count = NX * NZ;
    // Size, not only vertex count: the River (1 m cells) and Tsunami (10 m)
    // worlds are both 201 x 201, and reusing the water geometry across them
    // would leave its x/z at the other world's scale.
    const resized = NX !== this.nx || NZ !== this.nz
      || sizeX !== this.sizeX || sizeZ !== this.sizeZ;
    this.nx = NX;
    this.nz = NZ;
    this.sizeX = sizeX;
    this.sizeZ = sizeZ;
    const positions = new Float32Array(count * 3);
    const uvs = new Float32Array(count * 2);
    const colors = new Float32Array(count * 3);
    this.ground = new Float32Array(count);
    this.mapIndex = new Int32Array(count);
    this.sideX = new Int8Array(count);
    this.sideZ = new Int8Array(count);
    this.outward = new Float32Array(count);
    const axis = (t: number, cells: number, half: number) => {
      if (t < n) {
        const out = this.offsets[n - t];
        return { coord: -(half + out), map: 0, out, side: -1 };
      }
      if (t <= n + cells) return { coord: (t - n) * cell - half, map: t - n, out: 0, side: 0 };
      const out = this.offsets[t - (n + cells)];
      return { coord: half + out, map: cells, out, side: 1 };
    };
    for (let b = 0; b < NZ; b++) {
      const z = axis(b, H, sizeZ / 2);
      for (let a = 0; a < NX; a++) {
        const x = axis(a, W, sizeX / 2);
        let y = terrain.at(x.map, z.map);
        if (x.side > 0) y += this.slopeE[z.map] * Math.min(x.out, cap);
        else if (x.side < 0) y += this.slopeW[z.map] * Math.min(x.out, cap);
        if (z.side > 0) y += this.slopeN[x.map] * Math.min(z.out, cap);
        else if (z.side < 0) y += this.slopeS[x.map] * Math.min(z.out, cap);
        const v = b * NX + a;
        positions[v * 3] = x.coord;
        positions[v * 3 + 1] = y;
        positions[v * 3 + 2] = z.coord;
        // the same UV mapping as the map's PlaneGeometry, so the grass tiles
        // run across the seam without a jump
        uvs[v * 2] = (x.coord + sizeX / 2) / sizeX;
        uvs[v * 2 + 1] = (-z.coord + sizeZ / 2) / sizeZ;
        const out = Math.min(1, Math.max(x.out, z.out) / reach);
        colors[v * 3] = colors[v * 3 + 1] = colors[v * 3 + 2] = 1 - 0.25 * out;
        this.ground[v] = y;
        this.mapIndex[v] = z.map * (W + 1) + x.map;
        this.sideX[v] = x.side;
        this.sideZ[v] = z.side;
        this.outward[v] = out;
      }
    }
    const quads: number[] = [];
    for (let b = 0; b < NZ - 1; b++) {
      for (let a = 0; a < NX - 1; a++) {
        if (a >= n && a < n + W && b >= n && b < n + H) continue;   // the map itself
        quads.push(b * NX + a);
      }
    }
    this.outerQuads = new Uint32Array(quads);
    const indices = new Uint32Array(quads.length * 6);
    quads.forEach((v00, q) => {
      // wound so the face points up (x grows with a, z with b)
      const v10 = v00 + 1, v01 = v00 + NX, v11 = v01 + 1;
      indices.set([v00, v01, v10, v10, v01, v11], q * 6);
    });

    const terrainGeometry = new THREE.BufferGeometry();
    terrainGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    terrainGeometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
    terrainGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    terrainGeometry.setIndex(new THREE.BufferAttribute(indices, 1));
    terrainGeometry.computeVertexNormals();
    this.terrainMesh.geometry.dispose();
    this.terrainMesh.geometry = terrainGeometry;

    if (!resized && this.waterDepth.length === count) {
      // same grid: the water geometry's x/z are unchanged, y is rewritten per frame
      return;
    }
    const waterPositions = new Float32Array(positions);
    this.waterFlow = new Float32Array(count * 2);
    this.waterDepth = new Float32Array(count);
    this.wet = new Uint8Array(count);
    const water = new THREE.BufferGeometry();
    water.setAttribute('position', new THREE.BufferAttribute(waterPositions, 3));
    water.setAttribute('aFlow', new THREE.BufferAttribute(this.waterFlow, 2));
    water.setAttribute('aDepth', new THREE.BufferAttribute(this.waterDepth, 1));
    // lava never runs on past the edge; the shader still needs the attribute
    water.setAttribute('aLavaTemp', new THREE.BufferAttribute(new Float32Array(count), 1));
    water.setIndex(new THREE.BufferAttribute(new Uint32Array(quads.length * 6), 1));
    water.setDrawRange(0, 0);
    this.waterMesh.geometry.dispose();
    this.waterMesh.geometry = water;
  }
}
