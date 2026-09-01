/** Logical terrain heightfield, decoupled from the visual mesh.
 *  Mirrors backend TerrainGrid so brush edits stay numerically identical. */
export class TerrainGrid {
  readonly width: number;
  readonly height: number;
  readonly cellSize: number;
  readonly heights: Float32Array; // (width+1) x (height+1), row-major by z

  constructor(width = 100, height = 100, cellSize = 1) {
    this.width = width;
    this.height = height;
    this.cellSize = cellSize;
    this.heights = new Float32Array((width + 1) * (height + 1));
  }

  get sizeM(): number {
    return this.width * this.cellSize;
  }

  at(i: number, j: number): number {
    return this.heights[j * (this.width + 1) + i];
  }

  heightAt(x: number, z: number): number {
    const gx = clamp(x / this.cellSize + this.width / 2, 0, this.width);
    const gz = clamp(z / this.cellSize + this.height / 2, 0, this.height);
    const i0 = Math.floor(gx), j0 = Math.floor(gz);
    const i1 = Math.min(i0 + 1, this.width), j1 = Math.min(j0 + 1, this.height);
    const fx = gx - i0, fz = gz - j0;
    return (1 - fx) * (1 - fz) * this.at(i0, j0) + fx * (1 - fz) * this.at(i1, j0)
      + (1 - fx) * fz * this.at(i0, j1) + fx * fz * this.at(i1, j1);
  }

  /** Same falloff as backend TerrainGrid.brush (keeps local and authoritative grids in sync). */
  brush(x: number, z: number, radius: number, strength: number): void {
    const rCells = Math.max(1, radius / this.cellSize);
    const ci = x / this.cellSize + this.width / 2;
    const cj = z / this.cellSize + this.height / 2;
    const loI = Math.max(0, Math.floor(ci - rCells));
    const hiI = Math.min(this.width, Math.ceil(ci + rCells));
    const loJ = Math.max(0, Math.floor(cj - rCells));
    const hiJ = Math.min(this.height, Math.ceil(cj + rCells));
    for (let j = loJ; j <= hiJ; j++) {
      for (let i = loI; i <= hiI; i++) {
        const d = Math.hypot(i - ci, j - cj);
        if (d <= rCells) {
          const falloff = 0.5 * (1 + Math.cos((Math.PI * d) / rCells));
          const idx = j * (this.width + 1) + i;
          this.heights[idx] = clamp(this.heights[idx] + strength * falloff, -20, 60);
        }
      }
    }
  }

  loadHeights(flat: ArrayLike<number>): void {
    this.heights.set(flat as Float32Array);
  }

  toList(): number[] {
    return Array.from(this.heights);
  }
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}
