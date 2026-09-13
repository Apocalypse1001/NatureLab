/**
 * Falling rain streaks (RainLab-1, docs/14_rain_plan.md).
 *
 * A picture of the rain INPUT, never of the water. The water the rain adds is
 * the solver's `h`, drawn by the ordinary water mesh; these streaks only say
 * "it is raining, this hard". So the one number they read is the intensity the
 * backend reports it is actually applying (`fluid.rain_mm_h`), not the slider:
 * a rain control the solver rejected must not keep drawing rain.
 *
 * Density is proportional to intensity up to MAX_DROPS, so 50 mm/h visibly
 * reads as ten times 5 mm/h. Past the cap the picture stops scaling while the
 * physics does not -- stated here rather than discovered.
 *
 * Drops live in a box that follows the orbit target and is sized from the
 * camera distance, so the rain looks the same around a 200 m town and a 2 km
 * coast without spending drops on terrain nobody is looking at.
 */
import * as THREE from 'three';

export class RainField {
  readonly lines: THREE.LineSegments;

  static readonly MAX_DROPS = 15000;
  /** Visual density only: streaks per mm/h of rain. */
  static readonly DROPS_PER_MM_H = 150;

  private positions: Float32Array;
  private speeds: Float32Array;
  private floors: Float32Array;
  private intensity = 0;
  private count = 0;
  private half = 40;
  private height = 30;

  constructor() {
    const max = RainField.MAX_DROPS;
    this.positions = new Float32Array(max * 6);
    this.speeds = new Float32Array(max);
    this.floors = new Float32Array(max);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(this.positions, 3));
    geometry.setDrawRange(0, 0);
    this.lines = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({
      color: 0xaecbe6, transparent: true, opacity: 0.35, depthWrite: false,
    }));
    this.lines.frustumCulled = false;
    this.lines.visible = false;
  }

  /** Intensity in mm/h as reported by the solver; 0 hides the rain. */
  setIntensity(mmPerHour: number): void {
    this.intensity = Number.isFinite(mmPerHour) ? Math.max(0, mmPerHour) : 0;
  }

  /**
   * Advance and redraw. `running` false freezes the drops in place: a paused
   * simulation is not raining, it is a still frame of rain.
   */
  update(dt: number, centre: THREE.Vector3, cameraDistance: number, span: number,
         groundAt: (x: number, z: number) => number, running: boolean): void {
    const wanted = Math.min(RainField.MAX_DROPS,
                            Math.round(this.intensity * RainField.DROPS_PER_MM_H));
    this.half = THREE.MathUtils.clamp(cameraDistance * 0.7, 15, span / 2);
    this.height = this.half * 0.8;
    // newly requested drops start at random heights so a change of intensity
    // fills the box at once instead of arriving as one flat sheet from above
    for (let i = this.count; i < wanted; i++) this.respawn(i, centre, groundAt, true);
    this.count = wanted;
    const geometry = this.lines.geometry;
    geometry.setDrawRange(0, wanted * 2);
    this.lines.visible = wanted > 0;
    if (!wanted || !running) return;

    const step = Math.min(dt, 0.1);
    // heavier rain falls faster and draws longer: real drops run from ~2 m/s
    // (drizzle) to ~9 m/s (a downpour's large drops)
    const streak = 0.03 + 0.04 * Math.min(1, this.intensity / 50);
    const size = this.half * 2;
    for (let i = 0; i < wanted; i++) {
      const o = i * 6;
      let x = this.positions[o];
      let z = this.positions[o + 2];
      // keep drops inside the box as the camera pans, instead of leaving them
      // behind on the far side of the map
      if (x < centre.x - this.half) x += size; else if (x > centre.x + this.half) x -= size;
      if (z < centre.z - this.half) z += size; else if (z > centre.z + this.half) z -= size;
      const y = this.positions[o + 1] - this.speeds[i] * step;
      if (y <= this.floors[i]) {
        this.respawn(i, centre, groundAt, false);
        continue;
      }
      const tail = this.speeds[i] * streak;
      this.positions[o] = x; this.positions[o + 1] = y; this.positions[o + 2] = z;
      this.positions[o + 3] = x; this.positions[o + 4] = y + tail; this.positions[o + 5] = z;
    }
    (geometry.attributes.position as THREE.BufferAttribute).needsUpdate = true;
  }

  private respawn(i: number, centre: THREE.Vector3,
                  groundAt: (x: number, z: number) => number, anyHeight: boolean): void {
    const x = centre.x + (Math.random() * 2 - 1) * this.half;
    const z = centre.z + (Math.random() * 2 - 1) * this.half;
    const floor = groundAt(x, z);
    const top = Math.max(floor, centre.y) + this.height;
    const y = anyHeight ? floor + Math.random() * (top - floor) : top;
    this.floors[i] = floor;
    this.speeds[i] = 2 + 7 * Math.min(1, this.intensity / 50) * (0.7 + Math.random() * 0.6);
    const o = i * 6;
    this.positions[o] = x; this.positions[o + 1] = y; this.positions[o + 2] = z;
    this.positions[o + 3] = x; this.positions[o + 4] = y; this.positions[o + 5] = z;
  }
}
