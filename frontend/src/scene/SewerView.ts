/**
 * What the storm sewer is doing, drawn from the numbers the solver streams
 * (v0.17.0, backend/app/sewer.py).
 *
 * - Dots run along every working pipe at the mean velocity of the flow in a
 *   full pipe, Q / A -- a real speed, so a pipe carrying half its capacity
 *   visibly moves half as fast, and an idle one is still.
 * - The pipe is tinted by load: grey idle, blue working, orange-red full.
 *   A pipe that runs uphill or is not joined at both ends is drawn dark red
 *   or dark grey, and carries nothing.
 * - A splash of droplets leaves every outfall that is receiving water, thicker
 *   with more flow, in the direction the pipe arrives from.
 *
 * None of it feeds back into the physics; the water poured into the river is
 * the solver's, under the splash.
 */
import * as THREE from 'three';
import type { SewerLinkState } from '../world/types';

const DOTS_PER_PIPE = 20;
const MAX_PIPES = 32;
const JET_DROPS = 90;
const MAX_OUTFALLS = 16;

const IDLE = new THREE.Color(0x8a939c);
const WORKING = new THREE.Color(0x3f8fe0);
const FULL = new THREE.Color(0xf06a3a);
const UPHILL = new THREE.Color(0x7a2f2f);
const BROKEN = new THREE.Color(0x3c4046);

export class SewerView {
  readonly group = new THREE.Group();
  private dots: THREE.InstancedMesh;
  private jet: THREE.Points;
  private jetPositions = new Float32Array(JET_DROPS * MAX_OUTFALLS * 3);
  private links: SewerLinkState[] = [];
  private running = false;
  private phase = new Map<string, number>();
  private time = 0;
  private matrix = new THREE.Matrix4();
  private scratch = new THREE.Vector3();
  private tint = new THREE.Color();

  constructor(private objectsRoot: THREE.Group) {
    this.dots = new THREE.InstancedMesh(
      new THREE.SphereGeometry(1, 8, 6),
      new THREE.MeshBasicMaterial({ color: 0xcfe8ff }),
      DOTS_PER_PIPE * MAX_PIPES);
    this.dots.count = 0;
    this.dots.frustumCulled = false;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(this.jetPositions, 3));
    geometry.setDrawRange(0, 0);
    this.jet = new THREE.Points(geometry, new THREE.PointsMaterial({
      color: 0xd6ecff, size: 0.22, sizeAttenuation: true, transparent: true,
      opacity: 0.85, depthWrite: false,
    }));
    this.jet.frustumCulled = false;
    this.group.add(this.dots, this.jet);
  }

  setState(links: SewerLinkState[], running: boolean): void {
    this.links = links;
    this.running = running;
  }

  update(dt: number): void {
    if (this.running) this.time += dt;
    let dotCount = 0;
    let dropCount = 0;
    let outfalls = 0;
    for (const link of this.links.slice(0, MAX_PIPES)) {
      const pipe = this.objectsRoot.getObjectByName(link.pipe_id);
      const curve = pipe?.userData.curve as THREE.CurvePath<THREE.Vector3> | undefined;
      if (!pipe || !curve) continue;
      const radius = (pipe.userData.pipeRadius as number) ?? 0.12;
      const load = link.capacity_m3s > 0 ? Math.min(1, link.flow_m3s / link.capacity_m3s) : 0;
      this.tintPipe(pipe, link, load);
      if (link.status !== 'ok' || link.flow_m3s <= 1e-5) continue;

      // mean velocity of the flow if it filled the pipe: Q / (pi r^2), using the
      // real diameter, not the drawn radius (which has a visibility floor)
      const diameter = Math.max(0.05, 2 * Math.sqrt(link.capacity_m3s > 0
        ? (pipe.userData.pipeRadius as number) ** 2 : 0.01));
      const area = Math.PI * (diameter / 2) ** 2;
      const speed = Math.min(6, link.flow_m3s / area);
      const length = Math.max(1, curve.getLength());
      const phase = ((this.phase.get(link.pipe_id) ?? 0)
        + (this.running ? speed * dt / length : 0)) % 1;
      this.phase.set(link.pipe_id, phase);
      pipe.updateMatrixWorld();
      const size = radius * (0.35 + 0.35 * Math.sqrt(load));
      for (let k = 0; k < DOTS_PER_PIPE; k++) {
        const t = (phase + k / DOTS_PER_PIPE) % 1;
        curve.getPointAt(t, this.scratch);
        pipe.localToWorld(this.scratch);
        this.matrix.makeScale(size, size, size).setPosition(this.scratch);
        this.dots.setMatrixAt(dotCount++, this.matrix);
      }

      const outfall = this.objectsRoot.getObjectByName(link.outfall_id);
      if (outfall && outfalls < MAX_OUTFALLS) {
        outfalls++;
        const end = curve.getPointAt(1, new THREE.Vector3());
        const before = curve.getPointAt(Math.max(0, 1 - 2 / length), new THREE.Vector3());
        pipe.localToWorld(end);
        pipe.localToWorld(before);
        const dir = end.clone().sub(before).setY(0);
        if (dir.lengthSq() < 1e-6) dir.set(1, 0, 0);
        dir.normalize();
        outfall.rotation.set(0, Math.atan2(-dir.z, dir.x), 0);
        const mouth = outfall.position.clone().addScaledVector(dir, 0.45).setY(outfall.position.y + 0.45);
        const drops = Math.max(8, Math.round(JET_DROPS * Math.sqrt(load)));
        const reach = 0.6 + 1.6 * Math.sqrt(load);
        for (let k = 0; k < drops; k++) {
          const seed = Math.sin((k + 1) * 12.9898) * 43758.5453;
          const jitter = seed - Math.floor(seed);
          const age = (this.time * (0.9 + 0.4 * jitter) + k / drops) % 1;
          const side = (jitter - 0.5) * 0.5;
          const o = dropCount * 3;
          this.jetPositions[o] = mouth.x + dir.x * reach * age - dir.z * side * age;
          this.jetPositions[o + 1] = mouth.y - 1.2 * age * age;
          this.jetPositions[o + 2] = mouth.z + dir.z * reach * age + dir.x * side * age;
          dropCount++;
        }
      }
    }
    this.dots.count = dotCount;
    this.dots.instanceMatrix.needsUpdate = true;
    const attribute = this.jet.geometry.attributes.position as THREE.BufferAttribute;
    attribute.needsUpdate = true;
    this.jet.geometry.setDrawRange(0, dropCount);
    this.jet.visible = dropCount > 0;
  }

  private tintPipe(pipe: THREE.Object3D, link: SewerLinkState, load: number): void {
    const tube = pipe.getObjectByName('pipe-tube') as THREE.Mesh | undefined;
    const material = tube?.material as THREE.MeshStandardMaterial | undefined;
    if (!material) return;
    if (link.status === 'uphill') this.tint.copy(UPHILL);
    else if (link.status !== 'ok') this.tint.copy(BROKEN);
    else if (link.flow_m3s <= 1e-5) this.tint.copy(IDLE);
    else if (load < 0.9) this.tint.copy(IDLE).lerp(WORKING, Math.min(1, 0.4 + load));
    else this.tint.copy(WORKING).lerp(FULL, Math.min(1, (load - 0.9) * 10));
    material.color.copy(this.tint);
  }
}
