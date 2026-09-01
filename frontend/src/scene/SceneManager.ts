/** Three.js scene: renderer, camera, controls, lights, terrain mesh,
 *  water placeholder, particle points, object meshes. */
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { TerrainGrid } from '../world/TerrainGrid';
import { applyTransform, buildObjectMesh } from '../world/ObjectFactory';
import type { ObjectData } from '../world/types';

export class SceneManager {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  readonly controls: OrbitControls;
  readonly terrainMesh: THREE.Mesh;
  readonly waterMesh: THREE.Mesh;
  readonly objectsRoot = new THREE.Group();
  readonly points: THREE.Points;
  readonly selectionBox: THREE.BoxHelper | null = null;

  private raycaster = new THREE.Raycaster();
  private _selectionHelper: THREE.BoxHelper | null = null;
  private _waterTimeUniform: { value: number } | null = null;
  private _clockStart = performance.now();

  constructor(canvas: HTMLCanvasElement, private terrain: TerrainGrid) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x0e1420);

    this.scene.fog = new THREE.Fog(0x0e1420, 120, 400);

    this.camera = new THREE.PerspectiveCamera(55, 1, 0.1, 1000);
    this.camera.position.set(60, 55, 60);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.maxDistance = 300;

    this.scene.add(new THREE.HemisphereLight(0xbfd4ff, 0x3a3226, 0.7));
    const sun = new THREE.DirectionalLight(0xffffff, 1.2);
    sun.position.set(60, 90, 30);
    this.scene.add(sun);
    this.scene.add(new THREE.AxesHelper(10));

    // terrain mesh (geometry rebuilt from the logical grid)
    const geo = new THREE.PlaneGeometry(
      this.terrain.sizeM, this.terrain.sizeM,
      this.terrain.width, this.terrain.height);
    this.terrainMesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({
      color: 0x5d7a4a, roughness: 1.0, metalness: 0,
    }));
    this.terrainMesh.rotation.x = -Math.PI / 2;
    this.scene.add(this.terrainMesh);

    const grid = new THREE.GridHelper(this.terrain.sizeM, this.terrain.width, 0x223344, 0x1b2836);
    grid.position.y = 0.05;
    this.scene.add(grid);

    // Water surface: same vertex grid as terrain (not a flat plane) so it
    // can be deformed per-cell from the backend's real water-height field
    // once the simulation is RUNNING -- see updateWaterField(). Before that
    // (editor/IDLE), setFlatWater() shows a flat preview at the slider
    // level, matching the original placeholder behaviour.
    const waterGeo = new THREE.PlaneGeometry(
      this.terrain.sizeM, this.terrain.sizeM, this.terrain.width, this.terrain.height);
    const waterMat = new THREE.MeshStandardMaterial({
      color: 0x2f7fd0, transparent: true, opacity: 0.55,
      roughness: 0.15, metalness: 0.1, depthWrite: false, side: THREE.DoubleSide,
    });
    // Cheap cosmetic ripple: perturbs the already-real per-cell height
    // (set in updateWaterField from backend depth data) with a small
    // animated offset, GPU-side, so it doesn't touch the physics data or
    // need a per-frame CPU vertex pass. Purely decorative -- see
    // docs/04_TZ_v0.3_roadmap.md priorities: physics/visibility first,
    // "beautiful" last. This is the first (cheap-tier) pass at that.
    waterMat.onBeforeCompile = (shader) => {
      shader.uniforms.uTime = { value: 0 };
      shader.vertexShader = shader.vertexShader
        .replace('#include <common>', `
          #include <common>
          uniform float uTime;
        `)
        .replace('#include <beginnormal_vertex>', `
          #include <beginnormal_vertex>
          float dRippleX = cos(position.x * 0.8 + uTime * 1.6) * 0.024;
          float dRippleY = cos(position.y * 0.6 - uTime * 1.1) * 0.015;
          objectNormal = normalize(objectNormal + vec3(-dRippleX, -dRippleY, 0.0));
        `)
        .replace('#include <begin_vertex>', `
          #include <begin_vertex>
          transformed.z += sin(position.x * 0.8 + uTime * 1.6) * 0.03
                          + sin(position.y * 0.6 - uTime * 1.1) * 0.025;
        `);
      this._waterTimeUniform = shader.uniforms.uTime;
    };
    this.waterMesh = new THREE.Mesh(waterGeo, waterMat);
    this.waterMesh.rotation.x = -Math.PI / 2;
    this.scene.add(this.waterMesh);

    // particle points buffer (filled from backend binary frames). This is
    // the original Warp-connectivity demo particle stream, unrelated to
    // the real water field above -- kept simulating/streaming (tests and
    // the Warp selftest depend on it) but hidden from view by default,
    // see setParticles(): it visually read as unrelated "dust" once real
    // water rendering existed, not as water.
    const positions = new Float32Array(1024 * 3);
    const pgeo = new THREE.BufferGeometry();
    pgeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    pgeo.setDrawRange(0, 0);
    this.points = new THREE.Points(pgeo, new THREE.PointsMaterial({
      color: 0x6fc3ff, size: 0.25, sizeAttenuation: true,
      transparent: true, opacity: 0.9,
    }));
    this.points.frustumCulled = false;
    this.points.visible = false;
    this.scene.add(this.points);

    this.scene.add(this.objectsRoot);
    this.resize();
  }

  resize(): void {
    const parent = this.renderer.domElement.parentElement;
    if (!parent) return;
    const w = parent.clientWidth, h = parent.clientHeight;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  // ------------------------------------------------------------------ terrain
  rebuildTerrain(terrain: TerrainGrid): void {
    this.terrain = terrain;
    const geo = this.terrainMesh.geometry as THREE.PlaneGeometry;
    const pos = geo.attributes.position;
    const w = terrain.width, h = terrain.height;
    // plane row j==0 corresponds to local y=+h/2 i.e. world z=-h/2 — matches backend grid
    for (let j = 0; j <= h; j++) {
      for (let i = 0; i <= w; i++) {
        pos.setZ(j * (w + 1) + i, terrain.heights[j * (w + 1) + i]);
      }
    }
    pos.needsUpdate = true;
    geo.computeVertexNormals();
  }

  /** Flat editor-preview water (IDLE, before the sim owns the surface). */
  setWater(level: number, visible: boolean): void {
    const geo = this.waterMesh.geometry as THREE.PlaneGeometry;
    const pos = geo.attributes.position;
    for (let idx = 0; idx < pos.count; idx++) pos.setZ(idx, level);
    pos.needsUpdate = true;
    geo.computeVertexNormals();
    this.waterMesh.visible = visible;
  }

  /**
   * Deform the water mesh per-cell from the backend's real depth field
   * (WATER_HEIGHT bulk frames), so the surface actually dips to terrain
   * level wherever depth is ~0 -- around/inside obstacles included. This
   * is what makes water visibly go around an object instead of a flat
   * plane implying it flows straight through. `depths` is indexed exactly
   * like the terrain grid (see rebuildTerrain).
   */
  updateWaterField(depths: Float32Array): void {
    const geo = this.waterMesh.geometry as THREE.PlaneGeometry;
    const pos = geo.attributes.position;
    const terrainPos = (this.terrainMesh.geometry as THREE.PlaneGeometry).attributes.position;
    const count = Math.min(pos.count, depths.length);
    let anyWet = false;
    for (let idx = 0; idx < count; idx++) {
      const depth = Math.max(0, depths[idx]);
      if (depth > 0.01) anyWet = true;
      pos.setZ(idx, terrainPos.getZ(idx) + depth);
    }
    pos.needsUpdate = true;
    geo.computeVertexNormals();
    this.waterMesh.visible = anyWet;
  }

  setParticles(buffer: Float32Array, count: number): void {
    let attr = this.points.geometry.attributes.position as THREE.BufferAttribute;
    if (attr.count < count) {
      let capacity = Math.max(1024, attr.count);
      while (capacity < count) capacity *= 2;
      attr = new THREE.BufferAttribute(new Float32Array(capacity * 3), 3);
      this.points.geometry.setAttribute('position', attr);
    }
    attr.array.set(buffer.subarray(0, count * 3));
    attr.needsUpdate = true;
    this.points.geometry.setDrawRange(0, count);
    // stays hidden regardless of count -- see the constructor note: this
    // demo stream is superseded by the real water mesh, kept simulating
    // for the Warp-connectivity tests, not for display.
  }

  /** Toggle the raw Warp-connectivity demo particle stream (debug use). */
  setDebugParticlesVisible(visible: boolean): void {
    this.points.visible = visible;
  }

  // ------------------------------------------------------------------ objects
  setObject(obj: ObjectData): void {
    let group = this.objectsRoot.getObjectByName(obj.id) as THREE.Group | undefined;
    if (!group) {
      group = buildObjectMesh(obj);
      this.objectsRoot.add(group);
    }
    applyTransform(group, obj);
    if (this._selectionHelper && this._selectionHelper.userData.owner === obj.id) {
      this._selectionHelper.update();
    }
  }

  removeObject(id: string): void {
    const group = this.objectsRoot.getObjectByName(id);
    if (group) this.objectsRoot.remove(group);
    this.clearSelection();
  }

  clearObjects(): void {
    this.objectsRoot.clear();
    this.clearSelection();
  }

  // ------------------------------------------------------------------ selection
  highlight(id: string | null): void {
    this.clearSelection();
    if (!id) return;
    const group = this.objectsRoot.getObjectByName(id);
    if (!group) return;
    const helper = new THREE.BoxHelper(group, 0xffd24a);
    helper.userData.owner = id;
    this.scene.add(helper);
    this._selectionHelper = helper;
  }

  clearSelection(): void {
    if (this._selectionHelper) {
      this.scene.remove(this._selectionHelper);
      this._selectionHelper.dispose();
      this._selectionHelper = null;
    }
  }

  /** Raycast objects under the cursor. */
  pickObject(ndc: THREE.Vector2): ObjectData['id'] | null {
    this.raycaster.setFromCamera(ndc, this.camera);
    const hits = this.raycaster.intersectObjects(this.objectsRoot.children, true);
    for (const hit of hits) {
      let node: THREE.Object3D | null = hit.object;
      while (node) {
        if (node.userData.objectId) return node.userData.objectId as string;
        node = node.parent;
      }
    }
    return null;
  }

  /** Raycast the terrain surface. */
  pickTerrain(ndc: THREE.Vector2): THREE.Vector3 | null {
    this.raycaster.setFromCamera(ndc, this.camera);
    const hits = this.raycaster.intersectObject(this.terrainMesh, false);
    return hits.length ? hits[0].point : null;
  }

  render(): void {
    this.controls.update();
    if (this._selectionHelper) this._selectionHelper.update();
    if (this._waterTimeUniform) {
      this._waterTimeUniform.value = (performance.now() - this._clockStart) / 1000;
    }
    this.renderer.render(this.scene, this.camera);
  }
}
