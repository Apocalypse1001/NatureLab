/** Three.js scene: renderer, camera, controls, lights, terrain mesh,
 *  dynamic water surface, particle points, object meshes. */
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
  private waterBaseIndices: Uint16Array | Uint32Array;
  private waterDynamicIndices: Uint16Array | Uint32Array;
  private tracersVisible = true;
  private tracerDisplayLimit = 8000;
  private receivedTracerCount = 0;

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

    // Water vertices share terrain ordering, allowing direct bulk frame updates.
    const waterGeometry = new THREE.PlaneGeometry(this.terrain.sizeM, this.terrain.sizeM,
                                                  this.terrain.width, this.terrain.height);
    const sourceIndices = waterGeometry.index!.array;
    this.waterBaseIndices = sourceIndices instanceof Uint32Array
      ? new Uint32Array(sourceIndices) : new Uint16Array(sourceIndices);
    this.waterDynamicIndices = sourceIndices instanceof Uint32Array
      ? new Uint32Array(sourceIndices.length) : new Uint16Array(sourceIndices.length);
    waterGeometry.setIndex(new THREE.BufferAttribute(this.waterDynamicIndices, 1));
    this.waterMesh = new THREE.Mesh(
      waterGeometry,
      new THREE.MeshStandardMaterial({
        color: 0x2f7fd0, transparent: true, opacity: 0.45,
        roughness: 0.2, metalness: 0.1, depthWrite: false, side: THREE.DoubleSide,
      }));
    this.waterMesh.rotation.x = -Math.PI / 2;
    this.waterMesh.frustumCulled = false;
    this.scene.add(this.waterMesh);

    // particle points buffer (filled from backend binary frames)
    const positions = new Float32Array(1024 * 3);
    const pgeo = new THREE.BufferGeometry();
    pgeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    pgeo.setDrawRange(0, 0);
    this.points = new THREE.Points(pgeo, new THREE.PointsMaterial({
      color: 0xbde9ff, size: 0.14, sizeAttenuation: true,
      transparent: true, opacity: 0.8, depthWrite: false,
    }));
    this.points.frustumCulled = false;
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

  setWater(level: number, visible: boolean): void {
    this.waterMesh.position.y = 0;
    this.waterMesh.visible = visible;
    const pos = this.waterMesh.geometry.attributes.position as THREE.BufferAttribute;
    for (let i = 0; i < pos.count; i++) pos.setZ(i, level);
    pos.needsUpdate = true;
    this.waterMesh.geometry.setDrawRange(0, 0);
  }

  setWaterHeights(heights: Float32Array, count: number): boolean {
    const pos = this.waterMesh.geometry.attributes.position as THREE.BufferAttribute;
    if (count !== pos.count || heights.length < count) return false;
    this.waterMesh.position.y = 0;
    const wet = new Uint8Array(count);
    for (let i = 0; i < count; i++) {
      pos.setZ(i, heights[i]);
      wet[i] = heights[i] > this.terrain.heights[i] + 1e-4 ? 1 : 0;
    }
    pos.needsUpdate = true;
    let used = 0;
    for (let i = 0; i < this.waterBaseIndices.length; i += 3) {
      const a = this.waterBaseIndices[i], b = this.waterBaseIndices[i + 1];
      const c = this.waterBaseIndices[i + 2];
      if (wet[a] && wet[b] && wet[c]) {
        this.waterDynamicIndices[used++] = a;
        this.waterDynamicIndices[used++] = b;
        this.waterDynamicIndices[used++] = c;
      }
    }
    this.waterMesh.geometry.index!.needsUpdate = true;
    this.waterMesh.geometry.setDrawRange(0, used);
    this.waterMesh.geometry.computeVertexNormals();
    return true;
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
    this.receivedTracerCount = count;
    this.applyTracerDisplay();
  }

  setTracerVisible(visible: boolean): void {
    this.tracersVisible = visible;
    this.applyTracerDisplay();
  }

  setTracerDisplayLimit(count: number): void {
    this.tracerDisplayLimit = Math.max(0, Math.floor(count));
    this.applyTracerDisplay();
  }

  clearTracers(): void {
    this.receivedTracerCount = 0;
    this.applyTracerDisplay();
  }

  private applyTracerDisplay(): void {
    const count = Math.min(this.receivedTracerCount, this.tracerDisplayLimit);
    this.points.geometry.setDrawRange(0, count);
    this.points.visible = this.tracersVisible && count > 0;
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
    this.renderer.render(this.scene, this.camera);
  }
}
