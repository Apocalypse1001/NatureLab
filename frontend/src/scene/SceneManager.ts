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
  readonly sun: THREE.DirectionalLight;
  readonly selectionBox: THREE.BoxHelper | null = null;

  private raycaster = new THREE.Raycaster();
  private _selectionHelper: THREE.BoxHelper | null = null;
  private waterBaseIndices: Uint16Array | Uint32Array;
  private waterDynamicIndices: Uint16Array | Uint32Array;
  private gridHelper: THREE.GridHelper;
  private waterFlow = new Float32Array(0);
  private waterDepth = new Float32Array(0);
  private sprayPoints: THREE.Points | null = null;
  private sprayPositions = new Float32Array(0);
  private sprayPhase = 0;
  private static readonly MAX_SPRAY = 4000;
  // (1.4 m/s)^2 -- below this the surface stays unbroken
  private static readonly SPRAY_SPEED_SQ = 1.96;
  private waterTime = { value: 0 };
  private _clockStart = performance.now();
  private tracersVisible = true;
  private tracerDisplayLimit = 36000;   // matches config.FLOW_TRACER_COUNT
  private receivedTracerCount = 0;

  constructor(canvas: HTMLCanvasElement, private terrain: TerrainGrid) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x0e1420);
    // Shadows are the one cheap thing that makes an object look like it is ON
    // the terrain rather than floating in front of it. Nothing else in this
    // pass changes what the user can read off the scene as much as this does.
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    // Camera framing, fog and zoom limits are derived from the world size, not
    // fixed numbers: they were tuned for a 100 m map and left the 200 m map
    // half out of frame and inside the fog when it was doubled in v0.7.0.
    const span = terrain.sizeM;
    this.scene.fog = new THREE.Fog(0x0e1420, span * 1.2, span * 4);

    this.camera = new THREE.PerspectiveCamera(55, 1, 0.1, 1000);
    this.camera.position.set(span * 0.6, span * 0.55, span * 0.6);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.maxDistance = span * 3;

    // Three lights, and each one has a job. The sky fill lifts everything the
    // sun cannot reach; the sun is the only caster and supplies the shape; a
    // dim bounce from the opposite side keeps shadowed walls readable instead
    // of black, which matters when a child is being asked to look at the wall
    // the water is hitting.
    this.scene.add(new THREE.HemisphereLight(0xcfe0ff, 0x4a4032, 0.85));
    this.sun = new THREE.DirectionalLight(0xfff4e2, 2.1);
    this.sun.castShadow = true;
    this.scene.add(this.sun);
    this.scene.add(this.sun.target);
    this.configureSun();
    const bounce = new THREE.DirectionalLight(0x9fb8d8, 0.35);
    bounce.position.set(-span * 0.4, span * 0.25, -span * 0.5);
    this.scene.add(bounce);
    this.scene.add(new THREE.AxesHelper(span * 0.1));

    // terrain mesh (geometry rebuilt from the logical grid)
    const geo = new THREE.PlaneGeometry(
      this.terrain.sizeM, this.terrain.sizeM,
      this.terrain.width, this.terrain.height);
    this.terrainMesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({
      color: 0x6f8c56, roughness: 1.0, metalness: 0,
    }));
    this.terrainMesh.rotation.x = -Math.PI / 2;
    this.terrainMesh.receiveShadow = true;
    this.scene.add(this.terrainMesh);

    this.gridHelper = new THREE.GridHelper(
      this.terrain.sizeM, this.terrain.width, 0x223344, 0x1b2836);
    this.gridHelper.position.y = 0.05;
    this.scene.add(this.gridHelper);

    // Water vertices share terrain ordering, allowing direct bulk frame updates.
    const waterGeometry = new THREE.PlaneGeometry(this.terrain.sizeM, this.terrain.sizeM,
                                                  this.terrain.width, this.terrain.height);
    const sourceIndices = waterGeometry.index!.array;
    this.waterBaseIndices = sourceIndices instanceof Uint32Array
      ? new Uint32Array(sourceIndices) : new Uint16Array(sourceIndices);
    this.waterDynamicIndices = sourceIndices instanceof Uint32Array
      ? new Uint32Array(sourceIndices.length) : new Uint16Array(sourceIndices.length);
    waterGeometry.setIndex(new THREE.BufferAttribute(this.waterDynamicIndices, 1));
    this.attachWaterAttributes(waterGeometry);
    this.waterMesh = new THREE.Mesh(waterGeometry, this.buildWaterMaterial());
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

  /**
   * Place the sun and size its shadow frustum from the world, never from fixed
   * metres.
   *
   * The old `sun.position.set(60, 90, 30)` was the one thing in the constructor
   * that did NOT follow `terrain.sizeM`, and it survived only because a light
   * with no shadow does not care where it is -- direction is all that matters.
   * The moment it casts, position and frustum both matter: at the 200 m map a
   * default ortho shadow camera clips everything past ~5 m from the origin, so
   * the map would have come back with a rectangle of shadow in the middle and
   * nothing outside it. That is the same class of bug the v0.7.0 comments in
   * `rebuildGridGeometry` record, which is why this is a method called from
   * both places rather than two copies of the numbers.
   */
  private configureSun(): void {
    const span = this.terrain.sizeM;
    this.sun.position.set(span * 0.45, span * 0.75, span * 0.30);
    this.sun.target.position.set(0, 0, 0);
    this.sun.target.updateMatrixWorld();
    // Half-width covers the map diagonal (0.707 * span) with margin for tall
    // objects leaning their shadows outward.
    const half = span * 0.78;
    const cam = this.sun.shadow.camera;
    cam.left = -half; cam.right = half;
    cam.top = half; cam.bottom = -half;
    cam.near = span * 0.05;
    cam.far = span * 2.5;
    cam.updateProjectionMatrix();
    // 4096 over a 312 m frustum is ~7.6 cm per texel on the 200 m map -- enough
    // for a 4 m house to keep a recognisable shadow instead of a blob.
    this.sun.shadow.mapSize.set(4096, 4096);
    // normalBias, not bias: the terrain is a deformed height field, so slope
    // acne has to be fixed along the surface normal or steep banks stripe.
    this.sun.shadow.normalBias = 0.06;
    this.sun.shadow.bias = -0.0004;
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
  /**
   * Apply a terrain grid to the scene.
   *
   * If the grid RESOLUTION changed -- a different world size, a loaded world
   * built at another scale -- the meshes are rebuilt, not just re-heighted.
   * Before v0.7.0 this method only rewrote vertex Z, which worked purely
   * because the frontend's default TerrainGrid happened to match the backend's.
   * Doubling the map to 200 m exposed it: the backend streamed 40 401 heights
   * into a 10 201-vertex mesh, `setWaterHeights` rejected every frame on the
   * size check, and the water simply never appeared.
   */
  rebuildTerrain(terrain: TerrainGrid): void {
    const resized = terrain.width !== this.terrain.width
      || terrain.height !== this.terrain.height
      || terrain.cellSize !== this.terrain.cellSize;
    this.terrain = terrain;
    if (resized) this.rebuildGridGeometry();
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

  /**
   * Water material driven by the REAL velocity field, not a painted flow map.
   *
   * Every off-the-shelf option considered -- THREE.Water, FFT oceans, painted
   * flow-map textures -- assumes a flat plane with invented motion. This project
   * has the opposite situation: a deforming height-field mesh carrying a real
   * `u/v` field from the Warp solver. So the flow map is streamed from physics
   * (FrameKind.VELOCITY_FIELD, which existed in the protocol from the start and
   * had never been sent) and everything visual is derived from it:
   *
   * - ripples travel ALONG the direction the water actually moves, at a rate set
   *   by how fast it actually moves, so still water is visibly still;
   * - foam appears where the flow is genuinely fast or genuinely shallow-and-
   *   fast, which is where white water forms -- around piers, over rocks, along
   *   a wave front -- rather than wherever a texture happened to be painted;
   * - colour deepens with real depth, so a shallow margin reads as shallow.
   *
   * The consequence worth stating: if the physics is wrong, this looks wrong.
   * That is the point. A prettier shader that hid the physics would be the
   * decorative water docs/01_vision.md explicitly rules out.
   */
  private buildWaterMaterial(): THREE.MeshStandardMaterial {
    const material = new THREE.MeshStandardMaterial({
      color: 0x2f7fd0, transparent: true, opacity: 0.45,
      roughness: 0.2, metalness: 0.1, depthWrite: false, side: THREE.DoubleSide,
    });
    material.onBeforeCompile = (shader) => {
      shader.uniforms.uTime = this.waterTime;
      shader.vertexShader = shader.vertexShader
        .replace('#include <common>', `
          #include <common>
          attribute vec2 aFlow;
          attribute float aDepth;
          varying vec2 vFlow;
          varying float vDepth;
          varying float vSpeed;
        `)
        .replace('#include <begin_vertex>', `
          #include <begin_vertex>
          vFlow = aFlow;
          vDepth = aDepth;
          vSpeed = length(aFlow);
        `);
      shader.fragmentShader = shader.fragmentShader
        .replace('#include <common>', `
          #include <common>
          uniform float uTime;
          varying vec2 vFlow;
          varying float vDepth;
          varying float vSpeed;

          // cheap value noise, enough for surface texture at this scale
          float hash(vec2 p) {
            return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
          }
          float noise(vec2 p) {
            vec2 i = floor(p);
            vec2 f = fract(p);
            f = f * f * (3.0 - 2.0 * f);
            return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x),
                       mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y);
          }
        `)
        .replace('#include <dithering_fragment>', `
          #include <dithering_fragment>
          // Ripples advected along the real current: the sample point is pushed
          // backwards along the flow, so the pattern travels downstream at the
          // water's own speed. Still water gets a still surface, for free.
          vec2 world = vec2(vViewPosition.x, vViewPosition.z);
          vec2 drift = vFlow * uTime * 0.6;
          float ripple = noise(gl_FragCoord.xy * 0.05 - drift * 4.0)
                       + 0.5 * noise(gl_FragCoord.xy * 0.11 + drift * 2.0);
          // Foam where the water is genuinely fast, and more of it where fast
          // water is also shallow -- that is where white water actually breaks.
          float shallow = 1.0 - smoothstep(0.05, 0.6, vDepth);
          float foam = smoothstep(0.55, 1.9, vSpeed) * (0.45 + 0.55 * shallow);
          foam *= smoothstep(0.55, 1.15, ripple);
          gl_FragColor.rgb = mix(gl_FragColor.rgb, vec3(0.92, 0.96, 1.0), foam * 0.85);
          // depth colour: shallow margins read lighter and greener than the
          // channel, which is what makes a river's shape legible from above
          float deep = smoothstep(0.0, 1.6, vDepth);
          gl_FragColor.rgb *= mix(1.28, 0.82, deep);
          gl_FragColor.rgb += vec3(0.0, 0.05, 0.02) * (1.0 - deep);
          gl_FragColor.a = clamp(gl_FragColor.a + foam * 0.5 + 0.12 * (1.0 - deep), 0.0, 1.0);
        `);
    };
    return material;
  }

  /** Rebuild every piece of geometry whose resolution follows the grid. */
  private rebuildGridGeometry(): void {
    const { sizeM, width, height } = this.terrain;

    this.terrainMesh.geometry.dispose();
    this.terrainMesh.geometry = new THREE.PlaneGeometry(sizeM, sizeM, width, height);

    const waterGeometry = new THREE.PlaneGeometry(sizeM, sizeM, width, height);
    const sourceIndices = waterGeometry.index!.array;
    // Mirror whatever index width Three.js chose rather than assuming one.
    // 201x201 = 40 401 vertices still fits Uint16 (max 65 535), so the doubled
    // map did NOT need the 32-bit path -- but 401x401 would, and hard-coding
    // Uint16 here is exactly the kind of thing that fails silently later.
    this.waterBaseIndices = sourceIndices instanceof Uint32Array
      ? new Uint32Array(sourceIndices) : new Uint16Array(sourceIndices);
    this.waterDynamicIndices = sourceIndices instanceof Uint32Array
      ? new Uint32Array(sourceIndices.length) : new Uint16Array(sourceIndices.length);
    waterGeometry.setIndex(new THREE.BufferAttribute(this.waterDynamicIndices, 1));
    waterGeometry.setDrawRange(0, 0);
    this.attachWaterAttributes(waterGeometry);
    this.waterMesh.geometry.dispose();
    this.waterMesh.geometry = waterGeometry;

    this.scene.remove(this.gridHelper);
    this.gridHelper.geometry.dispose();
    this.gridHelper = new THREE.GridHelper(sizeM, width, 0x223344, 0x1b2836);
    this.gridHelper.position.y = 0.05;
    this.scene.add(this.gridHelper);

    // camera framing, fog and the sun's shadow frustum follow the world, see
    // the constructor note and configureSun()
    this.configureSun();
    this.scene.fog = new THREE.Fog(0x0e1420, sizeM * 1.2, sizeM * 4);
    this.camera.far = sizeM * 6;
    this.camera.position.set(sizeM * 0.6, sizeM * 0.55, sizeM * 0.6);
    this.camera.updateProjectionMatrix();
    this.controls.maxDistance = sizeM * 3;
    this.controls.target.set(0, 0, 0);
    this.controls.update();
  }

  private attachWaterAttributes(geometry: THREE.BufferGeometry): void {
    const count = geometry.attributes.position.count;
    this.waterFlow = new Float32Array(count * 2);
    this.waterDepth = new Float32Array(count);
    geometry.setAttribute('aFlow', new THREE.BufferAttribute(this.waterFlow, 2));
    geometry.setAttribute('aDepth', new THREE.BufferAttribute(this.waterDepth, 1));
  }

  /**
   * Apply a streamed VELOCITY_FIELD frame. Values are the solver's real u/v in
   * m/s, in terrain-vertex order -- the same ordering as the height frame.
   */
  setVelocityField(values: Float32Array, count: number): boolean {
    const attribute = this.waterMesh.geometry.attributes.aFlow as THREE.BufferAttribute;
    if (!attribute || count !== attribute.count) return false;
    for (let i = 0; i < count; i++) {
      // the field arrives as vec3 (u, 0, v); the shader only needs the plane
      this.waterFlow[i * 2] = values[i * 3];
      this.waterFlow[i * 2 + 1] = values[i * 3 + 2];
    }
    attribute.needsUpdate = true;
    this.updateSpray(values, count);
    return true;
  }

  /**
   * Spray at the genuinely violent places, picked from the real fields.
   *
   * A cell qualifies when it is fast AND shallow -- that is where water breaks
   * white in reality: over a submerged boulder, between bridge piers, along an
   * advancing front. Nothing is emitted for a broad slow flood however large it
   * is, which is correct and is the difference between this and a particle
   * effect sprinkled over the whole surface.
   */
  private updateSpray(velocities: Float32Array, count: number): void {
    if (!this.sprayPoints) {
      this.sprayPositions = new Float32Array(SceneManager.MAX_SPRAY * 3);
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position',
        new THREE.BufferAttribute(this.sprayPositions, 3));
      geometry.setDrawRange(0, 0);
      this.sprayPoints = new THREE.Points(geometry, new THREE.PointsMaterial({
        color: 0xffffff, size: 0.45, sizeAttenuation: true,
        transparent: true, opacity: 0.75, depthWrite: false,
      }));
      this.sprayPoints.frustumCulled = false;
      this.scene.add(this.sprayPoints);
    }
    const heightAttribute = this.waterMesh.geometry.attributes.position as THREE.BufferAttribute;
    const grid = this.terrain.width + 1;
    const half = this.terrain.sizeM / 2;
    const cell = this.terrain.cellSize;
    let emitted = 0;
    // stride keeps the scan cheap and the sampling even; the phase walks so the
    // spray shimmers instead of sitting on the same vertices every frame
    const stride = 7;
    const phase = (this.sprayPhase = (this.sprayPhase + 1) % stride);
    for (let i = phase; i < count && emitted < SceneManager.MAX_SPRAY; i += stride) {
      const depth = this.waterDepth[i];
      if (depth <= 0.02 || depth > 0.9) continue;
      const u = velocities[i * 3];
      const v = velocities[i * 3 + 2];
      if (u * u + v * v < SceneManager.SPRAY_SPEED_SQ) continue;
      const gx = i % grid;
      const gz = (i / grid) | 0;
      this.sprayPositions[emitted * 3] = gx * cell - half + (Math.random() - 0.5) * cell;
      this.sprayPositions[emitted * 3 + 1] = heightAttribute.getZ(i) + 0.05 + Math.random() * 0.3;
      this.sprayPositions[emitted * 3 + 2] = gz * cell - half + (Math.random() - 0.5) * cell;
      emitted++;
    }
    (this.sprayPoints.geometry.attributes.position as THREE.BufferAttribute).needsUpdate = true;
    this.sprayPoints.geometry.setDrawRange(0, emitted);
    this.sprayPoints.visible = this.tracersVisible && emitted > 0;
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
      const depth = heights[i] - this.terrain.heights[i];
      this.waterDepth[i] = depth > 0 ? depth : 0;
      wet[i] = depth > 1e-4 ? 1 : 0;
    }
    pos.needsUpdate = true;
    const depthAttribute = this.waterMesh.geometry.attributes.aDepth as THREE.BufferAttribute;
    if (depthAttribute) depthAttribute.needsUpdate = true;
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
    if (group) {
      this.objectsRoot.remove(group);
      SceneManager.disposeSubtree(group);
    }
    this.clearSelection();
  }

  clearObjects(): void {
    for (const child of this.objectsRoot.children) SceneManager.disposeSubtree(child);
    this.objectsRoot.clear();
    this.clearSelection();
  }

  /**
   * Release the GPU resources of a discarded object subtree.
   *
   * Removing a group from its parent does not free anything, which cost nothing
   * while every builder shared a handful of module-scope materials. Since
   * v0.12.1 that is no longer true: builders allocate per-instance materials so
   * each object can be tinted from its own id, and ROCK/DEBRIS allocate
   * per-instance deformed geometry. `applyWorld()` calls `clearObjects()` and
   * rebuilds the entire scene on every world replace -- every LOAD, every
   * reconnect -- so without this each reload would strand the previous set.
   */
  private static disposeSubtree(root: THREE.Object3D): void {
    root.traverse((node) => {
      const mesh = node as THREE.Mesh;
      if (!mesh.isMesh) return;
      mesh.geometry?.dispose();
      const material = mesh.material;
      if (Array.isArray(material)) material.forEach((m) => m.dispose());
      else material?.dispose();
    });
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
    // drives the advected ripple pattern in the water shader
    this.waterTime.value = (performance.now() - this._clockStart) / 1000;
    this.controls.update();
    if (this._selectionHelper) this._selectionHelper.update();
    this.renderer.render(this.scene, this.camera);
  }
}
