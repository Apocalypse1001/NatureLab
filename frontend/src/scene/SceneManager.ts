/** Three.js scene: renderer, camera, controls, lights, terrain mesh,
 *  dynamic water surface, particle points, object meshes. */
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { TerrainGrid } from '../world/TerrainGrid';
import { RainField } from './RainField';
import { EdgeSkirt, type EdgeWaterMode } from './EdgeSkirt';
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

  private composer: EffectComposer;
  private bloomPass: UnrealBloomPass;
  private skyDome: THREE.Mesh;
  private groundTexture: THREE.CanvasTexture;
  private raycaster = new THREE.Raycaster();
  private _selectionHelper: THREE.BoxHelper | null = null;
  private waterBaseIndices: Uint16Array | Uint32Array;
  private waterDynamicIndices: Uint16Array | Uint32Array;
  private gridHelper: THREE.GridHelper;
  private waterFlow = new Float32Array(0);
  private waterDepth = new Float32Array(0);
  private waterLavaTemp = new Float32Array(0);
  private sprayPoints: THREE.Points | null = null;
  private sprayPositions = new Float32Array(0);
  private sprayPhase = 0;
  private static readonly MAX_SPRAY = 4000;
  // (1.4 m/s)^2 -- below this the surface stays unbroken
  private static readonly SPRAY_SPEED_SQ = 1.96;
  private static readonly CHAR_COLOR = new THREE.Color(0x0a0806);
  private static readonly EMBER_COLOR = new THREE.Color(0xff4010);
  // Flow-tracer colours: the ordinary water droplet tint, and what it turns
  // into when the same tracer is riding over lava instead -- see
  // setParticles/lavaTempAt. Below LAVA_TRACER_MIN_C nothing changes, so a
  // river or dam world (aLavaTemp always 0) never pays for this or sees it.
  private static readonly TRACER_WATER_COLOR = new THREE.Color(0xbde9ff);
  private static readonly TRACER_SPARK_COLOR = new THREE.Color(0xffb347);
  private static readonly LAVA_TRACER_MIN_C = 650.0;
  private static readonly LAVA_TRACER_MAX_C = 1100.0;
  private tracerColors = new Float32Array(0);
  private waterTime = { value: 0 };
  private _clockStart = performance.now();
  private tracersVisible = true;
  private tracerDisplayLimit = 36000;   // matches config.FLOW_TRACER_COUNT
  private receivedTracerCount = 0;
  // RainLab-1: streaks drawn from the intensity the solver reports applying
  private rain = new RainField();
  private rainRunning = false;
  private lastRenderMs = performance.now();
  // v0.16.0: terrain and water continued past the map edge, drawn only
  private edgeSkirt: EdgeSkirt;

  constructor(canvas: HTMLCanvasElement, private terrain: TerrainGrid) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x0e1420);
    // Shadows are the one cheap thing that makes an object look like it is ON
    // the terrain rather than floating in front of it. Nothing else in this
    // pass changes what the user can read off the scene as much as this does.
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    // Filmic response instead of the flat linear default: highlights (lava
    // glow, sun-lit glass) roll off instead of clipping to flat white, and
    // mid-tones keep separation instead of crushing together.
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.1;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    // EffectComposer.render() calls renderer.render() once per pass, and each
    // call resets `info` when autoReset is on -- so by the time render() below
    // returns, `info.render` only reflects the LAST internal pass (the final
    // full-screen quad: 1 triangle, 1 draw call), not the scene. The driver's
    // "did it draw" smoke signal, and any future agent reading it, depends on
    // this staying a real triangle/draw-call count.
    this.renderer.info.autoReset = false;

    // A neutral studio environment, not a photo HDRI: every MeshStandardMaterial
    // (car glazing, window panes, metal hubs) picks up SOME reflection instead
    // of looking like flat-shaded paint, without importing a texture asset or
    // implying a specific sky that fights the fog/sun colours below.
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    this.scene.environmentIntensity = 0.6;
    pmrem.dispose();

    // Camera framing, fog and zoom limits are derived from the world size, not
    // fixed numbers: they were tuned for a 100 m map and left the 200 m map
    // half out of frame and inside the fog when it was doubled in v0.7.0.
    const span = terrain.sizeM;
    this.scene.fog = new THREE.Fog(0x0e1420, span * 1.2, span * 4);

    // A gradient dome, not the flat clear-colour backdrop the renderer painted
    // before: at ground level (the camera height a child would actually use)
    // a single flat colour reads as a wall behind the houses, not a sky. The
    // horizon stop matches the fog colour exactly so the dome and the
    // fog-swallowed terrain hand off with no seam.
    this.skyDome = this.buildSkyDome();
    this.scene.add(this.skyDome);
    this.updateSkyDome(span);

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
    // A speckled canvas texture, not a flat fill: up close (a child's eye
    // height) one solid green reads as a floor, not grass. Mottling is cheap
    // -- baked once, tiled by repeat -- and keeps the same low-poly toy read
    // the rest of the scene has, rather than importing a photo texture.
    this.groundTexture = this.buildGroundTexture();
    this.terrainMesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({
      map: this.groundTexture, roughness: 1.0, metalness: 0,
    }));
    this.updateGroundTextureRepeat(span);
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

    this.edgeSkirt = new EdgeSkirt(this.groundTexture, this.buildWaterMaterial());
    this.edgeSkirt.rebuild(this.terrain);
    this.scene.add(this.edgeSkirt.group);

    // particle points buffer (filled from backend binary frames)
    const positions = new Float32Array(1024 * 3);
    const pgeo = new THREE.BufferGeometry();
    pgeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    // VolcanoLab v0.14.0: per-particle colour, so a tracer riding the flow
    // over hot lava reads as a spark rather than a water droplet -- driven by
    // aLavaTemp's own grid (see setParticles/lavaTempAt), never a second
    // "is this a volcano" flag. Material colour is plain white so the vertex
    // colour attribute IS the colour, not a tint on top of one.
    this.tracerColors = new Float32Array(1024 * 3).fill(1);
    pgeo.setAttribute('color', new THREE.BufferAttribute(this.tracerColors, 3));
    pgeo.setDrawRange(0, 0);
    this.points = new THREE.Points(pgeo, new THREE.PointsMaterial({
      color: 0xffffff, vertexColors: true, size: 0.14, sizeAttenuation: true,
      transparent: true, opacity: 0.8, depthWrite: false,
    }));
    this.points.frustumCulled = false;
    this.scene.add(this.points);
    this.scene.add(this.rain.lines);

    this.scene.add(this.objectsRoot);

    // Bloom only, kept deliberately subtle: threshold above the brightness any
    // ordinary sunlit surface reaches post-tonemap, so it catches the genuinely
    // emissive things -- the VENT throat, SOURCE/DRAIN rings, car lamps -- as a
    // glow, without haloing every white wall the sun hits.
    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.bloomPass = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.22, 0.2, 1.0);
    this.composer.addPass(this.bloomPass);
    this.composer.addPass(new OutputPass());

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
    this.composer.setSize(w, h);
    this.composer.setPixelRatio(this.renderer.getPixelRatio());
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
    // A loaded world may not be a volcano; a stale glow from the previous one
    // would otherwise linger since LAVA_TEMPERATURE frames simply stop
    // arriving rather than sending zeros (see simulation.py's lava_active
    // gate). rebuildGridGeometry already zeroes this via attachWaterAttributes
    // when the grid resizes, so this only has work to do on same-size reloads.
    if (!resized) {
      const attribute = this.waterMesh.geometry.attributes.aLavaTemp as THREE.BufferAttribute;
      if (attribute) {
        this.waterLavaTemp.fill(0);
        attribute.needsUpdate = true;
      }
    }
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
    this.edgeSkirt.rebuild(terrain);
    // The helper grid sits at y = 0.05, i.e. under the ground of any map whose
    // terrain stays above it and on top of the ground of a flat one -- but
    // floating on the SEA of a coast, where the tsunami world showed it as a
    // black mesh of 10 m lines over the whole sea (measured: hiding it was the
    // only change that removed them). Where the ground dips below it, hide it.
    let lowest = Infinity;
    for (let i = 0; i < terrain.heights.length; i++) lowest = Math.min(lowest, terrain.heights[i]);
    this.gridHelper.visible = lowest >= this.gridHelper.position.y - 0.05;
  }

  /** What the water does past the east outlet, the west inflow and the north/south walls. */
  setEdgeWater(east: EdgeWaterMode, west: EdgeWaterMode, sides: EdgeWaterMode): void {
    this.edgeSkirt.setWaterModes(east, west, sides);
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
          attribute float aLavaTemp;
          varying vec2 vFlow;
          varying float vDepth;
          varying float vSpeed;
          varying float vLavaTemp;
        `)
        .replace('#include <begin_vertex>', `
          #include <begin_vertex>
          vFlow = aFlow;
          vDepth = aDepth;
          vSpeed = length(aFlow);
          vLavaTemp = aLavaTemp;
        `);
      shader.fragmentShader = shader.fragmentShader
        .replace('#include <common>', `
          #include <common>
          uniform float uTime;
          varying vec2 vFlow;
          varying float vDepth;
          varying float vSpeed;
          varying float vLavaTemp;

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

          // VolcanoLab (v0.13.0): colour AS a function of the streamed
          // temperature field, not a painted "lava texture" -- the same
          // principle buildWaterMaterial's docstring states for the flow map.
          // Thresholds mirror config.py: LAVA_SOLIDUS_TEMP_C = 980,
          // LAVA_ERUPTION_TEMP_C = 1150. Below ~700C nothing glows (real
          // basalt stops visibly radiating well before it solidifies), so a
          // cooling flow darkens itself all the way to black rock -- it is not
          // faded out or hidden, the temperature field just stops producing
          // colour for it.
          vec3 lavaColor(float tC) {
            vec3 dark   = vec3(0.05, 0.01, 0.01);
            vec3 cherry = vec3(0.55, 0.07, 0.02);
            vec3 orange = vec3(0.95, 0.35, 0.05);
            vec3 white  = vec3(1.00, 0.95, 0.65);
            vec3 c = mix(dark, cherry, smoothstep(700.0, 980.0, tC));
            c = mix(c, orange, smoothstep(980.0, 1080.0, tC));
            c = mix(c, white, smoothstep(1080.0, 1150.0, tC));
            return c;
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
          // RainLab-1: a film is not a flood. Rain wets every cell past the
          // solver's 0.1 mm dry threshold within seconds, and the lines above
          // make shallow water MORE opaque, so 45 s of a downpour (a 0.6 mm
          // sheet) drew the whole map as standing water. Real water under a
          // millimetre over ground is barely visible; fade it in between 1 mm
          // and 1 cm. Lava is untouched -- lavaMix below forces alpha to 1.
          gl_FragColor.a *= smoothstep(0.001, 0.01, vDepth);
          // Lava overrides the water treatment above entirely rather than
          // tinting it: foam and blue depth-shading are real-water phenomena
          // that mean nothing for a viscous melt. vLavaTemp is 0 for every
          // frame that never received a LAVA_TEMPERATURE stream, so this is a
          // no-op for every world that isn't a volcano.
          float lavaMix = smoothstep(650.0, 850.0, vLavaTemp);
          vec3 hotColor = lavaColor(vLavaTemp);
          // Crust cracks: thin dark veins, visible only in the crusting band
          // just below the solidus -- a real flow's surface only fractures
          // once it starts hardening, so fully molten (>1040C) and already-
          // cold (<920C) read smooth. Built from the same value noise as the
          // water ripples above rather than screen-space derivatives, which
          // this software-rendered target cannot be relied on to support.
          float crustBand = smoothstep(920.0, 980.0, vLavaTemp)
                           * (1.0 - smoothstep(980.0, 1040.0, vLavaTemp));
          float crackNoise = noise(gl_FragCoord.xy * 0.22 + vLavaTemp * 0.015);
          float crack = smoothstep(0.47, 0.5, crackNoise) * crustBand;
          hotColor = mix(hotColor, vec3(0.03, 0.01, 0.01), crack);
          gl_FragColor.rgb = mix(gl_FragColor.rgb, hotColor, lavaMix);
          gl_FragColor.a = mix(gl_FragColor.a, 1.0, lavaMix);
        `);
    };
    return material;
  }

  /**
   * A vertex-coloured sphere, seen from inside (`BackSide`), standing in for
   * a sky. `MeshBasicMaterial` with `fog: false`: it must ignore the scene's
   * own fog, since it IS the thing the fog fades into, not an object the fog
   * should dim as if it had depth.
   */
  private buildSkyDome(): THREE.Mesh {
    const geometry = new THREE.SphereGeometry(1, 24, 16);
    const position = geometry.attributes.position as THREE.BufferAttribute;
    const colors = new Float32Array(position.count * 3);
    const zenith = new THREE.Color(0x25446b);
    const horizon = new THREE.Color(0x0e1420);   // matches scene.fog exactly
    const c = new THREE.Color();
    // Steep on purpose: from eye height, most of what a camera frames is
    // within a few degrees of the horizon (y near 0), not the zenith. A
    // gradient that only finished at y=0.65 stayed flat across that whole
    // band and read as no gradient at all.
    for (let i = 0; i < position.count; i++) {
      const t = THREE.MathUtils.smoothstep(position.getY(i), -0.05, 0.3);
      c.copy(horizon).lerp(zenith, t);
      colors.set([c.r, c.g, c.b], i * 3);
    }
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    const material = new THREE.MeshBasicMaterial({
      vertexColors: true, side: THREE.BackSide, fog: false, depthWrite: false,
    });
    const dome = new THREE.Mesh(geometry, material);
    dome.renderOrder = -1;
    return dome;
  }

  /** Keep the dome outside `controls.maxDistance` but inside `camera.far`. */
  private updateSkyDome(span: number): void {
    this.skyDome.scale.setScalar(span * 4.5);
  }

  /**
   * A small tileable patch of mottled grass, baked once on a canvas.
   *
   * Splotches, not speckle: a first version scattered 5000 tiny
   * semi-transparent dots, which is exactly the setup that averages itself
   * back into a flat colour -- with that many independent low-alpha samples
   * overlapping, the law of large numbers wins and the result reads as
   * uniform again. A few dozen soft, larger, more opaque blobs stay visible
   * as patches instead of resolving into a new flat shade.
   */
  private buildGroundTexture(): THREE.CanvasTexture {
    const size = 512;
    const canvas = document.createElement('canvas');
    canvas.width = size; canvas.height = size;
    const ctx = canvas.getContext('2d')!;
    ctx.fillStyle = '#6f8c56';
    ctx.fillRect(0, 0, size, size);
    for (let i = 0; i < 90; i++) {
      const shade = 0.72 + Math.random() * 0.56;
      const r = Math.min(255, Math.round(111 * shade));
      const g = Math.min(255, Math.round(140 * shade));
      const b = Math.min(255, Math.round(86 * shade));
      const x = Math.random() * size, y = Math.random() * size;
      const radius = 22 + Math.random() * 46;
      const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius);
      gradient.addColorStop(0, `rgba(${r},${g},${b},0.45)`);
      gradient.addColorStop(1, `rgba(${r},${g},${b},0)`);
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();
    }
    // Fine dark speckle on top for close-up grain, kept sparse enough that it
    // does not average itself out the way the first attempt did.
    for (let i = 0; i < 700; i++) {
      ctx.fillStyle = `rgba(30,38,20,${0.08 + Math.random() * 0.1})`;
      const x = Math.random() * size, y = Math.random() * size;
      ctx.fillRect(x, y, 1.5, 1.5);
    }
    const texture = new THREE.CanvasTexture(canvas);
    texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
    texture.colorSpace = THREE.SRGBColorSpace;
    return texture;
  }

  /** One tile roughly every 10 m, regardless of the map's physical size. */
  private updateGroundTextureRepeat(sizeM: number): void {
    const tiles = Math.max(1, Math.round(sizeM / 10));
    this.groundTexture.repeat.set(tiles, tiles);
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
    this.updateSkyDome(sizeM);
    this.updateGroundTextureRepeat(sizeM);
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
    this.waterLavaTemp = new Float32Array(count);
    geometry.setAttribute('aFlow', new THREE.BufferAttribute(this.waterFlow, 2));
    geometry.setAttribute('aDepth', new THREE.BufferAttribute(this.waterDepth, 1));
    geometry.setAttribute('aLavaTemp', new THREE.BufferAttribute(this.waterLavaTemp, 1));
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

  /**
   * Apply a streamed LAVA_TEMPERATURE frame (degrees C, terrain-vertex order,
   * same grid as WATER_HEIGHT). Never sent for a lava-less world, so a river
   * or dam scene simply never calls this and the shader's ramp stays at its
   * cold default -- see buildWaterMaterial.
   */
  setLavaTemperature(values: Float32Array, count: number): boolean {
    const attribute = this.waterMesh.geometry.attributes.aLavaTemp as THREE.BufferAttribute;
    if (!attribute || count !== attribute.count) return false;
    this.waterLavaTemp.set(values.subarray(0, count));
    attribute.needsUpdate = true;
    return true;
  }

  setWater(level: number, visible: boolean): void {
    this.waterMesh.position.y = 0;
    this.waterMesh.visible = visible;
    const pos = this.waterMesh.geometry.attributes.position as THREE.BufferAttribute;
    for (let i = 0; i < pos.count; i++) pos.setZ(i, level);
    pos.needsUpdate = true;
    this.waterMesh.geometry.setDrawRange(0, 0);
    this.edgeSkirt.setWaterVisible(visible);
    this.edgeSkirt.clearWater();
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
    this.edgeSkirt.updateWater(heights, this.waterFlow);
    return true;
  }

  /**
   * Lava temperature (C) at a world (x, z), same grid `aLavaTemp`/`waterDepth`
   * use. 0 (below every ignition/spark threshold) whenever no LAVA_TEMPERATURE
   * frame has ever arrived, off the map, or outside the terrain -- a river or
   * dam world never sees a spark-coloured tracer.
   */
  private lavaTempAt(x: number, z: number): number {
    if (!this.waterLavaTemp.length) return 0;
    const grid = this.terrain.width + 1;
    const half = this.terrain.sizeM / 2;
    const gx = Math.round((x + half) / this.terrain.cellSize);
    const gz = Math.round((z + half) / this.terrain.cellSize);
    if (gx < 0 || gx >= grid || gz < 0 || gz >= grid) return 0;
    return this.waterLavaTemp[gz * grid + gx];
  }

  private static readonly _scratchColor = new THREE.Color();

  setParticles(buffer: Float32Array, count: number): void {
    let attr = this.points.geometry.attributes.position as THREE.BufferAttribute;
    let colorAttr = this.points.geometry.attributes.color as THREE.BufferAttribute;
    if (attr.count < count) {
      let capacity = Math.max(1024, attr.count);
      while (capacity < count) capacity *= 2;
      attr = new THREE.BufferAttribute(new Float32Array(capacity * 3), 3);
      this.points.geometry.setAttribute('position', attr);
      this.tracerColors = new Float32Array(capacity * 3).fill(1);
      colorAttr = new THREE.BufferAttribute(this.tracerColors, 3);
      this.points.geometry.setAttribute('color', colorAttr);
    }
    attr.array.set(buffer.subarray(0, count * 3));
    // VolcanoLab v0.14.0: a tracer riding the flow over hot lava becomes a
    // spark, not a water droplet -- driven by the same aLavaTemp grid the
    // surface shader reads, so this is a genuine physics readout (where is
    // it hot) rather than a decorative particle system bolted on separately.
    const positions = attr.array as Float32Array;
    const colors = this.tracerColors;
    const time = this.waterTime.value;
    for (let i = 0; i < count; i++) {
      const px = positions[i * 3], pz = positions[i * 3 + 2];
      const tempC = this.lavaTempAt(px, pz);
      const mix = THREE.MathUtils.clamp(
        (tempC - SceneManager.LAVA_TRACER_MIN_C)
          / (SceneManager.LAVA_TRACER_MAX_C - SceneManager.LAVA_TRACER_MIN_C), 0, 1);
      SceneManager._scratchColor.copy(SceneManager.TRACER_WATER_COLOR)
        .lerp(SceneManager.TRACER_SPARK_COLOR, mix);
      colors[i * 3] = SceneManager._scratchColor.r;
      colors[i * 3 + 1] = SceneManager._scratchColor.g;
      colors[i * 3 + 2] = SceneManager._scratchColor.b;
      // A spark also rises: a small time-varying lift on top of the real
      // advected position, recomputed fresh from the authoritative backend
      // Y every frame (never accumulated), so still water stays exactly
      // where the solver put it.
      if (mix > 0) {
        positions[i * 3 + 1] += (0.4 + 0.5 * Math.sin(time * 2.3 + i)) * mix;
      }
    }
    attr.needsUpdate = true;
    colorAttr.needsUpdate = true;
    this.receivedTracerCount = count;
    this.applyTracerDisplay();
  }

  /**
   * RainLab-1: intensity in mm/h as streamed in `fluid.rain_mm_h`. `running`
   * false freezes the streaks -- a paused simulation is not raining.
   */
  setRain(mmPerHour: number, running: boolean): void {
    this.rain.setIntensity(mmPerHour);
    this.rainRunning = running;
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
  /**
   * Returns true iff a pre-existing group was torn down and replaced (as
   * opposed to a brand-new object, or an existing one just moved/scaled).
   * A caller that has a TransformControls gizmo attached by object identity
   * (EditorController) needs this: `attach()`'d to the old THREE.Object3D
   * that just got disposed, the gizmo does not silently follow the id to the
   * new one -- it starts warning every frame that its target has left the
   * scene graph. See main.ts's updateObject callback.
   */
  setObject(obj: ObjectData): boolean {
    let group = this.objectsRoot.getObjectByName(obj.id) as THREE.Group | undefined;
    let rebuilt = false;
    // BUILDING's geometry is parametric on obj.metadata.floors, not just its
    // transform -- unlike every other builder, so a floor-count edit in the
    // properties panel has to force a real rebuild, not just applyTransform.
    if (group && obj.type === 'BUILDING' && group.userData.floors !== obj.metadata.floors) {
      this.objectsRoot.remove(group);
      SceneManager.disposeSubtree(group);
      group = undefined;
      rebuilt = true;
    }
    if (!group) {
      group = buildObjectMesh(obj);
      if (obj.type === 'BUILDING') group.userData.floors = obj.metadata.floors;
      this.objectsRoot.add(group);
    }
    applyTransform(group, obj);
    this.applyDamageVisual(group, obj);
    if (this._selectionHelper && this._selectionHelper.userData.owner === obj.id) {
      this._selectionHelper.update();
    }
    return rebuilt;
  }

  /**
   * VolcanoLab v0.14.0: charring driven by `obj.damage` (0..1, from lava
   * contact -- see SimulationManager._check_lava_ignition), not a separate
   * "burnt" mesh swapped in. Every material on the object darkens toward
   * char and embers along the way, because that is the physical consequence
   * of the fire the backend already measured, not a decoration layered on
   * top of it -- the same "colour AS a function of the field" principle
   * buildWaterMaterial's lava ramp uses for T.
   *
   * Recomputed from a CACHED base colour every call, not the material's
   * current (possibly already-darkened) one: `setObject` runs every stream
   * tick, and blending toward char from whatever colour is already on screen
   * would compound every frame instead of tracking `obj.damage` as an
   * absolute value.
   */
  private applyDamageVisual(group: THREE.Group, obj: ObjectData): void {
    const damage = THREE.MathUtils.clamp(obj.damage, 0, 1);
    group.traverse((node) => {
      const mesh = node as THREE.Mesh;
      if (!mesh.isMesh) return;
      const material = mesh.material as THREE.MeshStandardMaterial;
      if (!material || !(material as { isMeshStandardMaterial?: boolean }).isMeshStandardMaterial) return;
      if (!mesh.userData.baseColor) {
        mesh.userData.baseColor = material.color.clone();
        mesh.userData.baseEmissive = material.emissive.clone();
        mesh.userData.baseEmissiveIntensity = material.emissiveIntensity;
      }
      const base = mesh.userData.baseColor as THREE.Color;
      const baseEmissive = mesh.userData.baseEmissive as THREE.Color;
      const baseEmissiveIntensity = mesh.userData.baseEmissiveIntensity as number;
      material.color.copy(base).lerp(SceneManager.CHAR_COLOR, damage);
      // Embers peak partway through burning and fade again as the surface
      // finishes going to cold char -- a bump, not a straight ramp to zero.
      const ember = damage > 0 ? Math.sin(Math.min(damage, 1) * Math.PI) : 0;
      material.emissive.copy(baseEmissive).lerp(SceneManager.EMBER_COLOR, ember);
      material.emissiveIntensity = baseEmissiveIntensity + ember * 1.4;
    });
    // BROKEN collapses the object -- a consequence read off the object's own
    // scale, not new geometry: a house that burned down is not a house.
    group.scale.y = obj.scale[1] * (obj.state === 'BROKEN' ? 0.12 : 1.0);
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
    const now = performance.now();
    this.waterTime.value = (now - this._clockStart) / 1000;
    this.controls.update();
    this.rain.update((now - this.lastRenderMs) / 1000, this.controls.target,
                     this.camera.position.distanceTo(this.controls.target),
                     this.terrain.sizeM, (x, z) => this.terrain.heightAt(x, z),
                     this.rainRunning);
    this.lastRenderMs = now;
    if (this._selectionHelper) this._selectionHelper.update();
    this.renderer.info.reset();
    this.composer.render();
  }
}
