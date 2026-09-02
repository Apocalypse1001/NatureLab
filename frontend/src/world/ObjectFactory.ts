/** Primitive meshes per object type. Extend the registry to add new types.
 *
 *  Every builder is a pure function of the object's data, and every one of them
 *  is bound by the same rule: the dimensions the solver uses are not the
 *  builder's to change. `rigid_body.footprint_half_extents` gives HOUSE 2.0x2.0
 *  and CAR 2.2x1.0; `fluid_solver` uses ROCK_BASE_RADIUS_M = 1.5 and
 *  BRIDGE_SPAN_M = 24.0 with piers at the ends and the middle. Detail is added
 *  strictly INSIDE those envelopes -- windows, planks, rails, hubs -- so the
 *  thing the user sees stays the thing the water feels.
 */
import * as THREE from 'three';
import { OBJECT_COLORS, type ObjectData } from './types';

type Builder = (obj: ObjectData) => THREE.Group;

/**
 * A stable pseudo-random number in [0, 1) per object and channel.
 *
 * Used only where nature is not uniform -- trunk lean, boulder yaw, foliage
 * tint. Derived from the object id so it never changes between frames, reloads
 * or a saved world: a forest that reshuffled itself on every reconnect would be
 * worse than a cloned one.
 */
function variant(id: string, channel = 0): number {
  let hash = 2166136261 ^ (channel * 0x9e3779b1);
  for (let i = 0; i < id.length; i++) {
    hash ^= id.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return ((hash >>> 0) % 100000) / 100000;
}

/**
 * Push every vertex of a convex primitive in or out along its own radius, by a
 * stable per-vertex amount. Turns an icosahedron into a boulder.
 *
 * The scale range must never exceed 1.0: for a ROCK the sphere it deforms is
 * exactly `fluid_solver.ROCK_BASE_RADIUS_M`, and a vertex pushed past that
 * would draw stone outside the raised-bed dome the water actually climbs.
 */
function roughened(geometry: THREE.BufferGeometry, seed: number,
                   low: number, high: number): THREE.BufferGeometry {
  const position = geometry.attributes.position as THREE.BufferAttribute;
  for (let i = 0; i < position.count; i++) {
    const x = position.getX(i), y = position.getY(i), z = position.getZ(i);
    // hash the vertex itself, so shared vertices move together and the hull
    // stays closed
    const noise = Math.abs(Math.sin((x * 12.99 + y * 78.23 + z * 37.72 + seed) * 43758.5)) % 1;
    const scale = low + (high - low) * noise;
    position.setXYZ(i, x * scale, y * scale, z * scale);
  }
  geometry.computeVertexNormals();
  return geometry;
}

/** Nudge a base colour by a small hue/lightness offset, kept subtle. */
function tinted(base: number, amount: number, seed: number): THREE.Color {
  const color = new THREE.Color(base);
  const hsl = { h: 0, s: 0, l: 0 };
  color.getHSL(hsl);
  return color.setHSL(
    (hsl.h + (seed - 0.5) * amount + 1) % 1,
    THREE.MathUtils.clamp(hsl.s + (seed - 0.5) * amount, 0, 1),
    THREE.MathUtils.clamp(hsl.l + (seed - 0.5) * amount * 0.8, 0.05, 0.95));
}

const GLASS = () => new THREE.MeshStandardMaterial({
  color: 0x3d6180, roughness: 0.12, metalness: 0.25,
  emissive: 0x101c28, emissiveIntensity: 0.6,
});
const RUBBER = () => new THREE.MeshStandardMaterial({ color: 0x1e1e22, roughness: 0.95 });
const METAL = () => new THREE.MeshStandardMaterial({
  color: 0x9aa3ad, roughness: 0.35, metalness: 0.7,
});

const builders: Record<string, Builder> = {
  HOUSE: (obj) => {
    // 4 x 4 m base, 3 m walls, pyramid roof to 5 m. The base is exactly the
    // 2.0 x 2.0 half-extent the obstacle rasterizer stamps into the solid mask,
    // so the plinth, door and windows all sit inside it.
    const g = new THREE.Group();
    const seed = variant(obj.id);
    const wall = new THREE.MeshStandardMaterial({
      color: tinted(OBJECT_COLORS.HOUSE, 0.10, seed), roughness: 0.85,
    });
    const trim = new THREE.MeshStandardMaterial({ color: 0xf1e7d6, roughness: 0.8 });
    const stone = new THREE.MeshStandardMaterial({ color: 0x8d8578, roughness: 0.95 });
    const roofMat = new THREE.MeshStandardMaterial({
      color: tinted(0x8a4b32, 0.08, variant(obj.id, 1)), roughness: 0.8,
    });

    // foundation: 0.3 m, the same number as metadata.foundation_height, so
    // "raise the house" in the properties panel has something to read against
    const plinth = new THREE.Mesh(new THREE.BoxGeometry(4.04, 0.3, 4.04), stone);
    plinth.position.y = 0.15;
    const body = new THREE.Mesh(new THREE.BoxGeometry(4, 3, 4), wall);
    body.position.y = 1.5;
    const roof = new THREE.Mesh(new THREE.ConeGeometry(3.2, 2, 4), roofMat);
    roof.position.y = 4;
    roof.rotation.y = Math.PI / 4;
    g.add(plinth, body, roof);

    // door on the +z face, with a step and a handle
    const door = new THREE.Mesh(new THREE.BoxGeometry(1.0, 2.0, 0.12),
      new THREE.MeshStandardMaterial({ color: 0x6b4a2b, roughness: 0.7 }));
    door.position.set(0, 1.15, 2.0);
    const step = new THREE.Mesh(new THREE.BoxGeometry(1.4, 0.15, 0.5), stone);
    step.position.set(0, 0.22, 2.2);
    const handle = new THREE.Mesh(new THREE.SphereGeometry(0.07, 8, 6), METAL());
    handle.position.set(0.35, 1.15, 2.09);
    g.add(door, step, handle);

    // windows: two per side wall, two flanking the door, one at the back
    const glass = GLASS();
    // The frame is a flat border no deeper than the pane, and the pane is
    // pushed out along the WALL NORMAL only. Both matter: a frame that is
    // deeper than its pane swallows the glass completely (it did), and offset
    // by scaling from the centre would slide a side window along the wall as
    // well as through it.
    const place = (x: number, y: number, z: number, w: number, d: number) => {
      const frame = new THREE.Mesh(new THREE.BoxGeometry(w + 0.18, 1.08, d), trim);
      frame.position.set(x, y, z);
      const pane = new THREE.Mesh(new THREE.BoxGeometry(w, 0.9, d), glass);
      const nx = w < d ? Math.sign(x) * 0.04 : 0;
      const nz = d < w ? Math.sign(z) * 0.04 : 0;
      pane.position.set(x + nx, y, z + nz);
      g.add(frame, pane);
    };
    for (const z of [-1.05, 1.05]) {
      place(2.0, 1.85, z, 0.1, 0.95);
      place(-2.0, 1.85, z, 0.1, 0.95);
    }
    for (const x of [-1.3, 1.3]) place(x, 1.85, 2.0, 0.95, 0.1);
    place(0, 1.85, -2.0, 1.5, 0.1);

    const chimney = new THREE.Mesh(new THREE.BoxGeometry(0.5, 1.8, 0.5), stone);
    chimney.position.set(1.05, 3.85, 1.05);
    const cap = new THREE.Mesh(new THREE.BoxGeometry(0.66, 0.14, 0.66), trim);
    cap.position.set(1.05, 4.8, 1.05);
    g.add(chimney, cap);
    return g;
  },

  CAR: (obj) => {
    // 4.4 x 2.0 m -- the half-extents the fluid solver samples the body with.
    const g = new THREE.Group();
    const paint = new THREE.MeshStandardMaterial({
      color: tinted(OBJECT_COLORS.CAR, 0.22, variant(obj.id)),
      roughness: 0.35, metalness: 0.35,
    });
    const body = new THREE.Mesh(new THREE.BoxGeometry(4.4, 1.1, 2), paint);
    body.position.y = 0.9;
    const cabin = new THREE.Mesh(new THREE.BoxGeometry(2.2, 0.9, 1.8), paint);
    cabin.position.set(-0.3, 1.9, 0);
    // glazing sits proud of the cabin by 3 cm on each side: no z-fighting, and
    // the greenhouse reads as glass from any angle
    const glazing = new THREE.Mesh(new THREE.BoxGeometry(1.9, 0.5, 1.86), GLASS());
    glazing.position.set(-0.3, 1.98, 0);
    g.add(body, cabin, glazing);

    const dark = RUBBER();
    for (const x of [2.21, -2.21]) {
      const bumper = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.36, 2.02), dark);
      bumper.position.set(x, 0.62, 0);
      g.add(bumper);
    }
    const lamp = (x: number, z: number, color: number) => {
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(0.09, 0.22, 0.44),
        new THREE.MeshStandardMaterial({
          color, emissive: color, emissiveIntensity: 0.7, roughness: 0.3,
        }));
      mesh.position.set(x, 1.08, z);
      g.add(mesh);
    };
    for (const z of [0.62, -0.62]) {
      lamp(2.22, z, 0xfff0c4);
      lamp(-2.22, z, 0xd8402c);
    }

    for (const [x, z] of [[1.4, 1.1], [1.4, -1.1], [-1.4, 1.1], [-1.4, -1.1]]) {
      const wheel = new THREE.Mesh(new THREE.CylinderGeometry(0.4, 0.4, 0.3, 16), dark);
      wheel.rotation.x = Math.PI / 2;
      wheel.position.set(x, 0.4, z);
      const hub = new THREE.Mesh(new THREE.CylinderGeometry(0.17, 0.17, 0.34, 12), METAL());
      hub.rotation.x = Math.PI / 2;
      hub.position.set(x, 0.4, z);
      g.add(wheel, hub);
    }
    return g;
  },

  TREE: (obj) => {
    // Trunk footprint stays 0.25 m; only the crown varies between individuals,
    // and it varies because a cloned forest is the tell that nothing here is
    // really being simulated.
    const g = new THREE.Group();
    const seed = variant(obj.id);
    // The per-individual yaw lives on an inner node: `applyTransform` writes the
    // object's own rotation onto the group every frame, so anything set on `g`
    // here would be silently overwritten on the first update.
    const inner = new THREE.Group();
    inner.rotation.y = seed * Math.PI * 2;
    g.add(inner);
    const trunk = new THREE.Mesh(
      new THREE.CylinderGeometry(0.18, 0.25, 1.6, 8),
      new THREE.MeshStandardMaterial({
        color: tinted(0x6b4a2b, 0.12, variant(obj.id, 2)), roughness: 0.95,
      }));
    trunk.position.y = 0.8;
    inner.add(trunk);

    const foliage = new THREE.MeshStandardMaterial({
      color: tinted(OBJECT_COLORS.TREE, 0.16, seed), roughness: 0.9, flatShading: true,
    });
    const crown = new THREE.Group();
    for (const [radius, height, y] of
         [[1.15, 1.5, 2.05], [0.92, 1.35, 2.75], [0.58, 1.15, 3.42]]) {
      const tier = new THREE.Mesh(new THREE.ConeGeometry(radius, height, 9), foliage);
      tier.position.y = y;
      tier.rotation.y = seed * Math.PI;
      crown.add(tier);
    }
    crown.scale.setScalar(0.92 + seed * 0.16);
    inner.add(crown);
    return g;
  },

  BOX: (obj) => {
    // The visible 1.2 m crate whose defaults the BOX physics were matched to.
    const g = new THREE.Group();
    const planks = new THREE.MeshStandardMaterial({
      color: tinted(OBJECT_COLORS.BOX, 0.12, variant(obj.id)), roughness: 0.9,
    });
    const frame = new THREE.MeshStandardMaterial({ color: 0x7a5334, roughness: 0.9 });
    const box = new THREE.Mesh(new THREE.BoxGeometry(1.2, 1.2, 1.2), planks);
    box.position.y = 0.6;
    g.add(box);
    for (const y of [0.28, 0.92]) {
      const band = new THREE.Mesh(new THREE.BoxGeometry(1.21, 0.12, 1.21), frame);
      band.position.y = y;
      g.add(band);
    }
    for (const x of [-0.535, 0.535]) {
      for (const z of [-0.535, 0.535]) {
        const post = new THREE.Mesh(new THREE.BoxGeometry(0.14, 1.21, 0.14), frame);
        post.position.set(x, 0.6, z);
        g.add(post);
      }
    }
    return g;
  },

  DEBRIS: (obj) => {
    // Loose wreckage rather than one tidy pebble -- this is what the flood tore
    // off something else. Everything stays inside the 0.6 m half-extent.
    const g = new THREE.Group();
    const seed = variant(obj.id);
    const stone = new THREE.MeshStandardMaterial({
      color: tinted(OBJECT_COLORS.DEBRIS, 0.14, seed), flatShading: true, roughness: 0.95,
    });
    const timber = new THREE.MeshStandardMaterial({ color: 0x7d6242, roughness: 0.95 });
    const chunks: number[][] =
      [[0.24, 0.05, 0.18, 0.20], [0.17, -0.22, 0.12, -0.16], [0.13, 0.14, 0.10, -0.26]];
    for (const [radius, x, , z] of chunks) {
      const chunk = new THREE.Mesh(
        roughened(new THREE.IcosahedronGeometry(radius, 0), seed * 53 + x, 0.7, 1.0),
        stone);
      chunk.position.set(x, radius * 0.75, z);
      chunk.rotation.set(seed * 3, seed * 5, seed * 7);
      g.add(chunk);
    }
    for (const [angle, z] of [[0.6, 0.05], [-1.1, -0.12]]) {
      const plank = new THREE.Mesh(new THREE.BoxGeometry(0.56, 0.07, 0.13), timber);
      plank.position.set(0, 0.05, z);
      plank.rotation.y = angle + seed;
      g.add(plank);
    }
    return g;
  },

  ROCK: (obj) => {
    // A riverbed boulder. The visible radius matches
    // fluid_solver.ROCK_BASE_RADIUS_M so what the user sees is what the water
    // feels: the solver raises the effective bed by a dome of exactly this
    // footprint. Scaling the object scales both, horizontally and vertically.
    // The satellite stones sit well inside that radius and add nothing to it.
    const g = new THREE.Group();
    const seed = variant(obj.id);
    const material = new THREE.MeshStandardMaterial({
      color: tinted(OBJECT_COLORS.ROCK, 0.12, seed), flatShading: true, roughness: 0.95,
    });
    const rock = new THREE.Mesh(
      roughened(new THREE.IcosahedronGeometry(1.5, 1), seed * 97, 0.78, 1.0), material);
    rock.scale.set(1, 0.62, 1);   // a dome, not a ball -- water passes over it
    rock.position.y = 0.3;
    rock.rotation.y = seed * Math.PI;
    g.add(rock);
    for (const [radius, x, z] of [[0.42, 0.95, -0.62], [0.3, -0.82, 0.86]]) {
      const stone = new THREE.Mesh(
        roughened(new THREE.IcosahedronGeometry(radius, 0), seed * 31 + x, 0.75, 1.0),
        material);
      stone.scale.set(1, 0.7, 1);
      stone.position.set(x, radius * 0.35, z);
      stone.rotation.set(seed * 3, seed * 7 + radius, seed * 2);
      g.add(stone);
    }
    return g;
  },

  BRIDGE: () => {
    // Deck plus three piers, 24 m across -- the span and pier spacing match
    // fluid_solver.BRIDGE_SPAN_M and the pier rasterization, so the piers the
    // water is diverted by are the piers the user can see. Only the piers are
    // solid to the flow; the deck is drawn but never rasterized, because a
    // bridge that dams its own river is not a bridge.
    const g = new THREE.Group();
    const stone = new THREE.MeshStandardMaterial({
      color: OBJECT_COLORS.BRIDGE, roughness: 0.9,
    });
    const deck = new THREE.Mesh(new THREE.BoxGeometry(5, 0.5, 24), stone);
    deck.position.y = 3.2;
    g.add(deck);

    const timber = new THREE.MeshStandardMaterial({ color: 0x8a7350, roughness: 0.95 });
    for (let z = -11.25; z <= 11.25; z += 1.5) {
      const plank = new THREE.Mesh(new THREE.BoxGeometry(5.02, 0.07, 0.16), timber);
      plank.position.set(0, 3.47, z);
      g.add(plank);
    }
    for (const rail of [-2.2, 2.2]) {
      const bar = new THREE.Mesh(new THREE.BoxGeometry(0.25, 0.9, 24), stone);
      bar.position.set(rail, 3.9, 0);
      g.add(bar);
      for (let z = -12; z <= 12; z += 3) {
        const post = new THREE.Mesh(new THREE.BoxGeometry(0.34, 1.15, 0.34), stone);
        post.position.set(rail, 4.02, z);
        g.add(post);
      }
    }
    for (const z of [-12, 0, 12]) {
      const pier = new THREE.Mesh(
        new THREE.CylinderGeometry(0.9, 1.1, 3.2, 16), stone);
      pier.position.set(0, 1.6, z);
      // A cap where the pier meets the deck: the piers are what the flow piles
      // debris against, so they should read as structure, not as posts.
      const cap = new THREE.Mesh(
        new THREE.CylinderGeometry(1.08, 1.08, 0.28, 16), stone);
      cap.position.set(0, 3.05, z);
      g.add(pier, cap);
    }
    return g;
  },

  PERSON: (obj) => {
    // A blocky, Lego-style figure at human scale (about 1.7 m). Light, tall for
    // its footprint and draggy, which is exactly why moving water carries one
    // off so readily -- that is the lesson, not the model.
    const g = new THREE.Group();
    const seed = variant(obj.id);
    const shirt = new THREE.MeshStandardMaterial({
      color: tinted(OBJECT_COLORS.PERSON, 0.5, seed), roughness: 0.8,
    });
    const trousers = new THREE.MeshStandardMaterial({
      color: tinted(0x3a4a6b, 0.3, variant(obj.id, 3)), roughness: 0.85,
    });
    const skin = new THREE.MeshStandardMaterial({ color: 0xf6c177, roughness: 0.7 });
    const boots = new THREE.MeshStandardMaterial({ color: 0x2b2b30, roughness: 0.9 });

    const legs = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.75, 0.28), trousers);
    legs.position.y = 0.38;
    const torso = new THREE.Mesh(new THREE.BoxGeometry(0.55, 0.6, 0.32), shirt);
    torso.position.y = 1.05;
    const head = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.22, 0.3, 14), skin);
    head.position.y = 1.5;
    const stud = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.1, 0.08, 10), skin);
    stud.position.y = 1.69;
    g.add(legs, torso, head, stud);
    for (const x of [-0.38, 0.38]) {
      const arm = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.55, 0.2), shirt);
      arm.position.set(x, 1.05, 0);
      const hand = new THREE.Mesh(new THREE.BoxGeometry(0.17, 0.17, 0.21), skin);
      hand.position.set(x, 0.72, 0);
      g.add(arm, hand);
    }
    for (const x of [-0.11, 0.11]) {
      const foot = new THREE.Mesh(new THREE.BoxGeometry(0.19, 0.13, 0.34), boots);
      foot.position.set(x, 0.065, 0.04);
      g.add(foot);
    }
    // Face a stable random way. It has to go on an inner node, because
    // `applyTransform` overwrites the group's own rotation from the object data.
    const facing = new THREE.Group();
    facing.rotation.y = seed * Math.PI * 2;
    facing.add(...g.children.slice());
    g.add(facing);
    return g;
  },

  SOURCE: () => {
    // Where the water comes from. Drawn as an open ring with an arrow pointing
    // downstream, at the radius the solver actually uses for the inflow disc --
    // the request was literally "an indication of where the water flows from",
    // so the marker has to be the real footprint, not a decorative icon.
    const g = new THREE.Group();
    const material = new THREE.MeshStandardMaterial({
      color: OBJECT_COLORS.SOURCE, emissive: 0x0f5c3c, emissiveIntensity: 0.9,
    });
    const ring = new THREE.Mesh(new THREE.TorusGeometry(4, 0.28, 12, 48), material);
    ring.rotation.x = Math.PI / 2;
    ring.position.y = 0.35;
    const arrow = new THREE.Mesh(new THREE.ConeGeometry(0.9, 2.6, 14), material);
    arrow.rotation.z = -Math.PI / 2;   // points along +x, the flow direction
    arrow.position.set(2.2, 1.4, 0);
    const post = new THREE.Mesh(
      new THREE.CylinderGeometry(0.12, 0.12, 3.0, 10), material);
    post.position.y = 1.5;
    g.add(ring, arrow, post);
    return g;
  },

  DRAIN: () => {
    // Where the water goes. A recessed grate ring at the solver's drain radius,
    // plus a funnel that reads as "down" from any camera angle.
    const g = new THREE.Group();
    const material = new THREE.MeshStandardMaterial({
      color: OBJECT_COLORS.DRAIN, emissive: 0x5c1f13, emissiveIntensity: 0.8,
      metalness: 0.3, roughness: 0.6,
    });
    const rim = new THREE.Mesh(new THREE.TorusGeometry(5, 0.3, 12, 48), material);
    rim.rotation.x = Math.PI / 2;
    rim.position.y = 0.2;
    const funnel = new THREE.Mesh(
      new THREE.ConeGeometry(4.4, 2.4, 28, 1, true), material);
    funnel.position.y = -1.0;
    for (let i = 0; i < 4; i++) {
      const bar = new THREE.Mesh(new THREE.BoxGeometry(9.4, 0.12, 0.3), material);
      bar.position.set(0, 0.2, -3 + i * 2);
      g.add(bar);
    }
    g.add(rim, funnel);
    return g;
  },

  GAUGE: () => {
    // A staff gauge, so the pole is graduated: alternating 0.25 m bands from
    // the ground up. The sparkline carries the numbers, but a child reads depth
    // off a striped stick, and these stripes are real metres.
    const g = new THREE.Group();
    const staff = new THREE.MeshStandardMaterial({ color: 0xf2f6f8, roughness: 0.6 });
    const band = new THREE.MeshStandardMaterial({ color: 0xd8402c, roughness: 0.6 });
    const marker = new THREE.MeshStandardMaterial({
      color: OBJECT_COLORS.GAUGE, emissive: 0x123c4a, emissiveIntensity: 0.9,
    });
    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.08, 2.4, 12), staff);
    pole.position.y = 1.2;
    g.add(pole);
    for (let y = 0.125; y < 2.4; y += 0.5) {
      const stripe = new THREE.Mesh(
        new THREE.CylinderGeometry(0.085, 0.085, 0.25, 12), band);
      stripe.position.y = y;
      g.add(stripe);
    }
    const head = new THREE.Mesh(new THREE.TorusGeometry(0.35, 0.07, 10, 28), marker);
    head.position.y = 2.5;
    head.rotation.x = Math.PI / 2;
    g.add(head);
    return g;
  },
};

export function registerObjectBuilder(type: string, build: Builder): void {
  builders[type] = build;
}

export function buildObjectMesh(obj: ObjectData): THREE.Group {
  const build = builders[obj.type] ?? builders.BOX;
  const group = build(obj);
  group.name = obj.id;
  group.userData.objectId = obj.id;
  // Every object casts and receives. The terrain receives only; the water
  // surface and the tracer/spray point clouds take no part in shadowing at all
  // -- see SceneManager.configureSun.
  group.traverse((node) => {
    if ((node as THREE.Mesh).isMesh) {
      node.castShadow = true;
      node.receiveShadow = true;
    }
  });
  applyTransform(group, obj);
  return group;
}

export function applyTransform(group: THREE.Group, obj: ObjectData): void {
  group.position.set(obj.position[0], obj.position[1], obj.position[2]);
  group.rotation.set(obj.rotation[0], obj.rotation[1], obj.rotation[2]);
  group.scale.set(obj.scale[0], obj.scale[1], obj.scale[2]);
}
