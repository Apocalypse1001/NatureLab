/** Primitive meshes per object type. Extend the registry to add new types. */
import * as THREE from 'three';
import { OBJECT_COLORS, type ObjectData } from './types';

const builders: Record<string, () => THREE.Group> = {
  HOUSE: () => {
    const g = new THREE.Group();
    const body = new THREE.Mesh(
      new THREE.BoxGeometry(4, 3, 4),
      new THREE.MeshStandardMaterial({ color: OBJECT_COLORS.HOUSE }));
    body.position.y = 1.5;
    const roof = new THREE.Mesh(
      new THREE.ConeGeometry(3.2, 2, 4),
      new THREE.MeshStandardMaterial({ color: 0x8a4b32 }));
    roof.position.y = 4;
    roof.rotation.y = Math.PI / 4;
    g.add(body, roof);
    return g;
  },
  CAR: () => {
    const g = new THREE.Group();
    const mat = new THREE.MeshStandardMaterial({ color: OBJECT_COLORS.CAR });
    const body = new THREE.Mesh(new THREE.BoxGeometry(4.4, 1.1, 2), mat);
    body.position.y = 0.9;
    const cabin = new THREE.Mesh(new THREE.BoxGeometry(2.2, 0.9, 1.8), mat);
    cabin.position.set(-0.3, 1.9, 0);
    g.add(body, cabin);
    for (const [x, z] of [[1.4, 1.1], [1.4, -1.1], [-1.4, 1.1], [-1.4, -1.1]]) {
      const wheel = new THREE.Mesh(
        new THREE.CylinderGeometry(0.4, 0.4, 0.3, 12),
        new THREE.MeshStandardMaterial({ color: 0x222222 }));
      wheel.rotation.x = Math.PI / 2;
      wheel.position.set(x, 0.4, z);
      g.add(wheel);
    }
    return g;
  },
  TREE: () => {
    const g = new THREE.Group();
    const trunk = new THREE.Mesh(
      new THREE.CylinderGeometry(0.18, 0.25, 1.6, 8),
      new THREE.MeshStandardMaterial({ color: 0x6b4a2b }));
    trunk.position.y = 0.8;
    const leaves = new THREE.Mesh(
      new THREE.ConeGeometry(1.1, 2.6, 8),
      new THREE.MeshStandardMaterial({ color: OBJECT_COLORS.TREE }));
    leaves.position.y = 2.6;
    g.add(trunk, leaves);
    return g;
  },
  BOX: () => {
    const g = new THREE.Group();
    const box = new THREE.Mesh(
      new THREE.BoxGeometry(1.2, 1.2, 1.2),
      new THREE.MeshStandardMaterial({ color: OBJECT_COLORS.BOX }));
    box.position.y = 0.6;
    g.add(box);
    return g;
  },
  DEBRIS: () => {
    const g = new THREE.Group();
    const rock = new THREE.Mesh(
      new THREE.IcosahedronGeometry(0.6, 0),
      new THREE.MeshStandardMaterial({ color: OBJECT_COLORS.DEBRIS, flatShading: true }));
    rock.position.y = 0.4;
    g.add(rock);
    return g;
  },
  ROCK: () => {
    // A riverbed boulder. The visible radius matches
    // fluid_solver.ROCK_BASE_RADIUS_M so what the user sees is what the water
    // feels: the solver raises the effective bed by a dome of exactly this
    // footprint. Scaling the object scales both, horizontally and vertically.
    const g = new THREE.Group();
    const rock = new THREE.Mesh(
      new THREE.IcosahedronGeometry(1.5, 1),
      new THREE.MeshStandardMaterial({
        color: OBJECT_COLORS.ROCK, flatShading: true, roughness: 0.95,
      }));
    rock.scale.y = 0.55;          // a dome, not a ball -- water passes over it
    rock.position.y = 0.35;
    g.add(rock);
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
    for (const rail of [-2.2, 2.2]) {
      const bar = new THREE.Mesh(new THREE.BoxGeometry(0.25, 0.9, 24), stone);
      bar.position.set(rail, 3.9, 0);
      g.add(bar);
    }
    for (const z of [-12, 0, 12]) {
      const pier = new THREE.Mesh(
        new THREE.CylinderGeometry(0.9, 1.1, 3.2, 12), stone);
      pier.position.set(0, 1.6, z);
      g.add(pier);
    }
    return g;
  },
  PERSON: () => {
    // A blocky, Lego-style figure at human scale (about 1.7 m). Light, tall for
    // its footprint and draggy, which is exactly why moving water carries one
    // off so readily -- that is the lesson, not the model.
    const g = new THREE.Group();
    const body = new THREE.MeshStandardMaterial({ color: OBJECT_COLORS.PERSON });
    const skin = new THREE.MeshStandardMaterial({ color: 0xf6c177 });
    const legs = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.75, 0.28), body);
    legs.position.y = 0.38;
    const torso = new THREE.Mesh(new THREE.BoxGeometry(0.55, 0.6, 0.32), body);
    torso.position.y = 1.05;
    const head = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.22, 0.3, 12), skin);
    head.position.y = 1.5;
    const stud = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.1, 0.08, 10), skin);
    stud.position.y = 1.69;
    for (const x of [-0.38, 0.38]) {
      const arm = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.55, 0.2), body);
      arm.position.set(x, 1.05, 0);
      g.add(arm);
    }
    g.add(legs, torso, head, stud);
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
    const ring = new THREE.Mesh(new THREE.TorusGeometry(4, 0.28, 10, 40), material);
    ring.rotation.x = Math.PI / 2;
    ring.position.y = 0.35;
    const arrow = new THREE.Mesh(new THREE.ConeGeometry(0.9, 2.6, 12), material);
    arrow.rotation.z = -Math.PI / 2;   // points along +x, the flow direction
    arrow.position.set(2.2, 1.4, 0);
    const post = new THREE.Mesh(
      new THREE.CylinderGeometry(0.12, 0.12, 3.0, 8), material);
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
    const rim = new THREE.Mesh(new THREE.TorusGeometry(5, 0.3, 10, 44), material);
    rim.rotation.x = Math.PI / 2;
    rim.position.y = 0.2;
    const funnel = new THREE.Mesh(
      new THREE.ConeGeometry(4.4, 2.4, 24, 1, true), material);
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
    const g = new THREE.Group();
    const material = new THREE.MeshStandardMaterial({
      color: OBJECT_COLORS.GAUGE, emissive: 0x123c4a, emissiveIntensity: 0.8,
    });
    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.08, 2.4, 10), material);
    pole.position.y = 1.2;
    const marker = new THREE.Mesh(new THREE.TorusGeometry(0.35, 0.07, 8, 24), material);
    marker.position.y = 2.2;
    marker.rotation.x = Math.PI / 2;
    g.add(pole, marker);
    return g;
  },
};

export function registerObjectBuilder(type: string, build: () => THREE.Group): void {
  builders[type] = build;
}

export function buildObjectMesh(obj: ObjectData): THREE.Group {
  const build = builders[obj.type] ?? builders.BOX;
  const group = build();
  group.name = obj.id;
  group.userData.objectId = obj.id;
  applyTransform(group, obj);
  return group;
}

export function applyTransform(group: THREE.Group, obj: ObjectData): void {
  group.position.set(obj.position[0], obj.position[1], obj.position[2]);
  group.rotation.set(obj.rotation[0], obj.rotation[1], obj.rotation[2]);
  group.scale.set(obj.scale[0], obj.scale[1], obj.scale[2]);
}

/** Approximate half-heights for placing objects on the selected tool. */
export function objectHalfSize(type: string): number {
  return { HOUSE: 0, CAR: 0, TREE: 0, BOX: 0, DEBRIS: 0, ROCK: 0,
           BRIDGE: 0, PERSON: 0, SOURCE: 0, DRAIN: 0, GAUGE: 0 }[type] ?? 0;
}
