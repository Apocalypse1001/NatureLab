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
  return { HOUSE: 0, CAR: 0, TREE: 0, BOX: 0, DEBRIS: 0, GAUGE: 0 }[type] ?? 0;
}
