/** Mouse interaction: object select/move/rotate/scale, terrain brush, delete. */
import * as THREE from 'three';
import { TransformControls } from 'three/examples/jsm/controls/TransformControls.js';
import type { SceneManager } from '../scene/SceneManager';
import type { WorldStore } from '../world/WorldStore';
import type { BackendClient } from '../net/BackendClient';
import type { ObjectType } from '../world/types';

export type Tool = 'select' | 'raise' | 'lower';

export class EditorController {
  tool: Tool = 'select';
  brushRadius = 6;
  brushStrength = 0.4;
  terrainEditingEnabled = true;

  private transform: TransformControls;
  private painting = false;
  private lastBrushSent = 0;
  private pointer = new THREE.Vector2();

  constructor(
    private scene: SceneManager,
    private store: WorldStore,
    private net: BackendClient,
  ) {
    this.transform = new TransformControls(scene.camera, scene.renderer.domElement);
    this.transform.setSize(1.25);
    scene.scene.add(this.transform.getHelper());

    this.transform.addEventListener('dragging-changed', (e: { value?: unknown }) => {
      scene.controls.enabled = !e.value;
    });
    this.transform.addEventListener('objectChange', () => this.pushSelectedTransform());

    const canvas = scene.renderer.domElement;
    canvas.addEventListener('pointerdown', (e) => this.onPointerDown(e), true);
    canvas.addEventListener('pointermove', (e) => this.onPointerMove(e));
    window.addEventListener('pointerup', () => this.onPointerUp());
    window.addEventListener('keydown', (e) => this.onKeyDown(e));

    store.on('selection-changed', () => {
      const id = store.selectedId;
      if (id) {
        const group = scene.objectsRoot.getObjectByName(id);
        if (group) this.transform.attach(group);
        // SOURCE, DRAIN and VENT are ground fixtures: they belong on the terrain,
        // and dragging one up into the air would silently change what the solver
        // does (a source above its own water level fills nothing, a drain above
        // the surface drains nothing, a vent above the bed erupts into thin air)
        // with no visible reason. So the vertical gizmo handle is switched off
        // for them and their Y is pinned to the ground in pushSelectedTransform
        // -- they move on the plane only.
        const type = store.objects.get(id)?.type;
        this.transform.showY = !EditorController.GROUND_FIXTURES.has(type ?? '');
      } else {
        this.transform.detach();
        this.transform.showY = true;
      }
      scene.highlight(id);
    });
  }

  // ------------------------------------------------------------------ pointer
  private setPointer(e: PointerEvent): void {
    const rect = this.scene.renderer.domElement.getBoundingClientRect();
    this.pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  }

  private onPointerDown(e: PointerEvent): void {
    this.setPointer(e);
    if (this.tool === 'select') {
      if ((this.transform as unknown as { dragging: boolean }).dragging) return;
      const id = this.scene.pickObject(this.pointer);
      this.store.select(id);
      return;
    }
    if (e.button !== 0) return;
    this.painting = true;
    this.applyBrush();
  }

  private onPointerMove(e: PointerEvent): void {
    this.setPointer(e);
    if (this.painting) this.applyBrush();
  }

  private onPointerUp(): void {
    this.painting = false;
  }

  private onKeyDown(e: KeyboardEvent): void {
    if ((e.target as HTMLElement)?.tagName === 'INPUT') return;
    if (e.key === 'Delete' || e.key === 'Backspace') this.deleteSelected();
    else if (e.key === 'w') this.transform.setMode('translate');
    else if (e.key === 'e') this.transform.setMode('rotate');
    else if (e.key === 'r') this.transform.setMode('scale');
  }

  // ------------------------------------------------------------------ actions
  setTool(tool: Tool): void {
    this.tool = tool;
    // terrain painting owns the left mouse button; free the orbit camera
    this.scene.controls.enabled = tool === 'select';
    if (tool !== 'select') {
      this.transform.detach();
      this.store.select(null);
    }
  }

  addObject(type: ObjectType): void {
    // Auto-place on a grid SIZED TO THE MAP, not a fixed 3 columns. The fixed
    // version put every object in a 72 m-wide strip (3 * 36 m spacing) down
    // the map's centre and grew unboundedly in +z with every add: past
    // roughly a dozen objects the next one landed outside the terrain
    // entirely, while most of the map's width sat unused -- reported as
    // "objects start landing in empty space after 3-5 adds". Sizing the grid
    // to the terrain and wrapping (modulo) once it fills keeps every
    // auto-placed object on the map, at the cost of new objects eventually
    // overlapping older ones once the grid is full -- the user can still drag
    // them apart, which is a far smaller problem than an object placed off
    // the terrain.
    const idx = this.store.objects.size;
    const size = this.store.terrain.sizeM;
    const spacing = size * 0.18;
    const margin = spacing * 0.5;
    const columns = Math.max(3, Math.floor((size - 2 * margin) / spacing) + 1);
    const cell = idx % (columns * columns);
    const x = ((cell % columns) - (columns - 1) / 2) * spacing;
    const z = (Math.floor(cell / columns) - (columns - 1) / 2) * spacing;
    const y = this.store.terrain.heightAt(x, z);
    this.net.send({ op: 'object_add', object: { type, position: [x, y, z] } });
  }

  deleteSelected(): void {
    const id = this.store.selectedId;
    if (!id) return;
    this.net.send({ op: 'object_remove', id });
    this.store.removeObject(id);
    this.scene.removeObject(id);
  }

  private applyBrush(): void {
    if (!this.terrainEditingEnabled) return;
    const point = this.scene.pickTerrain(this.pointer);
    if (!point) return;
    const sign = this.tool === 'raise' ? 1 : -1;
    const now = performance.now();
    if (now - this.lastBrushSent > 40) {
      this.lastBrushSent = now;
      const sent = this.net.send({
        op: 'terrain_brush', x: point.x, z: point.z,
        radius: this.brushRadius, strength: this.brushStrength * sign,
      });
      if (sent) {
        // Optimistic edit is applied only for the exact command sent. The
        // authoritative terrain_patch response then removes numeric drift.
        this.store.brushTerrain(point.x, point.z, this.brushRadius,
                                this.brushStrength * sign);
        this.scene.rebuildTerrain(this.store.terrain);
      }
    }
  }

  /** Types that live on the ground and may only be moved in the XZ plane. */
  private static readonly GROUND_FIXTURES = new Set(['SOURCE', 'DRAIN', 'VENT']);

  private pushSelectedTransform(): void {
    const id = this.store.selectedId;
    const group = this.transform.object as THREE.Object3D | undefined;
    if (!id || !group) return;
    // Ground fixtures stay on the terrain however the gizmo was dragged: the
    // vertical handle is hidden, but a rotate/scale drag can still nudge Y.
    if (EditorController.GROUND_FIXTURES.has(this.store.objects.get(id)?.type ?? '')) {
      group.position.y = this.store.terrain.heightAt(group.position.x, group.position.z);
    }
    this.store.updateObject(id, {
      position: [group.position.x, group.position.y, group.position.z],
      rotation: [group.rotation.x, group.rotation.y, group.rotation.z],
      scale: [group.scale.x, group.scale.y, group.scale.z],
    });
    this.net.send({
      op: 'object_update', id,
      fields: {
        position: [group.position.x, group.position.y, group.position.z],
        rotation: [group.rotation.x, group.rotation.y, group.rotation.z],
        scale: [group.scale.x, group.scale.y, group.scale.z],
      },
    });
  }
}
