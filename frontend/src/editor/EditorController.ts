/** Mouse interaction: object select/move/rotate/scale, terrain brush, pipe laying, delete. */
import * as THREE from 'three';
import { TransformControls } from 'three/examples/jsm/controls/TransformControls.js';
import type { SceneManager } from '../scene/SceneManager';
import type { WorldStore } from '../world/WorldStore';
import type { BackendClient } from '../net/BackendClient';
import type { ObjectType } from '../world/types';

export type Tool = 'select' | 'raise' | 'lower' | 'pipe';

export class EditorController {
  tool: Tool = 'select';
  brushRadius = 6;
  brushStrength = 0.4;
  /** v0.17.0: diameter of the next pipe laid, metres. */
  pipeDiameter = 0.2;
  /** Told of every tool change -- a click, Esc, a finished pipe -- for the HUD. */
  onToolChange: ((tool: Tool) => void) | null = null;

  private transform: TransformControls;
  private painting = false;
  private lastBrushSent = 0;
  private pointer = new THREE.Vector2();
  // v0.17.0 pipe laying: the route clicked so far, what it starts on, and --
  // when continuing an existing pipe -- which one
  private pipePoints: number[][] = [];
  private pipeFrom: string | null = null;
  private extendingPipe: string | null = null;
  private downAt: { x: number; y: number; button: number } | null = null;

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
    canvas.addEventListener('contextmenu', (e) => {
      if (this.tool === 'pipe') e.preventDefault();   // right-click ends a pipe
    });
    window.addEventListener('pointerup', (e) => this.onPointerUp(e));
    window.addEventListener('keydown', (e) => this.onKeyDown(e));

    store.on('selection-changed', () => {
      const id = store.selectedId;
      if (id) {
        const type = store.objects.get(id)?.type;
        const group = scene.objectsRoot.getObjectByName(id);
        // A PIPE is its route; dragging the group would move the drawing away
        // from the points the solver and the save file hold. Its ends follow
        // the inlet and the outfall instead (main.ts).
        if (group && type !== 'PIPE') this.transform.attach(group);
        else this.transform.detach();
        // SOURCE, DRAIN and VENT are ground fixtures: they belong on the terrain,
        // and dragging one up into the air would silently change what the solver
        // does (a source above its own water level fills nothing, a drain above
        // the surface drains nothing, a vent above the bed erupts into thin air)
        // with no visible reason. So the vertical gizmo handle is switched off
        // for them and their Y is pinned to the ground in pushSelectedTransform
        // -- they move on the plane only.
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
    if (this.tool === 'pipe') {
      // decided on release: a drag is the camera orbiting, a click is a point
      this.downAt = { x: e.clientX, y: e.clientY, button: e.button };
      return;
    }
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
    if (this.tool === 'pipe' && this.pipePoints.length) {
      this.updatePipePreview(this.scene.pickTerrain(this.pointer));
    }
  }

  private onPointerUp(e: PointerEvent): void {
    this.painting = false;
    if (this.tool !== 'pipe' || !this.downAt) return;
    const moved = Math.hypot(e.clientX - this.downAt.x, e.clientY - this.downAt.y);
    const button = this.downAt.button;
    this.downAt = null;
    if (moved > 5 || e.target !== this.scene.renderer.domElement) return;
    this.setPointer(e);
    if (button === 2) this.finishPipe(null);
    else if (button === 0) this.addPipePoint();
  }

  private onKeyDown(e: KeyboardEvent): void {
    const tag = (e.target as HTMLElement)?.tagName;
    if (tag === 'INPUT' || tag === 'SELECT') return;
    if (this.tool === 'pipe') {
      if (e.key === 'Enter') this.finishPipe(null);
      else if (e.key === 'Escape') this.setTool('select');
      else if (e.key === 'Backspace' || e.key === 'Delete') {
        this.pipePoints.pop();
        if (!this.pipePoints.length) this.pipeFrom = null;
        this.updatePipePreview(null);
      }
      return;
    }
    if (e.key === 'Delete' || e.key === 'Backspace') this.deleteSelected();
    // By physical key, not by character: with a Russian layout W/E/R type
    // ц/у/к, and with Caps Lock or Shift they type W/E/R -- `e.key` matched
    // neither, so the gizmo never changed mode. Ctrl+R (reload) is not a mode.
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.code === 'KeyW') this.transform.setMode('translate');
    else if (e.code === 'KeyE') this.transform.setMode('rotate');
    else if (e.code === 'KeyR') this.transform.setMode('scale');
  }

  // ------------------------------------------------------------------ actions
  setTool(tool: Tool): void {
    if (this.tool === 'pipe' && tool !== 'pipe') this.cancelPipe();
    this.tool = tool;
    // terrain painting owns the left mouse button; free the orbit camera.
    // Laying a pipe keeps the camera: a drag orbits, a click places a point.
    this.scene.controls.enabled = tool === 'select' || tool === 'pipe';
    if (tool !== 'select') {
      this.transform.detach();
      this.store.select(null);
    }
    this.onToolChange?.(tool);
  }

  /** Carry an existing pipe on from its outfall: its route is picked back up. */
  extendPipe(id: string): void {
    const pipe = this.store.objects.get(id);
    const points = pipe?.metadata.points as number[][] | undefined;
    if (!pipe || pipe.type !== 'PIPE' || !points || points.length < 2) return;
    this.setTool('pipe');
    this.pipePoints = points.slice(0, -1).map((p) => [...p]);
    this.pipeFrom = String(pipe.metadata.from_id ?? '') || null;
    this.extendingPipe = id;
    this.updatePipePreview(null);
  }

  private addPipePoint(): void {
    // A pipe joined to a manhole ends on top of its cover, so the nearest hit
    // there is the pipe; the node under it is what a click means.
    const obj = this.scene.pickObjectIds(this.pointer)
      .map((id) => this.store.objects.get(id))
      .find((o) => o?.type === 'STORM_INLET' || o?.type === 'MANHOLE' || o?.type === 'OUTFALL');
    if (!this.pipePoints.length) {
      // v0.18.0: a pipe may also run on from a manhole
      if (obj?.type === 'STORM_INLET' || obj?.type === 'MANHOLE') {
        this.pipeFrom = obj.id;
        this.pipePoints.push([...obj.position]);
      } else {
        const point = this.scene.pickTerrain(this.pointer);
        if (!point) return;
        this.pipeFrom = null;          // the backend places an inlet here
        this.pipePoints.push([point.x, point.y, point.z]);
      }
    } else if ((obj?.type === 'OUTFALL' || obj?.type === 'MANHOLE') && !this.extendingPipe
               && obj.id !== this.pipeFrom) {
      this.pipePoints.push([...obj.position]);
      this.finishPipe(obj.id);
      return;
    } else {
      const point = this.scene.pickTerrain(this.pointer);
      if (!point) return;
      this.pipePoints.push([point.x, point.y, point.z]);
    }
    this.updatePipePreview(null);
  }

  private finishPipe(toId: string | null): void {
    const points = this.pipePoints;
    if (points.length >= 2) {
      if (this.extendingPipe) {
        this.net.send({ op: 'pipe_update', pipe: { id: this.extendingPipe, points } });
      } else {
        this.net.send({ op: 'pipe_add', pipe: {
          points, diameter_m: this.pipeDiameter,
          from_id: this.pipeFrom ?? '', to_id: toId ?? '',
        } });
      }
    }
    this.setTool('select');
  }

  private cancelPipe(): void {
    this.pipePoints = [];
    this.pipeFrom = null;
    this.extendingPipe = null;
    this.downAt = null;
    this.scene.setPipePreview(null);
  }

  private updatePipePreview(cursor: THREE.Vector3 | null): void {
    if (!this.pipePoints.length) {
      this.scene.setPipePreview(null);
      return;
    }
    const route = this.pipePoints.map((p) => new THREE.Vector3(p[0], p[1] + 0.3, p[2]));
    if (cursor) route.push(new THREE.Vector3(cursor.x, cursor.y + 0.3, cursor.z));
    this.scene.setPipePreview(route);
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
  private static readonly GROUND_FIXTURES = new Set(
    ['SOURCE', 'DRAIN', 'VENT', 'STORM_INLET', 'OUTFALL', 'MANHOLE', 'SECTION']);

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
