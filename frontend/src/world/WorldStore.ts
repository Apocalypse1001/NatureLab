/** Logical world state shared by all frontend modules.
 *  The backend stays authoritative for simulation; this store mirrors the
 *  editor state and applies streamed simulation updates. */
import { TerrainGrid } from './TerrainGrid';
import type { ObjectData, WorldData } from './types';

export type StoreEvent =
  | 'objects-changed' | 'object-updated' | 'terrain-changed'
  | 'water-changed' | 'world-replaced' | 'selection-changed';

type Listener = (payload?: unknown) => void;

export class WorldStore {
  terrain = new TerrainGrid();
  objects = new Map<string, ObjectData>();
  waterLevel = 0.5;
  waterVisible = true;
  selectedId: string | null = null;

  private listeners = new Map<StoreEvent, Set<Listener>>();

  on(event: StoreEvent, fn: Listener): void {
    if (!this.listeners.has(event)) this.listeners.set(event, new Set());
    this.listeners.get(event)!.add(fn);
  }

  private emit(event: StoreEvent, payload?: unknown): void {
    this.listeners.get(event)?.forEach((fn) => fn(payload));
  }

  // ------------------------------------------------------------------ bulk
  replaceWorld(world: WorldData): void {
    this.terrain = new TerrainGrid(world.terrain.width, world.terrain.height,
                                   world.terrain.cell_size);
    this.terrain.loadHeights(world.terrain.heights);
    this.objects = new Map(world.objects.map((o) => [o.id, o]));
    this.waterLevel = world.water.level;
    this.waterVisible = world.water.visible;
    this.selectedId = null;
    this.emit('world-replaced');
  }

  // ------------------------------------------------------------------ objects
  addObject(obj: ObjectData): void {
    this.objects.set(obj.id, obj);
    this.emit('objects-changed');
  }

  updateObject(id: string, patch: Partial<ObjectData>): void {
    const obj = this.objects.get(id);
    if (!obj) return;
    Object.assign(obj, patch);
    this.emit('object-updated', id);
  }

  removeObject(id: string): void {
    if (!this.objects.delete(id)) return;
    if (this.selectedId === id) this.select(null);
    this.emit('objects-changed');
  }

  select(id: string | null): void {
    if (id !== null && !this.objects.has(id)) id = null;
    this.selectedId = id;
    this.emit('selection-changed', id);
  }

  // ------------------------------------------------------------------ terrain / water
  brushTerrain(x: number, z: number, radius: number, strength: number): void {
    this.terrain.brush(x, z, radius, strength);
    this.emit('terrain-changed');
  }

  setWaterLevel(level: number): void {
    this.waterLevel = level;
    this.emit('water-changed');
  }
}
