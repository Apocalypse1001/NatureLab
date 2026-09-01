/** Shared data model — mirrors backend/app/world_state.py. */

export type ObjectType = 'HOUSE' | 'CAR' | 'TREE' | 'BOX' | 'DEBRIS' | 'ROCK' | 'GAUGE';

export type ObjectState =
  | 'INTACT' | 'MOVING' | 'FLOATING' | 'COLLIDING'
  | 'DAMAGED' | 'BROKEN' | 'SETTLED';

export interface ObjectData {
  id: string;
  type: ObjectType;
  position: number[];   // meters [x, y, z]
  rotation: number[];   // radians (euler XYZ)
  scale: number[];
  mass: number;         // kg
  friction: number;
  buoyancy: number;
  volume_m3: number;
  drag_coefficient: number;
  ground_contact_area: number;
  cross_sectional_area: number;
  is_static: boolean;
  damage: number;       // 0..1
  state: ObjectState;
  metadata: Record<string, number>;
}

export interface TerrainData {
  width: number;
  height: number;
  cell_size: number;
  heights: number[];    // (width+1) * (height+1), row-major, z rows
}

export interface WorldData {
  version: number;
  terrain: TerrainData;
  water: { level: number; visible: boolean; erosion_enabled?: boolean };
  environment: { gravity: number; wind: number[]; temperature: number };
  objects: ObjectData[];
}

export interface EngineInfo {
  engine: string;
  warp_available: boolean;
  cuda: boolean;
  device: string;
  gpu_name: string;
  selftest: Record<string, unknown>;
  dt: number;
}

export interface SimStateMessage {
  type: 'sim_state';
  status: 'IDLE' | 'RUNNING' | 'PAUSED';
  time: number;
  speed: number;
  sim_fps: number;
  objects: number;
  particles: number;
  events: SimEvent[];
  moved_objects: { id: string; position: number[]; state: ObjectState }[];
  gauge_history_capacity: number;
  gauges: GaugeState[];
  fluid?: { solver: string; device?: string; grid?: number[]; substeps: number;
            wet_cells?: number; volume_m3?: number; cfl_dt?: number;
            max_depth?: number; max_velocity?: number };
}

export interface GaugeSample {
  time_s: number;
  water_depth_m: number;
  surface_elevation_m: number | null;
  speed_m_s: number;
}

export interface GaugeState {
  id: string;
  arrival_time_s: number | null;
  latest: GaugeSample | null;
  samples: GaugeSample[];
}

export interface SimEvent {
  time: number;
  type: string;
  object_id: string | null;
  cause: string;
  parameters: Record<string, unknown>;
}

/** Add new object types here (+ backend defaults) without touching the core. */
export const OBJECT_TYPES: ObjectType[] = ['HOUSE', 'CAR', 'TREE', 'BOX', 'DEBRIS', 'ROCK', 'GAUGE'];

export const OBJECT_COLORS: Record<string, number> = {
  HOUSE: 0xc9a27a,
  CAR: 0x4d7fff,
  TREE: 0x3f9d4e,
  BOX: 0xb08050,
  DEBRIS: 0x808080,
  ROCK: 0x8d8577,
  GAUGE: 0x62e6ff,
};
