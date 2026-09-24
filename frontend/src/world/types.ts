/** Shared data model — mirrors backend/app/world_state.py. */

export type ObjectType = 'HOUSE' | 'CAR' | 'TREE' | 'BOX' | 'DEBRIS' | 'ROCK'
  | 'BRIDGE' | 'PERSON' | 'ROAD' | 'SOURCE' | 'DRAIN' | 'GAUGE' | 'VENT' | 'BUILDING'
  // v0.17.0 storm sewer (backend/app/sewer.py)
  | 'STORM_INLET' | 'OUTFALL' | 'PIPE'
  // v0.18.0 Sewer-2: where pipes join
  | 'MANHOLE'
  // v0.18.0: a gauging line across the flow (backend/app/sections.py)
  | 'SECTION';

/** What one PIPE is doing, streamed in sim_state and in the pipe_* replies. */
export interface SewerLinkState {
  pipe_id: string;
  /** a STORM_INLET or MANHOLE */
  from_id: string;
  /** a MANHOLE or OUTFALL */
  to_id: string;
  to_type: string;
  capacity_m3s: number;
  /** summed over every grate whose chain runs through this pipe */
  flow_m3s: number;
  length_m: number;
  fall_m: number;
  status: 'ok' | 'uphill' | 'disconnected' | 'second_pipe' | 'blocked' | 'dead_end' | 'loop';
  /** for "blocked": the pipe on the way that carries nothing */
  blocked_by: string;
  /** v0.18.0: pipe-bottom heights at the two ends; fall_m is their difference */
  from_invert_m: number;
  to_invert_m: number;
  /** for "uphill": how deep the end node would need to be for a 0.5% grade */
  suggested_to_depth_m: number;
  upstream_inlets: string[];
}

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
  // Numbers for every type, plus a PIPE's route and joints
  // (`points` [[x,y,z],...], `from_id`, `to_id`).
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  metadata: Record<string, any>;
}

export interface TerrainData {
  width: number;
  height: number;
  cell_size: number;
  heights: number[];    // (width+1) * (height+1), row-major, z rows
  /** Surface class per vertex (backend config.SURFACE_CLASSES), same order as
   *  heights; only on a surveyed site. */
  surface?: number[];
}

/** Beyond the open east edge: a sea or pool (free overfall), or a river running on. */
export type OutletKind = 'overfall' | 'river';

export interface WorldData {
  version: number;
  terrain: TerrainData;
  /** A surveyed world's provenance (tools/make_site_nl01.py); `parcel` is the
   *  plot boundary as [x, z] metres. Absent on generated worlds. */
  site?: { id?: string; parcel?: number[][]; [key: string]: unknown };
  water: { level: number; visible: boolean; erosion_enabled?: boolean;
           outflow_enabled?: boolean;
           // v0.12.0 river boundaries: part of the world, so a loaded world has
           // to be able to put its own numbers back on the controls
           inlet_enabled?: boolean; inlet_width_m?: number;
           inlet_discharge_m3s?: number; outlet_width_m?: number;
           // v0.18.0 flood hydrograph on the inlet (backend/app/hydrograph.py)
           hydrograph?: Hydrograph;
           // v0.18.1: whether the world starts with its river flowing
           initial_flow?: boolean;
           // v0.16.0: what lies beyond the open east edge
           outlet_kind?: OutletKind;
           // RainLab-1 (docs/14_rain_plan.md): rain in mm/h (= L/m² per hour),
           // and whether the west edge holds its inflow level at all
           rain_intensity_mm_h?: number; edge_inflow_enabled?: boolean;
           open_sides?: boolean;
           tsunami_enabled?: boolean };
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
  moved_objects: { id: string; position: number[]; state: ObjectState; damage: number }[];
  gauge_history_capacity: number;
  gauges: GaugeState[];
  sections?: SectionState[];
  sewer?: SewerLinkState[];
  fluid?: { solver: string; device?: string; grid?: number[]; substeps: number;
            wet_cells?: number; volume_m3?: number; cfl_dt?: number;
            max_depth?: number; max_velocity?: number;
            erosion?: boolean; outflow_columns?: number; cfl_limited?: boolean;
            open_sides?: boolean;
            outlet_kind?: OutletKind; lava_enabled?: boolean;
            // v0.12.0 river boundaries and their volume ledger
            inlet_enabled?: boolean; inlet_request_m3s?: number;
            inlet_discharge_m3s?: number; added_m3?: number; removed_m3?: number;
            volume_error_m3?: number; sediment_out_m3?: number;
            // RainLab-1: what the solver is actually applying, not the slider
            rain_mm_h?: number; rain_m3s?: number; edge_inflow?: boolean };
}

export interface GaugeSample {
  time_s: number;
  water_depth_m: number;
  surface_elevation_m: number | null;
  speed_m_s: number;
}

/** v0.18.0: a flood on top of the inlet's base discharge, in sim seconds. */
export interface Hydrograph {
  enabled: boolean;
  peak_m3s: number;
  start_s: number;
  rise_s: number;
  fall_s: number;
}

/** v0.18.0: what a gauging line reads over one frame. */
export interface SectionSample {
  time_s: number;
  /** net discharge back to front through the line, m3/s (negative = backwards) */
  flow_m3s: number;
  area_m2: number;
  wetted_width_m: number;
  mean_depth_m: number;
  mean_velocity_m_s: number;
  /** the SECTION's Froude number V / sqrt(g A/B), not a local maximum */
  froude_section: number;
  level_m: number | null;
  volume_m3: number;
}

export interface SectionState {
  id: string;
  latest: SectionSample | null;
  samples: SectionSample[];
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
export const OBJECT_TYPES: ObjectType[] = ['HOUSE', 'BUILDING', 'CAR', 'TREE', 'BOX', 'DEBRIS',
  'ROCK', 'BRIDGE', 'PERSON', 'ROAD', 'SOURCE', 'DRAIN', 'GAUGE', 'SECTION', 'VENT',
  // a PIPE is laid with the Pipe tool, not dropped on a grid
  'STORM_INLET', 'MANHOLE', 'OUTFALL'];

export const OBJECT_COLORS: Record<string, number> = {
  HOUSE: 0xc9a27a,
  CAR: 0x4d7fff,
  TREE: 0x3f9d4e,
  BOX: 0xb08050,
  DEBRIS: 0x808080,
  ROCK: 0x8d8577,
  BRIDGE: 0xb9a37e,
  PERSON: 0xf2d64b,
  ROAD: 0x4a4a52,
  SOURCE: 0x4fd8a0,
  STORM_INLET: 0x5b6470,
  OUTFALL: 0x9aa3ad,
  MANHOLE: 0x3d4248,
  PIPE: 0x8a939c,
  DRAIN: 0xd85f4f,
  GAUGE: 0x62e6ff,
  SECTION: 0xffd23f,
  VENT: 0xff5a1f,
  BUILDING: 0x9aa0ab,
};
