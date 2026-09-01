"""SimulationManager: fixed-timestep physics clock and authoritative world.

The loop runs in its own asyncio task, decoupled from the frontend FPS:
  real time -> accumulator -> whole fixed-dt steps -> Warp -> streamed state.
"""
from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional

from . import config, protocol
from .compute_engine import ComputeEngine, create_engine
from .events import EventLog, EventType
from .fluid_solver import SOLID_OBSTACLE_TYPES, FluidSolver, create_fluid_solver
from .persistence import load_world, save_world
from .rigid_body import PlaceholderRigidBodySystem, RigidBodySystem
from .world_state import WorldState, finite_number, vector3

SendText = Callable[[str], Coroutine[None, None, None]]
SendBytes = Callable[[bytes], Coroutine[None, None, None]]


@dataclass
class GaugeRuntime:
    arrival_time: Optional[float] = None
    latest: Optional[Dict[str, Any]] = None
    history: deque = field(default_factory=lambda: deque(
        maxlen=config.GAUGE_HISTORY_CAPACITY))
    pending: deque = field(default_factory=lambda: deque(
        maxlen=config.GAUGE_HISTORY_CAPACITY))
    next_history_time: float = 0.0


class SimulationManager:
    IDLE, RUNNING, PAUSED = "IDLE", "RUNNING", "PAUSED"

    def __init__(self) -> None:
        self.world = WorldState()
        self.initial: Optional[WorldState] = None
        self.engine: ComputeEngine = create_engine()
        self.fluid: FluidSolver = create_fluid_solver(self.engine.device)
        self.rigid: RigidBodySystem = PlaceholderRigidBodySystem()
        self.events = EventLog()
        self.status = self.IDLE
        self.sim_time = 0.0
        self.speed = 1.0
        self.sim_fps = 0.0
        self._flush_final_frame = False
        self._steps_in_window = 0
        self._fps_window_start = time.perf_counter()
        self._task: Optional[asyncio.Task] = None
        self._send_text: Optional[SendText] = None
        self._send_bytes: Optional[SendBytes] = None
        self.terrain_revision = 0
        self.obstacle_revision = 0
        self._obstacle_snapshot: Optional[dict] = None
        self._last_terrain_resync = 0.0
        self._flooded_decks: set = set()
        self._velocity_frame = 0
        self._gauges: Dict[str, GaugeRuntime] = {}
        self.selftest_result: Dict[str, Any] = {}
        try:
            self.selftest_result = self.engine.selftest()
        except Exception as exc:  # pragma: no cover
            self.selftest_result = {"error": str(exc)}

    # ------------------------------------------------------------------ wiring
    def attach(self, send_text: SendText, send_bytes: SendBytes) -> None:
        self._send_text, self._send_bytes = send_text, send_bytes
        self._ensure_loop()

    def _ensure_loop(self) -> None:
        if self._task is not None and not self._task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # no running loop (headless/test use)
            self._task = None
            return
        self._task = loop.create_task(self._loop())

    def engine_info(self) -> Dict[str, Any]:
        return {
            "version": config.VERSION,
            "engine": self.engine.kind,
            "warp_available": self.engine.warp_available,
            "cuda": self.engine.cuda,
            "device": self.engine.device,
            "gpu_name": self.engine.device_name,
            "selftest": self.selftest_result,
            "dt": config.FIXED_DT,
            "fluid": self.fluid.diagnostics(),
        }

    def _source_snapshot(self) -> list:
        """(centre, radius, level) for every SOURCE currently in the world."""
        return [(list(obj.position),
                 float(obj.metadata.get("inflow_radius", 0.0)) * float(obj.scale[0]),
                 float(obj.metadata.get("inflow_level", 0.0)))
                for obj in self.world.objects.values() if obj.type == "SOURCE"]

    def _drain_snapshot(self) -> list:
        """(centre, radius, strength) for every DRAIN currently in the world."""
        return [(list(obj.position),
                 float(obj.metadata.get("drain_radius", 0.0)) * float(obj.scale[0]),
                 float(obj.metadata.get("drain_strength", 0.0)))
                for obj in self.world.objects.values() if obj.type == "DRAIN"]

    def apply_water_outflow(self, enabled: bool) -> None:
        """Open/close the downstream map edge. Read live each tick."""
        self.world.water.outflow_enabled = bool(enabled)

    @staticmethod
    def _affects_fluid_boundary(obj) -> bool:
        """Does this object change what the fluid solver sees as a boundary?

        True for solid walls (HOUSE) and for riverbed bodies (a positive
        `bed_height`, i.e. ROCK). Both go through the same obstacle-revision
        path, so adding, moving, scaling or deleting either one re-rasterizes
        the solid mask AND the bed domes. This used to be a bare
        `type == "HOUSE"` check in three places, which meant a ROCK could be
        dragged across the map without the water ever noticing.
        """
        return (obj.type in SOLID_OBSTACLE_TYPES
                or float(obj.metadata.get("bed_height", 0.0)) > 0.0)

    # ------------------------------------------------------------------ world edits
    def apply_object_add(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(obj, dict):
            raise ValueError("object must be a JSON object")
        created = self.world.add_object(str(obj.get("type", "BOX")),
                                        vector3(obj.get("position", [0, 0, 0]), "position"))
        created.rotation = vector3(obj.get("rotation", [0, 0, 0]), "rotation")
        created.scale = vector3(obj.get("scale", [1, 1, 1]), "scale", positive=True)
        for key in ("mass", "friction", "buoyancy", "volume_m3",
                    "drag_coefficient", "ground_contact_area", "cross_sectional_area"):
            if key in obj:
                value = finite_number(obj[key], key)
                if key not in ("friction", "buoyancy") and value <= 0.0:
                    raise ValueError(f"{key} must be greater than zero")
                if key == "buoyancy" and not 0.0 <= value <= 1.0:
                    raise ValueError("buoyancy must be between 0 and 1")
                setattr(created, key, value)
        self.rigid.register_body(created)
        if created.type == "GAUGE":
            self._gauges[created.id] = GaugeRuntime(next_history_time=self.sim_time)
        if self._affects_fluid_boundary(created):
            self.obstacle_revision += 1
            self._obstacle_snapshot = None
        return created.to_dict()

    def apply_object_update(self, obj_id: str, fields: Dict[str, Any]) -> None:
        if not isinstance(fields, dict):
            raise ValueError("fields must be a JSON object")
        obj = self.world.objects.get(obj_id)
        if obj is None:
            raise ValueError(f"object not found: {obj_id}")
        for key, value in fields.items():
            if key in ("position", "rotation"):
                setattr(obj, key, vector3(value, key))
            elif key == "scale":
                obj.scale = vector3(value, key, positive=True)
            elif key in ("mass", "friction", "buoyancy", "damage", "volume_m3",
                         "drag_coefficient", "ground_contact_area",
                         "cross_sectional_area"):
                number = finite_number(value, key)
                if key in ("mass", "volume_m3", "drag_coefficient",
                           "ground_contact_area", "cross_sectional_area") and number <= 0.0:
                    raise ValueError(f"{key} must be greater than zero")
                if key == "friction" and number < 0.0:
                    raise ValueError("friction must be non-negative")
                if key == "buoyancy" and not 0.0 <= number <= 1.0:
                    raise ValueError("buoyancy must be between 0 and 1")
                setattr(obj, key, number)
            elif key == "metadata":
                if not isinstance(value, dict):
                    raise ValueError("metadata must be an object")
                obj.metadata.update(value)
            else:
                raise ValueError(f"field is not editable: {key}")
        self.rigid.update_body(obj)
        if obj.type == "GAUGE" and "position" in fields:
            self._gauges[obj.id] = GaugeRuntime(next_history_time=self.sim_time)
        if (self._affects_fluid_boundary(obj)
                and any(key in fields for key in ("position", "rotation", "scale", "metadata"))):
            self.obstacle_revision += 1
            self._obstacle_snapshot = None

    def apply_object_remove(self, obj_id: str) -> None:
        obj = self.world.objects.get(obj_id)
        self.rigid.unregister_body(obj_id)
        self.world.objects.pop(obj_id, None)
        self._gauges.pop(obj_id, None)
        if obj is not None and self._affects_fluid_boundary(obj):
            self.obstacle_revision += 1
            self._obstacle_snapshot = None

    def apply_terrain_brush(self, x: float, z: float, radius: float,
                            strength: float) -> Dict[str, Any]:
        if self.status == self.RUNNING:
            raise ValueError("terrain editing is disabled while simulation is RUNNING")
        x = finite_number(x, "terrain.x")
        z = finite_number(z, "terrain.z")
        radius = finite_number(radius, "terrain.radius")
        strength = finite_number(strength, "terrain.strength")
        if not 0.1 <= radius <= config.WORLD_SIZE_M:
            raise ValueError("terrain.radius out of range")
        if abs(strength) > 10.0:
            raise ValueError("terrain.strength out of range")
        self.world.terrain.brush(x, z, radius, strength)
        self.terrain_revision += 1
        return {"heights": self.world.terrain.to_list(),
                "checksum": self.world.terrain.checksum()}

    def apply_water_level(self, level: float) -> None:
        self.world.water.level = finite_number(level, "water.level")
        if hasattr(self.fluid, "set_level"):
            self.fluid.set_level(self.world.water.level)

    def apply_water_erosion(self, enabled: bool) -> None:
        """RiverLab erosion toggle. Read live each tick in _step_once(), so --
        unlike a terrain brush -- it takes effect immediately while RUNNING."""
        self.world.water.erosion_enabled = bool(enabled)

    # ------------------------------------------------------------------ clock control
    def start(self) -> None:
        if self.status == self.RUNNING:
            return
        if self.status == self.PAUSED:
            self.status = self.RUNNING
            self.events.record(self.sim_time, EventType.SIM_STARTED, cause="resume")
            return
        self.initial = self.world.clone()
        self.fluid.initialize(self.world)
        self.rigid.initialize(self.world, self.fluid, self.events)
        self._reset_gauges()
        self._obstacle_snapshot = self.rigid.obstacle_snapshot()
        self.fluid.set_boundaries(self.world.terrain, self._obstacle_snapshot,
                                  self.terrain_revision, self.obstacle_revision)
        if config.ENABLE_DEBUG_PARTICLES:
            self.engine.init_particles(config.PARTICLE_COUNT, self.world.water.level)
        self.sim_time = 0.0
        self.status = self.RUNNING
        self.events.record(self.sim_time, EventType.SIM_STARTED, cause="user",
                           particles=config.PARTICLE_COUNT if config.ENABLE_DEBUG_PARTICLES else 0,
                           engine=self.engine.kind, device=self.engine.device)
        if self._task is None or self._task.done():
            self._ensure_loop()

    def pause(self) -> None:
        if self.status == self.RUNNING:
            self.status = self.PAUSED
            self._flush_final_frame = True
            self.events.record(self.sim_time, EventType.SIM_PAUSED, cause="user")

    def reset(self) -> None:
        if self.initial is not None:
            self.world = self.initial.clone()
        self.status = self.IDLE
        self.sim_time = 0.0
        self.events.record(self.sim_time, EventType.SIM_RESET, cause="user")
        self.fluid.initialize(self.world)
        self.rigid.initialize(self.world, self.fluid, self.events)
        self._reset_gauges()
        self._obstacle_snapshot = self.rigid.obstacle_snapshot()
        self.fluid.set_boundaries(self.world.terrain, self._obstacle_snapshot,
                                  self.terrain_revision, self.obstacle_revision)
        if config.ENABLE_DEBUG_PARTICLES:
            self.engine.init_particles(config.PARTICLE_COUNT, self.world.water.level)

    def set_speed(self, value: float) -> None:
        value = finite_number(value, "speed")
        if value not in config.SPEED_OPTIONS:
            raise ValueError("unsupported simulation speed")
        self.speed = value

    # ------------------------------------------------------------------ persistence
    def save(self, name: str = "default") -> str:
        path = save_world(self.world, name)
        self.events.record(self.sim_time, EventType.WORLD_SAVED, cause="user",
                           name=name)
        return path

    def load(self, name: str = "default") -> None:
        loaded = load_world(name)
        if self.status != self.IDLE:
            self.pause()
            self.reset()
        self.world = loaded
        self.initial = None
        self.status = self.IDLE
        self.sim_time = 0.0
        self._last_terrain_resync = 0.0
        self.fluid.initialize(self.world)
        self.rigid.initialize(self.world, self.fluid, self.events)
        self._reset_gauges()
        self.terrain_revision += 1
        self.obstacle_revision += 1
        self._obstacle_snapshot = self.rigid.obstacle_snapshot()
        self.fluid.set_boundaries(self.world.terrain, self._obstacle_snapshot,
                                  self.terrain_revision, self.obstacle_revision)
        if config.ENABLE_DEBUG_PARTICLES:
            self.engine.init_particles(config.PARTICLE_COUNT, self.world.water.level)
        self.events.record(self.sim_time, EventType.WORLD_LOADED, cause="user",
                           name=name)

    # ------------------------------------------------------------------ physics loop
    def _step_once(self) -> None:
        dt = config.FIXED_DT
        if config.ENABLE_DEBUG_PARTICLES:
            self.engine.step_particles(dt)
        if hasattr(self.fluid, "set_erosion"):
            # read live, so the RiverLab toggle takes effect while RUNNING
            self.fluid.set_erosion(self.world.water.erosion_enabled)
        if hasattr(self.fluid, "set_outflow"):
            self.fluid.set_outflow(config.FLUID_OUTFLOW_COLUMNS
                                   if self.world.water.outflow_enabled else 0)
        if hasattr(self.fluid, "set_water_features"):
            # Positions are read live every tick rather than snapshotted, so a
            # SOURCE or DRAIN can be dragged while RUNNING and the water reacts
            # at once -- the reason they are objects and not settings.
            self.fluid.set_water_features(self._source_snapshot(),
                                          self._drain_snapshot())
        if self._obstacle_snapshot is None:
            self._obstacle_snapshot = self.rigid.obstacle_snapshot()
        self.fluid.set_boundaries(self.world.terrain, self._obstacle_snapshot,
                                  self.terrain_revision, self.obstacle_revision)
        self.fluid.advance(dt, config.FLUID_MAX_SUBSTEPS, config.FLUID_STABILITY_DT)
        samples = self.fluid.sample_for_bodies(
            self.rigid.buffer.positions, self.rigid.buffer.velocities,
            self.rigid.buffer.drag_coefficients, self.rigid.buffer.cross_areas,
            self.rigid.buffer.body_heights, self.rigid.buffer.rotations,
            self.rigid.buffer.half_extents)
        self.rigid.step(dt, self.sim_time, samples)
        self.sim_time += dt
        self._update_gauges(self.sim_time)
        self._check_bridge_decks(self.sim_time)
        self._steps_in_window += 1

    def _check_bridge_decks(self, sample_time: float) -> None:
        """Report the moment water rises to a bridge's deck.

        A bridge's piers obstruct the flow but its deck does not -- water passes
        underneath it, which is what a bridge is for. The deck becomes relevant
        exactly once: when the river reaches it. That is a real, observable
        cause-and-effect moment ("the bridge went under at 41 s") and it is the
        kind of thing docs/01_vision.md wants recorded, so it is an event rather
        than a silently changed number.

        Fires once per bridge per run, like the GAUGE arrival event, so a river
        lapping at the deck does not spam the log.
        """
        for oid, obj in self.world.objects.items():
            if obj.type != "BRIDGE" or oid in self._flooded_decks:
                continue
            sample = self.rigid.latest_fluid_samples.get(oid)
            if sample is None:
                continue
            deck = float(obj.metadata.get("deck_height", 0.0)) * float(obj.scale[1])
            surface = float(sample.get("surface", 0.0))
            if deck > 0.0 and surface >= obj.position[1] + deck:
                self._flooded_decks.add(oid)
                self.events.record(sample_time, EventType.BRIDGE_DECK_FLOODED,
                                   object_id=oid, cause="water_reached_deck",
                                   deck_height_m=round(deck, 3),
                                   water_surface_m=round(surface, 3))

    def _reset_gauges(self) -> None:
        self._flooded_decks = set()
        self._gauges = {oid: GaugeRuntime()
                        for oid, obj in self.world.objects.items()
                        if obj.type == "GAUGE"}

    def _update_gauges(self, sample_time: float) -> None:
        for oid, runtime in self._gauges.items():
            fluid_sample = self.rigid.latest_fluid_samples.get(oid)
            if fluid_sample is None:
                continue
            depth = float(fluid_sample["depth"])
            velocity = fluid_sample["velocity"]
            speed = float((velocity[0] ** 2 + velocity[2] ** 2) ** 0.5)
            wet = depth > config.FLUID_DRY_DEPTH
            sample = {
                "time_s": round(sample_time, 3),
                "water_depth_m": depth,
                "surface_elevation_m": float(fluid_sample["surface"]) if wet else None,
                "speed_m_s": speed,
            }
            runtime.latest = sample
            if wet and runtime.arrival_time is None:
                runtime.arrival_time = sample_time
                self.events.record(sample_time, EventType.WATER_ENTERED_AREA, oid,
                                   cause="gauge_depth_threshold_crossed",
                                   threshold_m=config.FLUID_DRY_DEPTH,
                                   water_depth_m=depth,
                                   surface_elevation_m=sample["surface_elevation_m"],
                                   speed_m_s=speed)
            if sample_time + 1.0e-9 >= runtime.next_history_time:
                runtime.history.append(sample)
                runtime.pending.append(sample)
                while runtime.next_history_time <= sample_time + 1.0e-9:
                    runtime.next_history_time += config.GAUGE_HISTORY_INTERVAL

    def _serialize_gauges(self) -> List[Dict[str, Any]]:
        result = []
        for oid, runtime in self._gauges.items():
            result.append({
                "id": oid,
                "arrival_time_s": (round(runtime.arrival_time, 3)
                                   if runtime.arrival_time is not None else None),
                "latest": runtime.latest,
                "samples": list(runtime.pending),
            })
            runtime.pending.clear()
        return result

    async def _loop(self) -> None:
        interval = 1.0 / config.STREAM_HZ
        last = time.perf_counter()
        accumulator = 0.0
        try:
            while True:
                await asyncio.sleep(interval)
                now = time.perf_counter()
                elapsed = now - last
                last = now
                if self.status == self.RUNNING:
                    accumulator = min(accumulator + elapsed * self.speed,
                                      config.MAX_SUBSTEPS * config.FIXED_DT)
                    while accumulator >= config.FIXED_DT:
                        self._step_once()
                        accumulator -= config.FIXED_DT
                if now - self._fps_window_start >= 0.5:
                    self.sim_fps = self._steps_in_window / (now - self._fps_window_start)
                    self._steps_in_window = 0
                    self._fps_window_start = now
                await self._stream()
        except asyncio.CancelledError:  # pragma: no cover
            self._task = None
            raise

    async def _stream(self) -> None:
        if self._send_text is None or self._send_bytes is None:
            return
        particle_count = 0
        if self.status == self.RUNNING or self._flush_final_frame:
            if self.status == self.PAUSED:
                self._flush_final_frame = False
            positions = self.fluid.get_flow_particles()
            particle_count = len(positions)
            if len(positions):
                try:
                    await self._send_bytes(protocol.encode_particles(positions, self.sim_time))
                except Exception:
                    return  # client disconnected; the loop must survive
            heights = self.fluid.get_water_height_field()
            if len(heights):
                try:
                    await self._send_bytes(protocol.encode_water_height(heights, self.sim_time))
                except Exception:
                    return
            # v0.10.0: the real velocity field, so the renderer can build its
            # flow map from physics instead of inventing one. Throttled to
            # config.VELOCITY_STREAM_EVERY frames: a flow map is a low-frequency
            # visual signal and does not need every tick, while the field is the
            # largest payload on the wire after the tracers.
            self._velocity_frame += 1
            if self._velocity_frame % config.VELOCITY_STREAM_EVERY == 0:
                velocities = self.fluid.get_velocity_field()
                if velocities is not None and len(velocities):
                    try:
                        await self._send_bytes(
                            protocol.encode_velocity_field(velocities, self.sim_time))
                    except Exception:
                        return
            # RiverLab (v0.6.0): erosion mutates the bed on the GPU every tick,
            # so without this the new channel would be physically real on the
            # backend and invisible on screen -- the exact class of bug this
            # project already hit once with water itself (see the architectural
            # principle in docs/04_TZ_v0.3_roadmap.md). Reuses the terrain_patch
            # message the frontend already applies, throttled because it is JSON
            # on the text channel plus a full device-to-host readback.
            if (self.status == self.RUNNING and self.world.water.erosion_enabled
                    and hasattr(self.fluid, "get_terrain_heights")
                    and self.sim_time - self._last_terrain_resync
                    >= config.TERRAIN_RESYNC_INTERVAL_S):
                self._last_terrain_resync = self.sim_time
                eroded = self.fluid.get_terrain_heights()
                if eroded.size == self.world.terrain.heights.size:
                    self.world.terrain.heights = eroded.reshape(
                        self.world.terrain.heights.shape)
                    try:
                        await self._send_text(json.dumps({
                            "type": "terrain_patch",
                            "heights": self.world.terrain.to_list(),
                            "checksum": self.world.terrain.checksum(),
                        }, separators=(",", ":")))
                    except Exception:
                        return
        moved = [(oid, obj.position, obj.state)
                 for oid, obj in self.world.objects.items()
                 if obj.state != "INTACT"]
        message = {
            "type": "sim_state",
            "status": self.status,
            "time": round(self.sim_time, 3),
            "speed": self.speed,
            "sim_fps": round(self.sim_fps, 1),
            "objects": len(self.world.objects),
            "particles": particle_count,
            "gauge_history_capacity": config.GAUGE_HISTORY_CAPACITY,
            "gauges": self._serialize_gauges(),
            "events": self.events.take_pending(),
            "moved_objects": [{"id": oid, "position": [float(round(p, 3)) for p in pos],
                                "state": state} for oid, pos, state in moved],
            "fluid": self.fluid.diagnostics(),
        }
        try:
            await self._send_text(json.dumps(message, separators=(",", ":")))
        except Exception:
            pass  # disconnected client must never kill the physics loop

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None
