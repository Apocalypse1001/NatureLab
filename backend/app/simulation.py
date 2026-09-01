"""SimulationManager: fixed-timestep physics clock and authoritative world.

The loop runs in its own asyncio task, decoupled from the frontend FPS:
  real time -> accumulator -> whole fixed-dt steps -> Warp -> streamed state.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Callable, Coroutine, Dict, List, Optional

from . import config, protocol
from .compute_engine import ComputeEngine, create_engine
from .events import EventLog, EventType
from .fluid_solver import FluidSolver, ShallowWaterFluidSolver
from .persistence import load_world, save_world
from .rigid_body import ForceRigidBodySystem, RigidBodySystem
from .world_state import WorldState, finite_number, vector3

SendText = Callable[[str], Coroutine[None, None, None]]
SendBytes = Callable[[bytes], Coroutine[None, None, None]]


class SimulationManager:
    IDLE, RUNNING, PAUSED = "IDLE", "RUNNING", "PAUSED"

    def __init__(self) -> None:
        self.world = WorldState()
        self.initial: Optional[WorldState] = None
        self.engine: ComputeEngine = create_engine()
        self.fluid: FluidSolver = ShallowWaterFluidSolver()
        self.rigid: RigidBodySystem = ForceRigidBodySystem()
        self.events = EventLog()
        self.status = self.IDLE
        self.sim_time = 0.0
        self.speed = 1.0
        self.sim_fps = 0.0
        self._flush_final_frame = False
        self._last_terrain_resync = 0.0
        self._steps_in_window = 0
        self._fps_window_start = time.perf_counter()
        self._task: Optional[asyncio.Task] = None
        self._send_text: Optional[SendText] = None
        self._send_bytes: Optional[SendBytes] = None
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
            self._task = asyncio.create_task(self._loop())
        except RuntimeError:  # no running loop (headless/test use)
            self._task = None

    def engine_info(self) -> Dict[str, Any]:
        return {
            "engine": self.engine.kind,
            "warp_available": self.engine.warp_available,
            "cuda": self.engine.cuda,
            "device": self.engine.device,
            "gpu_name": self.engine.device_name,
            "selftest": self.selftest_result,
            "dt": config.FIXED_DT,
        }

    # ------------------------------------------------------------------ world edits
    def apply_object_add(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(obj, dict):
            raise ValueError("object must be a JSON object")
        created = self.world.add_object(str(obj.get("type", "BOX")),
                                        vector3(obj.get("position", [0, 0, 0]), "position"))
        created.rotation = vector3(obj.get("rotation", [0, 0, 0]), "rotation")
        created.scale = vector3(obj.get("scale", [1, 1, 1]), "scale", positive=True)
        for key in ("mass", "friction", "buoyancy"):
            if key in obj:
                setattr(created, key, finite_number(obj[key], key))
        self.rigid.register_body(created)
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
            elif key in ("mass", "friction", "buoyancy", "damage"):
                setattr(obj, key, finite_number(value, key))
            elif key == "metadata":
                if not isinstance(value, dict):
                    raise ValueError("metadata must be an object")
                obj.metadata.update(value)
            else:
                raise ValueError(f"field is not editable: {key}")
        self.rigid.update_body(obj)

    def apply_object_remove(self, obj_id: str) -> None:
        self.rigid.unregister_body(obj_id)
        self.world.objects.pop(obj_id, None)

    def apply_terrain_brush(self, x: float, z: float, radius: float,
                            strength: float) -> Dict[str, Any]:
        x = finite_number(x, "terrain.x")
        z = finite_number(z, "terrain.z")
        radius = finite_number(radius, "terrain.radius")
        strength = finite_number(strength, "terrain.strength")
        if not 0.1 <= radius <= config.WORLD_SIZE_M:
            raise ValueError("terrain.radius out of range")
        if abs(strength) > 10.0:
            raise ValueError("terrain.strength out of range")
        self.world.terrain.brush(x, z, radius, strength)
        return {"heights": self.world.terrain.to_list(),
                "checksum": self.world.terrain.checksum()}

    def apply_water_level(self, level: float) -> None:
        self.world.water.level = finite_number(level, "water.level")

    def apply_water_flow(self, enabled: bool) -> None:
        """Continuous river current toggle -- read live each tick in
        _step_once(), so unlike water_level this takes effect immediately
        even while RUNNING."""
        self.world.water.flow_enabled = bool(enabled)

    def apply_environment_temperature(self, value: float) -> None:
        """Baseline water temperature (v0.4 RiverLab, Schauberger hypothesis).

        Read live each tick by ShallowWaterFluidSolver.set_environment() --
        no re-init needed, unlike water_level which only takes effect at
        start()/reset() (see docs/04_TZ_v0.3_roadmap.md v0.4 "честные
        нюансы" for that separate, still-open gap).
        """
        self.world.environment.temperature = finite_number(value, "environment.temperature")

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
        self.engine.init_particles(config.PARTICLE_COUNT, self.world.water.level)
        self.sim_time = 0.0
        self._last_terrain_resync = 0.0
        self.status = self.RUNNING
        self.events.record(self.sim_time, EventType.SIM_STARTED, cause="user",
                           particles=config.PARTICLE_COUNT,
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
        self._last_terrain_resync = 0.0
        self.events.record(self.sim_time, EventType.SIM_RESET, cause="user")
        self.fluid.initialize(self.world)
        self.rigid.initialize(self.world, self.fluid, self.events)
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
        self.engine.init_particles(config.PARTICLE_COUNT, self.world.water.level)
        self.events.record(self.sim_time, EventType.WORLD_LOADED, cause="user",
                           name=name)

    # ------------------------------------------------------------------ physics loop
    def _step_once(self) -> None:
        dt = config.FIXED_DT
        self.engine.step_particles(dt)
        self.fluid.set_boundaries(self.world.terrain, self.rigid.obstacle_snapshot())
        self.fluid.set_environment(self.world.environment.temperature, self.rigid.shade_snapshot())
        self.fluid.set_bed_obstructions(self.rigid.bed_snapshot())
        self.fluid.set_river_flow(self.world.water.flow_enabled)
        self.fluid.advance(dt, config.FLUID_MAX_SUBSTEPS, config.FLUID_STABILITY_DT)
        samples = self.fluid.sample_for_bodies(self.rigid.buffer.positions,
                                               self.rigid.buffer.footprint_radii)
        self.rigid.step(dt, self.sim_time, samples)
        self.sim_time += dt
        self._steps_in_window += 1

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
        if self.status == self.RUNNING or self._flush_final_frame:
            if self.status == self.PAUSED:
                self._flush_final_frame = False
            positions = self.engine.visualization_positions(config.VISUALIZATION_PARTICLE_LIMIT)
            if len(positions):
                try:
                    await self._send_bytes(protocol.encode_particles(positions, self.sim_time))
                except Exception:
                    return  # client disconnected; the loop must survive
            depth_grid = self.fluid.get_depth_grid()
            if depth_grid is not None:
                try:
                    await self._send_bytes(protocol.encode_float_frame(
                        protocol.FrameKind.WATER_HEIGHT, depth_grid, self.sim_time))
                except Exception:
                    return
            if (self.status == self.RUNNING
                    and self.sim_time - self._last_terrain_resync >= config.TERRAIN_RESYNC_INTERVAL_S):
                # RiverLab (v0.4): erosion/deposition mutates world.terrain
                # directly each tick (see ShallowWaterFluidSolver), but the
                # existing terrain sync is otherwise only a reply to an
                # explicit terrain_brush op -- without this, erosion would
                # be physically real on the backend and invisible on the
                # frontend, the exact class of bug already found and fixed
                # for water (see docs/04_TZ_v0.3_roadmap.md "Архитектурный
                # принцип"). Reuses the existing terrain_patch message the
                # frontend already knows how to apply, just throttled
                # (every TERRAIN_RESYNC_INTERVAL_S) since it's JSON, not
                # the binary bulk path.
                self._last_terrain_resync = self.sim_time
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
            "particles": min(config.PARTICLE_COUNT, config.VISUALIZATION_PARTICLE_LIMIT)
            if self.status != self.IDLE else 0,
            "events": self.events.take_pending(),
            "moved_objects": [{"id": oid, "position": [float(round(p, 3)) for p in pos],
                               "state": state} for oid, pos, state in moved],
        }
        try:
            await self._send_text(json.dumps(message, separators=(",", ":")))
        except Exception:
            pass  # disconnected client must never kill the physics loop

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None
