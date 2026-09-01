"""Data-oriented first-order rigid-body and fluid-force coupling."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from . import config
from .compute_engine import WARP_IMPORTED, wp
from .events import EventLog, EventType
from .world_state import ObjectState, WorldObject


STATE_NAMES = [ObjectState.INTACT.value, ObjectState.MOVING.value,
               ObjectState.FLOATING.value, ObjectState.SETTLED.value]
STATE_CODES = {name: i for i, name in enumerate(STATE_NAMES)}


if WARP_IMPORTED:
    @wp.kernel
    def _integrate_bodies(positions: wp.array(dtype=wp.vec3),
                          velocities: wp.array(dtype=wp.vec3),
                          masses: wp.array(dtype=float),
                           frictions: wp.array(dtype=float),
                           buoyancies: wp.array(dtype=float),
                           volumes: wp.array(dtype=float),
                          ground_areas: wp.array(dtype=float),
                          static: wp.array(dtype=wp.int32),
                          states: wp.array(dtype=wp.int32),
                           immersions: wp.array(dtype=float),
                          forces: wp.array(dtype=wp.vec3),
                          dynamic: wp.array(dtype=wp.int32),
                          dt: float, gravity: float, rho: float):
        i = wp.tid()
        depth = wp.max(0.0, immersions[i])
        height = volumes[i] / wp.max(ground_areas[i], 1.0e-6)
        submerged = wp.clamp(depth / wp.max(height, 1.0e-6), 0.0, 1.0)
        buoyancy = buoyancies[i] * rho * gravity * volumes[i] * submerged
        weight = masses[i] * gravity
        floating = static[i] == 0 and buoyancy >= weight * 0.999
        force = forces[i]
        drive = wp.sqrt(force.x * force.x + force.z * force.z)
        hold = frictions[i] * wp.max(weight - buoyancy, 0.0)
        sliding = static[i] == 0 and not floating and drive > hold
        moving = floating or sliding
        velocity = velocities[i]
        position = positions[i]
        if moving:
            scale = float(1.0)
            if sliding and drive > 1.0e-8:
                scale = wp.max(0.0, 1.0 - hold / drive)
            acceleration = wp.vec3(force.x * scale / masses[i], 0.0,
                                   force.z * scale / masses[i])
            velocity = velocity + acceleration * dt
            position = position + velocity * dt
            if floating:
                states[i] = 2
            else:
                states[i] = 1
            dynamic[i] = 1
        else:
            velocity = wp.vec3(0.0, 0.0, 0.0)
            if states[i] == 1 or states[i] == 2:
                states[i] = 3
            dynamic[i] = 0
        positions[i] = position
        velocities[i] = velocity


@dataclass
class RigidStateBuffer:
    ids: List[str] = field(default_factory=list)
    index: Dict[str, int] = field(default_factory=dict)
    positions: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    velocities: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    rotations: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    scales: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    half_extents: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float32))
    masses: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    frictions: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    buoyancies: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    volumes: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    drag_coefficients: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    ground_areas: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    cross_areas: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    static: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.bool_))
    states: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int16))

    def register(self, obj: WorldObject) -> int:
        if obj.id in self.index:
            self.update(obj)
            return self.index[obj.id]
        idx = len(self.ids)
        self.ids.append(obj.id)
        self.index[obj.id] = idx
        self.positions = np.vstack((self.positions, np.asarray(obj.position, dtype=np.float32)))
        self.velocities = np.vstack((self.velocities, np.zeros(3, dtype=np.float32)))
        self.rotations = np.vstack((self.rotations, np.asarray(obj.rotation, dtype=np.float32)))
        self.scales = np.vstack((self.scales, np.asarray(obj.scale, dtype=np.float32)))
        self.half_extents = np.vstack((self.half_extents, footprint_half_extents(obj)))
        self.masses = np.append(self.masses, np.float32(obj.mass))
        self.frictions = np.append(self.frictions, np.float32(obj.friction))
        self.buoyancies = np.append(self.buoyancies, np.float32(obj.buoyancy))
        self.volumes = np.append(self.volumes, np.float32(obj.volume_m3))
        self.drag_coefficients = np.append(self.drag_coefficients,
                                           np.float32(obj.drag_coefficient))
        self.ground_areas = np.append(self.ground_areas,
                                      np.float32(obj.ground_contact_area))
        self.cross_areas = np.append(self.cross_areas,
                                     np.float32(obj.cross_sectional_area))
        self.static = np.append(self.static, np.bool_(obj.is_static))
        self.states = np.append(self.states,
                                np.int16(STATE_CODES.get(obj.state, 0)))
        return idx

    def update(self, obj: WorldObject) -> None:
        idx = self.index.get(obj.id)
        if idx is None:
            self.register(obj)
            return
        self.positions[idx] = obj.position
        self.rotations[idx] = obj.rotation
        self.scales[idx] = obj.scale
        self.half_extents[idx] = footprint_half_extents(obj)
        self.masses[idx] = obj.mass
        self.frictions[idx] = obj.friction
        self.buoyancies[idx] = obj.buoyancy
        self.volumes[idx] = obj.volume_m3
        self.drag_coefficients[idx] = obj.drag_coefficient
        self.ground_areas[idx] = obj.ground_contact_area
        self.cross_areas[idx] = obj.cross_sectional_area
        self.static[idx] = obj.is_static
        self.states[idx] = STATE_CODES.get(obj.state, int(self.states[idx]))

    def unregister(self, object_id: str) -> None:
        idx = self.index.pop(object_id, None)
        if idx is None:
            return
        last = len(self.ids) - 1
        arrays = (self.positions, self.velocities, self.rotations, self.scales,
                  self.half_extents, self.masses, self.frictions, self.buoyancies,
                  self.volumes, self.drag_coefficients,
                  self.ground_areas, self.cross_areas, self.static, self.states)
        if idx != last:
            moved_id = self.ids[last]
            self.ids[idx] = moved_id
            self.index[moved_id] = idx
            for array in arrays:
                array[idx] = array[last]
        self.ids.pop()
        for name in ("positions", "velocities", "rotations", "scales", "half_extents",
                     "masses", "frictions", "buoyancies", "volumes",
                     "drag_coefficients", "ground_areas", "cross_areas",
                     "static", "states"):
            setattr(self, name, getattr(self, name)[:-1])

    @property
    def body_heights(self) -> np.ndarray:
        return self.volumes / np.maximum(self.ground_areas, 1.0e-6)


def footprint_half_extents(obj: WorldObject) -> np.ndarray:
    base = {"HOUSE": (2.0, 2.0), "CAR": (2.2, 1.0), "TREE": (0.25, 0.25),
            "BOX": (0.6, 0.6), "DEBRIS": (0.6, 0.6), "GAUGE": (0.0, 0.0)}
    hx, hz = base.get(obj.type, (0.5, 0.5))
    return np.asarray((hx * obj.scale[0], hz * obj.scale[2]), dtype=np.float32)


class RigidBodySystem:
    def initialize(self, world, fluid, events: EventLog) -> None: ...
    def register_body(self, obj: WorldObject) -> None: ...
    def update_body(self, obj: WorldObject) -> None: ...
    def unregister_body(self, object_id: str) -> None: ...
    def obstacle_snapshot(self) -> dict: ...
    def step(self, dt: float, sim_time: float, fluid_samples=None) -> List[str]: ...


class PlaceholderRigidBodySystem(RigidBodySystem):
    """Vectorized force model; angular dynamics remain outside 0.4."""

    def __init__(self) -> None:
        self._world = self._fluid = None
        self._events: EventLog | None = None
        self.buffer = RigidStateBuffer()
        self.latest_fluid_samples: Dict[str, dict] = {}

    def initialize(self, world, fluid, events: EventLog) -> None:
        self._world, self._fluid, self._events = world, fluid, events
        self.buffer = RigidStateBuffer()
        self.latest_fluid_samples = {}
        self._device_count = 0
        for obj in world.objects.values():
            self.buffer.register(obj)

    def register_body(self, obj: WorldObject) -> None: self.buffer.register(obj)
    def update_body(self, obj: WorldObject) -> None: self.buffer.update(obj)
    def unregister_body(self, object_id: str) -> None: self.buffer.unregister(object_id)

    def obstacle_snapshot(self) -> dict:
        """What the fluid solver needs to know about bodies as boundaries.

        `bed_heights` (v0.6.0) carries each body's `metadata.bed_height`: zero
        for everything that acts as a wall, positive for a ROCK, which the
        solver turns into a raised bed dome instead of a solid cell. Reported
        for every body rather than only for ROCK so the solver decides what is
        solid and what is bed from the data, not from a type name.
        """
        objects = [self._world.objects[oid] for oid in self.buffer.ids]
        return {"ids": list(self.buffer.ids),
                "positions": self.buffer.positions.copy(),
                "rotations": self.buffer.rotations.copy(),
                "types": [obj.type for obj in objects],
                "bed_heights": [float(obj.metadata.get("bed_height", 0.0))
                                for obj in objects],
                 "scales": self.buffer.scales.copy()}

    def _resolve_collisions(self, dynamic: np.ndarray) -> None:
        count = len(self.buffer.ids)
        if count < 2 or not np.any(dynamic):
            return
        base = {"HOUSE": 2.8, "CAR": 2.3, "TREE": 0.55,
                "BOX": 0.85, "DEBRIS": 0.65}
        radii = np.asarray([base.get(self._world.objects[oid].type, 0.85)
                            * max(self._world.objects[oid].scale[0],
                                  self._world.objects[oid].scale[2])
                            for oid in self.buffer.ids], dtype=np.float32)
        xz = self.buffer.positions[:, (0, 2)]
        delta = xz[:, None, :] - xz[None, :, :]
        distance = np.linalg.norm(delta, axis=2)
        minimum = radii[:, None] + radii[None, :]
        pair_i, pair_j = np.where(np.triu(distance < minimum, 1))
        collidable = np.asarray([self._world.objects[oid].type != "GAUGE"
                                 for oid in self.buffer.ids], dtype=np.bool_)
        active = ((dynamic[pair_i] | dynamic[pair_j])
                  & collidable[pair_i] & collidable[pair_j])
        pair_i, pair_j = pair_i[active], pair_j[active]
        if not len(pair_i):
            return
        pair_delta = delta[pair_i, pair_j].copy()
        pair_distance = distance[pair_i, pair_j].copy()
        zero = pair_distance < 1.0e-6
        pair_delta[zero] = [1.0, 0.0]
        pair_distance[zero] = 1.0
        normal = pair_delta / pair_distance[:, None]
        overlap = minimum[pair_i, pair_j] - distance[pair_i, pair_j]
        inv_i = np.where(dynamic[pair_i], 1.0 / self.buffer.masses[pair_i], 0.0)
        inv_j = np.where(dynamic[pair_j], 1.0 / self.buffer.masses[pair_j], 0.0)
        total = np.maximum(inv_i + inv_j, 1.0e-12)
        correction = np.zeros((count, 2), dtype=np.float32)
        np.add.at(correction, pair_i, normal * (overlap * inv_i / total)[:, None])
        np.add.at(correction, pair_j, -normal * (overlap * inv_j / total)[:, None])
        self.buffer.positions[:, 0] += correction[:, 0]
        self.buffer.positions[:, 2] += correction[:, 1]

    def _step_device(self, dt: float, sim_time: float, samples: dict) -> List[str]:
        count = len(self.buffer.ids)
        if getattr(self, "_device_count", 0) != count:
            device = samples["device"]
            self._d_positions = wp.empty(count, dtype=wp.vec3, device=device)
            self._d_velocities = wp.empty(count, dtype=wp.vec3, device=device)
            self._d_masses = wp.empty(count, dtype=float, device=device)
            self._d_frictions = wp.empty(count, dtype=float, device=device)
            self._d_buoyancies = wp.empty(count, dtype=float, device=device)
            self._d_volumes = wp.empty(count, dtype=float, device=device)
            self._d_ground_areas = wp.empty(count, dtype=float, device=device)
            self._d_static = wp.empty(count, dtype=wp.int32, device=device)
            self._d_states = wp.empty(count, dtype=wp.int32, device=device)
            self._d_dynamic = wp.empty(count, dtype=wp.int32, device=device)
            self._device_count = count
        self._d_positions.assign(self.buffer.positions)
        self._d_velocities.assign(self.buffer.velocities)
        self._d_masses.assign(self.buffer.masses)
        self._d_frictions.assign(self.buffer.frictions)
        self._d_buoyancies.assign(self.buffer.buoyancies)
        self._d_volumes.assign(self.buffer.volumes)
        self._d_ground_areas.assign(self.buffer.ground_areas)
        self._d_static.assign(self.buffer.static.astype(np.int32))
        self._d_states.assign(self.buffer.states.astype(np.int32))
        old_states = self.buffer.states.copy()
        wp.launch(_integrate_bodies, dim=count, inputs=[self._d_positions,
            self._d_velocities, self._d_masses, self._d_frictions,
                   self._d_buoyancies, self._d_volumes, self._d_ground_areas,
                   self._d_static, self._d_states, samples["immersions_device"],
                   samples["forces_device"],
                  self._d_dynamic, dt, float(self._world.environment.gravity),
                  config.WATER_DENSITY], device=samples["device"])
        self.buffer.positions[:] = np.asarray(self._d_positions.numpy(), dtype=np.float32)
        self.buffer.velocities[:] = np.asarray(self._d_velocities.numpy(), dtype=np.float32)
        self.buffer.states[:] = np.asarray(self._d_states.numpy(), dtype=np.int16)
        dynamic = np.asarray(self._d_dynamic.numpy(), dtype=np.int32).astype(bool)
        depths = np.asarray(samples["depths_device"].numpy(), dtype=np.float32)
        immersions = np.asarray(samples["immersions_device"].numpy(), dtype=np.float32)
        surfaces = np.asarray(samples["surface_elevations_device"].numpy(), dtype=np.float32)
        supports = np.asarray(samples["support_elevations_device"].numpy(), dtype=np.float32)
        flow_velocities = np.asarray(samples["velocities_device"].numpy(), dtype=np.float32)
        self._cache_fluid_samples(depths, surfaces, supports, flow_velocities)
        self._resolve_collisions(dynamic)

        for idx, oid in enumerate(self.buffer.ids):
            obj = self._world.objects[oid]
            terrain_y = float(supports[idx])
            if self.buffer.states[idx] == STATE_CODES[ObjectState.FLOATING.value]:
                draft = self.buffer.masses[idx] / (config.WATER_DENSITY
                                                    * self.buffer.buoyancies[idx]
                                                    * self.buffer.ground_areas[idx])
                self.buffer.positions[idx, 1] = max(terrain_y, surfaces[idx] - draft)
            else:
                self.buffer.positions[idx, 1] = terrain_y
            obj.position = self.buffer.positions[idx].astype(float).tolist()
            obj.state = STATE_NAMES[int(self.buffer.states[idx])]

        changed = np.flatnonzero((old_states != self.buffer.states) | dynamic)
        for idx in np.flatnonzero(old_states != self.buffer.states):
            oid = self.buffer.ids[int(idx)]
            state = STATE_NAMES[int(self.buffer.states[idx])]
            if not self._events:
                continue
            if state == ObjectState.FLOATING.value:
                self._events.record(sim_time, EventType.OBJECT_FLOATING, oid,
                                     cause="gpu_buoyancy_supports_weight",
                                     water_depth=float(depths[idx]),
                                     immersion=float(immersions[idx]))
            elif state == ObjectState.MOVING.value:
                self._events.record(sim_time, EventType.OBJECT_STARTED_MOVING, oid,
                                    cause="gpu_drag_exceeds_friction")
            elif state == ObjectState.SETTLED.value:
                self._events.record(sim_time, EventType.OBJECT_SETTLED, oid,
                                    cause="gpu_force_below_friction")
        return [self.buffer.ids[int(i)] for i in changed]

    def step(self, dt: float, sim_time: float, fluid_samples=None) -> List[str]:
        count = len(self.buffer.ids)
        if self._world is None or not count:
            return []
        if fluid_samples is not None and "depths_device" in fluid_samples:
            return self._step_device(dt, sim_time, fluid_samples)
        depths = np.zeros(count, dtype=np.float32)
        immersions = np.zeros(count, dtype=np.float32)
        forces = np.zeros((count, 3), dtype=np.float32)
        if fluid_samples is not None:
            depths[:] = fluid_samples["depths"]
            immersions[:] = fluid_samples.get("immersions", fluid_samples["depths"])
            forces[:] = fluid_samples["forces"]
            surfaces = np.asarray(fluid_samples.get("surface_elevations", depths),
                                  dtype=np.float32)
            supports = np.asarray(fluid_samples.get("support_elevations",
                                  np.zeros(count)), dtype=np.float32)
            flow_velocities = np.asarray(fluid_samples.get("velocities",
                                         np.zeros((count, 3))), dtype=np.float32)
            self._cache_fluid_samples(depths, surfaces, supports, flow_velocities)

        gravity = float(self._world.environment.gravity)
        submerged = np.clip(immersions / np.maximum(self.buffer.body_heights, 1.0e-6), 0.0, 1.0)
        buoyancy_force = (self.buffer.buoyancies * config.WATER_DENSITY * gravity
                          * self.buffer.volumes * submerged)
        weight = self.buffer.masses * gravity
        floating = (~self.buffer.static) & (buoyancy_force >= weight * 0.999)
        normal_force = np.maximum(weight - buoyancy_force, 0.0)
        horizontal_force = forces[:, (0, 2)].copy()
        drive = np.linalg.norm(horizontal_force, axis=1)
        speed = np.linalg.norm(self.buffer.velocities[:, (0, 2)], axis=1)
        friction_limit = self.buffer.frictions * normal_force
        sliding = (~self.buffer.static) & (~floating) & (drive > friction_limit)
        moving = floating | sliding

        direction = np.zeros_like(horizontal_force)
        force_nonzero = drive > 1.0e-8
        direction[force_nonzero] = horizontal_force[force_nonzero] / drive[force_nonzero, None]
        net = horizontal_force.copy()
        grounded = sliding & (friction_limit > 0.0)
        net[grounded] -= direction[grounded] * friction_limit[grounded, None]
        net[~moving] = 0.0
        acceleration = net / np.maximum(self.buffer.masses[:, None], 1.0e-6)
        self.buffer.velocities[:, 0] += acceleration[:, 0] * dt
        self.buffer.velocities[:, 2] += acceleration[:, 1] * dt
        self.buffer.velocities[~moving, 0] = 0.0
        self.buffer.velocities[~moving, 2] = 0.0
        self.buffer.positions[moving] += self.buffer.velocities[moving] * dt
        self._resolve_collisions(moving)

        old_states = self.buffer.states.copy()
        for idx in range(count):
            obj = self._world.objects[self.buffer.ids[idx]]
            terrain_y = self._world.terrain.height_at(float(self.buffer.positions[idx, 0]),
                                                       float(self.buffer.positions[idx, 2]))
            if floating[idx]:
                draft = self.buffer.masses[idx] / (config.WATER_DENSITY
                                                    * self.buffer.buoyancies[idx]
                                                    * self.buffer.ground_areas[idx])
                surface = (fluid_samples.get("surface_elevations", depths)[idx]
                           if fluid_samples is not None else terrain_y + depths[idx])
                self.buffer.positions[idx, 1] = max(terrain_y, surface - draft)
                new_state = STATE_CODES[ObjectState.FLOATING.value]
            elif sliding[idx] or speed[idx] > config.RIGID_STOP_SPEED:
                self.buffer.positions[idx, 1] = terrain_y
                new_state = STATE_CODES[ObjectState.MOVING.value]
            elif old_states[idx] in (STATE_CODES[ObjectState.MOVING.value],
                                     STATE_CODES[ObjectState.FLOATING.value]):
                self.buffer.positions[idx, 1] = terrain_y
                new_state = STATE_CODES[ObjectState.SETTLED.value]
            else:
                self.buffer.positions[idx, 1] = terrain_y
                new_state = int(old_states[idx])
            self.buffer.states[idx] = new_state
            obj.position = self.buffer.positions[idx].astype(float).tolist()
            obj.state = STATE_NAMES[new_state]

        changed = np.flatnonzero((old_states != self.buffer.states) | moving)
        for idx in np.flatnonzero(old_states != self.buffer.states):
            oid = self.buffer.ids[int(idx)]
            state = STATE_NAMES[int(self.buffer.states[idx])]
            if not self._events:
                continue
            if state == ObjectState.FLOATING.value:
                self._events.record(sim_time, EventType.OBJECT_FLOATING, oid,
                                     cause="buoyancy_supports_weight",
                                     water_depth=float(depths[idx]),
                                     immersion=float(immersions[idx]))
            elif state == ObjectState.MOVING.value:
                self._events.record(sim_time, EventType.OBJECT_STARTED_MOVING, oid,
                                    cause="drag_exceeds_friction",
                                    force=float(drive[idx]))
            elif state == ObjectState.SETTLED.value:
                self._events.record(sim_time, EventType.OBJECT_SETTLED, oid,
                                    cause="force_below_friction")
        return [self.buffer.ids[int(i)] for i in changed]

    def _cache_fluid_samples(self, depths: np.ndarray, surfaces: np.ndarray,
                             supports: np.ndarray, velocities: np.ndarray) -> None:
        self.latest_fluid_samples = {
            oid: {"depth": float(depths[i]), "surface": float(surfaces[i]),
                  "support": float(supports[i]),
                  "velocity": np.asarray(velocities[i], dtype=np.float32).copy()}
            for i, oid in enumerate(self.buffer.ids)
        }

    def get_transforms(self) -> List[Tuple[str, List[float]]]:
        return [(oid, self.buffer.positions[i].astype(float).tolist())
                for i, oid in enumerate(self.buffer.ids)]
