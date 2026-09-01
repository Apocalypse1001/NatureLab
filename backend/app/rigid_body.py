"""Data-oriented rigid-body foundation.

The placeholder keeps contiguous arrays and an ID-to-index map. A future Warp
solver can upload these buffers directly without changing SimulationManager.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from . import config
from .events import EventLog, EventType
from .world_state import ObjectState, WorldObject


@dataclass
class RigidStateBuffer:
    ids: List[str] = field(default_factory=list)
    index: Dict[str, int] = field(default_factory=dict)
    positions: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    velocities: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    rotations: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    masses: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    buoyancies: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    states: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int16))
    frictions: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    drags: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    foundation_heights: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    footprint_radii: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))

    @staticmethod
    def _footprint_radius(obj: WorldObject) -> float:
        base = float(obj.metadata.get("footprint_radius", 1.0))
        return base * max(float(obj.scale[0]), float(obj.scale[2]))

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
        self.masses = np.append(self.masses, np.float32(obj.mass))
        self.buoyancies = np.append(self.buoyancies, np.float32(obj.buoyancy))
        self.states = np.append(self.states, np.int16(0))
        self.frictions = np.append(self.frictions, np.float32(obj.friction))
        self.drags = np.append(self.drags, np.float32(obj.metadata.get("drag", 0.5)))
        self.foundation_heights = np.append(
            self.foundation_heights, np.float32(obj.metadata.get("foundation_height", 0.0)))
        self.footprint_radii = np.append(self.footprint_radii, np.float32(self._footprint_radius(obj)))
        return idx

    def update(self, obj: WorldObject) -> None:
        idx = self.index.get(obj.id)
        if idx is None:
            self.register(obj)
            return
        self.positions[idx] = obj.position
        self.rotations[idx] = obj.rotation
        self.masses[idx] = obj.mass
        self.buoyancies[idx] = obj.buoyancy
        self.frictions[idx] = obj.friction
        self.drags[idx] = obj.metadata.get("drag", self.drags[idx])
        self.foundation_heights[idx] = obj.metadata.get("foundation_height", self.foundation_heights[idx])
        self.footprint_radii[idx] = self._footprint_radius(obj)

    def unregister(self, object_id: str) -> None:
        idx = self.index.pop(object_id, None)
        if idx is None:
            return
        last = len(self.ids) - 1
        if idx != last:
            moved_id = self.ids[last]
            self.ids[idx] = moved_id
            self.index[moved_id] = idx
            for array in (self.positions, self.velocities, self.rotations,
                          self.masses, self.buoyancies, self.states,
                          self.frictions, self.drags, self.foundation_heights,
                          self.footprint_radii):
                array[idx] = array[last]
        self.ids.pop()
        self.positions = self.positions[:-1]
        self.velocities = self.velocities[:-1]
        self.rotations = self.rotations[:-1]
        self.masses = self.masses[:-1]
        self.buoyancies = self.buoyancies[:-1]
        self.states = self.states[:-1]
        self.frictions = self.frictions[:-1]
        self.drags = self.drags[:-1]
        self.foundation_heights = self.foundation_heights[:-1]
        self.footprint_radii = self.footprint_radii[:-1]


class RigidBodySystem:
    def initialize(self, world, fluid, events: EventLog) -> None: ...
    def register_body(self, obj: WorldObject) -> None: ...
    def update_body(self, obj: WorldObject) -> None: ...
    def unregister_body(self, object_id: str) -> None: ...
    def obstacle_snapshot(self) -> dict: ...
    def step(self, dt: float, sim_time: float, fluid_samples=None) -> List[str]: ...
    def reset(self) -> None: ...
    def get_transforms(self) -> List[Tuple[str, List[float]]]: ...


class _ArrayRigidBodySystem(RigidBodySystem):
    """Shared array-backed bookkeeping; subclasses only differ in step()."""

    def __init__(self) -> None:
        self._world = None
        self._fluid = None
        self._events: EventLog | None = None
        self.buffer = RigidStateBuffer()

    def initialize(self, world, fluid, events: EventLog) -> None:
        self._world, self._fluid, self._events = world, fluid, events
        self.buffer = RigidStateBuffer()
        for obj in world.objects.values():
            self.buffer.register(obj)

    def register_body(self, obj: WorldObject) -> None:
        self.buffer.register(obj)

    def update_body(self, obj: WorldObject) -> None:
        self.buffer.update(obj)

    def unregister_body(self, object_id: str) -> None:
        self.buffer.unregister(object_id)

    def obstacle_snapshot(self) -> dict:
        return {
            "ids": list(self.buffer.ids),
            "positions": self.buffer.positions.copy(),
            "rotations": self.buffer.rotations.copy(),
            "masses": self.buffer.masses.copy(),
            "radii": self.buffer.footprint_radii.copy(),
        }

    def reset(self) -> None:
        if self._world is not None and self._fluid is not None and self._events is not None:
            self.initialize(self._world, self._fluid, self._events)

    def get_transforms(self) -> List[Tuple[str, List[float]]]:
        return [(oid, self.buffer.positions[i].astype(float).tolist())
                for i, oid in enumerate(self.buffer.ids)]


class PlaceholderRigidBodySystem(_ArrayRigidBodySystem):
    """Binary buoyancy-threshold placeholder, kept for reference/tests.

    Superseded by ForceRigidBodySystem (gravity/buoyancy/drag/friction) as of
    v0.3 -- see docs/04_TZ_v0.3_roadmap.md.
    """

    def step(self, dt: float, sim_time: float, fluid_samples=None) -> List[str]:
        if self._world is None or not self.buffer.ids:
            return []
        positions = self.buffer.positions
        depths = np.zeros(len(self.buffer.ids), dtype=np.float32)
        flow = np.zeros((len(self.buffer.ids), 3), dtype=np.float32)
        if fluid_samples is not None:
            depths[:] = fluid_samples["depths"]
            flow[:] = fluid_samples["velocities"]
        floating = self.buffer.buoyancies * depths > 0.3
        alpha = min(1.0, dt)
        self.buffer.velocities[floating] += (flow[floating] - self.buffer.velocities[floating]) * alpha
        positions[floating] += self.buffer.velocities[floating] * dt

        changed: List[str] = []
        for idx in np.flatnonzero(floating):
            oid = self.buffer.ids[int(idx)]
            obj = self._world.objects.get(oid)
            if obj is None:
                continue
            if obj.state != ObjectState.FLOATING.value and self._events:
                self._events.record(sim_time, EventType.OBJECT_STARTED_MOVING, oid,
                                    cause="water_buoyancy", water_depth=float(depths[idx]))
                self._events.record(sim_time, EventType.OBJECT_FLOATING, oid,
                                    cause="buoyancy_gt_threshold", buoyancy=obj.buoyancy)
            obj.state = ObjectState.FLOATING.value
            obj.position = positions[idx].astype(float).tolist()
            changed.append(oid)
        for idx in np.flatnonzero(~floating):
            oid = self.buffer.ids[int(idx)]
            obj = self._world.objects.get(oid)
            if obj is not None and obj.state == ObjectState.FLOATING.value:
                obj.state = ObjectState.SETTLED.value
                self.buffer.velocities[idx] = 0.0
                if self._events:
                    self._events.record(sim_time, EventType.OBJECT_SETTLED, oid,
                                        cause="water_receded")
                changed.append(oid)
        return changed


class ForceRigidBodySystem(_ArrayRigidBodySystem):
    """gravity + buoyancy + hydrodynamic drag + ground friction, vectorized.

    Per-object force model (see docs/04_TZ_v0.3_roadmap.md and
    docs/01_vision.md "Поведение автомобиля"):

    - buoyancy reduces how much of the object's weight still rests on the
      ground (normal force), scaled by how deep it sits past its own
      foundation_height -- this is what makes Experiment A/B from
      docs/01_vision.md (house foundation 0.2 m vs 1.0 m) actually change
      the outcome instead of being an unused metadata field.
    - ground friction is Coulomb friction against that reduced normal force:
      a static object stays put while drag stays under the friction limit,
      and starts moving once drag exceeds it.
    - hydrodynamic drag pulls the object toward the local water velocity,
      scaled by how submerged it is.

    `drag` is a per-object scalar proxy (Cd * area), not a full aerodynamic
    breakdown -- see the causal- vs engineering-realism note in
    docs/04_TZ_v0.3_roadmap.md section 3. Object-object collision is not
    handled here yet (two floating bodies can overlap) -- that is a
    follow-up, not silently claimed as done.
    """

    def step(self, dt: float, sim_time: float, fluid_samples=None) -> List[str]:
        if self._world is None or not self.buffer.ids:
            return []
        n = len(self.buffer.ids)
        depths = np.zeros(n, dtype=np.float32)
        flow = np.zeros((n, 3), dtype=np.float32)
        if fluid_samples is not None:
            depths[:] = fluid_samples["depths"]
            flow[:] = fluid_samples["velocities"]

        gravity = float(self._world.environment.gravity)
        buf = self.buffer
        weight = buf.masses * gravity  # N

        effective_depth = np.maximum(0.0, depths - buf.foundation_heights)
        depth_ratio = effective_depth / config.RIGID_REFERENCE_DEPTH_M
        # Buoyant support is NOT capped at buoyancy_coeff * weight: real water
        # keeps adding lift as depth increases, so any object with buoyancy >
        # 0 eventually floats given enough depth -- only how much depth it
        # takes differs per object. buoyancy_coeff instead sets how quickly
        # support grows with depth. (Capping it at depth_ratio==1 was a real
        # bug: a box with buoyancy=0.8 could never drop contact_fraction
        # below 0.2, so it could never reach RIGID_FLOAT_CONTACT_THRESHOLD
        # and would never float no matter how deep the water got.)
        buoyant = np.minimum(weight, buf.buoyancies * depth_ratio * weight)
        normal = np.maximum(0.0, weight - buoyant)
        submerge = np.clip(depth_ratio, 0.0, 1.0)  # drag exposure caps once fully wet
        contact_fraction = np.where(weight > 1e-9, normal / np.maximum(weight, 1e-9), 0.0)
        friction_max = buf.frictions * normal

        rel = flow.copy()
        rel[:, 1] = 0.0
        rel[:, 0] -= buf.velocities[:, 0]
        rel[:, 2] -= buf.velocities[:, 2]
        drag = config.RIGID_WATER_DRAG_SCALE * buf.drags[:, None] * submerge[:, None] * rel
        drag_mag = np.linalg.norm(drag[:, [0, 2]], axis=1)

        prev_speed = np.linalg.norm(buf.velocities[:, [0, 2]], axis=1)
        was_moving = prev_speed > config.RIGID_MOVE_EPS_MPS

        net = np.zeros_like(drag)
        # at rest: static friction holds unless drag overcomes it
        drag_dir = np.zeros_like(drag)
        nonzero_drag = drag_mag > 1e-9
        drag_dir[nonzero_drag] = drag[nonzero_drag] / drag_mag[nonzero_drag, None]
        at_rest = ~was_moving
        overcomes_static = at_rest & (drag_mag > friction_max)
        net[overcomes_static] = (drag[overcomes_static]
                                  - drag_dir[overcomes_static] * friction_max[overcomes_static, None])
        # already moving: kinetic friction opposes current velocity direction
        vel_dir = np.zeros_like(buf.velocities)
        nonzero_speed = prev_speed > 1e-9
        vel_xz = buf.velocities.copy()
        vel_xz[:, 1] = 0.0
        vel_dir[nonzero_speed] = vel_xz[nonzero_speed] / prev_speed[nonzero_speed, None]
        net[was_moving] = drag[was_moving] - vel_dir[was_moving] * friction_max[was_moving, None]

        accel = net / np.maximum(buf.masses[:, None], 1e-6)
        new_velocities = buf.velocities + accel * dt
        new_velocities[:, 1] = 0.0
        holding = at_rest & ~overcomes_static
        new_velocities[holding] = 0.0
        new_speed = np.linalg.norm(new_velocities[:, [0, 2]], axis=1)
        # a moving body whose drag can no longer beat friction comes to rest
        # rather than oscillating around zero at fixed dt
        stopping = was_moving & (drag_mag <= friction_max) & (new_speed < config.RIGID_MOVE_EPS_MPS)
        new_velocities[stopping] = 0.0

        buf.velocities = new_velocities.astype(np.float32)
        buf.positions = (buf.positions + buf.velocities * dt).astype(np.float32)
        new_speed = np.linalg.norm(buf.velocities[:, [0, 2]], axis=1)

        floating_mask = contact_fraction < config.RIGID_FLOAT_CONTACT_THRESHOLD
        moving_mask = (~floating_mask) & (new_speed > config.RIGID_MOVE_EPS_MPS)

        changed: List[str] = []
        for idx in range(n):
            oid = buf.ids[idx]
            obj = self._world.objects.get(oid)
            if obj is None:
                continue
            prev_state = obj.state
            if floating_mask[idx]:
                new_state = ObjectState.FLOATING.value
            elif moving_mask[idx]:
                new_state = ObjectState.MOVING.value
            elif prev_state in (ObjectState.FLOATING.value, ObjectState.MOVING.value,
                                 ObjectState.SETTLED.value):
                new_state = ObjectState.SETTLED.value
            else:
                new_state = prev_state
            if new_state != prev_state and self._events:
                cause_params = dict(water_depth=float(depths[idx]),
                                     flow_speed=float(np.linalg.norm(flow[idx, [0, 2]])),
                                     drag_force=float(drag_mag[idx]),
                                     ground_friction=float(friction_max[idx]),
                                     contact_fraction=float(contact_fraction[idx]))
                if new_state == ObjectState.MOVING.value:
                    self._events.record(sim_time, EventType.OBJECT_STARTED_MOVING, oid,
                                        cause="drag_exceeded_friction", **cause_params)
                elif new_state == ObjectState.FLOATING.value:
                    self._events.record(sim_time, EventType.OBJECT_FLOATING, oid,
                                        cause="buoyancy_exceeded_weight", **cause_params)
                elif new_state == ObjectState.SETTLED.value:
                    self._events.record(sim_time, EventType.OBJECT_SETTLED, oid,
                                        cause="water_receded", **cause_params)
            obj.state = new_state
            obj.position = buf.positions[idx].astype(float).tolist()
            if new_state != prev_state or moving_mask[idx] or floating_mask[idx]:
                changed.append(oid)
        return changed
