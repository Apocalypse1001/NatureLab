"""Data-oriented rigid-body foundation.

The placeholder keeps contiguous arrays and an ID-to-index map. A future Warp
solver can upload these buffers directly without changing SimulationManager.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

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
                          self.masses, self.buoyancies, self.states):
                array[idx] = array[last]
        self.ids.pop()
        self.positions = self.positions[:-1]
        self.velocities = self.velocities[:-1]
        self.rotations = self.rotations[:-1]
        self.masses = self.masses[:-1]
        self.buoyancies = self.buoyancies[:-1]
        self.states = self.states[:-1]


class RigidBodySystem:
    def initialize(self, world, fluid, events: EventLog) -> None: ...
    def register_body(self, obj: WorldObject) -> None: ...
    def update_body(self, obj: WorldObject) -> None: ...
    def unregister_body(self, object_id: str) -> None: ...
    def obstacle_snapshot(self) -> dict: ...
    def step(self, dt: float, sim_time: float, fluid_samples=None) -> List[str]: ...
    def reset(self) -> None: ...
    def get_transforms(self) -> List[Tuple[str, List[float]]]: ...


class PlaceholderRigidBodySystem(RigidBodySystem):
    """Vectorized placeholder over contiguous arrays, not per-object Python state."""

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
        }

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

    def reset(self) -> None:
        if self._world is not None and self._fluid is not None and self._events is not None:
            self.initialize(self._world, self._fluid, self._events)

    def get_transforms(self) -> List[Tuple[str, List[float]]]:
        return [(oid, self.buffer.positions[i].astype(float).tolist())
                for i, oid in enumerate(self.buffer.ids)]
