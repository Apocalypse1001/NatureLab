"""World state model: terrain, water, objects.

The backend owns the authoritative simulation state. The frontend only
edits/visualises it (see ARCHITECTURE.md). All values are plain
JSON-serialisable so the whole state can be saved/loaded as one document.
"""
from __future__ import annotations

import copy
import enum
import hashlib
import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from . import config


def finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def vector3(value: Any, name: str, *, positive: bool = False) -> List[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly 3 numbers")
    result = [finite_number(v, f"{name}[{i}]") for i, v in enumerate(value)]
    if positive and any(v <= 0.0 for v in result):
        raise ValueError(f"{name} values must be greater than zero")
    return result


class ObjectType(str, enum.Enum):
    HOUSE = "HOUSE"
    CAR = "CAR"
    TREE = "TREE"
    BOX = "BOX"
    DEBRIS = "DEBRIS"
    ROCK = "ROCK"

    @classmethod
    def register(cls, name: str) -> "ObjectType":
        """Allow adding new types without touching the core."""
        if name in cls.__members__:
            return cls[name]
        member: Any = str.__new__(cls, name)
        member._name_ = name
        member._value_ = name
        cls._member_map_[name] = member
        cls._value2member_map_[name] = member
        return member


class ObjectState(str, enum.Enum):
    INTACT = "INTACT"
    MOVING = "MOVING"
    FLOATING = "FLOATING"
    COLLIDING = "COLLIDING"
    DAMAGED = "DAMAGED"
    BROKEN = "BROKEN"
    SETTLED = "SETTLED"


# Default physical properties per type. New types plug in here.
# `drag` is a single scalar proxy for (Cd * cross-sectional area), not a real
# aerodynamic/hydrodynamic decomposition -- see docs/04_TZ_v0.3_roadmap.md
# section 3 on the causal- vs engineering-realism bar this project targets.
# `footprint_radius` is the XZ half-extent used to carve the object out of the
# fluid grid (see fluid_solver.ShallowWaterFluidSolver) -- matches the actual
# primitive sizes in frontend/src/world/ObjectFactory.ts so a house blocks
# more of the flow than a box, not a single fixed radius for every object.
# `root_strength` is extra static resistance (Newtons) ON TOP OF normal
# Coulomb friction, applied only while the object is still "rooted" (see
# rigid_body.ForceRigidBodySystem). 0 for everything except TREE -- this is
# a generic mechanic (any type could opt in), not a TREE special case, and
# folds docs/01_vision.md's separate "root strength" and "break strength"
# TREE properties into one threshold: how much force (from water drag OR
# a body impact) before the anchor is permanently gone and it becomes an
# ordinary movable/floating body (matching "дерево упало -> стало
# препятствием" from 01_vision.md, since a broken tree is still a
# registered rigid body and keeps carving its obstacle hole in the water).
OBJECT_DEFAULTS: Dict[ObjectType, Dict[str, float]] = {
    ObjectType.HOUSE: {"mass": 20000.0, "friction": 0.7, "buoyancy": 0.1, "drag": 0.2,
                       "foundation_height": 0.3, "damage_resistance": 0.8,
                       "footprint_radius": 2.4, "root_strength": 0.0,
                       "shade_radius": 0.0, "shade_cooling": 0.0},
    ObjectType.CAR:   {"mass": 1500.0, "friction": 0.6, "buoyancy": 0.55, "drag": 1.5,
                       "foundation_height": 0.0, "damage_resistance": 0.3,
                       "footprint_radius": 2.2, "root_strength": 0.0,
                       "shade_radius": 0.0, "shade_cooling": 0.0},
    # shade_radius/shade_cooling (RiverLab, Schauberger): a tree canopy casts
    # shade well beyond its own footprint_radius (trunk) -- see
    # docs/04_TZ_v0.3_roadmap.md v0.4 and world_state note near
    # ShallowWaterFluidSolver._update_temperature_factor for how this is used.
    ObjectType.TREE:  {"mass": 800.0, "friction": 0.8, "buoyancy": 0.6, "drag": 0.9,
                       "foundation_height": 0.0, "damage_resistance": 0.4,
                       "footprint_radius": 1.2, "root_strength": 15000.0,
                       "shade_radius": 4.0, "shade_cooling": 3.0},
    ObjectType.BOX:   {"mass": 50.0, "friction": 0.5, "buoyancy": 0.8, "drag": 0.6,
                       "foundation_height": 0.0, "damage_resistance": 0.5,
                       "footprint_radius": 0.7, "root_strength": 0.0,
                       "shade_radius": 0.0, "shade_cooling": 0.0},
    ObjectType.DEBRIS: {"mass": 10.0, "friction": 0.4, "buoyancy": 0.9, "drag": 0.4,
                        "foundation_height": 0.0, "damage_resistance": 0.2,
                        "footprint_radius": 0.7, "root_strength": 0.0,
                        "shade_radius": 0.0, "shade_cooling": 0.0},
    # Riverbed rock (RiverLab, docs/04_TZ_v0.3_roadmap.md v0.4): not a Schauberger
    # special case, just root_strength turned up so high it's permanently
    # immovable by anything this sim can produce -- reuses 100% of the existing
    # obstacle/root machinery. footprint_radius already scales with obj.scale
    # (already editable in the UI), so "effect scales with rock size" falls
    # out for free instead of needing a new property.
    # bed_height: how far the boulder stands proud of the bed. The only type with
    # a non-zero one -- it makes the rock part of the riverbed rather than an
    # infinitely tall wall, so deep water passes over it and shallow water is
    # deflected around it. See config.BED_DOME_EXPONENT and
    # ShallowWaterFluidSolver.set_bed_obstructions.
    ObjectType.ROCK: {"mass": 2000.0, "friction": 0.9, "buoyancy": 0.0, "drag": 0.1,
                      "foundation_height": 0.0, "damage_resistance": 1.0,
                      "footprint_radius": 1.0, "root_strength": 1e8,
                      "shade_radius": 0.0, "shade_cooling": 0.0,
                      "bed_height": 0.8},
}


def default_properties(obj_type: str) -> Dict[str, float]:
    base = {"mass": 100.0, "friction": 0.5, "buoyancy": 0.5, "drag": 0.5,
            "foundation_height": 0.0, "damage_resistance": 0.5, "footprint_radius": 1.0,
            "root_strength": 0.0, "shade_radius": 0.0, "shade_cooling": 0.0,
            "bed_height": 0.0}
    base.update(OBJECT_DEFAULTS.get(ObjectType.register(obj_type), {}))
    return base


@dataclass
class WorldObject:
    id: str
    type: str
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    scale: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    mass: float = 100.0
    friction: float = 0.5
    buoyancy: float = 0.5
    damage: float = 0.0
    state: str = ObjectState.INTACT.value
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.__dict__)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "WorldObject":
        if not isinstance(data, dict):
            raise ValueError("object must be a JSON object")
        allowed = {f for f in WorldObject.__dataclass_fields__}
        values = {k: v for k, v in data.items() if k in allowed}
        values["id"] = str(values.get("id", "")).strip()
        if not values["id"]:
            raise ValueError("object.id is required")
        values["type"] = str(values.get("type", "")).upper()
        ObjectType.register(values["type"])
        values["position"] = vector3(values.get("position", [0, 0, 0]), "position")
        values["rotation"] = vector3(values.get("rotation", [0, 0, 0]), "rotation")
        values["scale"] = vector3(values.get("scale", [1, 1, 1]), "scale", positive=True)
        for key in ("mass", "friction", "buoyancy", "damage"):
            values[key] = finite_number(values.get(key, WorldObject.__dataclass_fields__[key].default), key)
        state = str(values.get("state", ObjectState.INTACT.value))
        if state not in {member.value for member in ObjectState}:
            raise ValueError(f"invalid object state: {state}")
        values["state"] = state
        metadata = values.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be an object")
        values["metadata"] = metadata
        return WorldObject(**values)


@dataclass
class TerrainGrid:
    """Logical heightfield: numeric array decoupled from the visual mesh."""
    width: int = config.TERRAIN_CELLS
    height: int = config.TERRAIN_CELLS
    cell_size: float = config.TERRAIN_CELL_SIZE
    heights: np.ndarray = field(
        default_factory=lambda: np.zeros((config.TERRAIN_CELLS + 1, config.TERRAIN_CELLS + 1), dtype=np.float32))

    @property
    def size_m(self) -> float:
        return self.width * self.cell_size

    def height_at(self, x: float, z: float) -> float:
        gx = np.clip(x / self.cell_size + self.width / 2, 0, self.width)
        gz = np.clip(z / self.cell_size + self.height / 2, 0, self.height)
        i0, j0 = int(np.floor(gx)), int(np.floor(gz))
        i1, j1 = min(i0 + 1, self.width), min(j0 + 1, self.height)
        fx, fz = gx - i0, gz - j0
        h = self.heights
        return float(
            (1 - fx) * (1 - fz) * h[j0, i0] + fx * (1 - fz) * h[j0, i1]
            + (1 - fx) * fz * h[j1, i0] + fx * fz * h[j1, i1])

    def brush(self, x: float, z: float, radius: float, strength: float) -> None:
        """Raise (or lower, with negative strength) the terrain with a smooth brush."""
        r_cells = max(1.0, radius / self.cell_size)
        ci = x / self.cell_size + self.width / 2
        cj = z / self.cell_size + self.height / 2
        lo_i, hi_i = int(max(0, ci - r_cells)), int(min(self.width, ci + r_cells))
        lo_j, hi_j = int(max(0, cj - r_cells)), int(min(self.height, cj + r_cells))
        for j in range(lo_j, hi_j + 1):
            for i in range(lo_i, hi_i + 1):
                d = ((i - ci) ** 2 + (j - cj) ** 2) ** 0.5
                if d <= r_cells:
                    falloff = 0.5 * (1 + np.cos(np.pi * d / r_cells))
                    self.heights[j, i] = float(np.clip(
                        self.heights[j, i] + strength * falloff,
                        config.HEIGHT_MIN, config.HEIGHT_MAX))

    def to_list(self) -> List[float]:
        return self.heights.ravel().tolist()

    def checksum(self) -> str:
        canonical = np.ascontiguousarray(self.heights, dtype="<f4")
        return hashlib.sha256(canonical.tobytes()).hexdigest()

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "TerrainGrid":
        if not isinstance(data, dict):
            raise ValueError("terrain must be an object")
        grid = TerrainGrid(
            width=int(data.get("width", config.TERRAIN_CELLS)),
            height=int(data.get("height", config.TERRAIN_CELLS)),
            cell_size=float(data.get("cell_size", config.TERRAIN_CELL_SIZE)))
        flat = data.get("heights")
        if flat is not None:
            expected = (grid.width + 1) * (grid.height + 1)
            if not isinstance(flat, list) or len(flat) != expected:
                raise ValueError(f"terrain.heights must contain {expected} values")
            arr = np.asarray(flat, dtype=np.float32)
            if not np.isfinite(arr).all():
                raise ValueError("terrain heights must be finite")
            grid.heights = arr.reshape(grid.height + 1, grid.width + 1)
        return grid


@dataclass
class WaterState:
    level: float = 0.5          # meters above terrain datum
    visible: bool = True
    flow_enabled: bool = False  # continuous river current, see ShallowWaterFluidSolver

    def to_dict(self) -> Dict[str, Any]:
        return {"level": self.level, "visible": self.visible,
                "flow_enabled": self.flow_enabled}


@dataclass
class EnvironmentState:
    gravity: float = 9.81
    wind: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    temperature: float = 15.0

    def to_dict(self) -> Dict[str, Any]:
        return {"gravity": self.gravity, "wind": self.wind,
                "temperature": self.temperature}


@dataclass
class WorldState:
    terrain: TerrainGrid = field(default_factory=TerrainGrid)
    water: WaterState = field(default_factory=WaterState)
    objects: Dict[str, WorldObject] = field(default_factory=dict)
    environment: EnvironmentState = field(default_factory=EnvironmentState)
    _counters: "itertools.count[int]" = field(default_factory=itertools.count, repr=False)

    # ------------------------------------------------------------------ objects
    def add_object(self, obj_type: str, position: List[float]) -> WorldObject:
        obj_type = ObjectType.register(obj_type).value
        idx = next(self._counters)
        props = default_properties(obj_type)
        name_prefix = obj_type.capitalize()
        # everything default_properties() knows that is not already a
        # first-class field goes to metadata. Derived rather than listed by
        # hand: the old explicit list silently dropped each newly added
        # property (bed_height was added and simply never reached any object).
        promoted = ("mass", "friction", "buoyancy")
        obj = WorldObject(
            id=f"{name_prefix}_{idx:03d}", type=obj_type, position=list(position),
            mass=props["mass"], friction=props["friction"], buoyancy=props["buoyancy"],
            metadata={k: v for k, v in props.items() if k not in promoted},
        )
        self.objects[obj.id] = obj
        return obj

    def sync_counter(self) -> None:
        """After a load: continue numbering after the highest existing index."""
        max_idx = 0
        for obj in self.objects.values():
            try:
                max_idx = max(max_idx, int(obj.id.rsplit("_", 1)[1]))
            except (ValueError, IndexError):
                pass
        self._counters = itertools.count(max_idx + 1)

    # ------------------------------------------------------------------ (de)serialise
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "terrain": {"width": self.terrain.width, "height": self.terrain.height,
                        "cell_size": self.terrain.cell_size,
                        "heights": self.terrain.to_list()},
            "water": self.water.to_dict(),
            "environment": self.environment.to_dict(),
            "objects": [o.to_dict() for o in self.objects.values()],
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "WorldState":
        if not isinstance(data, dict):
            raise ValueError("world must be a JSON object")
        state = WorldState()
        state.terrain = TerrainGrid.from_dict(data.get("terrain", {}))
        water = data.get("water", {})
        state.water = WaterState(level=finite_number(water.get("level", 0.5), "water.level"),
                                 visible=bool(water.get("visible", True)),
                                 flow_enabled=bool(water.get("flow_enabled", False)))
        env = data.get("environment", {})
        state.environment = EnvironmentState(
            gravity=finite_number(env.get("gravity", 9.81), "environment.gravity"),
            wind=vector3(env.get("wind", [0.0, 0.0, 0.0]), "environment.wind"),
            temperature=finite_number(env.get("temperature", 15.0), "environment.temperature"))
        objects = data.get("objects", [])
        if not isinstance(objects, list):
            raise ValueError("objects must be an array")
        parsed = [WorldObject.from_dict(o) for o in objects]
        if len({o.id for o in parsed}) != len(parsed):
            raise ValueError("object IDs must be unique")
        state.objects = {o.id: o for o in parsed}
        state.sync_counter()
        return state

    def clone(self) -> "WorldState":
        return WorldState.from_dict(self.to_dict())
