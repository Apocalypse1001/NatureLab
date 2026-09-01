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
    SOURCE = "SOURCE"
    DRAIN = "DRAIN"
    GAUGE = "GAUGE"

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
OBJECT_DEFAULTS: Dict[ObjectType, Dict[str, float]] = {
    ObjectType.HOUSE: {"mass": 20000.0, "friction": 0.7, "buoyancy": 1.0,
                       "volume_m3": 15.0, "drag_coefficient": 1.2,
                       "ground_contact_area": 16.0, "cross_sectional_area": 12.0,
                       "is_static": True,
                       "foundation_height": 0.3, "damage_resistance": 0.8},
    ObjectType.CAR:   {"mass": 1500.0, "friction": 0.6, "buoyancy": 1.0,
                       "volume_m3": 1.8, "drag_coefficient": 0.9,
                       "ground_contact_area": 3.5, "cross_sectional_area": 2.4,
                       "is_static": False,
                       "foundation_height": 0.0, "damage_resistance": 0.3},
    ObjectType.TREE:  {"mass": 800.0, "friction": 0.8, "buoyancy": 1.0,
                       "volume_m3": 1.0, "drag_coefficient": 1.1,
                       "ground_contact_area": 0.2, "cross_sectional_area": 0.8,
                       "is_static": False,
                       "foundation_height": 0.0, "damage_resistance": 0.4},
    ObjectType.BOX:   {"mass": 50.0, "friction": 0.5, "buoyancy": 1.0,
                       "volume_m3": 1.728, "drag_coefficient": 1.05,
                       "ground_contact_area": 1.44, "cross_sectional_area": 1.44,
                       "is_static": False,
                       "foundation_height": 0.0, "damage_resistance": 0.5},
    ObjectType.DEBRIS: {"mass": 10.0, "friction": 0.4, "buoyancy": 1.0,
                         "volume_m3": 0.025, "drag_coefficient": 1.2,
                         "ground_contact_area": 0.1, "cross_sectional_area": 0.2,
                         "is_static": False,
                          "foundation_height": 0.0, "damage_resistance": 0.2},
    # A riverbed boulder: part of the bed, not a rigid body and not a wall.
    # `bed_height` is what the fluid solver raises the effective bed by; the
    # dome's radius comes from the object's horizontal scale and its height
    # from the vertical one, so Scale Y in the properties panel is already the
    # control for "how much of the channel does this rock block".
    ObjectType.ROCK:  {"mass": 4000.0, "friction": 0.9, "buoyancy": 0.0,
                       "volume_m3": 1.5, "drag_coefficient": 0.9,
                       "ground_contact_area": 3.0, "cross_sectional_area": 1.5,
                       "is_static": True, "bed_height": 0.8,
                       "foundation_height": 0.0, "damage_resistance": 1.0},
    # Placeable inflow. Holds water at `inflow_level` inside `inflow_radius`,
    # using the same rule as the map-edge inflow (h = max(0, level - bed)), so a
    # source on a hillside fills to the height asked for instead of drowning the
    # hill. Not a rigid body, not an obstacle: it is a boundary condition the
    # user can pick up and move.
    ObjectType.SOURCE: {"mass": 1.0, "friction": 0.0, "buoyancy": 0.0,
                        "volume_m3": 1.0, "drag_coefficient": 1.0,
                        "ground_contact_area": 1.0, "cross_sectional_area": 1.0,
                        "is_static": True, "inflow_level": 1.5,
                        "inflow_radius": 4.0,
                        "foundation_height": 0.0, "damage_resistance": 1.0},
    # Placeable outlet. Removes water through a smooth radial sink and spins the
    # flow around it from the circulation it measures -- see config.py's
    # DRAIN_SWIRL_GAIN for why the rotation cannot be a constant.
    ObjectType.DRAIN:  {"mass": 1.0, "friction": 0.0, "buoyancy": 0.0,
                        "volume_m3": 1.0, "drag_coefficient": 1.0,
                        "ground_contact_area": 1.0, "cross_sectional_area": 1.0,
                        "is_static": True, "drain_radius": 5.0,
                        "drain_strength": 1.2,
                        "foundation_height": 0.0, "damage_resistance": 1.0},
    ObjectType.GAUGE: {"mass": 1.0, "friction": 0.0, "buoyancy": 0.0,
                       "volume_m3": 1.0, "drag_coefficient": 1.0,
                       "ground_contact_area": 1.0, "cross_sectional_area": 1.0,
                       "is_static": True,
                       "foundation_height": 0.0, "damage_resistance": 1.0},
}


# Properties that live as real columns on WorldObject rather than in metadata.
# Everything else in default_properties() lands in metadata automatically -- the
# list used to be written out by hand in add_object(), which meant every new
# property (bed_height was the one that bit) silently never reached an object.
_COLUMN_PROPERTIES = frozenset({
    "mass", "friction", "buoyancy", "volume_m3", "drag_coefficient",
    "ground_contact_area", "cross_sectional_area", "is_static",
})


def default_properties(obj_type: str) -> Dict[str, float]:
    base = {"mass": 100.0, "friction": 0.5, "buoyancy": 1.0,
            "volume_m3": 0.1, "drag_coefficient": 1.0,
            "ground_contact_area": 0.5, "cross_sectional_area": 0.5,
            "is_static": False, "bed_height": 0.0,
            "inflow_level": 0.0, "inflow_radius": 0.0,
            "drain_radius": 0.0, "drain_strength": 0.0,
            "foundation_height": 0.0, "damage_resistance": 0.5}
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
    volume_m3: float = 0.1
    drag_coefficient: float = 1.0
    ground_contact_area: float = 0.5
    cross_sectional_area: float = 0.5
    is_static: bool = False
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
        defaults = default_properties(values["type"])
        for key in ("mass", "friction", "buoyancy", "damage", "volume_m3",
                    "drag_coefficient", "ground_contact_area", "cross_sectional_area"):
            fallback = defaults.get(key, WorldObject.__dataclass_fields__[key].default)
            values[key] = finite_number(values.get(key, fallback), key)
        for key in ("mass", "volume_m3", "drag_coefficient", "ground_contact_area",
                    "cross_sectional_area"):
            if values[key] <= 0.0:
                raise ValueError(f"{key} must be greater than zero")
        if values["friction"] < 0.0:
            raise ValueError("friction must be non-negative")
        if not 0.0 <= values["buoyancy"] <= 1.0:
            raise ValueError("buoyancy must be between 0 and 1")
        values["is_static"] = bool(values.get("is_static", defaults.get("is_static", False)))
        state = str(values.get("state", ObjectState.INTACT.value))
        if state not in {member.value for member in ObjectState}:
            raise ValueError(f"invalid object state: {state}")
        values["state"] = state
        metadata = values.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be an object")
        # Backfill metadata keys the saved world predates, from this type's
        # defaults rather than from zero. A ROCK saved before v0.6.0 has no
        # bed_height; defaulting it to 0 would silently turn the boulder into a
        # flat patch of riverbed with no effect at all, which reads as "the
        # feature is broken" rather than "this save is older".
        for key, fallback in default_properties(values["type"]).items():
            if key not in _COLUMN_PROPERTIES and key not in metadata:
                metadata[key] = fallback
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
    # RiverLab (v0.6.0): let the river cut and fill its own bed. Off by default
    # so an existing FloodLab experiment behaves exactly as it did in 0.5.1 and
    # the terrain the user built stays the terrain they built.
    erosion_enabled: bool = False
    # v0.8.0: let water leave through the east edge. On by default -- a river
    # that cannot run off the map just fills it, which is what the closed
    # boundary did for every version up to 0.7.0.
    outflow_enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {"level": self.level, "visible": self.visible,
                "erosion_enabled": self.erosion_enabled,
                "outflow_enabled": self.outflow_enabled}


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
        obj = WorldObject(
            id=f"{name_prefix}_{idx:03d}", type=obj_type, position=list(position),
            mass=props["mass"], friction=props["friction"], buoyancy=props["buoyancy"],
            volume_m3=props["volume_m3"], drag_coefficient=props["drag_coefficient"],
            ground_contact_area=props["ground_contact_area"],
            cross_sectional_area=props["cross_sectional_area"],
            is_static=props["is_static"],
            metadata={key: props[key] for key in props if key not in _COLUMN_PROPERTIES},
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
            "version": 2,
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
                                 visible=bool(water.get("visible", True)))
        env = data.get("environment", {})
        state.environment = EnvironmentState(
            gravity=finite_number(env.get("gravity", 9.81), "environment.gravity"),
            wind=vector3(env.get("wind", [0.0, 0.0, 0.0]), "environment.wind"),
            temperature=finite_number(env.get("temperature", 15.0), "environment.temperature"))
        objects = data.get("objects", [])
        if not isinstance(objects, list):
            raise ValueError("objects must be an array")
        version = int(data.get("version", 1))
        if version < 2:
            objects = [{**o, "buoyancy": 1.0} if isinstance(o, dict) else o
                       for o in objects]
        parsed = [WorldObject.from_dict(o) for o in objects]
        if len({o.id for o in parsed}) != len(parsed):
            raise ValueError("object IDs must be unique")
        state.objects = {o.id: o for o in parsed}
        state.sync_counter()
        return state

    def clone(self) -> "WorldState":
        return WorldState.from_dict(self.to_dict())
