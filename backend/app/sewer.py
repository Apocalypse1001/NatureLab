"""Storm sewer primitives (v0.17.0, docs/16_sewer_plan.md).

The user asked for simple elements in the spirit of Cities: Skylines: a pipe
laid from point A to point B, joined to a drain, so that you can SEE water go
down the grate and come out into the river. So this is not a pipe solver:

- a STORM_INLET is a grate on the surface. It takes water through the same
  radial sink a DRAIN uses, but never more than its pipe can carry;
- a PIPE is a graph edge from one inlet to one outfall. It holds no water. Its
  waypoints are the route the user drew and are drawn as a tube; nothing reads
  a gradient off them. What it carries is capped by Manning's formula for a
  pipe running full, from its diameter and the fall between its two ends;
- an OUTFALL is where the pipe ends. Whatever the inlets feeding it took this
  substep is poured out there, over a small disc, in the same substep.

A pipe running uphill carries nothing -- a gravity sewer cannot lift water --
and the UI says so rather than quietly pumping it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from . import config

INLET_TYPE = "STORM_INLET"
OUTFALL_TYPE = "OUTFALL"
PIPE_TYPE = "PIPE"
SEWER_TYPES = frozenset({INLET_TYPE, OUTFALL_TYPE, PIPE_TYPE})


def pipe_capacity_m3s(diameter_m: float, fall_m: float, length_m: float) -> float:
    """Manning, circular pipe running full: Q = (1/n) * A * R^(2/3) * S^(1/2).

    A = pi d^2 / 4 and the hydraulic radius of a full circle is R = d / 4.
    No fall (or a rise) is no capacity at all.
    """
    if diameter_m <= 0.0 or length_m <= 0.0 or fall_m <= 0.0:
        return 0.0
    slope = fall_m / length_m
    area = math.pi * diameter_m * diameter_m / 4.0
    radius = diameter_m / 4.0
    return area * radius ** (2.0 / 3.0) * math.sqrt(slope) / config.SEWER_MANNING_N


def path_length(points: List[List[float]]) -> float:
    total = 0.0
    for a, b in zip(points, points[1:]):
        total += math.hypot(b[0] - a[0], b[2] - a[2])
    return total


def validate_pipe_metadata(metadata: Dict[str, Any]) -> None:
    """Raise ValueError unless a PIPE's metadata is well formed."""
    points = metadata.get("points")
    if not isinstance(points, list) or not 2 <= len(points) <= config.PIPE_MAX_POINTS:
        raise ValueError(f"pipe.points must be a list of 2..{config.PIPE_MAX_POINTS} points")
    for k, point in enumerate(points):
        if (not isinstance(point, (list, tuple)) or len(point) != 3
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(float(v)) for v in point)):
            raise ValueError(f"pipe.points[{k}] must be three finite numbers")
    diameter = metadata.get("diameter_m", config.PIPE_DEFAULT_DIAMETER_M)
    if (isinstance(diameter, bool) or not isinstance(diameter, (int, float))
            or not config.PIPE_MIN_DIAMETER_M <= float(diameter) <= config.PIPE_MAX_DIAMETER_M):
        raise ValueError(f"pipe.diameter_m must be within "
                         f"[{config.PIPE_MIN_DIAMETER_M}, {config.PIPE_MAX_DIAMETER_M}] m")
    for key in ("from_id", "to_id"):
        if not isinstance(metadata.get(key, ""), str):
            raise ValueError(f"pipe.{key} must be a string")


@dataclass
class SewerLink:
    """What one PIPE is doing, for the solver and for the UI."""
    pipe_id: str
    inlet_id: str
    outfall_id: str
    capacity_m3s: float
    length_m: float
    fall_m: float
    status: str          # "ok" | "uphill" | "disconnected" | "second_pipe"

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def resolve(world) -> Tuple[list, list, List[SewerLink], List[str]]:
    """Build the network the solver runs from the objects in the world.

    Returns (inlets, outfalls, links, inlet_ids):
    - inlets: (centre_xyz, radius_m, capacity_m3s, outfall_index) per
      STORM_INLET, in world order; outfall_index is -1 when no working pipe
      leaves it, and its capacity is then 0 -- a grate with no pipe behind it
      takes nothing;
    - outfalls: (centre_xyz, radius_m) per OUTFALL;
    - links: one SewerLink per PIPE;
    - inlet_ids: the object id behind each inlet row, for mapping flows back.

    Fall is the terrain height under the inlet minus under the outfall, not
    the objects' y, so a fixture nudged off the ground cannot change it.
    """
    objects = world.objects
    terrain = world.terrain
    inlet_objs = [o for o in objects.values() if o.type == INLET_TYPE]
    outfall_objs = [o for o in objects.values() if o.type == OUTFALL_TYPE]
    outfall_index = {o.id: k for k, o in enumerate(outfall_objs)}
    capacity: Dict[str, float] = {}
    target: Dict[str, int] = {}
    links: List[SewerLink] = []
    for pipe in (o for o in objects.values() if o.type == PIPE_TYPE):
        meta = pipe.metadata
        inlet = objects.get(str(meta.get("from_id", "")))
        outfall = objects.get(str(meta.get("to_id", "")))
        points = meta.get("points") or []
        if (inlet is None or inlet.type != INLET_TYPE
                or outfall is None or outfall.type != OUTFALL_TYPE or len(points) < 2):
            links.append(SewerLink(pipe.id, str(meta.get("from_id", "")),
                                   str(meta.get("to_id", "")), 0.0, 0.0, 0.0,
                                   "disconnected"))
            continue
        route = ([list(inlet.position)] + [list(p) for p in points[1:-1]]
                 + [list(outfall.position)])
        length = path_length(route)
        fall = (terrain.height_at(inlet.position[0], inlet.position[2])
                - terrain.height_at(outfall.position[0], outfall.position[2]))
        q = pipe_capacity_m3s(float(meta.get("diameter_m", config.PIPE_DEFAULT_DIAMETER_M)),
                              fall, length)
        if inlet.id in capacity:
            status, q = "second_pipe", 0.0
        else:
            status = "ok" if q > 0.0 else "uphill"
            capacity[inlet.id] = q
            target[inlet.id] = outfall_index[outfall.id] if q > 0.0 else -1
        links.append(SewerLink(pipe.id, inlet.id, outfall.id, q, length, fall, status))

    inlets = [(list(o.position),
               float(o.metadata.get("inlet_radius", config.STORM_INLET_RADIUS_M))
               * float(o.scale[0]),
               capacity.get(o.id, 0.0), target.get(o.id, -1))
              for o in inlet_objs]
    outfalls = [(list(o.position),
                 float(o.metadata.get("outfall_radius", config.OUTFALL_RADIUS_M))
                 * float(o.scale[0]))
                for o in outfall_objs]
    return inlets, outfalls, links, [o.id for o in inlet_objs]
