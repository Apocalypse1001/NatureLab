"""Storm sewer primitives (v0.17.0 Sewer-1, v0.18.0 Sewer-2; docs/16_sewer_plan.md).

The user asked for simple elements in the spirit of Cities: Skylines: a pipe
laid from point A to point B, joined to a drain, so that you can SEE water go
down the grate and come out into the river. So this is not a pipe solver:

- a STORM_INLET is a grate on the surface. It takes water through the same
  radial sink a DRAIN uses, but never more than the network below it can carry;
- a PIPE is a graph edge from a grate or a manhole to a manhole or an outfall.
  It holds no water. Its waypoints are the route the user drew and are drawn as
  a tube; nothing reads a gradient off them. What it carries is capped by
  Manning's formula for a pipe running full, from its diameter and the fall
  between its two ends;
- a MANHOLE is a junction: pipes run into it and one pipe runs on. It holds no
  water and does not overflow yet (Sewer-4), so what the network below cannot
  carry simply never leaves the grates -- it stays on the street;
- an OUTFALL is where a chain ends. Whatever the grates feeding it took this
  substep is poured out there, over a small disc, in the same substep.

A grate's water goes to the OUTFALL at the end of its chain, capped by the
narrowest pipe on the way. Where chains join, each shared pipe's capacity is
split between the grates above it by what they WANT -- measured on the GPU a
frame before (`allocate`) -- so a dry grate does not hold back a share a
flooded one needs.

A pipe running uphill carries nothing -- a gravity sewer cannot lift water --
and the UI says so rather than quietly pumping it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from . import config

INLET_TYPE = "STORM_INLET"
OUTFALL_TYPE = "OUTFALL"
MANHOLE_TYPE = "MANHOLE"
PIPE_TYPE = "PIPE"
SEWER_TYPES = frozenset({INLET_TYPE, OUTFALL_TYPE, MANHOLE_TYPE, PIPE_TYPE})
# what a pipe may start from and run into
PIPE_SOURCES = frozenset({INLET_TYPE, MANHOLE_TYPE})
PIPE_TARGETS = frozenset({MANHOLE_TYPE, OUTFALL_TYPE})
# resolve() runs every tick: a chain longer than this is reported as a loop
MAX_CHAIN = 256


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
    """What one PIPE is doing, for the solver and for the UI.

    status: "ok"; "uphill" (this pipe rises); "disconnected" (an end is not a
    node a pipe may join); "second_pipe" (its start already has a pipe running
    on); "blocked" (fine itself, but no grate's water reaches the river
    through it because a pipe on the way carries nothing -- `blocked_by` names
    it, above or below); "dead_end" (the chain stops at a manhole with no pipe
    out); "loop" (the chain comes back on itself). The last three are given
    only when every grate above the pipe is cut off.
    """
    pipe_id: str
    from_id: str
    to_id: str
    to_type: str
    capacity_m3s: float
    length_m: float
    fall_m: float
    status: str
    blocked_by: str = ""
    # the grates whose water runs through this pipe
    upstream_inlets: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        out = dict(self.__dict__)
        out["upstream_inlets"] = list(self.upstream_inlets)
        return out


@dataclass
class Network:
    """The resolved sewer: rows for the solver plus what the UI reads."""
    inlet_ids: List[str]
    inlet_centres: List[List[float]]
    inlet_radii: List[float]
    # the narrowest pipe on each grate's way to its outfall; 0 when it has none
    path_capacity: List[float]
    outfall_of: List[int]
    outfalls: List[Tuple[List[float], float]]
    links: List[SewerLink]

    def inlet_rows(self, capacities: List[float]) -> list:
        """(centre, radius, capacity, outfall_index) per grate, for set_sewer."""
        return list(zip(self.inlet_centres, self.inlet_radii, capacities, self.outfall_of))


def resolve(world) -> Network:
    """Build the network the solver runs from the objects in the world.

    Fall is the terrain height under a pipe's start node minus under its end
    node, not the objects' y, so a fixture nudged off the ground cannot change
    it. This runs every tick, so every walk is bounded: a loop is reported,
    never followed.
    """
    objects = world.objects
    terrain = world.terrain
    inlet_objs = [o for o in objects.values() if o.type == INLET_TYPE]
    outfall_objs = [o for o in objects.values() if o.type == OUTFALL_TYPE]
    outfall_index = {o.id: k for k, o in enumerate(outfall_objs)}

    links: List[SewerLink] = []
    out_link: Dict[str, SewerLink] = {}      # node id -> the pipe running on from it
    for pipe in (o for o in objects.values() if o.type == PIPE_TYPE):
        meta = pipe.metadata
        start = objects.get(str(meta.get("from_id", "")))
        end = objects.get(str(meta.get("to_id", "")))
        points = meta.get("points") or []
        if (start is None or start.type not in PIPE_SOURCES
                or end is None or end.type not in PIPE_TARGETS or len(points) < 2):
            links.append(SewerLink(pipe.id, str(meta.get("from_id", "")),
                                   str(meta.get("to_id", "")),
                                   end.type if end is not None else "",
                                   0.0, 0.0, 0.0, "disconnected"))
            continue
        route = ([list(start.position)] + [list(p) for p in points[1:-1]]
                 + [list(end.position)])
        length = path_length(route)
        fall = (terrain.height_at(start.position[0], start.position[2])
                - terrain.height_at(end.position[0], end.position[2]))
        q = pipe_capacity_m3s(float(meta.get("diameter_m", config.PIPE_DEFAULT_DIAMETER_M)),
                              fall, length)
        link = SewerLink(pipe.id, start.id, end.id, end.type, q, length, fall,
                         "ok" if q > 0.0 else "uphill")
        if start.id in out_link:
            link.status, link.capacity_m3s = "second_pipe", 0.0
        else:
            out_link[start.id] = link
        links.append(link)

    path_capacity: List[float] = []
    outfall_of: List[int] = []
    failures: Dict[str, Tuple[str, str]] = {}   # grate id -> (why, the pipe at fault)
    for inlet in inlet_objs:
        chain: List[SewerLink] = []
        seen = {inlet.id}
        node = inlet.id
        ending = "dead_end"
        while node in out_link:
            if len(chain) >= MAX_CHAIN:
                ending = "loop"
                break
            link = out_link[node]
            chain.append(link)
            if link.to_type == OUTFALL_TYPE:
                ending = "ok"
                break
            if link.to_id in seen:
                ending = "loop"
                break
            seen.add(link.to_id)
            node = link.to_id
        for link in chain:
            if inlet.id not in link.upstream_inlets:
                link.upstream_inlets.append(inlet.id)
        narrowest = min((link.capacity_m3s for link in chain), default=0.0)
        if chain and ending == "ok" and narrowest > 0.0:
            path_capacity.append(narrowest)
            outfall_of.append(outfall_index[chain[-1].to_id])
        else:
            failures[inlet.id] = (ending, next((l.pipe_id for l in chain
                                                if l.capacity_m3s <= 0.0), ""))
            path_capacity.append(0.0)
            outfall_of.append(-1)

    # A pipe that is fine in itself but carries no grate's water says why, so
    # the panel never reads "ok, 0.0 L/s" without a reason -- but only when
    # EVERY grate above it is cut off: a trunk still fed by one working chain
    # is ok, whatever a second chain into it does.
    for link in links:
        if link.status != "ok" or not link.upstream_inlets:
            continue
        if not all(i in failures for i in link.upstream_inlets):
            continue
        ending, culprit = failures[link.upstream_inlets[0]]
        if ending != "ok":
            link.status = ending
        else:
            link.status, link.blocked_by = "blocked", culprit

    return Network(
        inlet_ids=[o.id for o in inlet_objs],
        inlet_centres=[list(o.position) for o in inlet_objs],
        inlet_radii=[float(o.metadata.get("inlet_radius", config.STORM_INLET_RADIUS_M))
                     * float(o.scale[0]) for o in inlet_objs],
        path_capacity=path_capacity,
        outfall_of=outfall_of,
        outfalls=[(list(o.position),
                   float(o.metadata.get("outfall_radius", config.OUTFALL_RADIUS_M))
                   * float(o.scale[0])) for o in outfall_objs],
        links=links)


def shared(network: Network) -> bool:
    """Whether any pipe carries more than one grate's water."""
    return any(link.status == "ok" and len(link.upstream_inlets) > 1
               for link in network.links)


def allocate(network: Network, demand: Dict[str, float]) -> List[float]:
    """How much each grate may take this tick.

    Every grate may take up to the narrowest pipe on its own way down. Where
    grates share a pipe and together WANT more than it carries, they are cut
    max-min fair: to a common level, so a grate that wants less than the level
    keeps what it wants and the rest is split among the others. What a grate
    wants is measured on the GPU with it allowed its whole path
    (`sewer_inlet_demand_m3s`, one frame old) -- not what it took, because a
    sink allowed less takes less at the same depth, and splitting by takings
    spirals down. A grate with no measurement yet wants its whole path.
    """
    ids = network.inlet_ids
    allowed = list(network.path_capacity)
    wants = [min(path, demand.get(i, path)) for i, path in zip(ids, allowed)]
    row = {inlet_id: k for k, inlet_id in enumerate(ids)}
    links = [link for link in network.links
             if link.status == "ok" and len(link.upstream_inlets) > 1]
    for link in links:
        rows = [row[i] for i in link.upstream_inlets if i in row]
        if sum(wants[k] for k in rows) <= link.capacity_m3s:
            continue
        values = sorted(wants[k] for k in rows)
        remaining, level = link.capacity_m3s, 0.0
        for n, value in enumerate(values):
            share = remaining / (len(values) - n)
            if value > share:
                level = share
                break
            remaining -= value
        for k in rows:
            # a grate wanting less than the level may still grow to it; the
            # others are held at it
            allowed[k] = min(allowed[k], level)
            wants[k] = min(wants[k], level)
    return allowed
