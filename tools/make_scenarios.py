"""Write the ready-to-load scenario worlds into data/.

    python tools/make_scenarios.py            # both scenarios
    python tools/make_scenarios.py --only dam

Two scenarios ship: `scenario_river` (a town on the bank of a running river) and
`scenario_dam` (the same town, below a dam holding back a reservoir). Both are
ordinary saved worlds -- the save format already carries the whole terrain and
every object, so a scenario needs no new file format, no new WebSocket op and no
new load path. The frontend's Scenarios buttons send exactly the `load` the
existing LOAD button sends.

This lives in `tools/` for the reason `make_river_world.py` does: the layout used
in tests and screenshots has to be reproducible from a command line, and a broken
frontend must never be what stands between someone and a scenario on screen.

Two rules the layout obeys, both of which are easy to get wrong invisibly:

* **Every object is seated on the ground it stands on.** The floodplain here sits
  near 2.5-2.9 m, so an object written with `y = 0` is buried three metres deep
  and simply looks *missing* in a screenshot from above -- not broken, missing.
  Every placement goes through `seat()`, which reads `terrain.height_at`.
* **Nothing is placed in the channel.** The trapezoidal channel plus its banks
  occupy |z| < bed_width/2 + bank_run, and a house dropped in there is a dam the
  user did not ask for.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app import persistence                                    # noqa: E402
from app.terrain_gen import dam_ridge, river_valley           # noqa: E402
from app.world_state import WorldState                         # noqa: E402

# The town sits on the north bank, downstream of the dam station, in both
# scenarios: it is the same town, so the two runs are comparable, and putting it
# below the dam is the whole point of the dam scenario.
STREET_Z = 24.0            # centreline of the main street (world z, m)
STREET_HALF = 3.5          # street half-width, matches the ROAD builder's 7 m
SOUTH_ROW_Z = 17.0         # houses between the street and the river
NORTH_ROW_Z = 31.0         # houses on the far side of the street
DUMP_AT = (-20.0, 18.0)    # upstream of the town, downstream of the dam crest
DUMP_RADIUS = 5.0          # keeps the scatter clear of the bank top at z = 10


def seat(world: WorldState, obj_type: str, x: float, z: float,
         yaw: float = 0.0, scale: List[float] | None = None,
         **fields: Any):
    """Add an object standing on the terrain at (x, z), not floating over it."""
    y = world.terrain.height_at(x, z)
    obj = world.add_object(obj_type, [float(x), float(y), float(z)])
    obj.rotation = [0.0, float(yaw), 0.0]
    if scale is not None:
        obj.scale = [float(v) for v in scale]
    for key, value in fields.items():
        setattr(obj, key, value)
    return obj


def build_town(world: WorldState) -> Dict[str, int]:
    """Place the shared town: houses, street, figures, forest and the dump."""
    rng = np.random.RandomState(20260902)   # fixed: the same town every time
    counts: Dict[str, int] = {}

    def tally(kind: str, n: int = 1) -> None:
        counts[kind] = counts.get(kind, 0) + n

    # --- the street -------------------------------------------------------
    # Six 12 m segments rather than one 72 m slab: the floodplain falls, and a
    # single slab would be buried at one end and floating at the other.
    for x in (0.0, 12.0, 24.0, 36.0, 48.0, 60.0):
        seat(world, "ROAD", x, STREET_Z)
        tally("road")
    # a spur running down to the bank, so the flood has a way up into the town
    seat(world, "ROAD", 30.0, 15.0, yaw=np.pi / 2)
    tally("road")

    # --- ten houses, five a side, doors facing the street -----------------
    for x in (0.0, 13.0, 26.0, 39.0, 52.0):
        seat(world, "HOUSE", x, SOUTH_ROW_Z)                  # door faces +z
        tally("house")
    for x in (6.0, 19.0, 32.0, 45.0, 58.0):
        seat(world, "HOUSE", x, NORTH_ROW_Z, yaw=np.pi)       # door faces -z
        tally("house")

    # --- cars on the street ----------------------------------------------
    for x, z in ((9.0, 22.2), (33.0, 25.8), (55.0, 22.2)):
        seat(world, "CAR", x, z)
        tally("car")

    # --- figures ----------------------------------------------------------
    for x, z in ((3.0, 20.6), (22.0, 27.4), (35.0, 20.4),
                 (50.0, 27.0), (63.0, 24.0), (16.0, 13.0)):
        seat(world, "PERSON", x, z)
        tally("person")

    # --- the wood behind the town, and a fringe along the bank ------------
    for _ in range(18):
        x = rng.uniform(-14.0, 72.0)
        z = rng.uniform(40.0, 60.0)
        seat(world, "TREE", x, z)
        tally("tree")
    for _ in range(7):
        x = rng.uniform(-6.0, 68.0)
        z = rng.uniform(11.5, 13.5)
        seat(world, "TREE", x, z)
        tally("tree")

    # --- the dump ---------------------------------------------------------
    # Placed UPSTREAM of the town on purpose. Everything in it is light and
    # draggy, so when the water arrives the rubbish is what moves first and it
    # moves into the town -- which is the causal chain docs/01_vision.md asks
    # for, not scenery: flood arrives -> the dump goes -> it piles against the
    # houses and the bridge piers.
    cx, cz = DUMP_AT
    for _ in range(12):
        angle = rng.uniform(0.0, 2.0 * np.pi)
        radius = rng.uniform(0.0, DUMP_RADIUS)
        seat(world, "DEBRIS", cx + radius * np.cos(angle), cz + radius * np.sin(angle))
        tally("debris")
    for dx, dz in ((-4.0, -3.0), (0.5, -4.0), (4.5, 1.0)):
        seat(world, "BOX", cx + dx, cz + dz)
        tally("crate")
    # three skips: a scaled crate is a bigger crate, so its mass and volume are
    # written to match rather than left at the 50 kg of a 1.2 m box
    for dx, dz in ((-5.5, 3.0), (2.0, 4.5)):
        seat(world, "BOX", cx + dx, cz + dz, yaw=0.4,
             scale=[2.4, 1.2, 1.6], mass=520.0, volume_m3=8.0,
             ground_contact_area=5.5, cross_sectional_area=4.0)
        tally("skip")

    # --- instruments ------------------------------------------------------
    # One in the channel (what the river is doing) and one on the street (what
    # is happening to the town). Two gauges is the smallest set that lets a
    # child compare "the river rose by X" with "the street went under".
    seat(world, "GAUGE", 20.0, 0.0)
    seat(world, "GAUGE", 30.0, STREET_Z)
    tally("gauge", 2)
    return counts


def build_river(world: WorldState) -> Dict[str, Any]:
    river = river_valley(world.terrain, None)
    water = world.water
    water.level = 0.0
    water.visible = True
    water.erosion_enabled = False       # see docs/07_river_plan.md: the incision
    water.outflow_enabled = True        # feedback is not calibrated yet
    water.inlet_enabled = True
    water.inlet_centre_z = 0.0
    water.inlet_width_m = 12.0
    water.inlet_discharge_m3s = 12.0
    return river


def build_dam(world: WorldState) -> Dict[str, Any]:
    effective = dam_ridge(world.terrain, None, None)
    water = world.water
    water.level = 0.0
    water.visible = True
    water.erosion_enabled = False
    water.outflow_enabled = True
    water.inlet_enabled = True
    water.inlet_centre_z = 0.0
    water.inlet_width_m = 12.0
    # 30 m3/s is measured, not picked. `docs/probe_dam_v0121.py` ran the real
    # solver on this exact terrain:
    #
    #   Q = 30: reservoir tops the spillway lip at 120 s and then settles at
    #           3.73 m against a 3.76 m crest -- the spillway carries the whole
    #           inflow and the dam holds, indefinitely.
    #   Q = 60: same lip at 120 s, but the crest is overtopped at 360 s and the
    #           water below the dam reaches 3.85 m. That is the flood.
    #
    # So the scenario ships at the discharge where everything works, and the
    # experiment is the child's: push Q up the slider (it goes to 80) and watch
    # the crest go under, or cut the crest with the terrain brush and watch it
    # go at once. A scenario that arrives already broken has nothing to ask.
    water.inlet_discharge_m3s = 30.0
    return effective


SCENARIOS = {
    "river": ("scenario_river", build_river),
    "dam": ("scenario_dam", build_dam),
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=sorted(SCENARIOS),
                        help="build just one scenario (default: both)")
    args = parser.parse_args(argv)

    wanted = [args.only] if args.only else sorted(SCENARIOS)
    for key in wanted:
        name, build = SCENARIOS[key]
        world = WorldState()
        effective = build(world)          # terrain first: the town is seated on it
        counts = build_town(world)
        path = persistence.save_world(world, name)

        print(f"wrote {path}")
        print("  " + ", ".join(f"{n} {kind}" for kind, n in sorted(counts.items())))
        print(f"  {len(world.objects)} objects, "
              f"inlet Q {world.water.inlet_discharge_m3s:g} m3/s")
        if "crest_elevation" in effective:
            print(f"  dam crest {effective['crest_elevation']:.2f} m, spillway lip "
                  f"{effective['spill_elevation']:.2f} m, channel bed "
                  f"{effective['channel_bed_at_dam']:.2f} m")
            print(f"  reservoir {effective['reservoir_volume_m3']:.0f} m3 below "
                  f"the lip; measured: spills at 120 s, holds at Q=30, "
                  f"crest overtopped at 360 s at Q=60")
        else:
            print(f"  channel bed {effective['inlet_bed']:.2f} m (inlet) -> "
                  f"{effective['outlet_bed_actual']:.2f} m (outlet)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
