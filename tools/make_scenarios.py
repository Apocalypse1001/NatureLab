"""Write the ready-to-load scenario worlds into data/.

    python tools/make_scenarios.py            # all scenarios
    python tools/make_scenarios.py --only dam

Three scenarios ship: `scenario_river` (a town on the bank of a running river),
`scenario_dam` (the same town, below a dam holding back a reservoir), and
`scenario_volcano` (a settlement on the flank of a generated cone, graded by
distance from the vent -- see docs/10_volcano2_plan.md). All are ordinary saved
worlds -- the save format already carries the whole terrain and every object, so
a scenario needs no new file format, no new WebSocket op and no new load path.
The frontend's Scenarios buttons send exactly the `load` the existing LOAD
button sends.

This lives in `tools/` for the reason `make_river_world.py` does: the layout used
in tests and screenshots has to be reproducible from a command line, and a broken
frontend must never be what stands between someone and a scenario on screen.

Rules the layout obeys, all of which are easy to get wrong invisibly:

* **Every object is seated on the ground it stands on.** The floodplain here sits
  near 2.5-2.9 m (and the volcano's flank varies continuously from 0 to 36 m), so
  an object written with `y = 0` is buried or floating and simply looks *missing*
  in a screenshot -- not broken, missing. Every placement goes through `seat()`,
  which reads `terrain.height_at`.
* **Nothing is placed in the river channel.** The trapezoidal channel plus its
  banks occupy |z| < bed_width/2 + bank_run, and a house dropped in there is a
  dam the user did not ask for.
* **The volcano settlement is placed at graded, measured distance from the vent**
  (build_volcano_settlement's docstring), not scattered -- the whole point is a
  legible causal experiment, the same role the dam scenario's discharge slider
  plays for build_dam.
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
from app.terrain_gen import coastline, dam_ridge, river_valley, volcano_cone  # noqa: E402
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


# VolcanoLab (v0.14.0). Kīlauea is the PHYSICAL reference, not a 1:1 scale
# model: it already supplies config.py's real numbers (LAVA_ERUPTION_TEMP_C =
# 1150, LAVA_SOLIDUS_TEMP_C = 980 -- genuinely its erupting/solidus basalt
# temperatures) and its behaviour (effusive summit-vent flows, a lava lake at
# the source). Its real flank slope is ~2-10 deg, which would need a base
# radius near 460 m for a 40 m cone -- more than twice this 200 m map, so the
# VERTICAL proportion is compressed to fit, the same way every other
# dimension in this world is not drawn at real-world scale. See
# docs/10_volcano2_plan.md.
VOLCANO_PEAK_M = 36.0
VOLCANO_BASE_RADIUS_M = 90.0
# Measured, not picked: docs/10_volcano2_plan.md's calibration probe ran the
# real solver (SimulationManager + this exact cone + a default VENT) to
# convergence at three slopes between 18.9 deg and this cone's 21.8 deg, and
# the lava front settled at L = 47-52 m in every one of them regardless of
# slope -- the same "not sharply sensitive to geometry" result
# docs/08_volcano_plan.md's Замер 2 already found on a gentler cone.
VOLCANO_MEASURED_RUNOUT_M = 48.0


def build_volcano(world: WorldState) -> Dict[str, Any]:
    effective = volcano_cone(world.terrain, {"peak_height": VOLCANO_PEAK_M,
                                              "base_radius": VOLCANO_BASE_RADIUS_M})
    vent_x, _, vent_z = effective["vent_position"]
    seat(world, "VENT", vent_x, vent_z)
    water = world.water
    water.level = 0.0
    water.visible = True
    water.erosion_enabled = False
    water.outflow_enabled = True
    return effective


def build_volcano_settlement(world: WorldState) -> Dict[str, int]:
    """Place a settlement around the cone at graded distance from the vent,
    rather than a random scatter -- the point is a legible causal experiment,
    the same one the dam scenario's discharge slider sets up (see build_dam's
    comment): this ships at a distance where the default VENT survives, and
    raising `vent_discharge_m3s` in the properties panel (or just waiting) is
    the user's to try.

    Three rings, all south-east of the vent so they read as one settlement
    rather than a test rig scattered on every compass point:

    * ~18-24 m -- an isolated homestead, inside the measured 48 m run-out.
      Expected to burn; it is the demonstration that the causal chain works.
    * ~55-65 m -- the village, just past the measured run-out. Safe at the
      shipped discharge, endangered by raising it or by waiting long enough
      for a slower creep past 48 m -- the actual experiment.
    * ~78-85 m -- a few outlying trees on the outer flank, first casualties
      if the flow ever reaches that far; nothing built there, on purpose.
    """
    rng = np.random.RandomState(20260906)
    counts: Dict[str, int] = {}

    def tally(kind: str, n: int = 1) -> None:
        counts[kind] = counts.get(kind, 0) + n

    def polar(distance: float, degrees: float) -> tuple:
        angle = np.radians(degrees)
        return distance * np.sin(angle), distance * np.cos(angle)

    # --- the homestead, inside the measured run-out ------------------------
    x, z = polar(20.0, 45.0)
    seat(world, "HOUSE", x, z, yaw=np.radians(225))
    tally("house")
    x, z = polar(24.0, 60.0)
    seat(world, "CAR", x, z)
    tally("car")
    x, z = polar(16.0, 30.0)
    seat(world, "TREE", x, z)
    tally("tree")

    # --- the village, just past it ------------------------------------------
    for i, deg in enumerate((35.0, 48.0, 62.0, 75.0)):
        x, z = polar(58.0 + (i % 2) * 5.0, deg)
        seat(world, "HOUSE", x, z, yaw=np.radians(deg + 180.0))
        tally("house")
    for x, z in (polar(60.0, 42.0), polar(66.0, 68.0)):
        seat(world, "CAR", x, z)
        tally("car")
    for x, z in (polar(52.0, 40.0), polar(56.0, 70.0), polar(50.0, 55.0)):
        seat(world, "PERSON", x, z)
        tally("person")

    # --- outer flank fringe --------------------------------------------------
    for _ in range(14):
        deg = rng.uniform(15.0, 85.0)
        distance = rng.uniform(70.0, 85.0)
        x, z = polar(distance, deg)
        seat(world, "TREE", x, z)
        tally("tree")

    # --- instruments -----------------------------------------------------
    # One at the measured run-out itself (does the flow reach here at all?),
    # one in the village (what is happening to the settlement?).
    x, z = polar(VOLCANO_MEASURED_RUNOUT_M, 52.0)
    seat(world, "GAUGE", x, z)
    x, z = polar(60.0, 52.0)
    seat(world, "GAUGE", x, z)
    tally("gauge", 2)
    return counts


# TsunamiLab, rebuilt for v0.14.1. The v0.14.0 version put the whole thing in
# the default 200 m world and measured only the depth at a fixed point, which
# hid that the scene did not show a tsunami at all: the sea retreated 1.4 m and
# the land flooded 4.6 m, on a "beach" sloping 1:2.3 -- a 24-degree cliff.
# docs/13_tsunami2_plan.md has the measurements; the two things that matter
# here are that how far the sea goes OUT is the drawdown divided by the beach
# SLOPE, and that a wave long enough to draw the sea back does not fit inside a
# 200 m box at all.
#
# So this scenario, and only this scenario, is built at a kilometre scale.
# `TerrainGrid.cell_size` is per-world and serialized, so the river, the dam
# and the volcano keep their 1 m cells and are not touched. The cost is stated
# rather than hidden: at 10 m cells a 4 m house is smaller than one cell, so
# the buildings here are scenery and landmarks for scale -- they do not split
# the flow the way they do in the river scenario, because the obstacle mask
# cannot resolve them. What this scale buys is the phenomenon itself.
TSUNAMI_CELL_SIZE_M = 10.0
TSUNAMI_OCEAN_DEPTH_M = 30.0
TSUNAMI_INLAND_HEIGHT_M = 12.0
TSUNAMI_LAND_EDGE_X = -950.0
TSUNAMI_BEACH_RUN_M = 900.0
TSUNAMI_AMPLITUDE_M = 6.0
TSUNAMI_PERIOD_S = 200.0
# Measured on the real solver by docs/probe_tsunami_v2.py, not chosen:
# amplitude 6 m and period 200 s were picked because they are the point
# where both halves of the signature are large AND max|u| stays near
# 5 m/s. Amplitude 9, or period 100, drives it to the
# FLUID_MAX_VELOCITY = 20 clamp exactly, where a numerical guard rather
# than the physics is shaping what you would see.
TSUNAMI_MEASURED_RETREAT_M = 62.0
TSUNAMI_MEASURED_FLOOD_M = 258.0


def _coastline_params() -> Dict[str, float]:
    return {"ocean_depth_m": TSUNAMI_OCEAN_DEPTH_M,
            "inland_height_m": TSUNAMI_INLAND_HEIGHT_M,
            "land_edge_x": TSUNAMI_LAND_EDGE_X,
            "beach_run_m": TSUNAMI_BEACH_RUN_M}


def build_tsunami(world: WorldState) -> Dict[str, Any]:
    # The cell size has to be set BEFORE the coastline is written: coastline()
    # reads it to lay the profile out in metres, and validate_coastline() sizes
    # its own bounds from the resulting world span rather than from
    # config.WORLD_SIZE_M, which describes the default world and not this one.
    world.terrain.cell_size = TSUNAMI_CELL_SIZE_M
    effective = coastline(world.terrain, _coastline_params())
    water = world.water
    water.level = 0.0
    water.visible = True
    water.erosion_enabled = False
    # The east edge is the wavemaker now, not an open outlet -- the two would
    # write the same columns. `_apply_tsunami_edge` suppresses the outlet while
    # it owns them; this flag is left off so the world file says the same thing
    # the solver does.
    water.outflow_enabled = False
    water.tsunami_enabled = True
    water.tsunami_amplitude_m = TSUNAMI_AMPLITUDE_M
    water.tsunami_period_s = TSUNAMI_PERIOD_S
    return effective


def build_beachfront_town(world: WorldState) -> Dict[str, int]:
    """A shore settlement placed against the MEASURED shoreline.

    Everything is positioned relative to `shore_x`, which coastline() measures
    off the bed it just wrote, never from a guessed offset -- the v1 probe's
    own change log has an entry for exactly that mistake, where a guessed
    shoreline put every "shore" reading 9 m underwater.
    """
    rng = np.random.RandomState(20260909)
    counts: Dict[str, int] = {}

    def tally(kind: str, n: int = 1) -> None:
        counts[kind] = counts.get(kind, 0) + n

    # main() calls build(world) then build_objects(world), and the latter gets
    # no access to the first one's return value. coastline() is a pure function
    # of the module constants above, so calling it again rewrites the identical
    # array and hands back the same measured shore_x -- cheaper than plumbing a
    # second argument through, and it cannot drift out of sync with the terrain
    # the way a hand-copied number would.
    shore_x = coastline(world.terrain, _coastline_params())["shore_x"]

    # --- the waterfront: loose, light, draggy, right at the water line ------
    for _ in range(14):
        seat(world, "DEBRIS", shore_x + rng.uniform(-40.0, 20.0),
             rng.uniform(-260.0, 260.0))
        tally("debris")
    for dz in (-150.0, -40.0, 70.0, 190.0):
        seat(world, "BOX", shore_x - rng.uniform(5.0, 45.0), dz)
        tally("crate")
    for dz in (-200.0, -90.0, 30.0, 140.0, 240.0):
        seat(world, "CAR", shore_x - rng.uniform(10.0, 60.0), dz,
             yaw=np.radians(rng.uniform(0.0, 360.0)))
        tally("car")
    for dz in (-230.0, -120.0, -20.0, 60.0, 180.0, 270.0):
        seat(world, "PERSON", shore_x - rng.uniform(5.0, 50.0), dz)
        tally("person")
    for _ in range(12):
        seat(world, "TREE", shore_x - rng.uniform(20.0, 120.0),
             rng.uniform(-280.0, 280.0))
        tally("tree")

    # --- the town, on the low coastal plain the wave runs over --------------
    # Towers rather than cottages: at 10 m cells a house is sub-cell, so what
    # a viewer can actually pick out from the camera distance this world forces
    # is a building with real height. BUILDING carries that on `floors`.
    for i, dz in enumerate((-320.0, -180.0, -60.0, 60.0, 180.0, 320.0)):
        x = shore_x - 90.0 - (i % 3) * 70.0
        seat(world, "BUILDING", x, dz, yaw=np.pi,
             metadata={"floors": float(4 + (i % 4) * 5)})
        tally("building")
    for i, dz in enumerate((-250.0, -110.0, 10.0, 130.0, 260.0)):
        seat(world, "HOUSE", shore_x - 250.0 - (i % 2) * 60.0, dz, yaw=np.pi)
        tally("house")

    # --- instruments -------------------------------------------------------
    # One in the shallows, which is the readout for "did the sea leave"; one on
    # dry land well inland, which only ever wets if the wave genuinely runs up
    # that far, and is therefore the readout for "did it reach the town".
    seat(world, "GAUGE", shore_x + 30.0, 0.0)
    seat(world, "GAUGE", shore_x - 120.0, 0.0)
    tally("gauge", 2)
    return counts


SCENARIOS = {
    "river": ("scenario_river", build_river, build_town),
    "dam": ("scenario_dam", build_dam, build_town),
    "volcano": ("scenario_volcano", build_volcano, build_volcano_settlement),
    "tsunami": ("scenario_tsunami", build_tsunami, build_beachfront_town),
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=sorted(SCENARIOS),
                        help="build just one scenario (default: all)")
    args = parser.parse_args(argv)

    wanted = [args.only] if args.only else sorted(SCENARIOS)
    for key in wanted:
        name, build, build_objects = SCENARIOS[key]
        world = WorldState()
        effective = build(world)          # terrain first: objects are seated on it
        counts = build_objects(world)
        path = persistence.save_world(world, name)

        print(f"wrote {path}")
        print("  " + ", ".join(f"{n} {kind}" for kind, n in sorted(counts.items())))
        if "vent_position" in effective:
            vent = next(o for o in world.objects.values() if o.type == "VENT")
            print(f"  {len(world.objects)} objects, "
                  f"vent Q {vent.metadata['vent_discharge_m3s']:g} m3/s")
            print(f"  cone peak {effective['peak_height']:.1f} m, base radius "
                  f"{effective['base_radius']:.1f} m; measured run-out "
                  f"{VOLCANO_MEASURED_RUNOUT_M:.0f} m at the shipped discharge "
                  f"(docs/10_volcano2_plan.md)")
        elif "shore_x" in effective:
            span = world.terrain.width * world.terrain.cell_size
            print(f"  {len(world.objects)} objects, world {span:.0f} m across at "
                  f"{world.terrain.cell_size:.0f} m cells (this scenario only)")
            print(f"  shore measured at x={effective['shore_x']:.1f} m, ocean depth "
                  f"{TSUNAMI_OCEAN_DEPTH_M:.0f} m")
            print(f"  east-edge wavemaker: amplitude {TSUNAMI_AMPLITUDE_M:.1f} m "
                  f"(peak {TSUNAMI_AMPLITUDE_M * 0.607:.1f} m), period "
                  f"{TSUNAMI_PERIOD_S:.0f} s; measured sea retreat "
                  f"{TSUNAMI_MEASURED_RETREAT_M:.0f} m, flood "
                  f"{TSUNAMI_MEASURED_FLOOD_M:.0f} m inland "
                  f"(docs/13_tsunami2_plan.md)")
        else:
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
