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


# TsunamiLab. Measured, not picked: docs/probe_tsunami_v1.py swept
# (ocean_depth, amplitude, half_width) on the real solver and reported this
# exact point as depth=15m amp=2.0m hw=15m -> dry_at=7.2s, wave_at=8.5s,
# max|u|=1.8 m/s, run-up to x=-25 (measured on the probe's own bed, which is
# the same bed coastline() writes -- cross-checked equal to 5e-7 m). Also
# note what this does NOT do, because a real player will ask "why didn't the
# house fall over": HOUSE/BUILDING/BRIDGE are `is_static: True`
# (`_integrate_bodies` gates both floating and sliding on `static[i] == 0`),
# and `obj.damage` has exactly one driver in this codebase -- lava contact
# (see `_check_lava_ignition`) -- never hydrodynamic load. So the wave sweeps
# CAR / PERSON / DEBRIS / BOX / TREE (all dynamic) and the houses stand,
# splitting the flow around them. That is not a bug in this scenario; it is
# an honest limit of the physics, stated in the scenario's own hint text the
# same way the Volcano scenario states its lava run-out limit.
TSUNAMI_OCEAN_DEPTH_M = 15.0
TSUNAMI_AMPLITUDE_M = 2.0
TSUNAMI_HALF_WIDTH_M = 15.0
TSUNAMI_CENTRE_X = 60.0
TSUNAMI_MEASURED_RUNUP_X = -25.0


def build_tsunami(world: WorldState) -> Dict[str, Any]:
    effective = coastline(world.terrain, {"ocean_depth_m": TSUNAMI_OCEAN_DEPTH_M})
    water = world.water
    water.level = 0.0
    water.visible = True
    water.erosion_enabled = False
    water.outflow_enabled = True          # east edge stays open -- see coastline()'s
                                           # docstring: the ocean sits there on purpose,
                                           # so half the seeded pulse that heads back out
                                           # to sea simply leaves instead of reflecting
    water.tsunami_enabled = True
    water.tsunami_amplitude_m = TSUNAMI_AMPLITUDE_M
    water.tsunami_half_width_m = TSUNAMI_HALF_WIDTH_M
    water.tsunami_centre_x = TSUNAMI_CENTRE_X
    return effective


def build_beachfront_town(world: WorldState) -> Dict[str, int]:
    """Place a beach (swept) and a village (stands, splits the flow) relative
    to the MEASURED shore_x and run-up, not guessed offsets -- the probe's
    own change log has a bug entry for exactly this mistake (an early draft
    guessed a shore position and was ~20 m off, silently sampling open ocean)."""
    rng = np.random.RandomState(20260909)
    counts: Dict[str, int] = {}

    def tally(kind: str, n: int = 1) -> None:
        counts[kind] = counts.get(kind, 0) + n

    # `main()` calls build(world) then build_objects(world) -- the latter gets
    # no access to build_tsunami's own `effective` dict (build_volcano_settlement
    # has the same shape, and works around it by recomputing from fixed
    # constants). coastline() is a pure function of these same module
    # constants, so calling it again here just rewrites terrain.heights with
    # the identical array it already holds -- cheap, and the one honest way to
    # get shore_x without either a second scenario-plumbing change or a
    # hand-copied number that can drift out of sync with the real terrain.
    shore_x = coastline(world.terrain, {"ocean_depth_m": TSUNAMI_OCEAN_DEPTH_M})["shore_x"]

    # Layout distances below are measured, not guessed, and NOT the same
    # number as TSUNAMI_MEASURED_RUNUP_X: that constant is an ABSOLUTE world
    # x (-25), and shore_x is itself ~-20.4, so the run-up is only ~4.6-5.6 m
    # INLAND of the shoreline -- not the 13-26 m an earlier draft of this
    # function placed the beach scatter and village at, which put the entire
    # village and most of the "beach" outside anything the wave could ever
    # reach. Caught by loading the actual generated scenario in a browser and
    # reading back where the swept objects actually settled (~4.5-5.6 m
    # inland) -- the live scene, with a village obstructing the flow, is the
    # more trustworthy measurement here, not the bare-bed probe alone.

    # --- the beach: loose, light, draggy -- exactly what a wave carries off ---
    for _ in range(10):
        z = rng.uniform(-16.0, 16.0)
        x = shore_x + rng.uniform(-5.0, 3.0)      # a little seaward of the shore
        seat(world, "DEBRIS", x, z)                # to right at the run-up edge
        tally("debris")
    for dx, dz in ((-1.0, -8.0), (-3.0, 4.0), (-4.5, -3.0)):
        seat(world, "BOX", shore_x + dx, dz)
        tally("crate")
    for x, z in ((shore_x + 1.0, -10.0), (shore_x - 2.0, 8.0), (shore_x - 4.0, -2.0)):
        seat(world, "CAR", x, z, yaw=np.radians(rng.uniform(0.0, 360.0)))
        tally("car")
    for x, z in ((shore_x + 1.5, 3.0), (shore_x - 1.0, -6.0), (shore_x - 2.5, 9.0),
                 (shore_x - 4.0, -9.0), (shore_x - 0.5, 12.0)):
        seat(world, "PERSON", x, z)
        tally("person")
    for _ in range(6):
        x = shore_x - rng.uniform(0.0, 5.0)
        z = rng.uniform(-16.0, 16.0)
        seat(world, "TREE", x, z)
        tally("tree")

    # --- the village: static, right at the measured run-up edge -------------
    # Close enough that the wave's leading edge genuinely reaches (or laps
    # at) the front row -- the flow splits around them, which is the honest
    # version of "hits the village" this build can actually show (see the
    # module-level comment on TSUNAMI_* about static bodies and damage).
    for i, z in enumerate((-14.0, -4.0, 6.0, 16.0)):
        x = shore_x - 6.0 - (i % 2) * 2.0
        seat(world, "HOUSE", x, z, yaw=np.pi)     # doors face the sea, on purpose:
        tally("house")                             # the view of the thing that hits them
    for x, z in ((shore_x - 8.0, -9.0), (shore_x - 7.5, 11.0)):
        seat(world, "CAR", x, z)
        tally("car")

    # --- instruments ----------------------------------------------------
    # One a couple of metres seaward of the measured shoreline -- this is the
    # readout for "did it recede, then did a wave arrive" (surface_elevation_m
    # goes to ~0 during the drawback, then spikes). One at the village's own
    # front edge -- this is "did the wave actually reach the houses".
    seat(world, "GAUGE", shore_x + 2.0, 0.0)
    seat(world, "GAUGE", shore_x - 5.0, 0.0)
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
            print(f"  {len(world.objects)} objects, shore at x={effective['shore_x']:.1f} m, "
                  f"ocean depth {TSUNAMI_OCEAN_DEPTH_M:.0f} m")
            print(f"  seeded pulse: amplitude {TSUNAMI_AMPLITUDE_M:.1f} m, half-width "
                  f"{TSUNAMI_HALF_WIDTH_M:.0f} m; measured: drawback at t=7.2s, wave at "
                  f"t=8.5s, run-out to x={TSUNAMI_MEASURED_RUNUP_X:.0f} m "
                  f"(docs/probe_tsunami_v1.py)")
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
