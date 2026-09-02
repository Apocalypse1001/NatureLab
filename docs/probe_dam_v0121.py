"""How long does the dam scenario take to fill, spill, and overtop?

Written for the v0.12.2 scenarios; the file keeps its v0121 name because that
is the measurement the shipped discharge is quoted from.

    python docs/probe_dam_v0121.py [--q 30 60] [--seconds 600]

The question this answers is a scenario-design one, not a solver one: the dam
scenario is worth shipping only if a child sees the causal chain
*inflow -> reservoir rises -> spillway runs -> crest is topped -> the town
floods* inside a sitting. `dam_ridge` reports `reservoir_volume_m3` as the
volume below the spillway lip, and dividing that by Q gives the time to FIRST
SPILL -- but not the time to overtopping, because once the lip is passed the
reservoir keeps rising over a floodplain far wider than the channel, and the
spillway bleeds off a share of the inflow that grows as the head over the lip
grows. That balance is what decides whether a given Q ever tops the crest at
all, and it is not something to guess from a weir formula on this geometry.

So this runs the real solver on the real generated terrain, headless, and prints
the reservoir surface against the two elevations that matter. No browser, no
objects: this is about water and bed only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app import config                              # noqa: E402
from app.compute_engine import create_engine        # noqa: E402
from app.fluid_solver import create_fluid_solver    # noqa: E402
from app.terrain_gen import dam_ridge               # noqa: E402
from app.world_state import WorldState              # noqa: E402


def run(discharge: float, seconds: float, report_every: float) -> None:
    world = WorldState()
    effective = dam_ridge(world.terrain, None, None)
    crest = effective["crest_elevation"]
    lip = effective["spill_elevation"]
    dam_x = effective["dam_x"]

    world.water.level = 0.0
    world.water.outflow_enabled = True
    world.water.erosion_enabled = False
    world.water.inlet_enabled = True
    world.water.inlet_centre_z = 0.0
    world.water.inlet_width_m = 12.0
    world.water.inlet_discharge_m3s = discharge

    engine = create_engine()
    solver = create_fluid_solver(engine.device)
    solver.initialize(world)
    solver.set_river_inlet(True, 0.0, 12.0, discharge)

    terrain = world.terrain
    cell = terrain.cell_size
    columns = terrain.width + 1
    x = np.arange(columns) * cell
    upstream = x < dam_x
    downstream = x > dam_x + 20.0
    bed = np.asarray(terrain.heights, dtype=np.float64)

    print(f"Q = {discharge:g} m3/s   crest {crest:.2f} m   spillway lip {lip:.2f} m")
    print(f"{'t (s)':>7} {'reservoir (m)':>14} {'over lip (m)':>13} "
          f"{'to crest (m)':>13} {'downstream max (m)':>19}")

    dt = 1.0 / 60.0
    steps = int(seconds / dt)
    next_report = 0.0
    first_spill = None
    first_overtop = None
    for step in range(1, steps + 1):
        solver.set_boundaries(terrain, {}, 0, 0)
        solver.advance(dt, config.FLUID_MAX_SUBSTEPS, config.FLUID_STABILITY_DT)
        t = step * dt
        if t + 1e-9 < next_report:
            continue
        next_report += report_every
        # get_water_height_field() is bed + depth for wet cells and bed - 0.05
        # for dry ones (the WATER_HEIGHT frame's own convention), so depth comes
        # back through a clamp rather than by trusting the dry sentinel.
        surface_grid = np.asarray(solver.get_water_height_field(),
                                  dtype=np.float64).reshape(bed.shape)
        depth = np.clip(surface_grid - bed, 0.0, None)
        surface = bed + depth
        reservoir = float(surface[:, upstream][depth[:, upstream] > 1e-3].max()) \
            if (depth[:, upstream] > 1e-3).any() else float("nan")
        down = float(depth[:, downstream].max())
        if first_spill is None and reservoir >= lip:
            first_spill = t
        if first_overtop is None and reservoir >= crest:
            first_overtop = t
        print(f"{t:7.0f} {reservoir:14.3f} {reservoir - lip:13.3f} "
              f"{crest - reservoir:13.3f} {down:19.3f}")

    print(f"  first spill over the lip: "
          f"{'never' if first_spill is None else f'{first_spill:.0f} s'}")
    print(f"  crest overtopped:         "
          f"{'never' if first_overtop is None else f'{first_overtop:.0f} s'}")
    print()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q", type=float, nargs="+", default=[30.0, 60.0],
                        help="discharges to try, m3/s")
    parser.add_argument("--seconds", type=float, default=600.0,
                        help="simulated seconds per run")
    parser.add_argument("--every", type=float, default=60.0,
                        help="reporting interval, simulated seconds")
    args = parser.parse_args(argv)
    for discharge in args.q:
        run(discharge, args.seconds, args.every)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
