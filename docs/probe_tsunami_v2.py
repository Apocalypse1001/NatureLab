"""Diagnostic probe: can a coastal domain 200 m wide show a real drawback?

The v1 probe (`probe_tsunami_v1.py`) asked "does depth at a fixed point go down
and then up". It does, and that answer was accepted -- but it is a much weaker
question than the one the scenario actually claims to answer, which is the one
a person watching asks: **does the sea visibly leave the beach, and does the
wave then flood the land?** Measured as waterline movement rather than depth at
a point, what v1 shipped does neither:

    sea retreated 1.4 m, flooded 4.6 m inland, on a beach sloping 1:2.3 (24 deg)

Two things are wrong with v1, and only one of them is a constant.

1. **The beach is a cliff.** 24 degrees. Real tsunami-vulnerable coast is 1:50
   to 1:1000. On a 1:2.3 face a 1.2 m drawdown moves the waterline 2.7 m, so
   the drawback is invisible no matter how the wave is tuned.

2. **The wave is far too short, and short enough to be outside the solver's own
   validity.** Shallow-water equations assume wavelength >> depth. v1 seeds a
   pulse of half-width 15 m into water 15 m deep: wavelength/depth ~ 4, where
   the theory needs that ratio to be large. A tsunami's defining property is
   the opposite extreme -- wavelength of 100+ km, period of 10-60 minutes --
   and THAT is why the real sea withdraws for many minutes and goes far out.
   A 15 m pulse draws the water down for about two seconds.

   Sweeping the half-width confirms it is the dominant term, not the amplitude:
   at half-width 15 m the retreat is 0.1-1.4 m across every geometry tried; at
   50 m it reaches 8-18 m.

The second point is fatal to v1's whole approach, not just its numbers. A wave
long enough to draw the sea back does not FIT in a 200 m box together with the
shore and the land behind it: pushing the pulse far enough offshore to start
calm clips its crest off the east edge, and making it long enough clips it
anyway. v1 seeded the wave inside the domain, so the domain had to hold the
whole wave.

**So this probe tests a different mechanism: the wave ARRIVES from outside.**
The east edge is driven as a prescribed sea level over time -- exactly the
mechanism `_apply_source` already implements on the west edge, only time-varying
and on the other side. The open ocean beyond the map draws down and then surges;
the map is a window onto a coast, not a container for a whole wave. The wave
period is then a free parameter in TIME and is no longer limited by how much
ocean fits on screen, which is the entire point.

    eta(t) = A * ((t - t0) / T) * exp(-0.5 * ((t - t0) / T)^2),   t0 = 2T

The same N-shape v1 used, in time instead of space: negative (trough, the sea
withdrawing) before t0, positive (crest, the wave) after it. Trough first is
again the whole mechanism, not a special case in the code. Peak magnitude is
A*exp(-0.5) = 0.607*A, not A -- worth stating, since v1's plan doc quotes A as
though it were the wave height.

Read-only: imports the real solver, drives it exactly as a kernel would, and
writes nothing. Run from the repo root:
    python docs/probe_tsunami_v2.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app import config                              # noqa: E402
from app.compute_engine import create_engine        # noqa: E402
from app.fluid_solver import create_fluid_solver    # noqa: E402
from app.terrain_gen import coastline               # noqa: E402
from app.world_state import WorldState              # noqa: E402

N = config.TERRAIN_CELLS + 1
CELL = config.TERRAIN_CELL_SIZE
DT = config.FIXED_DT
DRY = config.FLUID_DRY_DEPTH
XS = (np.arange(N) - (N - 1) * 0.5) * CELL
EDGE_COLUMNS = config.FLUID_OUTFLOW_COLUMNS


def edge_level(t, amplitude, period):
    """Sea level outside the map at time t. See the module docstring."""
    z = (t - 2.0 * period) / period
    return amplitude * z * np.exp(-0.5 * z * z)


def run(ocean_depth, land_edge_x, beach_run, amplitude, period,
        seconds=90.0, verbose=False):
    world = WorldState()
    effective = coastline(world.terrain, {"ocean_depth_m": ocean_depth,
                                          "land_edge_x": land_edge_x,
                                          "beach_run_m": beach_run})
    shore_x = effective["shore_x"]
    world.water.level = 0.0
    solver = create_fluid_solver(create_engine().device)
    solver.initialize(world)

    bed_row = np.asarray(world.terrain.heights, dtype=np.float64)[N // 2]
    bed_grid = np.asarray(world.terrain.heights, dtype=np.float64)

    # The sea exists BEFORE the wave: still water everywhere the bed is below
    # datum 0. v1 got this part right and it is kept unchanged.
    still = np.maximum(0.0 - bed_grid, 0.0)
    solver._h.assign(still.astype(np.float32).ravel())
    solver._u.assign(np.zeros(N * N, dtype=np.float32))
    solver._v.assign(np.zeros(N * N, dtype=np.float32))
    solver._source_enabled = False      # the west edge is land here anyway
    solver._measure()
    solver._volume_at_start = float(solver.diagnostics()["volume_m3"])

    i_shore = int(np.argmin(np.abs(XS - shore_x)))
    slope = abs((bed_row[i_shore + 1] - bed_row[i_shore - 1]) / (2.0 * CELL))
    band = np.arange(int(land_edge_x + (N - 1) * 0.5 * CELL), N)
    j = N // 2

    retreat = runup = 0.0
    t_retreat = t_runup = None
    vmax = 0.0
    trace = []
    steps = int(seconds / DT)
    for step in range(steps):
        t = step * DT
        level = float(edge_level(t, amplitude, period))
        # what the kernel will do: hold the east columns at the outside level
        h = np.asarray(solver._h.numpy()).reshape(N, N)
        target = np.maximum(level - bed_grid[:, N - EDGE_COLUMNS:], 0.0)
        h[:, N - EDGE_COLUMNS:] = target
        solver._h.assign(h.astype(np.float32).ravel())

        solver.advance(DT, config.FLUID_MAX_SUBSTEPS, config.FLUID_STABILITY_DT)
        vmax = max(vmax, float(solver.diagnostics()["max_velocity"]))

        if step % 15 != 14:
            continue
        hr = np.asarray(solver._h.numpy()).reshape(N, N)[j]
        wet = hr > DRY
        wb = wet[band]
        if wb.any():
            waterline = float(XS[band[int(np.argmax(wb))]])
            if waterline > shore_x and waterline - shore_x > retreat:
                retreat, t_retreat = waterline - shore_x, round(t, 1)
            if waterline < shore_x and shore_x - waterline > runup:
                runup, t_runup = shore_x - waterline, round(t, 1)
            trace.append((round(t, 1), round(waterline, 1), round(level, 2)))

    if verbose:
        print("    t(s)  waterline_x(m)  outside_level(m)")
        for row in trace[::max(1, len(trace) // 12)]:
            print("  %6.1f  %13.1f  %15.2f" % row)
    return dict(shore_x=shore_x, slope=slope, retreat=retreat, runup=runup,
                t_retreat=t_retreat, t_runup=t_runup, vmax=vmax,
                wavelength_over_depth=(4.0 * period * np.sqrt(9.81 * ocean_depth)
                                       / ocean_depth))


def main():
    print("grid %dx%d, dx=%.2f m, dt=%.4f s, driving the east %d columns"
          % (N, N, CELL, DT, EDGE_COLUMNS))
    print("")
    print("  depth  land_x  beach   amp  period  shore_x   slope   lam/H"
          "   RETREAT   FLOOD   max|u|")
    best = None
    for ocean_depth, land_edge_x, beach_run in ((5.0, -90.0, 165.0),
                                                (8.0, -80.0, 130.0)):
        for amplitude in (2.0, 3.0, 4.0):
            for period in (6.0, 12.0, 20.0):
                r = run(ocean_depth, land_edge_x, beach_run, amplitude, period)
                print("  %5.1f  %6.1f  %5.1f  %4.1f  %6.1f  %7.1f  1:%-5.1f  %5.0f"
                      "   %6.1f  %6.1f   %6.2f"
                      % (ocean_depth, land_edge_x, beach_run, amplitude, period,
                         r["shore_x"], 1.0 / max(r["slope"], 1e-9),
                         r["wavelength_over_depth"], r["retreat"], r["runup"],
                         r["vmax"]))
                score = min(r["retreat"], r["runup"])
                if r["vmax"] < config.FLUID_MAX_VELOCITY * 0.9 and (
                        best is None or score > best[0]):
                    best = (score, ocean_depth, land_edge_x, beach_run,
                            amplitude, period)
    print("")
    if best:
        print("best (largest smaller-of-retreat-and-flood, velocity clamp not "
              "reached): depth=%.1f land_x=%.1f beach=%.1f amp=%.1f period=%.1f"
              % best[1:])
        print("")
        r = run(best[1], best[2], best[3], best[4], best[5], verbose=True)
        print("  shore_x=%.1f  slope 1:%.1f  RETREAT %.1f m at t=%ss  "
              "FLOOD %.1f m inland at t=%ss  max|u|=%.2f m/s"
              % (r["shore_x"], 1.0 / r["slope"], r["retreat"], r["t_retreat"],
                 r["runup"], r["t_runup"], r["vmax"]))


if __name__ == "__main__":
    main()
