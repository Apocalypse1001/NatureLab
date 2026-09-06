"""Diagnostic probe: how far does lava run before it solidifies?

Blocking measurement for v0.13.0, specified in docs/08_volcano_plan.md, item 1.
Before any lava kernel is written, the plan asks a narrower question first:
on this map (200 m, dx = 1 m), does a physically plausible (eruption
temperature, cooling rate) pair even put the solidification front in a
range a player can see -- roughly 40-150 m from the vent? Two failure modes
kill the scene before a single shader is written: the flow freezes at the
vent (a bump, not a stream), or it never freezes on the map at all (no
"cooled -> slowed -> stopped" story to see).

Read-only: imports the real solver from backend/app, changes nothing on disk.
Run from the repo root:   python docs/probe_lava_v013.py

What is measured, and in what form. There is no temperature field or
temperature-dependent friction in the solver yet (that is the kernel work
this probe exists to justify or head off), so this probe does not modify
fluid_solver.py. Instead it:

  - drives the REAL Warp velocity/depth solver on an inclined bed with a
    point vent (the placeable level-held SOURCE from v0.8.0), so the flow
    speed the lava would ride on is the real answer to "water on this
    slope with this friction", not a guess;
  - carries a temperature field on top, advected once per frame by the
    same semi-Lagrangian kernel the sediment system already uses
    (`_advect_sediment` is generic in what scalar it moves -- this is
    exactly the "same kernel, different field" reuse the plan calls for
    in item 2), called directly rather than folded into the compiled
    solver loop, since this is a probe and not the kernel itself;
  - cools that field with the CLOSED-FORM solution of pure T^4 radiative
    loss from a well-mixed column of thickness h:

        dT/dt = -(k0/h) * T^4,   k0 = emissivity * sigma / (rho * cp)
        T(t)  = T0 * (1 + 3*k0*T0^3*t/h) ** (-1/3)

    which is unconditionally stable and can only ever cool, matching the
    plan's trap #2 ("cooling cannot be subtracted, only divided"). h in the
    denominator is exactly the causal claim in item 3: a thin sheet cools
    faster than a thick one because the same radiating surface serves less
    mass underneath it. Ambient T^4 is dropped from the balance (it is
    ~0.2% of erupting T^4 and this is a feasibility scan, not the
    calibrated law that ships).

A bulk (single-T-through-depth) model cannot represent a real lava crust,
which insulates the hot interior and cools far slower than a well-mixed
column of the same thickness would. That gap is exactly why "cooling
coefficient" is swept as a multiplier on the pure-radiation k0 rather than
taken as fixed physics: the multiplier stands in for whatever combination
of crust insulation, ground conduction and convective loss the calibrated
model will need. Finding which multiplier lands the front in 40-150 m is
the deliverable; that is a starting point for LAVA_EMISSIVITY /
LAVA_GROUND_CONDUCTION, not their final value.

The vent keeps refilling both fields it creates -- level AND temperature --
inside its disc every frame, which is the discipline item "trap 1" in the
plan names by number: a source that writes h and not T reproduces the
0.3989 bug from v0.11.0 exactly, a vent that looks like it is erupting while
the lava itself never gets hot.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app import config                              # noqa: E402
from app import fluid_solver as fs                  # noqa: E402
from app.compute_engine import create_engine        # noqa: E402
from app.world_state import WorldState              # noqa: E402

N = config.TERRAIN_CELLS + 1
CELL = config.TERRAIN_CELL_SIZE
DT = config.FIXED_DT
DRY = config.FLUID_DRY_DEPTH

# ------------------------------------------------------------- lava physics
# Basalt-order values, not tuned to anything: this is the "what is even
# plausible" pass the plan's item 1 asks for before a config knob exists.
SIGMA = 5.670374419e-8       # Stefan-Boltzmann, W/(m^2 K^4)
RHO_LAVA = 2700.0            # kg/m^3
CP_LAVA = 1200.0             # J/(kg K)
EMISSIVITY = 0.9
T_ERUPT_C = 1150.0
T_SOLIDUS_C = 980.0
T_AMBIENT_C = 20.0
K0 = EMISSIVITY * SIGMA / (RHO_LAVA * CP_LAVA)   # pure-radiation coefficient

VENT_I = 4                 # a few cells in from the west edge, off the boundary strip
VENT_RADIUS_M = 3.0
VENT_HEAD_M = 1.0           # depth the vent holds -- what drives the flow downhill


def c2k(celsius):
    return celsius + 273.15


def _vent_geometry():
    centre_j = N // 2
    vent_x = (VENT_I - (N - 1) * 0.5) * CELL
    vent_z = (centre_j - (N - 1) * 0.5) * CELL
    return centre_j, vent_x, vent_z


def _vent_mask():
    centre_j, vent_x, vent_z = _vent_geometry()
    ii, jj = np.meshgrid(np.arange(N), np.arange(N))
    x = (ii - (N - 1) * 0.5) * CELL
    z = (jj - (N - 1) * 0.5) * CELL
    return (x - vent_x) ** 2 + (z - vent_z) ** 2 <= VENT_RADIUS_M ** 2


def build(slope):
    """A single-direction flank: high at the vent (i=0), falling toward i=N-1.

    Planar, not conical -- the same simplification the erosion probe used
    before terrain_gen grew a real channel (docs/probe_erosion_v012.py). The
    per-length physics of a flow riding down a slope does not depend on
    whether the slope curves into a cone a few hundred meters further out.
    """
    world = WorldState()
    fall = np.arange(N, dtype=np.float32) * CELL * slope
    bed = np.tile((fall[-1] - fall)[None, :], (N, 1)).astype(np.float32)
    world.terrain.heights[:, :] = bed
    solver = fs.create_fluid_solver(create_engine().device)
    solver.initialize(world)
    solver._source_enabled = False
    solver.set_boundaries(world.terrain, {}, 0, 0)
    solver.set_outflow(config.FLUID_OUTFLOW_COLUMNS)
    solver.set_erosion(False)
    centre_j, vent_x, vent_z = _vent_geometry()
    vent_level = float(bed[centre_j, VENT_I] + VENT_HEAD_M)
    solver.set_water_features(
        sources=[(np.array([vent_x, 0.0, vent_z], dtype=np.float32),
                  VENT_RADIUS_M, vent_level)],
        drains=[])
    return world, solver


def run(label, slope, manning_n, cooling_multiplier,
        max_seconds=300.0, sample_every=5.0, settle_window=10, settle_tol=1.0):
    """Run one (slope, friction, cooling) point and report the solidification front.

    `manning_n` overrides config.FLUID_MANNING_N for the run's duration and is
    restored after -- the solver reads the module constant directly inside
    `advance()`, and this is a probe process, so the swap is local and undone,
    never a file edit. Stops early once the hot front has not moved by more
    than `settle_tol` metres over the last `settle_window` samples: this is a
    relaxation front (fed continuously, cooled continuously), so it converges
    rather than oscillating once the vent's supply and the flow's cooling
    reach balance, and waiting past that point only spends GPU time.
    """
    orig_n = config.FLUID_MANNING_N
    config.FLUID_MANNING_N = manning_n
    try:
        world, solver = build(slope)
        mask = _vent_mask().ravel()
        centre_j, _, _ = _vent_geometry()
        t_erupt_k = c2k(T_ERUPT_C)
        t_solidus_k = c2k(T_SOLIDUS_C)
        T = fs.wp.array(np.full(solver._count, c2k(T_AMBIENT_C), dtype=np.float32),
                        dtype=float, device=solver.device)
        next_T = fs.wp.empty(solver._count, dtype=float, device=solver.device)

        steps = int(max_seconds / DT)
        sample_steps = max(1, int(sample_every / DT))
        trace = []
        hot_history = []
        for step in range(steps):
            solver.advance(DT, config.FLUID_MAX_SUBSTEPS, config.FLUID_STABILITY_DT)
            fs.wp.launch(fs._advect_sediment, dim=solver._count,
                        inputs=[T, next_T, solver._u, solver._v, solver._h,
                                solver._obstacles, solver._width, solver._height,
                                CELL, DT, DRY], device=solver.device)
            T, next_T = next_T, T
            t_np = np.asarray(T.numpy())
            h_np = np.asarray(solver._h.numpy())
            t_np[mask] = t_erupt_k               # trap 1: refill T, not just h
            wet = h_np > DRY
            hh = np.maximum(h_np, 0.02)
            growth = 1.0 + 3.0 * K0 * cooling_multiplier * (t_np ** 3) * DT / hh
            t_np[wet] = t_np[wet] / np.cbrt(growth[wet])
            T.assign(t_np.astype(np.float32))

            if step % sample_steps == sample_steps - 1:
                row_h = h_np.reshape(N, N)[centre_j]
                row_t = t_np.reshape(N, N)[centre_j]
                wetf = row_h > DRY
                hot = wetf & (row_t >= t_solidus_k)
                hot_front = VENT_I
                for i in range(VENT_I, N):
                    if not hot[i]:
                        break
                    hot_front = i
                wet_front = VENT_I
                for i in range(VENT_I, N):
                    if not wetf[i]:
                        break
                    wet_front = i
                speed = np.hypot(np.asarray(solver._u.numpy()),
                                  np.asarray(solver._v.numpy())).reshape(N, N)[centre_j]
                trace.append((solver._time, (hot_front - VENT_I) * CELL,
                              (wet_front - VENT_I) * CELL,
                              float(speed[wetf].max()) if wetf.any() else 0.0))
                hot_history.append(trace[-1][1])
                if (len(hot_history) >= settle_window
                        and max(hot_history[-settle_window:])
                        - min(hot_history[-settle_window:]) < settle_tol):
                    break

        diag = solver.diagnostics()
        final_h = np.asarray(solver._h.numpy()).reshape(N, N)[centre_j]
        final_t = np.asarray(T.numpy()).reshape(N, N)[centre_j]
        final_u = np.hypot(np.asarray(solver._u.numpy()),
                            np.asarray(solver._v.numpy())).reshape(N, N)[centre_j]

        print("")
        print("=== %s ===" % label)
        print("  slope=%.1f%%  manning_n=%.3f  cooling_x=%g  T_erupt=%.0fC  T_solidus=%.0fC"
              % (slope * 100, manning_n, cooling_multiplier, T_ERUPT_C, T_SOLIDUS_C))
        print("  ran t=%.1f s (of %.1f s cap)  substeps=%d  cfl_limited=%s  max|u|=%.2f m/s"
              % (solver._time, max_seconds, diag["substeps"], diag["cfl_limited"],
                 diag["max_velocity"]))
        print("   t(s)   hot_front(m)  wet_front(m)  max|u| on row (m/s)")
        for t, hf, wf, spd in trace[-8:]:
            print("  %6.1f  %11.1f  %11.1f  %10.2f" % (t, hf, wf, spd))
        # profile at the finish, every 10th cell out to the wet front
        wf_i = VENT_I + int(round(trace[-1][2] / CELL)) if trace else VENT_I
        print("  profile at finish (x from vent, m | T C | |u| m/s | h m):")
        for i in range(VENT_I, min(N, wf_i + 6), max(1, (wf_i - VENT_I) // 12 or 1)):
            print("    x=%6.1f  T=%7.1f  |u|=%5.2f  h=%5.3f"
                  % ((i - VENT_I) * CELL, final_t[i] - 273.15, final_u[i], final_h[i]))
        solver.reset()
        result_L = trace[-1][1] if trace else 0.0
        return result_L, trace
    finally:
        config.FLUID_MANNING_N = orig_n


if __name__ == "__main__":
    print("K0 (pure blackbody, rho=%.0f cp=%.0f eps=%.2f) = %.3e m/(s K^3)"
          % (RHO_LAVA, CP_LAVA, EMISSIVITY, K0))
    print("vent: i=%d (x=0 m), radius=%.1f m, head=%.1f m above local bed"
          % (VENT_I, VENT_RADIUS_M, VENT_HEAD_M))

    results = []
    for slope in (0.10, 0.20):
        for manning_n in (0.03,):
            for mult in (1.0, 30.0, 300.0, 3000.0, 30000.0):
                label = "slope=%.0f%% n=%.2f cooling x%g" % (slope * 100, manning_n, mult)
                L, _ = run(label, slope, manning_n, mult)
                results.append((slope, manning_n, mult, L))

    print("")
    print("=== summary: hot-front distance L (m) by (slope, manning_n, cooling_x) ===")
    for slope, manning_n, mult, L in results:
        flag = " <-- in 40-150 m target" if 40.0 <= L <= 150.0 else ""
        print("  slope=%5.1f%%  n=%.2f  cooling_x=%8g  L=%7.1f m%s"
              % (slope * 100, manning_n, mult, L, flag))
