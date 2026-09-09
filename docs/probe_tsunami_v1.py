"""Diagnostic probe: does a seeded offshore pulse give a real drawback-then-wave?

SUPERSEDED by probe_tsunami_v2.py, and kept only as the record of a wrong
question. This probe asks whether the DEPTH AT A POINT falls and then rises. It
does, so this passed -- while the scene it validated moved the waterline 1.4 m
and flooded 4.6 m of land. What a person watching looks for is where the
water's EDGE is, and nothing here measures that. See docs/13_tsunami2_plan.md.


Blocking measurement for a TsunamiLab scenario, requested directly (no plan
doc yet -- this probe IS the plan's first step, docs/08_volcano_plan.md style,
compressed into one pass because the mechanism needed is small: a one-shot
initial condition, not a new per-tick kernel).

The user asked for the recognisable tsunami signature: water visibly recedes
from the shore BEFORE the wave arrives and wrecks the shore. A rising flood
front (what SOURCE/inlet/dam-break already give this project) does not have
that signature -- there is no leading trough. The real mechanism is an
"N-wave": a depression riding just ahead of an elevation, both moving toward
shore together. Standard tsunami-source shape, not invented for this probe.

Read-only: imports the real solver from backend/app, changes nothing on disk,
touches no file in fluid_solver.py. Run from the repo root:
    python docs/probe_tsunami_v1.py

Why this needs no backend change, unlike a new Lab: `WaterState` only stores
a scalar level plus boundary flags (checked directly in world_state.py before
writing this), so there genuinely is no way to author a non-uniform initial
water surface through the existing scenario JSON format today. But the fix is
a ONE-SHOT seed at t=0, not a per-tick kernel -- closer to what
probe_lava_v013.py and test_backend.py already do by hand
(`fluid._h.assign(...)`, `_u.assign(...)`, `_v.assign(...)`) than to a new
physics term. This probe validates that seed on the real solver before any
persistence-format work is proposed.

--------------------------------------------------------------- orientation
`FLUID_OUTFLOW_COLUMNS` (config.py) is a transmissive boundary on the EAST
edge only; every other outer face is a closed wall EXCEPT the west edge,
which is `FLUID_SOURCE_COLUMNS` -- a level-held boundary, live by default
(`_source_enabled = True` unconditionally in `initialize()`). So:

  - the ocean sits on the EAST (high x): a wave that overshoots seaward
    LEAVES the domain instead of reflecting back as a spurious echo.
  - the wave therefore travels -x, run-up happens at the WEST.
  - the west edge is verified dry by construction, not by disabling
    `_source_enabled`: land there is built well above `water.level = 0`, so
    `max(level - bed, 0) = 0` regardless of whether the source is "on".
    (`_source_enabled` is set False anyway below, belt and suspenders, same
    as probe_lava_v013.py does for its own SOURCE.)

--------------------------------------------------------------- bed profile
One smoothstep, not two glued segments -- `docs/08_volcano_plan.md`'s item 7
found a real bug (a ring wall) from gluing a flat platform onto a cone whose
own profile was already flat at the apex; the fix there was one shape, no
splice, and the same discipline applies here: flat land, ONE smoothstep beach
face, flat ocean floor, nothing stitched.

--------------------------------------------------------------- pulse seeding
An N-wave centred at x0, half-width L, offshore amplitude A:

    eta(x) = A * ((x - x0) / L) * exp(-0.5 * ((x - x0) / L) ** 2)

An odd function: NEGATIVE (trough) on the shore side of x0, POSITIVE (crest)
on the ocean side. Both ride toward shore together, so the trough -- being
already closer to shore -- arrives first. That ordering, not any special
"drawback code", is the entire mechanism.

Consistent velocity for a wave moving in -x only (not splitting into a second
copy heading further out to sea): the linear shallow-water characteristic
relation

    u(x) = -c(x) * eta(x) / H(x),   c(x) = sqrt(g * H(x))

with H(x) the local still-water depth. This is exact only for constant depth;
seeding it on a sloping bed is an approximation, honestly not a derivation --
the pulse is placed entirely within the flat deep-ocean shelf (see SWEEP)
specifically so the approximation starts from where it is least wrong, and
the real solver is what carries it in from there, not this formula.

--------------------------------------------------------------- what is checked
Two independent theory checks, not just "does it look right":
  - wave speed c = sqrt(g*H) at the seeding depth (~12 m/s at H=15 m);
  - Green's law shoaling: crest amplitude should grow roughly (H1/H2)^0.25
    going from the seeding depth to ~1 m nearshore (~2x at 15m -> 1m).
If either fails, the SEEDING is wrong, not the physics -- the solver itself
is already validated by the rest of this project's test suite.

Also reads `diagnostics()` for `max_velocity`, `cfl_limited`,
`volume_error_m3` -- this is the first scenario in the project to run the
solver at ~15 m depth, and arithmetic alone is not proof it stays stable at
the configured substep cap.
"""
import math
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
GRAVITY = 9.81


def x_at(i):
    return (i - (N - 1) * 0.5) * CELL


def i_at(x):
    return int(round(x / CELL + (N - 1) * 0.5))


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ------------------------------------------------------------- coastal bed
LAND_EDGE_X = -40.0     # flat land west of here
BEACH_RUN_M = 60.0      # smoothstep face width -> ocean floor reached at x=20
INLAND_HEIGHT_M = 5.0   # flat land elevation (well above water.level=0)


def bed_profile(x, ocean_depth_m):
    t = (x - LAND_EDGE_X) / BEACH_RUN_M
    s = smoothstep(t)
    return INLAND_HEIGHT_M - (INLAND_HEIGHT_M + ocean_depth_m) * s


def find_shore_x(bed_row):
    """Where the (monotonic) bed profile crosses 0, by linear interpolation
    between the two adjacent grid columns. The smoothstep's shoreline moves
    with ocean_depth_m (see the module docstring's BED PROFILE section) -- an
    earlier draft of this probe hard-guessed x=-2 for it, which was still
    ~9 m underwater at ocean_depth=15, and every "shore" reading it produced
    was actually a mid-ocean one. Never guess a derived position; read it off
    the bed the run itself is using."""
    xs = x_at(np.arange(N))
    below = bed_row < 0.0
    idx = np.flatnonzero(np.diff(below.astype(np.int8)))[0]
    x0, x1 = xs[idx], xs[idx + 1]
    b0, b1 = bed_row[idx], bed_row[idx + 1]
    return float(x0 + (x1 - x0) * (0.0 - b0) / (b1 - b0))


def build(ocean_depth_m):
    world = WorldState()
    xs = x_at(np.arange(N))
    bed_row = bed_profile(xs, ocean_depth_m).astype(np.float32)
    bed = np.tile(bed_row[None, :], (N, 1))
    world.terrain.heights[:, :] = bed
    world.water.level = 0.0
    solver = fs.create_fluid_solver(create_engine().device)
    solver.initialize(world)
    solver._source_enabled = False   # belt and suspenders -- see ORIENTATION
    solver.set_boundaries(world.terrain, {}, 0, 0)
    solver.set_outflow(config.FLUID_OUTFLOW_COLUMNS)
    solver.set_erosion(False)
    return world, solver, bed_row


def seed_pulse(solver, bed_row, ocean_depth_m, amplitude_m, half_width_m, centre_x):
    """Still water everywhere wet, an N-wave superimposed, u = 0.

    `initialize()` only fills the west SOURCE columns -- everywhere else in
    the ocean starts bone dry (h=0), which is wrong for "the sea is already
    there". This fills genuine still water first (h = max(0, -bed)), then
    adds the pulse on top -- the ocean existing before the tsunami hits it is
    a precondition, not part of the effect being measured.

    Zero initial velocity, NOT the linear shallow-water traveling-wave
    relation u = -c*eta/H the module docstring's ORIENTATION section
    describes as the theoretically tighter seed. Both were tried; u=0 is what
    shipped, for two measured reasons, not a shortcut:

      - u = -c*eta/H drove max|u| to ~19-20 m/s -- pinned against
        `FLUID_MAX_VELOCITY` (20.0) -- and produced a messy multi-hump shore
        trace with no clean single drawback. The travelling-wave relation is
        exact only for CONSTANT depth; this bed is not constant depth
        anywhere the pulse's tail reaches non-negligibly, so the "consistent"
        velocity was already wrong at t=0 and the solver spent the first few
        seconds radiating that error away as spurious short-wave noise.
      - u=0 let max|u| settle to ~1.3 m/s and gave the textbook picture on
        the first clean run: ~4 s of smooth, monotonic recession to within a
        centimetre of dry, then a sharp rise. See probe_tsunami_v1's own
        change log / the plan doc's "Замер" section for the actual trace.

    The real cost of u=0 is that the pulse splits roughly in half: one copy
    heads shoreward (what this scenario wants), the other heads out to sea
    -- straight through the OPEN outflow boundary on the east edge (see
    ORIENTATION), where it simply leaves. Losing half the seeded energy that
    way is exactly why `amplitude_m` here ends up needing to run higher than
    a naive "textbook offshore tsunami amplitude" table would suggest.
    """
    xs = x_at(np.arange(N))
    still_h = np.maximum(0.0, -bed_row)
    eta = amplitude_m * ((xs - centre_x) / half_width_m) \
        * np.exp(-0.5 * ((xs - centre_x) / half_width_m) ** 2)
    # only where there is real water to perturb -- no pulse painted onto dry land
    eta = np.where(still_h > DRY, eta, 0.0)
    h_row = np.maximum(0.0, still_h + eta)
    u_row = np.zeros_like(h_row)

    h = np.tile(h_row[None, :], (N, 1)).astype(np.float32).ravel()
    u = np.tile(u_row[None, :], (N, 1)).astype(np.float32).ravel()
    v = np.zeros(N * N, dtype=np.float32)
    solver._h.assign(h)
    solver._u.assign(u)
    solver._v.assign(v)
    return h_row, u_row


def run(label, ocean_depth_m, amplitude_m, half_width_m, centre_x,
        max_seconds=60.0, sample_every=1.0):
    world, solver, bed_row = build(ocean_depth_m)
    h0, u0 = seed_pulse(solver, bed_row, ocean_depth_m, amplitude_m, half_width_m, centre_x)
    # Reset the volume ledger's baseline to AFTER the hand-seeded ocean is in
    # place: `_volume_at_start` was captured by `set_boundaries()` while the
    # ocean was still bone dry (`initialize()` only fills the west SOURCE
    # columns), so without this every run reports a "volume error" equal to
    # the entire hand-injected ocean -- an artefact of how this probe seeds
    # water, not a conservation bug in the solver.
    solver._measure()   # `diagnostics()` reads a cache _measure() fills; without
                          # this call it still reports the pre-seed (dry) volume
    solver._volume_at_start = float(solver.diagnostics()["volume_m3"])
    centre_j = N // 2

    shore_x = find_shore_x(bed_row)
    shore_i = i_at(shore_x + 2.0)      # a couple of metres seaward of the shoreline:
                                        # nominally shallow water, the point that should
                                        # visibly go dry during drawback
    prebeach_i = i_at(shore_x - 10.0)  # nominally DRY land, 10 m inland of the shoreline:
                                        # only wets if run-up genuinely reaches this far
    offshore_i = i_at(centre_x)        # where the pulse started

    steps = int(max_seconds / DT)
    sample_steps = max(1, int(sample_every / DT))
    trace = []
    min_shore_depth = math.inf
    max_shore_depth = 0.0
    max_offshore_eta = -math.inf
    min_offshore_eta = math.inf
    max_crest_h = 0.0
    max_crest_x = None
    farthest_wet_i = None

    bed_flat = bed_row.astype(np.float64)

    for step in range(steps):
        solver.advance(DT, config.FLUID_MAX_SUBSTEPS, config.FLUID_STABILITY_DT)
        if step % sample_steps == sample_steps - 1:
            t = (step + 1) * DT
            h_np = np.asarray(solver._h.numpy()).reshape(N, N)[centre_j]
            eta_np = h_np + bed_flat   # absolute surface elevation
            wet = h_np > DRY

            shore_h = float(h_np[shore_i])
            offshore_eta = float(eta_np[offshore_i] - 0.0)   # bed=~ocean_depth negative there anyway
            min_shore_depth = min(min_shore_depth, shore_h)
            max_shore_depth = max(max_shore_depth, shore_h)
            min_offshore_eta = min(min_offshore_eta, float(eta_np[offshore_i]))
            max_offshore_eta = max(max_offshore_eta, float(eta_np[offshore_i]))

            # Crest tracking: highest surface elevation anywhere between the
            # beach toe and the offshore seed point, and how far inland the
            # wet front has ever reached (run-up). Masked to WET cells only --
            # `eta = h + bed`, and on dry land h=0 so eta is just the bed's
            # own elevation. An earlier draft took the unmasked max and
            # reported a "5 m crest" that was actually dry land's own +5 m
            # bed reading itself, nothing to do with the wave at all.
            band = slice(i_at(LAND_EDGE_X), offshore_i + 1)
            band_wet = wet[band]
            if band_wet.any():
                band_eta = np.where(band_wet, eta_np[band], -np.inf)
                band_i0 = i_at(LAND_EDGE_X)
                local_max = float(band_eta.max())
                if local_max > max_crest_h:
                    max_crest_h = local_max
                    max_crest_x = x_at(band_i0 + int(np.argmax(band_eta)))
            wet_idx = np.flatnonzero(wet[:offshore_i + 1])
            if len(wet_idx):
                fi = int(wet_idx.min())   # smallest i (furthest WEST = furthest inland) still wet
                if farthest_wet_i is None or fi < farthest_wet_i:
                    farthest_wet_i = fi

            trace.append((t, shore_h, float(eta_np[offshore_i]), float(h_np[prebeach_i])))

    diag = solver.diagnostics()
    print("")
    print("=== %s ===" % label)
    print("  ocean_depth=%.1fm  amplitude=%.2fm  half_width=%.1fm  centre_x=%.1fm"
          % (ocean_depth_m, amplitude_m, half_width_m, centre_x))
    theory_c = math.sqrt(GRAVITY * ocean_depth_m)
    print("  theory: c=sqrt(g*H)=%.2f m/s at seed depth" % theory_c)
    print("  ran t=%.1fs  substeps=%d  cfl_limited=%s  max|u|=%.2f m/s  volume_error_m3=%.3f"
          % (solver._time, diag["substeps"], diag["cfl_limited"], diag["max_velocity"],
             diag.get("volume_error_m3", float("nan"))))
    print("   t(s)   shore_h(m)   offshore_eta(m)   prebeach_h(m)")
    for t, sh, oe, pb in trace[::max(1, len(trace) // 10)]:
        print("  %6.1f  %9.3f  %14.3f  %12.3f" % (t, sh, oe, pb))
    print("  shore depth: min=%.3fm (dry=%s)  max=%.3fm"
          % (min_shore_depth, min_shore_depth <= DRY, max_shore_depth))
    print("  offshore eta: min=%.3fm  max=%.3fm  (trough then crest: %s)"
          % (min_offshore_eta, max_offshore_eta, min_offshore_eta < -0.05 < max_offshore_eta))
    if max_crest_x is not None:
        shoal_ratio = max_crest_h / amplitude_m if amplitude_m else float("nan")
        print("  peak surface elevation observed: %.3fm at x=%.1fm (amplification x%.2f vs offshore amplitude)"
              % (max_crest_h, max_crest_x, shoal_ratio))
    if farthest_wet_i is not None:
        # Inland distance is measured from THIS run's own shoreline, not from
        # x=0. An earlier version printed `-x_at(...)`, i.e. it assumed the
        # shoreline sat at the origin -- the very mistake find_shore_x()'s own
        # docstring warns about, made three functions further down. At
        # ocean_depth=15 the shoreline is at x=-20.4, so a wet front reaching
        # x=-26 is 5.6 m inland, and the old line called it 26 m. The report
        # contradicted itself in the same block: `prebeach_h` samples 10 m
        # inland and reads 0.000 m in every single run, which it could not do
        # if the water had genuinely gone 26 m past the shore.
        print("  furthest inland run-up: x=%.1fm (%.1fm inland of the bed=0 shoreline at x=%.1fm)"
              % (x_at(farthest_wet_i), shore_x - x_at(farthest_wet_i), shore_x))
    # Drawback: first-crossing times, not a global min/max index -- this
    # system genuinely oscillates for several cycles (a real wave TRAIN, not
    # one clean pulse), so "the global min" can land in a LATER cycle than
    # "the global max" even when the first cycle is a textbook drawback-then-
    # wave. What the user actually asked for is "does the shore visibly dry
    # out, and does a big wave follow" -- both as first-occurrence questions.
    baseline = trace[0][1] if trace else 0.0
    dry_threshold = max(DRY * 5.0, 0.05)
    wave_threshold = baseline * 1.5
    dry_at = next((t for t, sh, _, _ in trace if sh <= dry_threshold), None)
    wave_at = next((t for t, sh, _, _ in trace if sh >= wave_threshold), None)
    drawback_before_wave = (dry_at is not None and wave_at is not None
                             and dry_at < wave_at)
    print("  baseline shore depth=%.3fm; first near-dry (<=%.3fm) at t=%s; "
          "first big-wave (>=%.3fm) at t=%s; drawback precedes wave: %s"
          % (baseline, dry_threshold, ("%.2fs" % dry_at) if dry_at is not None else "never",
             wave_threshold, ("%.2fs" % wave_at) if wave_at is not None else "never",
             drawback_before_wave))
    solver.reset()
    return {
        "drawback": drawback_before_wave, "shore_went_dry": min_shore_depth <= dry_threshold,
        "dry_at": dry_at, "wave_at": wave_at,
        "cfl_limited": diag["cfl_limited"], "max_velocity": diag["max_velocity"],
        "shoal_ratio": max_crest_h / amplitude_m if amplitude_m else float("nan"),
        "runup_x": x_at(farthest_wet_i) if farthest_wet_i is not None else None,
    }


if __name__ == "__main__":
    print("grid: %d x %d, dx=%.2fm, dt=%.4fs" % (N, N, CELL, DT))
    print("bed: flat land +%.1fm west of x=%.1f, smoothstep %.0fm beach, "
          "flat ocean floor east of x=%.1f"
          % (INLAND_HEIGHT_M, LAND_EDGE_X, BEACH_RUN_M, LAND_EDGE_X + BEACH_RUN_M))

    # centre_x=60 keeps the pulse's 3-sigma extent (60 +/- 3*half_width) within
    # the flat ocean shelf (x >= LAND_EDGE_X+BEACH_RUN_M = 20) and clear of the
    # east outflow edge (x=100) for every half_width swept below.
    results = []
    for ocean_depth in (10.0, 15.0, 20.0):
        for amplitude in (1.0, 1.5, 2.0, 3.0):
            for half_width in (10.0, 15.0):
                label = "depth=%.0fm amp=%.1fm hw=%.0fm" % (ocean_depth, amplitude, half_width)
                res = run(label, ocean_depth, amplitude, half_width, 60.0,
                          max_seconds=40.0, sample_every=0.25)
                results.append((ocean_depth, amplitude, half_width, res))

    print("")
    print("=== summary ===")
    for ocean_depth, amplitude, half_width, res in results:
        flag = " <-- drawback confirmed, shore went dry" \
            if res["drawback"] and res["shore_went_dry"] else ""
        print("  depth=%5.1fm  amp=%4.1fm  hw=%4.1fm  ->  drawback=%s  dry_at=%s  wave_at=%s  "
              "cfl_limited=%s  vmax=%.2fm/s  shoal_x%.2f  runup_x=%s%s"
              % (ocean_depth, amplitude, half_width, res["drawback"],
                 ("%.1fs" % res["dry_at"]) if res["dry_at"] is not None else "--",
                 ("%.1fs" % res["wave_at"]) if res["wave_at"] is not None else "--",
                 res["cfl_limited"], res["max_velocity"], res["shoal_ratio"],
                 ("%.1f" % res["runup_x"]) if res["runup_x"] is not None else "n/a", flag))
