"""Diagnostic probe: does a swale's moisture plume reach a tree in game time?

Blocking measurement for PermacultureLab-1, specified in
docs/11_permaculture_plan.md, "С чего начать: замер, не код". Before any
soil_moisture kernel is written into fluid_solver.py, the plan asks a
narrower question first: for a capacity-limited bucket model (infiltration
rate capped by K_SAT, lateral spread as explicit diffusion at K_LATERAL), is
there a pair of values that (a) infiltrates a swale's held water at a rate
visible over tens of seconds rather than instantly or glacially, (b) spreads
that moisture sideways far enough to reach a tree's root radius within a
session's worth of game time, and (c) stays numerically stable at this
project's dt? Three failure modes kill the scene before a single Warp kernel
is written: the plume never leaves the swale (a wet line, not a story), it
takes longer than anyone will sit and watch, or the explicit diffusion step
overshoots into a checkerboard artefact -- the same class of numerical
failure docs/07_river_plan.md already named for the solver's own 2dx comb.

Read-only: touches no file in backend/app. Run from the repo root:
    python docs/probe_permaculture_v1.py

Why this probe does NOT use the real Warp solver, unlike probe_lava_v013.py:
lava's flow speed depended on real shallow-water hydraulics (a slope, a
Froude-limited vent, real friction), so that probe had to drive the actual
velocity/depth kernels to get an honest answer. A swale's water column is a
held boundary condition, not a flow -- nothing here depends on u/v or the
pressure-gradient term, only on a capped vertical flux (infiltration) and an
explicit horizontal diffusion (lateral spread). Both are plain numpy over a
small grid; pulling in Warp/CUDA would not make either number more honest,
only slower to iterate on.

Model, spelled out (mirrors what the plan's kernel section describes):

    infiltration:  flux = min(K_SAT, (capacity - moisture) / dt)   [m/s]
                    moisture[swale] += flux * dt
    lateral spread: moisture += r * laplacian4(moisture),  r = K_LATERAL*dt/dx^2
    root uptake:    moisture[tree ring] -= min(moisture, ROOT_UPTAKE_RATE*dt)

The swale refills its own held water level every step -- the same "a source
must fill every field it creates" discipline the volcano plan named after
the vent-that-erupts-cold-lava bug (v0.11.0's 0.3989 datum). Here there is
only one field (moisture), but the discipline is the same: the swale cell's
capacity is topped up unconditionally each step, standing in for "an
upstream catchment keeps this ditch wet", which is the actual permaculture
claim being tested, not an infinite-reservoir shortcut.

K_SAT is swept across the textbook order-of-magnitude range for common soils
(clay ~1e-7, loam ~1e-6..1e-5, sand ~1e-4 m/s) -- grounded in soil science,
not guessed. K_LATERAL has no such table: unsaturated lateral conductivity is
itself moisture-dependent in reality (Richards equation territory), and this
bucket model replaces that with one constant standing in for whatever the
real soil does -- exactly the role LAVA_COOLING_ENHANCEMENT plays for crust
insulation in the volcano model. Finding which K_LATERAL lands the plume at
a tree in a plausible time is the deliverable; it is a starting point for
SOIL_K_LATERAL, not a measured soil property.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app import config                              # noqa: E402

DX = config.TERRAIN_CELL_SIZE      # 1.0 m on the real map
DT = config.FIXED_DT               # 1/60 s, the project's real physics step

# ------------------------------------------------------------- soil physics
# SOIL_DEPTH_M / SOIL_POROSITY match the plan's docs/11_permaculture_plan.md
# config.py starting numbers -- not re-derived here, just reused so the probe
# answers the question the plan actually poses.
SOIL_DEPTH_M = 0.6
SOIL_POROSITY = 0.4
CAPACITY = SOIL_DEPTH_M * SOIL_POROSITY            # m of water-equivalent per cell

ROOT_RADIUS_M = 1.5
ROOT_UPTAKE_RATE = 5.0e-6          # m/s, order-of-magnitude placeholder -- see plan's
                                    # TREE_ROOT_UPTAKE_RATE note; a real tree's peak
                                    # transpiration is on this order per unit root area
VISIBLE_FRACTION = 0.10            # "visibly moist" threshold, as a fraction of capacity
VISIBLE_THRESHOLD = CAPACITY * VISIBLE_FRACTION
# A second, much lower threshold. The bucket model has no wetting-front depth
# (see plan's "Честное ограничение" -- named there as a real, not incidental,
# loss): a real swale visibly darkens the surface within minutes while the
# full profile below takes hours to fill, and one scalar per cell cannot
# represent "surface wet, column not full" at all. FIRST_ARRIVAL stands in
# for "the surface has visibly darkened" against VISIBLE's "the column reads
# as genuinely moist" -- reporting both is how this probe surfaces that gap
# instead of picking one number and hiding it.
FIRST_ARRIVAL_FRACTION = 0.01
FIRST_ARRIVAL_THRESHOLD = CAPACITY * FIRST_ARRIVAL_FRACTION

# Grid: a strip running away from a swale, wide enough that the 2D 4-neighbour
# Laplacian (the real kernel's shape) has real neighbours on both sides of the
# measured row, not an edge artefact.
LENGTH_M = 20.0
WIDTH_M = 8.0
NX = int(LENGTH_M / DX) + 1
NY = int(WIDTH_M / DX) + 1
SWALE_WIDTH_CELLS = 2              # ~1.2-2 m ditch, matches plan's SWALE_WIDTH_M


def _laplacian4(field):
    """Explicit 4-neighbour Laplacian with reflective (no-flux) edges.

    No-flux at the domain edges, not periodic or absorbing: the probe's grid
    is a cut-out of a larger field, and an edge should neither leak moisture
    nor manufacture a sink the real map would not have.
    """
    up = np.roll(field, -1, axis=0)
    down = np.roll(field, 1, axis=0)
    left = np.roll(field, -1, axis=1)
    right = np.roll(field, 1, axis=1)
    up[-1, :] = field[-1, :]
    down[0, :] = field[0, :]
    left[:, -1] = field[:, -1]
    right[:, 0] = field[:, 0]
    return (up + down + left + right - 4.0 * field)


def run(label, k_sat, k_lateral, tree_distance_m, max_seconds=180.0,
        sample_every=2.0):
    r = k_lateral * DT / (DX * DX)
    stable_theory = r <= 0.25   # standard von Neumann bound for 2D explicit FTCS

    moisture = np.zeros((NY, NX), dtype=np.float64)
    swale_cols = slice(0, SWALE_WIDTH_CELLS)
    tree_col = min(NX - 1, int(round(tree_distance_m / DX)) + SWALE_WIDTH_CELLS)
    tree_row = NY // 2
    tree_row_lo = max(0, tree_row - int(round(ROOT_RADIUS_M / DX)))
    tree_row_hi = min(NY, tree_row + int(round(ROOT_RADIUS_M / DX)) + 1)
    tree_col_lo = max(0, tree_col - int(round(ROOT_RADIUS_M / DX)))
    tree_col_hi = min(NX, tree_col + int(round(ROOT_RADIUS_M / DX)) + 1)

    steps = int(max_seconds / DT)
    sample_steps = max(1, int(sample_every / DT))
    trace = []
    first_arrival_at = None
    reached_at = None
    blew_up = False

    for step in range(steps):
        # Infiltration: ONLY the swale cells have standing surface water (a
        # held boundary condition -- trap 1, "a source must fill every field
        # it creates", stood in for here as an unlimited catchment feeding
        # the ditch). Every other cell in this probe has no surface water of
        # its own and can gain moisture ONLY through lateral spread from
        # wetter neighbours below. An earlier draft applied the K_SAT flux
        # law to the WHOLE grid, which silently modelled rain falling
        # everywhere rather than a swale -- every cell filled itself from
        # nothing, which masked whether the diffusion term was doing any of
        # the work the plan actually asked this probe to measure.
        swale = moisture[:, swale_cols]
        flux = np.minimum(k_sat, (CAPACITY - swale) / DT)
        swale += np.maximum(flux, 0.0) * DT
        np.clip(swale, 0.0, CAPACITY, out=swale)

        # Lateral spread (trap 2: this is the term the stability bound below
        # is checked against, not asserted blind). Deliberately NOT clipped
        # back into [0, capacity] here -- an unstable explicit step overshoots
        # into a checkerboard that a clip would silently launder into
        # something that *looks* bounded and converged every sample, while
        # actually oscillating between clip calls. The first probe draft did
        # exactly that and reported "reached in 2 s" for r=0.83, nearly 3x
        # past the 0.25 stability bound -- caught only by checking the raw
        # field before any clip runs.
        moisture += r * _laplacian4(moisture)

        if not np.isfinite(moisture).all() or moisture.max() > CAPACITY * 1.01 \
                or moisture.min() < -CAPACITY * 0.01:
            blew_up = True
            break
        np.clip(moisture, 0.0, CAPACITY, out=moisture)

        # Root uptake: a sink, never past zero.
        ring = moisture[tree_row_lo:tree_row_hi, tree_col_lo:tree_col_hi]
        np.subtract(ring, np.minimum(ring, ROOT_UPTAKE_RATE * DT), out=ring)

        if step % sample_steps == sample_steps - 1:
            t = (step + 1) * DT
            tree_moisture = float(moisture[tree_row_lo:tree_row_hi,
                                            tree_col_lo:tree_col_hi].mean())
            trace.append((t, tree_moisture))
            if first_arrival_at is None and tree_moisture >= FIRST_ARRIVAL_THRESHOLD:
                first_arrival_at = t
            if reached_at is None and tree_moisture >= VISIBLE_THRESHOLD:
                reached_at = t

    print("")
    print("=== %s ===" % label)
    print("  K_SAT=%.1e m/s  K_LATERAL=%.1e m^2/s  r=K*dt/dx^2=%.2e (stable<=0.25: %s)"
          % (k_sat, k_lateral, r, stable_theory))
    print("  tree at %.1f m, root radius %.1f m, capacity=%.3f m, first-arrival@%.4f m, visible@%.3f m"
          % (tree_distance_m, ROOT_RADIUS_M, CAPACITY, FIRST_ARRIVAL_THRESHOLD, VISIBLE_THRESHOLD))
    if blew_up:
        print("  UNSTABLE: field left [0, capacity] or went non-finite -- diffusion step too large")
        return None, None, trace
    print("   t(s)   tree-footprint moisture (m)")
    for t, m in trace[-6:]:
        print("  %6.1f  %8.4f" % (t, m))
    if first_arrival_at is not None:
        print("  first arrival (surface just wet) at t=%.1f s" % first_arrival_at)
    else:
        print("  first arrival NOT reached within %.0f s cap" % max_seconds)
    if reached_at is not None:
        print("  reached visible threshold (column genuinely moist) at t=%.1f s" % reached_at)
    else:
        print("  visible threshold NOT reached within %.0f s cap" % max_seconds)
    return first_arrival_at, reached_at, trace


if __name__ == "__main__":
    print("grid: %d x %d cells, dx=%.2f m, dt=%.4f s (%.0f Hz)" % (NX, NY, DX, DT, 1.0 / DT))
    print("capacity=%.3f m (SOIL_DEPTH_M=%.2f x SOIL_POROSITY=%.2f)"
          % (CAPACITY, SOIL_DEPTH_M, SOIL_POROSITY))

    print("")
    print("=== Part A: diffusion stability alone (K_SAT fixed, mid-range) ===")
    for k_lateral in (1.0, 5.0, 10.0, 15.0, 20.0, 50.0):
        _, _, _ = run("stability K_LATERAL=%.0e" % k_lateral, 1.0e-5, k_lateral, 3.0,
                       max_seconds=5.0, sample_every=1.0)

    print("")
    print("=== Part B: (K_SAT x K_LATERAL x distance), extended window ===")
    # K_SAT extended past undisturbed-soil textbook range (clay..sand,
    # 1e-7..1e-4) to include what a real permaculture swale is actually
    # built with -- backfilled mulch/coarse organic matter or gravel, which
    # is standard practice specifically BECAUSE undisturbed soil infiltrates
    # too slowly for the plants to benefit before runoff moves on. 600 s
    # game time = 10 min at 1x, or 2.5 min at the project's existing 4x speed
    # option (config.SPEED_OPTIONS) -- a plausible upper bound for "watched
    # in one sitting" without inventing a new acceleration mechanism.
    results = []
    for k_sat in (1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2):
        for k_lateral in (1.0e-2, 1.0e-1, 1.0, 5.0, 10.0):
            for tree_distance in (1.5, 3.0):
                label = ("K_SAT=%.0e K_LATERAL=%.0e tree@%.1fm"
                         % (k_sat, k_lateral, tree_distance))
                first_arrival, reached, _ = run(label, k_sat, k_lateral, tree_distance,
                                                 max_seconds=600.0, sample_every=5.0)
                results.append((k_sat, k_lateral, tree_distance, first_arrival, reached))

    print("")
    print("=== summary: first-arrival / visible-threshold time (s) at the tree ===")
    print("  (K_SAT, K_LATERAL, distance) -> first-arrival (surface wet) | visible (column moist)")
    for k_sat, k_lateral, dist, first_arrival, reached in results:
        fa = "%.0fs" % first_arrival if first_arrival is not None else "--"
        vt = "%.0fs" % reached if reached is not None else "--"
        flag = " <-- both within 600 s" if (first_arrival is not None and reached is not None) else ""
        print("  K_SAT=%.0e  K_LATERAL=%.0e  tree@%.1fm  ->  first=%-6s visible=%-6s%s"
              % (k_sat, k_lateral, dist, fa, vt, flag))
