"""Central configuration for the NatureLab backend.

All world units: 1 unit = 1 meter, speeds m/s, masses kg, time seconds.
"""
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
FRONTEND_DIST = PROJECT_DIR / "frontend" / "dist"

HOST = "127.0.0.1"
PORT = 8756

# World / terrain
WORLD_SIZE_M = 100.0          # 100 x 100 meters
TERRAIN_CELLS = 100           # grid resolution in cells per side
TERRAIN_CELL_SIZE = WORLD_SIZE_M / TERRAIN_CELLS
HEIGHT_MIN = -20.0
HEIGHT_MAX = 60.0

# Simulation
FIXED_DT = 1.0 / 60.0         # physics timestep (s)
SPEED_OPTIONS = [0.25, 0.5, 1.0, 2.0, 4.0]
MAX_SUBSTEPS = 10             # catch-up limit per broadcast tick
STREAM_HZ = 30.0              # how often particle frames are pushed to clients
PARTICLE_COUNT = 100_000
VISUALIZATION_PARTICLE_LIMIT = 25_000
FLUID_MAX_SUBSTEPS = 8
FLUID_STABILITY_DT = 1.0 / 120.0

# Shallow-water solver (v0.3): outflow-limited height-field method (Mei/Decaudin/Hu
# "virtual pipes" style). See docs/04_TZ_v0.3_roadmap.md, milestone v0.3.
#
# Recalibrated 2026-09-01 (was 1.6): the original value made the solver
# correct but effectively invisible in a live demo -- measured directly, a
# 100-cell-wide grid with River flow on still had its MIDDLE column sitting
# at its 30-seconds-ago value after a full 30 real seconds of RUNNING (only
# the few cells nearest each forced edge had moved at all). Matches the
# user-reported symptom exactly: with flow on, it still just looked like a
# still pool ("просто бассейн"), not the edge-to-edge current asked for.
# 10.0 was chosen as the highest value that stayed stable under the worst
# combination this project actually produces: an obstacle AND a bed-height
# ROCK AND full Schauberger shade cooling (temperature_factor clamped at
# TEMP_FACTOR_MAX=2.0, which multiplies this constant directly in _step) all
# at once, run 90 simulated seconds. Swept in steps of 2: instability first
# appeared at effective gain (this constant * TEMP_FACTOR_MAX) of 16 -- a
# checkerboard blow-up next to the obstacle within a couple of seconds, not a
# slow drift, so there is no "it'll be fine over a longer run" margin to
# rely on above that line. 10.0 keeps roughly 30% headroom under it while
# still being a ~6x speedup over the old value. Raising this further without
# re-running that same three-way stress sweep will eventually reproduce the
# blow-up.
FLUID_FLOW_GAIN = 10.0         # outflow rate per meter of height difference (1/s)
FLUID_OBSTACLE_RADIUS_M = 1.2  # fallback footprint if an object has no footprint_radius
FLUID_MIN_DEPTH = 1e-4         # below this, a cell is treated as dry

# Continuous river current (optional, off by default -- see WaterState.flow_enabled).
# initialize() fills depth = water_level - terrain, which makes the surface flat
# from tick 1 (zero gradient, zero flow -- documented in
# docs/04_TZ_v0.3_roadmap.md v0.4 "Важная находка"). Enabling flow_enabled clamps
# the west edge column (i=0) to at least world.water.level (read live every tick,
# see ShallowWaterFluidSolver._step -- so the Water Level slider now directly
# controls the river's source height, not a value disconnected from it) and the
# east edge column (i=-1) to at most FLUID_RIVER_SINK_FRACTION of that same
# source, i.e. an upstream reservoir that never runs dry and a downstream outlet
# that never backs up. That keeps a permanent height difference across the grid,
# which the existing unconditionally-conservative interior flux scheme then
# carries as real, continuous west->east flow. Mass is intentionally NOT
# conserved at those two edge columns while flow_enabled is on (water enters at
# the source, leaves at the sink) -- everywhere else the same conservative
# scheme as always applies.
#
# Recalibrated 2026-09-01 (user tested the running app a second time): the
# previous design pre-filled the ENTIRE grid to a fixed source/sink ramp the
# instant flow was enabled, specifically so the gradient was visible right
# away (see the FLUID_FLOW_GAIN comment below for why that mattered). But the
# user's actual ask was the opposite of "instantly full": water should visibly
# enter FROM the edge and flow across, at the height they set -- not appear
# everywhere at once. So enabling flow now instead RESETS the grid to dry
# (ShallowWaterFluidSolver._seed_river_profile) and lets the real flux physics
# advance a genuine wavefront in from the source column each tick. Measured
# directly at FLUID_FLOW_GAIN=10 starting bone dry: front reaches 15% of the
# domain by 1s, 43% by 10s, 77% by 40s -- a real, watchable flood wave, not an
# instant fill and not an imperceptibly slow one either.
FLUID_RIVER_SINK_FRACTION = 0.1  # east-edge cap, as a fraction of the west-edge source

# Sediment transport / erosion / deposition (v0.4 RiverLab). See
# fluid_solver.ShallowWaterFluidSolver._step and
# docs/04_TZ_v0.3_roadmap.md v0.4. Tuned deliberately slow relative to a
# single flood event -- real fluvial erosion is slow compared to one flood,
# and it keeps short FloodLab-style runs (seconds of sim time) from being
# measurably disturbed while still being clearly visible over the longer
# runs a RiverLab "River A vs River B" comparison would actually use.
SEDIMENT_CAPACITY_SCALE = 0.05  # capacity = this * flow_speed * depth
SEDIMENT_ERODE_RATE = 0.3       # fraction of the capacity gap eroded per second
SEDIMENT_DEPOSIT_RATE = 0.3     # fraction of the excess deposited per second
TERRAIN_RESYNC_INTERVAL_S = 1.0  # how often erosion-driven terrain changes are re-broadcast

# Riverbed rock obstruction (v0.4 RiverLab -- docs/04_TZ_v0.3_roadmap.md v0.4,
# the "камни на дне реки меняют русло" item). Its explicit note: a rock is not a
# rigid body with mass and buoyancy, it is part of the terrain/riverbed. So a
# body with bed_height > 0 is deliberately NOT rasterized into the binary
# obstacle mask (which is an infinitely tall wall -- correct for a house, wrong
# for a boulder). Instead it raises the *effective bed* by a dome of that
# height, so water is deflected around it near the bed and still passes over it
# when the river is deeper than the rock is tall. That single change is what
# produces meandering together with the existing sediment mechanic: the flow
# accelerates around the flanks (erosion) and stalls in the lee (deposition).
# The dome is added to terrain for flow purposes only -- world terrain is never
# mutated by it, so a rock can be moved or deleted without leaving a crater.
BED_DOME_EXPONENT = 0.5      # 0.5 = hemispherical profile; 1.0 would be a cone
BED_EROSION_SHIELD = 0.05    # bed offset (m) above which terrain under a rock is unerodible

# Water temperature (v0.4 RiverLab, Schauberger hypothesis -- docs/04_TZ_v0.3_roadmap.md
# v0.4 and docs/01_vision.md "Viktor Schauberger Lab"; NOT asserted as physically
# correct, only implemented as a testable, comparable effect per that section's own
# HYPOTHESIS->BUILD->SIMULATE->MEASURE->COMPARE stance). Deliberately NOT a persistent
# diffusing field -- ShallowWaterFluidSolver recomputes a per-cell multiplier from
# tree shade_snapshot() each tick (see _update_temperature_factor). Calibrated
# (approximately, not rigorously) from a real historical reference: Schauberger's
# Neuberg log flume moved a block 2km in 29 min at ~9.5C vs 40 min at ~14C -- about
# a 38% speed change over ~4.5C, i.e. roughly TEMP_EFFECT_PER_DEGREE_C below.
# Positive local cooling (shade below environment.temperature) increases both flow
# gain and sediment carrying capacity; warming decreases them; the factor is clamped
# so extreme temperatures can't destabilise the solver.
TEMP_EFFECT_PER_DEGREE_C = 0.075   # multiplicative change per +-1C vs environment.temperature
TEMP_FACTOR_MIN = 0.5
TEMP_FACTOR_MAX = 2.0

# Rigid body force model (v0.3): gravity + buoyancy + hydrodynamic drag +
# ground friction. See backend/app/rigid_body.py:ForceRigidBodySystem and
# docs/04_TZ_v0.3_roadmap.md.
RIGID_REFERENCE_DEPTH_M = 1.0     # depth at which buoyancy coefficient reaches full effect
RIGID_WATER_DRAG_SCALE = 1500.0   # N per (drag_coeff * m/s), tuned so a CAR moves at flood depth
RIGID_FLOAT_CONTACT_THRESHOLD = 0.15  # ground contact fraction below which a body is FLOATING
RIGID_MOVE_EPS_MPS = 0.02         # speed below which a body counts as "at rest"

# Impulse-based body<->body collision (v0.3 interim -- disk footprints in the
# XZ plane, mass-weighted, no rotation/torque). A full warp.sim rigid-body
# port is a separate scoped future milestone; see docs/04_TZ_v0.3_roadmap.md.
RIGID_COLLISION_RESTITUTION = 0.2  # 0 = perfectly inelastic, 1 = perfectly elastic
