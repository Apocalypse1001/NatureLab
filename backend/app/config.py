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
FLUID_FLOW_GAIN = 1.6          # outflow rate per meter of height difference (1/s)
FLUID_OBSTACLE_RADIUS_M = 1.2  # fallback footprint if an object has no footprint_radius
FLUID_MIN_DEPTH = 1e-4         # below this, a cell is treated as dry

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
