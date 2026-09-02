"""Central configuration for the NatureLab backend.

All world units: 1 unit = 1 meter, speeds m/s, masses kg, time seconds.
"""
from pathlib import Path

# Single source of truth for the release version. tools/make_release.py reads
# this to name the archive, and the frontend shows it in the title bar, so a
# running build always says which version it is -- the donor 0.5.0 tree carried
# no version string anywhere, which made "which build am I looking at?"
# answerable only by file timestamps.
VERSION = "0.12.0"

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
FRONTEND_DIST = PROJECT_DIR / "frontend" / "dist"

HOST = "127.0.0.1"
PORT = 8756

# World / terrain
# v0.7.0: doubled from 100 m at the user's request -- a river needs room to
# meander, and at 100 m a channel that curves at all runs out of map. Cell size
# is deliberately held at 1 m rather than stretched, so channel detail, single
# boulders and narrow side-channels stay as legible as they were; the cost is
# 4x the cells (40 401 vertices), which the RTX 5090 does not notice and the
# browser main thread does -- see TEST_REPORT.md for the measured frame times.
WORLD_SIZE_M = 200.0          # 200 x 200 meters
TERRAIN_CELLS = 200           # grid resolution in cells per side
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
ENABLE_DEBUG_PARTICLES = False
# Scaled with the map area (4x cells) and then raised again on top of that:
# the user's report was that it is "not always clear the water is moving", and
# tracer density is the thing that communicates it. These are advected by the
# real velocity field on the GPU, not decorative sprites -- the cost is one
# kernel over this many particles per substep plus 12 bytes each per stream
# frame (~430 KB/s at 30 Hz), not a per-particle CPU update.
FLOW_TRACER_COUNT = 36_000

# --------------------------------------------------- Water rendering (v0.10.0)
# How often the real velocity field is streamed, in broadcast frames. The
# frontend uses it as a physics-derived flow map (ripple direction, foam,
# spray); that is a low-frequency visual signal, so every third frame at 30 Hz
# is ample and keeps the largest remaining payload off the wire two frames in
# three. Set to 1 to stream every frame.
VELOCITY_STREAM_EVERY = 3
FLUID_MAX_SUBSTEPS = 8
FLUID_STABILITY_DT = 1.0 / 120.0
FLUID_CFL = 0.45
FLUID_DRY_DEPTH = 1.0e-4
FLUID_MAX_VELOCITY = 20.0
# Bed friction: Manning's law, not a uniform velocity damping. Bed stress is
# shared over the whole water column, so the deceleration goes as |u|/h^(4/3):
# a deep channel keeps its speed while a thin sheet is held back. That is why a
# river stays in its bed and only creeps out across a floodplain. The depth-blind
# damping that stood here up to v0.10.1 (FLUID_DAMPING = 0.15) gave a 1.5 m
# channel and a 5 cm sheet the identical answer, and no value of it could
# produce the difference -- see FrictionLawTests.
FLUID_MANNING_N = 0.03           # s/m^(1/3): a clean, straight natural channel
# Depth floor for the friction term. This is a physics knob, not a guard against
# dividing by zero: it alone decides how fast a wetting front creeps forward, and
# it sets the depth below which the new law brakes HARDER than the old damping
# did -- about 0.12 m at 1 m/s and 0.20 m at 2 m/s. Above that the new law is the
# weaker of the two (17x weaker at 1 m depth), so the change is not "more drag"
# or "less drag": it is drag that finally reads the depth.
FLUID_FRICTION_MIN_DEPTH = 0.02  # m
FLUID_SOURCE_COLUMNS = 2     # prescribed inflow starts at the left map edge
WATER_DENSITY = 1000.0
RIGID_STOP_SPEED = 1.0e-3
GAUGE_HISTORY_CAPACITY = 600
GAUGE_HISTORY_INTERVAL = 0.1

# --------------------------------------------------------------- RiverLab (v0.6.0)
# Sediment transport / erosion / deposition on the same grid as h/u/v.
# capacity = SEDIMENT_CAPACITY_SCALE * |velocity| * depth: a cell carrying less
# than its capacity picks material up off the bed, a cell carrying more drops it.
# On the pre-0.5.1 diffusion solver this had to use a flux surrogate in place of
# velocity (see docs/05_audit_v0.4_water.md, P1-1); with the Warp solver |u| is a
# real m/s, so the capacity law is the textbook one rather than a proxy.
#
# Deliberately slow relative to a single flood: real fluvial erosion is slow
# compared with one flood event, and keeping it slow means a short FloodLab run
# is not measurably disturbed while a longer RiverLab "River A vs River B"
# comparison still shows a clear difference.
SEDIMENT_CAPACITY_SCALE = 0.05   # capacity = this * |velocity| * depth
SEDIMENT_ERODE_RATE = 0.3        # fraction of the capacity gap eroded per second
SEDIMENT_DEPOSIT_RATE = 0.3      # fraction of the excess deposited per second
SEDIMENT_MAX_BED_CHANGE = 0.02   # m per second, clamp so one tick cannot dig a pit
TERRAIN_RESYNC_INTERVAL_S = 1.0  # how often eroded terrain is re-broadcast while RUNNING

# Riverbed rocks (ROCK). A boulder is not a rigid body with mass and buoyancy and
# it is not a wall either -- it is part of the bed. So ROCK is deliberately NOT
# rasterized into the binary solid mask (which is an infinitely tall wall, right
# for a house, wrong for a boulder); instead it raises the *effective* bed by a
# dome of `bed_height`. Everything the RiverLab item asks for falls out of that
# one choice: flow is deflected around the dome near the bed, deep water still
# passes over it, the effect scales with both the rock's size and where it sits,
# and the flanks scour while the lee fills -- which is what makes a channel move.
#
# The dome lives in a separate array from the erodible bed, so erosion can mutate
# the terrain under a rock while the dome itself is recomputed from live positions
# every tick -- a rock that is moved or deleted therefore leaves no crater.
# --------------------------------------------------- Water controls (v0.8.0)
# Transmissive outlet on the east edge, in cells. Before this every outer face
# was no-flux, so water that entered could never leave and the map filled
# forever (measured: 2290 -> 3484 -> 4311 m3, never decreasing). A river running
# off the edge of the domain is exactly this boundary condition.
FLUID_OUTFLOW_COLUMNS = 2

# A placed DRAIN removes water through a smooth radial sink and spins the flow
# around it. The spin is NOT a constant: it comes from the ambient circulation
# the drain measures in the annulus just outside itself, amplified as 1/r by
# convergence -- conservation of angular momentum. A perfectly symmetric
# approach therefore produces no rotation, which is physically right: in a
# depth-averaged model a purely radial sink cannot manufacture spin.
DRAIN_SWIRL_GAIN = 0.35      # how strongly measured circulation is fed back as spin

BED_DOME_EXPONENT = 0.5      # 0.5 = hemispherical profile; 1.0 would be a cone
BED_EROSION_SHIELD = 0.05    # bed offset (m) above which the terrain under a rock is unerodible
