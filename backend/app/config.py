"""Central configuration for the NatureLab backend.

All world units: 1 unit = 1 meter, speeds m/s, masses kg, time seconds.
"""
from pathlib import Path

# Single source of truth for the release version. tools/make_release.py reads
# this to name the archive, and the frontend shows it in the title bar, so a
# running build always says which version it is -- the donor 0.5.0 tree carried
# no version string anywhere, which made "which build am I looking at?"
# answerable only by file timestamps.
VERSION = "0.5.1"

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
ENABLE_DEBUG_PARTICLES = False
FLOW_TRACER_COUNT = 8_000
FLUID_MAX_SUBSTEPS = 8
FLUID_STABILITY_DT = 1.0 / 120.0
FLUID_CFL = 0.45
FLUID_DRY_DEPTH = 1.0e-4
FLUID_MAX_VELOCITY = 20.0
FLUID_DAMPING = 0.15
FLUID_SOURCE_COLUMNS = 2     # prescribed inflow starts at the left map edge
WATER_DENSITY = 1000.0
RIGID_STOP_SPEED = 1.0e-3
GAUGE_HISTORY_CAPACITY = 600
GAUGE_HISTORY_INTERVAL = 0.1
