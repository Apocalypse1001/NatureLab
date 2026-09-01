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
FLUID_OBSTACLE_RADIUS_M = 1.2  # placeholder footprint until object scale is wired in
FLUID_MIN_DEPTH = 1e-4         # below this, a cell is treated as dry
