"""Write a ready-to-load river valley into data/river_valley.json.

    python tools/make_river_world.py [--slope 0.002] [--bed-width 12] ...

This is the cheap path docs/07_river_plan.md asked for: the save format already
carries the whole terrain (40 401 elevations), so a generated world needs no new
file format and no new load path -- press LOAD and choose `river_valley`. The
same generator is also reachable live from the Terrain panel; this script exists
so the geometry used in tests and screenshots is reproducible from a command
line, and so a broken frontend never blocks getting a river on screen.

The water level is set to run the channel about half full at the inlet, which is
what `terrain_gen.river_valley` reports back as `inlet_level`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app import persistence                      # noqa: E402
from app.terrain_gen import DEFAULTS, river_valley  # noqa: E402
from app.world_state import WorldState           # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="river_valley",
                        help="world name under data/ (default: river_valley)")
    for key, value in DEFAULTS.items():
        parser.add_argument("--" + key.replace("_", "-"), type=float, default=value,
                            help=f"default {value}")
    parser.add_argument("--dry", action="store_true",
                        help="leave the map dry instead of filling the channel")
    args = parser.parse_args(argv)

    world = WorldState()
    params = {key: getattr(args, key) for key in DEFAULTS}
    effective = river_valley(world.terrain, params)
    world.water.level = 0.0 if args.dry else effective["inlet_level"]
    world.water.outflow_enabled = True
    path = persistence.save_world(world, args.name)

    print(f"wrote {path}")
    print(f"  slope {effective['slope'] * 100:.2f}%  bed {effective['bed_width']:.1f} m  "
          f"incision {effective['incision']:.1f} m  banks over {effective['bank_run']:.1f} m")
    print(f"  bed falls {effective['inlet_bed']:.2f} m (inlet) -> "
          f"{effective['outlet_bed_actual']:.2f} m (outlet)")
    print(f"  water level {world.water.level:.2f} m "
          f"= {effective['operating_depth']:.2f} m deep at the inlet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
