"""Package a runnable NatureLab release archive.

    python tools/make_release.py            # uses backend/app/config.py VERSION
    python tools/make_release.py 0.6.0      # also rewrites VERSION first

Produces `releases/NatureLab_v<version>.zip` and appends its SHA-256 to
`releases/CHECKSUMS.txt`, so any past version can be unpacked and run without
rebuilding anything.

The archive is deliberately *runnable on unpack*: it carries `frontend/dist/`
(the built bundle the backend serves) even though git ignores it. It does not
carry `.git/`, `node_modules/`, `__pycache__/`, previous releases, PyInstaller
build output, or driver screenshots -- those are either rebuildable or huge.
"""
from __future__ import annotations

import hashlib
import re
import sys
import zipfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "backend" / "app" / "config.py"
RELEASES = ROOT / "releases"

# Directory names pruned anywhere in the tree.
SKIP_DIRS = {".git", "node_modules", "__pycache__", "releases", "build",
             ".pytest_cache", ".mypy_cache", "shots", ".vite"}
# Paths (relative to ROOT) pruned specifically. `dist/` at the root is
# PyInstaller output; `frontend/dist/` is the served bundle and IS included.
SKIP_PATHS = {Path("dist")}
SKIP_SUFFIXES = {".pyc", ".pyo", ".zip", ".exe", ".log", ".err"}


def read_version() -> str:
    match = re.search(r'^VERSION = "([^"]+)"', CONFIG.read_text(encoding="utf-8"),
                      re.MULTILINE)
    if not match:
        raise SystemExit("VERSION not found in backend/app/config.py")
    return match.group(1)


def write_version(version: str) -> None:
    text = CONFIG.read_text(encoding="utf-8")
    updated = re.sub(r'^VERSION = "[^"]+"', f'VERSION = "{version}"', text,
                     count=1, flags=re.MULTILINE)
    if updated != text:
        CONFIG.write_text(updated, encoding="utf-8")
        print(f"config.VERSION -> {version}")


def wanted(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if relative in SKIP_PATHS or any(part in SKIP_DIRS for part in relative.parts):
        return False
    if relative.parts and relative.parts[0] == "dist":
        return False
    return path.suffix not in SKIP_SUFFIXES


def collect() -> list[Path]:
    files: list[Path] = []
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and wanted(path):
            files.append(path)
    return files


def main() -> int:
    if len(sys.argv) > 2:
        raise SystemExit(__doc__)
    if len(sys.argv) == 2:
        write_version(sys.argv[1])
    version = read_version()

    bundle = ROOT / "frontend" / "dist" / "index.html"
    if not bundle.exists():
        raise SystemExit("frontend/dist is missing -- run `cd frontend && npm run build` "
                         "first, or the archive will not run on unpack")

    RELEASES.mkdir(exist_ok=True)
    archive = RELEASES / f"NatureLab_v{version}.zip"
    files = collect()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in files:
            zf.write(path, Path(f"NatureLab_v{version}") / path.relative_to(ROOT))

    digest = hashlib.sha256(archive.read_bytes()).hexdigest().upper()
    size_mb = archive.stat().st_size / (1024 * 1024)
    line = f"{date.today().isoformat()}  NatureLab_v{version}.zip  {digest}"
    checksums = RELEASES / "CHECKSUMS.txt"
    existing = checksums.read_text(encoding="utf-8") if checksums.exists() else ""
    kept = [row for row in existing.splitlines()
            if row.strip() and f"NatureLab_v{version}.zip" not in row]
    checksums.write_text("\n".join(kept + [line]) + "\n", encoding="utf-8")

    print(f"{archive.relative_to(ROOT)}  {len(files)} files  {size_mb:.1f} MB")
    print(f"SHA-256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
