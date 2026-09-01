"""NatureLab 0.2 launcher: non-recursive backend ownership and shutdown."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request
import webbrowser

HOST, PORT = "127.0.0.1", 8756
URL = f"http://{HOST}:{PORT}/"


def install_root() -> Path:
    # A one-file PyInstaller __file__ points into _MEIPASS. External project
    # folders are deliberately resolved beside NatureLab.exe instead.
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


ROOT = install_root()
BACKEND = ROOT / "backend"
TEST_MODE = os.environ.get("NATURELAB_TEST_MODE") == "1"


def test_log(message: str) -> None:
    if TEST_MODE:
        with (ROOT / "launcher_test.log").open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")


def status() -> dict | None:
    try:
        import json
        with urllib.request.urlopen(URL + "api/status", timeout=1) as response:
            payload = json.loads(response.read())
            return payload if payload.get("app") == "NatureLab" else None
    except Exception:
        return None


def find_python() -> str | None:
    bundled = ROOT / "runtime" / "python" / "pythonw.exe"
    if bundled.exists():
        return str(bundled)
    # Never use sys.executable in a frozen launcher: that is NatureLab.exe.
    for name in ("python.exe", "pythonw.exe"):
        found = shutil.which(name)
        if found and Path(found).resolve() != Path(sys.executable).resolve():
            return found
    return None


def stop_backend(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def show_error(message: str) -> None:
    import tkinter as tk
    from tkinter import messagebox
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("NatureLab", message)
    root.destroy()


def main() -> None:
    test_log(f"launcher root={ROOT}")
    owned_process: subprocess.Popen | None = None
    if status() is None:
        python = find_python()
        if python is None or not BACKEND.is_dir():
            if TEST_MODE:
                test_log("python/backend not found")
                return
            show_error("Python/backend not found. Run start.bat after installing requirements.")
            return
        owned_process = subprocess.Popen(
            [python, "-m", "uvicorn", "app.main:app", "--host", HOST,
             "--port", str(PORT)],
            cwd=BACKEND,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        test_log(f"backend pid={owned_process.pid} python={python}")
        deadline = time.time() + 60
        while time.time() < deadline and status() is None and owned_process.poll() is None:
            time.sleep(0.25)
    if status() is None:
        stop_backend(owned_process)
        if TEST_MODE:
            test_log("backend startup failed")
            return
        show_error("Backend did not start. Install backend\\requirements.txt.")
        return

    if TEST_MODE:
        test_log("backend ready; stopping owned process")
        stop_backend(owned_process)
        test_log("launcher done")
        return
    webbrowser.open(URL)
    if owned_process is None:
        return  # existing verified NatureLab backend is not ours to stop

    import tkinter as tk
    root = tk.Tk()
    root.title("NatureLab 0.2")
    root.resizable(False, False)
    tk.Label(root, text=f"NatureLab is running\n{URL}", padx=28, pady=16).pack()
    tk.Button(root, text="Stop NatureLab", width=20, command=root.destroy).pack(pady=(0, 16))
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    try:
        root.mainloop()
    finally:
        stop_backend(owned_process)


if __name__ == "__main__":
    main()
