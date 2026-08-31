#!/usr/bin/env python3
"""Open xfce4-terminal, run the demo, grab running + success screenshots via ffmpeg x11grab."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

DISPLAY = ":1"
os.environ["DISPLAY"] = DISPLAY
ROOT = Path("/workspace")
OUT = ROOT / "run_shots"
OUT.mkdir(parents=True, exist_ok=True)
RUNNING = Path("/tmp/shot_running.flag")
SUCCESS = Path("/tmp/shot_success.flag")
for flag in (RUNNING, SUCCESS):
    flag.unlink(missing_ok=True)

terminal = subprocess.Popen(
    [
        "xfce4-terminal",
        f"--display={DISPLAY}",
        "--geometry=118x28+260+110",
        "--hide-menubar",
        "--hide-toolbar",
        "--hide-scrollbar",
        "--show-borders",
        "--font=JetBrains Mono 15",
        "--color-bg=#0d1117",
        "--color-text=#e6edf3",
        "--title=python3 -m leaf_disease.train --demo",
        f"--working-directory={ROOT}",
        "--hold",
        "-e",
        f"python3 -u {ROOT / 'scripts' / 'live_train_for_shot.py'}",
    ]
)


def terminal_rect(pad: int = 18) -> tuple[int, int, int, int]:
    raw = subprocess.check_output(["xdotool", "search", "--name", "leaf_disease.train"], text=True)
    wid = raw.strip().split()[-1]
    info = subprocess.check_output(["xwininfo", "-id", wid], text=True)
    vals: dict[str, int] = {}
    for line in info.splitlines():
        if "Absolute upper-left X" in line:
            vals["x"] = int(line.split(":")[-1])
        elif "Absolute upper-left Y" in line:
            vals["y"] = int(line.split(":")[-1])
        elif line.strip().startswith("Width:"):
            vals["w"] = int(line.split(":")[-1])
        elif line.strip().startswith("Height:"):
            vals["h"] = int(line.split(":")[-1])
    x = max(vals["x"] - pad, 0)
    y = max(vals["y"] - pad, 0)
    w = vals["w"] + pad * 2
    h = vals["h"] + pad * 2
    return x, y, w, h


def grab(path: Path) -> None:
    x, y, w, h = terminal_rect()
    tmp = Path("/tmp/_x11_shot.png")
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "x11grab",
        "-video_size",
        f"{w}x{h}",
        "-i",
        f"{DISPLAY}.0+{x},{y}",
        "-frames:v",
        "1",
        "-update",
        "1",
        str(tmp),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    tmp.replace(path)
    print(f"wrote {path} ({path.stat().st_size} bytes) size={w}x{h} at {x},{y}")


def wait_flag(path: Path, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists():
            return
        if terminal.poll() is not None:
            raise RuntimeError(f"terminal exited early with {terminal.returncode}, waiting for {path.name}")
        time.sleep(0.2)
    raise TimeoutError(f"timed out waiting for {path}")


wait_flag(RUNNING, 90)
time.sleep(2.5)
grab(OUT / "shot_running.png")
wait_flag(SUCCESS, 120)
time.sleep(2.5)
grab(OUT / "shot_success.png")
print("done")
