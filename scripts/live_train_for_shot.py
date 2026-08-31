#!/usr/bin/env python3
"""Run the demo in a visible terminal and pause so we can screenshot mid-run and after success."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

os.chdir("/workspace")
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TQDM_DISABLE"] = "1"
os.environ["PYTHONPATH"] = "/workspace"

CYAN = "\033[1;36m"
DIM = "\033[90m"
GREEN = "\033[1;32m"
RESET = "\033[0m"
BOLD = "\033[1m"

print(f"{DIM}ubuntu@cloud-agent:/workspace${RESET} {CYAN}python3 -m leaf_disease.train --demo --epochs 5 --output-dir outputs/demo --seed 42{RESET}")
print()

Path("/tmp/shot_running.flag").unlink(missing_ok=True)
Path("/tmp/shot_success.flag").unlink(missing_ok=True)
Path("outputs/demo").mkdir(parents=True, exist_ok=True)

proc = subprocess.Popen(
    [
        sys.executable,
        "-u",
        "-m",
        "leaf_disease.train",
        "--demo",
        "--epochs",
        "5",
        "--output-dir",
        "outputs/demo",
        "--seed",
        "42",
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
assert proc.stdout is not None
last_lines: list[str] = []
for line in proc.stdout:
    sys.stdout.write(line)
    sys.stdout.flush()
    last_lines.append(line.rstrip())
    if line.startswith("epoch 002"):
        Path("/tmp/shot_running.flag").write_text("running\n", encoding="utf-8")
        time.sleep(22)
rc = proc.wait()

print()
if rc == 0:
    print(f"{GREEN}================================================{RESET}")
    print(f"{GREEN}  ✓  代码跑通    PIPELINE OK    exit code = 0{RESET}")
    print(f"{GREEN}================================================{RESET}")
    listing = subprocess.check_output(["ls", "-lh", "outputs/demo/best.pt", "outputs/demo/history.json"], text=True)
    print(listing.rstrip())
else:
    print(f"train failed with exit code {rc}")
print()
print(f"{DIM}ubuntu@cloud-agent:/workspace${RESET}")
Path("/tmp/shot_success.flag").write_text(f"rc={rc}\n", encoding="utf-8")
time.sleep(90)
sys.exit(rc)
