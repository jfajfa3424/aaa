#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 PYTHONPATH=.
mkdir -p outputs/demo
python3 -m leaf_disease.train --demo --epochs 5 --output-dir outputs/demo --seed 42 | tee outputs/demo/train.log
python3 -m leaf_disease.evaluate --checkpoint outputs/demo/best.pt --demo --output-dir outputs/demo | tee outputs/demo/eval.log
python3 scripts/render_run_shots.py
echo "OK: see NOTES.md and run_shots/"
