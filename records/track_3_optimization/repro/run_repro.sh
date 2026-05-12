#!/usr/bin/env bash
set -euo pipefail

trials="${1:-20}"
gpus="$(nvidia-smi -L | wc -l)"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
marker="logs/repro_marker_${timestamp}"

if [[ "$gpus" -ne 8 ]]; then
  echo "Expected 8 GPUs for this replication, found ${gpus}" >&2
  exit 1
fi

pixi install
pixi run python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count()); assert torch.cuda.device_count() == 8'
pixi run python data/cached_fineweb10B.py 20

mkdir -p logs
touch "$marker"

pixi run torchrun --standalone --nproc_per_node=8 \
  records/track_3_optimization/repro/train_normuon_repro.py "$trials" \
  2>&1 | tee "logs/repro_normuon_console_${timestamp}.log"

pixi run torchrun --standalone --nproc_per_node=8 \
  records/track_3_optimization/repro/train_aurora_pure_repro.py "$trials" \
  2>&1 | tee "logs/repro_aurora_pure_console_${timestamp}.log"

mapfile -t new_logs < <(find logs -maxdepth 1 -type f -name '*.txt' -newer "$marker" | sort)
pixi run python records/track_3_optimization/repro/summarize_runs.py "${new_logs[@]}"
