#!/usr/bin/env bash
# Submit from the repo root with the S11 ANG/UP1 basename as the argument.
set -euo pipefail
scan=${1:?Supply absolute scan basename without extension}
repo=$PWD
root=${KIKUCHIPY_CUDA_ROOT:-$HOME/kikuchipyCUDA}
for file in "$scan.ang" "$scan.up1"; do test -s "$file" || { echo "Missing $file" >&2; exit 1; }; done
jobs=()
masters=()
for phase in alpha beta; do
  master=$(find "$repo" -type f -name "Co ($phase-cobalt)*{20kV}.sht" -print -quit)
  test -s "$master" || { echo "Missing 20 kV $phase cobalt master" >&2; exit 1; }
  masters+=("$master")
done
for master in "${masters[@]}"; do
  id=$(sbatch --parsable --gres=gpu:1 --time=02:00:00 hpc/submit_two_gpu.slurm \
    --master "$master" --patterns "$scan.up1" --ang "$scan.ang" --devices 1 --limit 0 --resolution 1 --npa)
  jobs+=("${id%%;*}")
  echo "$master: job ${id%%;*}"
done
python_bin="${KIKUCHIPY_CUDA_ENV:-$root/KikuchiPyCUDA}/bin/python"
printf -v merge '%q ' "$python_bin" "$repo/tools/merge_cobalt.py" \
  "$root/dictionary-run-${jobs[0]}/indexed.h5" "$root/dictionary-run-${jobs[1]}/indexed.h5" \
  "$root/cobalt-pair-${jobs[0]}-${jobs[1]}"
sbatch --partition=kamiak --cpus-per-task=2 --mem=8G --time=00:30:00 \
  --job-name=cobalt-merge --dependency="afterok:${jobs[0]}:${jobs[1]}" --wrap="$merge"
