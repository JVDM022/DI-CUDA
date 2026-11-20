#!/bin/bash
set -euo pipefail
# The caller invokes this from its private run folder.
release="$HOME/kikuchipyCUDA/desktop-backends/$1"
shift
npa=$1
shift
export KIKUCHIPY_CUDA_ROOT="$HOME/kikuchipyCUDA"
export KIKUCHIPY_CUDA_ENV="$release/env"
run=$PWD
exec 8>submission.lock
flock 8
# Save each accepted job before submitting its dependents. Reconnects reuse IDs.
queue() {
  local label=$1 id
  shift
  if [[ ! -s "$label.job" ]]; then
    id=$(sbatch --parsable "$@")
    id=${id%%;*}
    [[ $id =~ ^[0-9]+$ ]]
    printf '%s\n' "$id" > "$label.job.tmp"
    mv "$label.job.tmp" "$label.job"
  fi
  cat "$label.job"
}
deps=()
if [[ ! -e "$release/ready" ]]; then
  setup=$(queue setup --job-name=di-setup --output="$run/setup-%j.log" setup.slurm "$release" "$run/backend.tgz")
  deps=(--dependency="afterok:$setup" --kill-on-invalid-dep=yes)
fi
flags=()
[[ $npa == 1 ]] && flags+=(--npa)
jobs=()
for phase in "$@"; do
  n=$(( ${#jobs[@]} + 1 ))
  # Submit the bundled launcher immediately; setup installs its matching runtime.
  id=$(queue "phase$n" "${deps[@]}" --gres=gpu:1 --time=02:00:00 submit_two_gpu.slurm \
    --patterns "$run/scan.up1" --ang "$run/scan.ang" --master "$run/master$n.sht" \
    --output "$run/result$n" --devices 1 --limit 0 --resolution 1 "${flags[@]}")
  jobs+=("$id")
done
if [[ ${#jobs[@]} == 2 ]]; then
  printf -v cmd '%q ' "$KIKUCHIPY_CUDA_ENV/bin/python" "$release/source/tools/merge_cobalt.py" \
    "$run/result1/indexed.h5" "$run/result2/indexed.h5" "$run/combined" "$1" "$2" "$run/master1.sht" "$run/master2.sht"
  queue merge --partition=kamiak --cpus-per-task=2 --mem=8G --time=00:30:00 --job-name=phase-merge \
    --dependency="afterok:${jobs[0]}:${jobs[1]}" --kill-on-invalid-dep=yes --wrap="$cmd"
fi
for job in *.job; do echo "${job%.job} $(cat "$job")"; done
