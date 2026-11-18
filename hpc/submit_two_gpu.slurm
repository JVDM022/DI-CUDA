#!/bin/bash
#SBATCH --job-name=dictionary-indexing
#SBATCH --partition=camas
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=dictionary-indexing-%j.log

set -euo pipefail
module load miniconda3/26.5
module load gcc
module load cuda/12.2.0
source "$(conda info --base)/etc/profile.d/conda.sh"
export KIKUCHIPY_CUDA_ROOT="${KIKUCHIPY_CUDA_ROOT:-$HOME/kikuchipyCUDA}"
conda activate "${KIKUCHIPY_CUDA_ENV:-$KIKUCHIPY_CUDA_ROOT/KikuchiPyCUDA}"
cd "$KIKUCHIPY_CUDA_ROOT"
scratch_dir="$(mkworkspace -n "di-${SLURM_JOB_ID}")"
[[ "$scratch_dir" == /scratch/* && -d "$scratch_dir" && -w "$scratch_dir" ]] || { echo "Invalid scratch workspace: $scratch_dir" >&2; exit 1; }
echo "Kamiak scratch workspace: $scratch_dir"
export TMPDIR="$scratch_dir/tmp"
mkdir "$TMPDIR"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=4
export PYTHONUNBUFFERED=1
# Build the bundled SHT reader on the compute node for SHT input.
master_input="$KIKUCHIPY_CUDA_ROOT/master.h5"
previous_arg=""
for argument in "$@"; do
  if [[ "$previous_arg" == --master ]]; then master_input="$argument"; fi
  if [[ "$argument" == --master=* ]]; then master_input="${argument#--master=}"; fi
  previous_arg="$argument"
done
if [[ "$master_input" == *.[sS][hH][tT] ]]; then
  decoder_source="$(python -c 'import emsoft_di; from pathlib import Path; print(Path(emsoft_di.__file__).resolve().parents[2] / "tools/sht/read_sht.cpp")')"
  c++ -std=c++14 -O2 "$decoder_source" -o "$scratch_dir/emsoft-di-read-sht"
  export PATH="$scratch_dir:$PATH"
fi
# Default: coarse 128-pattern pilot. Arguments override the pilot settings.
srun --ntasks=1 python -m emsoft_di.hpc \
  --devices 2 --limit 128 --resolution 3 --cpu-threads 8 \
  --pattern-batch 256 --dictionary-batch 16384 \
  --work-dir "$scratch_dir/index" --output "dictionary-run-${SLURM_JOB_ID}" "$@"
