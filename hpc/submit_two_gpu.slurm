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
export MPLBACKEND=Agg
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=4
export PYTHONUNBUFFERED=1
# Default: coarse 128-pattern pilot. Arguments override the pilot settings.
srun --ntasks=1 python -m emsoft_di.hpc \
  --devices 2 --limit 128 --resolution 3 --cpu-threads 8 \
  --output "dictionary-run-${SLURM_JOB_ID}" "$@"
