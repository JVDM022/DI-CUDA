#!/bin/bash
# Generate an EMsoft-style dynamical master in Python/CuPy, then run the
# existing two-GPU dictionary generation and indexing pipeline.
# Set KIKUCHIPY_CUDA_MC to an EMsoft HDF5 file containing CrystalData and
# EMData/MCOpenCL/accum_z before submitting.
#SBATCH --job-name=python-master-index
#SBATCH --partition=camas
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=python-master-index-%j.log

set -euo pipefail
module load miniconda3/26.5
module load gcc
module load cuda/12.2.0
source "$(conda info --base)/etc/profile.d/conda.sh"

export KIKUCHIPY_CUDA_ROOT="${KIKUCHIPY_CUDA_ROOT:-$HOME/kikuchipyCUDA}"
conda activate "${KIKUCHIPY_CUDA_ENV:-$KIKUCHIPY_CUDA_ROOT/KikuchiPyCUDA}"
cd "$KIKUCHIPY_CUDA_ROOT/dictionary-indexing-cuda"

export MPLBACKEND=Agg
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=4
export PYTHONUNBUFFERED=1

mc_file="${KIKUCHIPY_CUDA_MC:-$KIKUCHIPY_CUDA_ROOT/monte-carlo.h5}"
pattern_file="${KIKUCHIPY_CUDA_PATTERNS:-$KIKUCHIPY_CUDA_ROOT/patterns.up1}"
ang_file="${KIKUCHIPY_CUDA_ANG:-$KIKUCHIPY_CUDA_ROOT/scan.ang}"
master_file="$KIKUCHIPY_CUDA_ROOT/python-master-${SLURM_JOB_ID}.h5"
result_dir="$KIKUCHIPY_CUDA_ROOT/dictionary-run-${SLURM_JOB_ID}"
penetration_map="$KIKUCHIPY_CUDA_ROOT/monte-carlo-penetration-${SLURM_JOB_ID}.html"

echo "Exporting OpenCL Monte Carlo electron penetration heat map: $penetration_map"
srun --ntasks=1 python -m emsoft_di.master.volume \
  --monte-carlo "$mc_file" \
  --output "$penetration_map"

echo "Stage 1/2: Python/CuPy dynamical master generation from $mc_file"
srun --ntasks=1 python -m emsoft_di.master \
  --monte-carlo "$mc_file" \
  --output "$master_file" \
  --backend cuda --devices 2 \
  --npx "${KIKUCHIPY_CUDA_MASTER_NPX:-500}" \
  --dmin "${KIKUCHIPY_CUDA_MASTER_DMIN:-0.05}" \
  --direction-batch "${KIKUCHIPY_CUDA_MASTER_BATCH:-4}" \
  --max-beams "${KIKUCHIPY_CUDA_MASTER_MAX_BEAMS:-512}"

echo "Stage 2/2: GPU dictionary generation and pattern indexing"
srun --ntasks=1 python -m emsoft_di.hpc \
  --patterns "$pattern_file" \
  --master "$master_file" \
  --ang "$ang_file" \
  --output "$result_dir" \
  --devices 2 --limit 0 --resolution 1 "$@"

echo "Pipeline complete: $result_dir/indexed.h5"
echo "Electron penetration heat map: $penetration_map"
