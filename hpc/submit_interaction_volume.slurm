#!/bin/bash
# Generate a Monte Carlo interaction-volume cross-section and HTML/SVG plots.
# Set KIKUCHIPY_CUDA_CRYSTAL and KIKUCHIPY_CUDA_MC_ELECTRONS before sbatch.
#SBATCH --job-name=mc-interaction-volume
#SBATCH --partition=camas
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=mc-interaction-volume-%j.log

set -euo pipefail
module load miniconda3/26.5
module load gcc
module load cuda/12.2.0
source "$(conda info --base)/etc/profile.d/conda.sh"

export KIKUCHIPY_CUDA_ROOT="${KIKUCHIPY_CUDA_ROOT:-$HOME/kikuchipyCUDA}"
conda activate "${KIKUCHIPY_CUDA_ENV:-$KIKUCHIPY_CUDA_ROOT/KikuchiPyCUDA}"
cd "$KIKUCHIPY_CUDA_ROOT/dictionary-indexing-cuda"

: "${KIKUCHIPY_CUDA_CRYSTAL:?Set KIKUCHIPY_CUDA_CRYSTAL to an HDF5 file containing CrystalData}"
: "${KIKUCHIPY_CUDA_MC_ELECTRONS:?Set KIKUCHIPY_CUDA_MC_ELECTRONS to the trajectory count}"

mc_file="${KIKUCHIPY_CUDA_MC_OUTPUT:-$KIKUCHIPY_CUDA_ROOT/interaction-volume-mc-${SLURM_JOB_ID}.h5}"
html_file="${KIKUCHIPY_CUDA_VOLUME_HTML:-$KIKUCHIPY_CUDA_ROOT/interaction-volume-${SLURM_JOB_ID}.html}"
svg_file="${KIKUCHIPY_CUDA_VOLUME_SVG:-$KIKUCHIPY_CUDA_ROOT/interaction-volume-${SLURM_JOB_ID}.svg}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Generating tracked Monte Carlo trajectories from $KIKUCHIPY_CUDA_CRYSTAL"
srun --ntasks=1 python -m emsoft_di.monte_carlo \
  --crystal "$KIKUCHIPY_CUDA_CRYSTAL" \
  --output "$mc_file" \
  --electrons "$KIKUCHIPY_CUDA_MC_ELECTRONS" \
  --energy-kev "${KIKUCHIPY_CUDA_MC_ENERGY_KEV:-30}" \
  --energy-min-kev "${KIKUCHIPY_CUDA_MC_ENERGY_MIN_KEV:-15}" \
  --energy-bin-kev "${KIKUCHIPY_CUDA_MC_ENERGY_BIN_KEV:-1}" \
  --depth-max-nm "${KIKUCHIPY_CUDA_MC_DEPTH_MAX_NM:-100}" \
  --depth-step-nm "${KIKUCHIPY_CUDA_MC_DEPTH_STEP_NM:-1}" \
  --detector-size "${KIKUCHIPY_CUDA_MC_DETECTOR_SIZE:-501}" \
  --sample-tilt-deg "${KIKUCHIPY_CUDA_MC_TILT_DEG:-70}" \
  --omega-deg "${KIKUCHIPY_CUDA_MC_OMEGA_DEG:-0}" \
  --steps "${KIKUCHIPY_CUDA_MC_STEPS:-300}" \
  --seed "${KIKUCHIPY_CUDA_MC_SEED:-20261022}" \
  --batch-electrons "${KIKUCHIPY_CUDA_MC_BATCH:-262144}" \
  --devices 0 1

echo "Exporting interactive HTML and static SVG interaction-volume maps"
srun --ntasks=1 python -m emsoft_di.master.volume \
  --monte-carlo "$mc_file" \
  --output "$html_file" \
  --svg "$svg_file" \
  --view interaction

echo "Monte Carlo HDF5: $mc_file"
echo "Interactive HTML: $html_file"
echo "Vector SVG: $svg_file"
