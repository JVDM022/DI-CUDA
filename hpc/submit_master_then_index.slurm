#!/bin/bash
# Optionally generate Monte Carlo HDF5 from an EMsoft .xtal in Python/CuPy,
# then generate the master and run the two-GPU dictionary/indexing pipeline.
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
python_bin="$CONDA_PREFIX/bin/python"
cd "$KIKUCHIPY_CUDA_ROOT/dictionary-indexing-cuda"

export MPLBACKEND=Agg
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=4
export PYTHONUNBUFFERED=1

master_stage=1
index_stage=2
stage_total=2
if [[ -n "${KIKUCHIPY_CUDA_CRYSTAL:-}" ]]; then
  : "${KIKUCHIPY_CUDA_MC_ELECTRONS:?Set KIKUCHIPY_CUDA_MC_ELECTRONS when generating Monte Carlo data}"
  master_stage=2
  index_stage=3
  stage_total=3
  mc_file="${KIKUCHIPY_CUDA_MC_OUTPUT:-$KIKUCHIPY_CUDA_ROOT/monte-carlo-python-${SLURM_JOB_ID}.h5}"
  echo "Stage 1/3: Python/CuPy Monte Carlo generation from $KIKUCHIPY_CUDA_CRYSTAL"
  srun --ntasks=1 "$python_bin" -m emsoft_di.monte_carlo \
    --crystal "$KIKUCHIPY_CUDA_CRYSTAL" \
    --output "$mc_file" \
    --electrons "$KIKUCHIPY_CUDA_MC_ELECTRONS" \
    --energy-kev "${KIKUCHIPY_CUDA_MC_ENERGY_KEV:-30}" \
    --energy-min-kev "${KIKUCHIPY_CUDA_MC_ENERGY_MIN_KEV:-15}" \
    --energy-bin-kev "${KIKUCHIPY_CUDA_MC_ENERGY_BIN_KEV:-1}" \
    --depth-max-nm "${KIKUCHIPY_CUDA_MC_DEPTH_MAX_NM:-100}" \
    --depth-step-nm "${KIKUCHIPY_CUDA_MC_DEPTH_STEP_NM:-1}" \
    --lateral-max-nm "${KIKUCHIPY_CUDA_MC_LATERAL_MAX_NM:-${KIKUCHIPY_CUDA_MC_DEPTH_MAX_NM:-100}}" \
    --detector-size "${KIKUCHIPY_CUDA_MC_DETECTOR_SIZE:-501}" \
    --sample-tilt-deg "${KIKUCHIPY_CUDA_MC_TILT_DEG:-70}" \
    --omega-deg "${KIKUCHIPY_CUDA_MC_OMEGA_DEG:-0}" \
    --steps "${KIKUCHIPY_CUDA_MC_STEPS:-300}" \
    --seed "${KIKUCHIPY_CUDA_MC_SEED:-20261022}" \
    --batch-electrons "${KIKUCHIPY_CUDA_MC_BATCH:-262144}" \
    --devices 0 1
else
  mc_file="${KIKUCHIPY_CUDA_MC:-$KIKUCHIPY_CUDA_ROOT/monte-carlo.h5}"
fi
pattern_file="${KIKUCHIPY_CUDA_PATTERNS:-$KIKUCHIPY_CUDA_ROOT/patterns.up1}"
ang_file="${KIKUCHIPY_CUDA_ANG:-$KIKUCHIPY_CUDA_ROOT/scan.ang}"
master_file="$KIKUCHIPY_CUDA_ROOT/python-master-${SLURM_JOB_ID}.h5"
result_dir="$KIKUCHIPY_CUDA_ROOT/dictionary-run-${SLURM_JOB_ID}"
penetration_map="$KIKUCHIPY_CUDA_ROOT/monte-carlo-penetration-${SLURM_JOB_ID}.html"
penetration_svg="$KIKUCHIPY_CUDA_ROOT/monte-carlo-penetration-${SLURM_JOB_ID}.svg"

echo "Exporting OpenCL Monte Carlo electron penetration heat map: $penetration_map"
srun --ntasks=1 "$python_bin" -m emsoft_di.master.volume \
  --monte-carlo "$mc_file" \
  --output "$penetration_map" \
  --svg "$penetration_svg"

echo "Stage $master_stage/$stage_total: Python/CuPy dynamical master generation from $mc_file"
srun --ntasks=1 "$python_bin" -m emsoft_di.master \
  --monte-carlo "$mc_file" \
  --output "$master_file" \
  --backend cuda --devices 2 \
  --npx "${KIKUCHIPY_CUDA_MASTER_NPX:-500}" \
  --dmin "${KIKUCHIPY_CUDA_MASTER_DMIN:-0.05}" \
  --direction-batch "${KIKUCHIPY_CUDA_MASTER_BATCH:-4}" \
  --max-beams "${KIKUCHIPY_CUDA_MASTER_MAX_BEAMS:-512}"

echo "Stage $index_stage/$stage_total: GPU dictionary generation and pattern indexing"
srun --ntasks=1 "$python_bin" -m emsoft_di.hpc \
  --patterns "$pattern_file" \
  --master "$master_file" \
  --ang "$ang_file" \
  --output "$result_dir" \
  --devices 2 --limit 0 --resolution 1 "$@"

echo "Pipeline complete: $result_dir/indexed.h5"
echo "Electron penetration heat map: $penetration_map"
