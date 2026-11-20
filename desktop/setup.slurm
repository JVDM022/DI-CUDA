#!/bin/bash
#SBATCH --partition=kamiak
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
release=$1
archive=$2
mkdir -p "$release"
exec 9>"$release/setup.lock"
flock 9
[[ -e "$release/ready" ]] && exit 0
module load miniconda3/26.5
source "$(conda info --base)/etc/profile.d/conda.sh"
test "$(sha256sum "$archive" | cut -d' ' -f1)" = "${release##*/}"
mkdir -p "$release/source"
tar -xzf "$archive" -C "$release/source"
envdir="$release/env"
if [[ ! -x "$envdir/bin/python" ]]; then
  conda create -y -p "$envdir" --override-channels -c https://conda.anaconda.org/conda-forge python=3.11 numpy=2.3 pip
fi
"$envdir/bin/python" -m pip install -e "$release/source[hpc]" cupy-cuda12x==13.6.0
"$envdir/bin/python" -c 'import emsoft_di, kikuchipy, cupy'
touch "$release/ready"
