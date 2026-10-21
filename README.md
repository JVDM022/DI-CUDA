# GPU EBSD dictionary indexing on HPC

Generate an EBSD pattern dictionary and match experimental patterns on NVIDIA
GPUs. This package combines kikuchipy 0.13.1 detector geometry and master-file
reading with a CuPy projector and batched NCC matcher. Two GPUs share dictionary
projection and then process separate experimental pattern ranges.

The Python import name remains `emsoft_di`. The repository folder can have any
name. Upstream EMsoft attribution and license are retained in `LICENSE`.

## 1. Prepare the files

The default layout is:

```text
~/kikuchipyCUDA/
  patterns.up1
  master.h5
  scan.ang
  dictionary-indexing-cuda/       # clone this repository here
  KikuchiPyCUDA/                 # Conda environment, created below
```

Inputs are raw UP1 experimental patterns, an EMsoft master-pattern HDF5 file,
and ANG scan coordinates with detector calibration in the header. These inputs
are not included in the repository. This workflow assumes alternating hex-row scan coordinates and a rectangular
UP1 raster. It is not a generic reader for every vendor scan layout; validate
the coordinate mapping for your acquisition.

Use `KIKUCHIPY_CUDA_ROOT` to change the data root. The Python command also accepts
explicit `--patterns`, `--master`, and `--ang` paths. See `--help` for all options.

## 2. Create the environment on HPC

On Kamiak, from your cloned repository:

```bash
module load miniconda3/26.5
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env create --prefix "$HOME/kikuchipyCUDA/KikuchiPyCUDA" -f environment.yml
conda activate "$HOME/kikuchipyCUDA/KikuchiPyCUDA"
python -m pip install -e '.[hpc]'
```

If that environment already exists, activate it and install the editable package;
do not recreate it. The environment uses Python 3.11, kikuchipy 0.13.1 and
CuPy 13.6.0 for CUDA 12.x. The system Python module is not needed. On the tested
compute node, its Python 3.11.4 installation could not find `libffi.so.6`.

## 3. Check inputs and run tests

Input inspection works on a login node and does not allocate a GPU:

```bash
python -m emsoft_di.hpc --dry-run --output "$HOME/kikuchipyCUDA/input-check"
```

For GPU validation, request a compute-node allocation:

```bash
srun --partition=camas --nodes=1 --ntasks=1 --gres=gpu:2 \
  --cpus-per-task=8 --mem=32G --time=00:15:00 --pty bash
module load miniconda3/26.5
module load gcc
module load cuda/12.2.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/kikuchipyCUDA/KikuchiPyCUDA"
cd "$HOME/kikuchipyCUDA/dictionary-indexing-cuda"
python -m unittest discover -s tests -v
```

CPU tests can run without GPUs. A skipped CUDA test means GPU execution was not
validated in that session. The projection test compares against kikuchipy;
when GPUs are visible it checks them as well. Tests use synthetic data.

## 4. Submit a pilot

From the repository directory:

```bash
sbatch hpc/submit_two_gpu.slurm
```

The default pilot indexes 128 experimental patterns using a coarse 3-degree
dictionary and 2x detector binning. The batch script requests two GPUs, 16 CPU
cores, 64 GB host RAM and a four-hour limit. Each GPU compares projected
reference patterns against kikuchipy before generating its dictionary portion.
A failed comparison stops the job.

For another cluster, edit the Slurm partition and resource requests and replace
the site-specific module names in `hpc/submit_two_gpu.slurm`.

## 5. Submit all points

After reviewing the pilot:

```bash
sbatch hpc/submit_two_gpu.slurm --limit 0 --resolution 1
```

`--limit 0` selects every UP1 slot. `--resolution 1` samples orientations at an
approximately 1-degree resolution; it does not promise 1-degree accuracy.
There is no orientation refinement. Large dictionaries require substantial disk
space; the job prints their size before generation. Every run generates its own
dictionary; dictionary reuse and automatic resume are not implemented.

For a different data root or environment:

```bash
export KIKUCHIPY_CUDA_ROOT=/path/to/data-root
export KIKUCHIPY_CUDA_ENV=/path/to/conda-environment
sbatch hpc/submit_two_gpu.slurm --limit 0 --resolution 1
```

## 6. Monitor and verify

Replace `JOBID` with the number printed by `sbatch`:

```bash
squeue -j JOBID
tail -f dictionary-indexing-JOBID.log
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed
```

Logs are written in the submission directory. Results are written under the data
root in `dictionary-run-JOBID/`. A successful run publishes `indexed.h5` only
after validating and merging the GPU shards. Existing output directories and
final results are not overwritten. Intermediate files remain for diagnosis.

## 7. Read the output

This is a custom results HDF5 file, not an OIM-compatible export. Key datasets:

| Dataset | Meaning |
| --- | --- |
| `simulation_indices` | Zero-based best dictionary indices, sorted by score |
| `scores` | NCC scores; column 0 is the best score |
| `valid` | Pattern was processable; this does not certify a correct orientation |
| `best_quaternion` | Scalar-first orix quaternion in its stored convention |
| `best_euler_bunge_radians` | Bunge Euler angles from orix `Rotation.to_euler()` |
| `dictionary_quaternions` | Dictionary orientations |
| `dictionary_euler_bunge_radians` | Dictionary Bunge Euler angles |
| `ang_mapping/pattern_index` | ANG-row to UP1-slot mapping |
| `ang_mapping/xy_um` | Measured spatial coordinates |
| `ang_mapping/included_in_run` | Whether a mapped point was included in this run |

The root `complete` attribute must be true. Blank patterns retain their rows with
invalid flags, -1 indices, and NaN scores/orientations. Use the mapping rather than
reshaping UP1 rows into an ANG image. A pilot may not cover every mapped index.
The output folder also contains `run.json`, the dictionary, energy weights and
per-GPU projection-check reports.

## Plotting with MTEX

See the official [MTEX instructions](https://mtex-toolbox.github.io/EBSDTutorial_py.html)
and [orientation-map plotting guide](https://mtex-toolbox.github.io/EBSDPlotting_py.html).
No plotting notebook is included. Import the custom datasets explicitly, apply
the saved spatial mapping, and check the Euler/reference-frame convention before
plotting. MTEX's generic HDF5 reader should not be assumed to recognize this schema.

## Validation and limits

CPU/GPU matching and projection comparisons are covered by tests. A completed
job establishes pipeline execution, not scientific indexing accuracy or a
performance guarantee. Review experimental/simulated pattern agreement and
validate orientations independently before interpreting results.

Master energy integration uses spatially summed Monte Carlo energy weights, an
approximation to detector-dependent weighting. `--energy-mode highest-bin` is
available as an alternative. The highest available master bin may differ from the simulation voltage. Gaussian background subtraction is
applied to both dictionary and experimental patterns; sigma defaults to 3 binned
pixels. Projection geometry uses float64, while matching uses float32. Reading,
orientation sampling, reference checks and experimental preprocessing use CPU.

The geometry bridge uses private kikuchipy helpers, so keep the pinned version.
Always review geometry, preprocessing, simulated/experimental pattern agreement
and indexing quality for a new scan. See [API reference](docs/API_REFERENCE.md)
for lower-level matching and simulation functions.

## Development

```bash
python -m unittest discover -s tests -v
python examples/match_synthetic.py --device cpu
```

GitHub Actions runs CPU checks. GPU validation requires NVIDIA hardware.
Experimental data, environments, dictionaries, results and logs are excluded
through `.gitignore`.
