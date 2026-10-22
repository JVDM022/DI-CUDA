# EMsoft dictionary matching in Python / CUDA

This package ports the normalized inner-product matching core from EMsoft
5.0.4's `opencl/DictIndx.cl` (`InnerProd`) and dictionary match selection to
Python using CuPy on NVIDIA CUDA. It also provides an explicit NumPy CPU
reference backend. CUDA is the default; it never silently falls back to CPU.

## Install on the NVIDIA machine

```sh
cd emsoft-di
python -m pip install '.[cuda12]'
```

Use `.[cuda13]` instead for a compatible CUDA 13 environment. Apple Silicon
cannot execute this NVIDIA backend. For CPU verification, install `.`.

## Use

```python
import numpy as np
from emsoft_di import index_patterns

# Arrays: (patterns, height, width), or (patterns, pixels).
# Files must already be decoded with the correct pixel ordering.
experimental = np.load('experimental.npy', mmap_mode='r')
dictionary = np.load('dictionary.npy', mmap_mode='r')
eulers = np.load('dictionary_eulers.npy')  # (dictionary patterns, 3)
result = index_patterns(
    experimental, dictionary, orientations=eulers,
    top_k=20, pattern_batch=128, dictionary_batch=4096,
)
np.savez('matches.npz', indices=result.indices, scores=result.scores,
         orientations=result.orientations)
```

Indices are **zero-based** (EMsoft Fortran output typically uses one-based
indices). Orientation columns, units, and conventions pass through unchanged.
Scores are normalized dot products, not calibrated confidence indices.
Exact ties favor the lower dictionary index. Near ties may differ with GPU
floating-point rounding. Two-level batching avoids storing the full
experimental-by-dictionary score matrix. Reduce batch sizes if VRAM is limited.
Dictionary blocks are transferred again for each experimental batch.
Sliceable HDF5 datasets are accepted directly if their first axis is patterns;
dataset selection and any transposition must be done explicitly by the caller.

## Scope and preprocessing

### Two-GPU Kamiak workflow

`python -m emsoft_di.hpc` connects the UP1 reader, master loading, detector
calibration, dictionary projection, and NCC matching. Both dictionary projection
and matching are sharded over the requested GPUs. Experimental patterns are
split into contiguous ranges and merged in original UP1 order. The defaults use
the files in `~/kikuchipyCUDA`. Install `.[hpc]` and
`cupy-cuda12x==13.6.0` in the existing Conda environment with CUDA 12.2 loaded.

On the allocated GPU node, after uploading the updated package:

```bash
cd ~/kikuchipyCUDA/dictionary-indexing-cuda
python -m pip install -e '.[hpc]'
python -m unittest discover -s tests -v
python -m emsoft_di.hpc --output ../input-check --dry-run
sbatch hpc/submit_two_gpu.slurm
```

The launcher loads Miniconda, GCC and CUDA 12.2, activates
`~/kikuchipyCUDA/KikuchiPyCUDA`, and requests two Camas GPUs,
16 CPU cores, 64 GB host RAM and four hours. The time is a limit, not an estimate.
It first runs a **128-pattern pilot** at coarse 3-degree orientation sampling
and 2x binning. Each GPU checks its projector against kikuchipy CPU reference
patterns before generating its dictionary range. A failed reference check stops
the run. Geometric calculations use float64; saved dictionary patterns and NCC
matching use float32. Only the small reference check uses CPU projection.

Output is `dictionary-run-JOBID/` beside the inputs, including `indexed.h5`,
`run.json`, a disk-backed dictionary, quaternions, Bunge Euler angles in radians,
and intermediate GPU shards. Logs go to `dictionary-indexing-JOBID.log` in the
submission directory. Blank experimental patterns have `valid=False`, NaN
scores and orientations, and simulation indices -1. These rows are preserved.
`ang_mapping/pattern_index` maps ANG rows to original UP1 indices; for pilot
runs, use `included_in_run` before indexing the output arrays. Header NCOLS are not used; the staggered-row mapping is checked against all
ANG coordinates and the UP1 raster. Validate this acquisition-layout assumption
for your scan.

After reviewing pilot scores, geometry and orientations, a full scan can be
requested explicitly:

```bash
sbatch hpc/submit_two_gpu.slurm --limit 0 --resolution 1
```

One degree is a configurable initial sampling choice, not a validated accuracy
target. There is no orientation refinement or automatic ANG export. Fine
sampling can require substantial disk space and runtime: the script prints the
dictionary count and disk size before projecting. It does not resume incomplete
runs or overwrite output directories. Keep failed shards for diagnosis and use
a new output directory when retrying. There is no automatic reuse of dictionaries
between runs. Runtime must be measured on the target GPUs.

Energy integration uses **spatially summed Monte Carlo energy weights** by
default (`--energy-mode mc-global`). This is an explicit approximation; it does
not model detector-position-dependent energy distributions. Weights, energies
and settings are saved. `--energy-mode highest-bin` selects the highest available
master energy instead; this can differ from the nominal simulation voltage.
Already-weighted masters are rejected in `mc-global` mode to prevent double
weighting. File loading, orientation sampling, reference geometry and merging
remain on CPU. GPU dictionary preprocessing subtracts a Gaussian background
with sigma 3 binned pixels; experimental preprocessing does the same on CPU.
NCC mean subtraction and normalization occur on GPU. Background settings are
configurable and require review against your actual experimental patterns.

This is a new pipeline, not a scientifically validated replacement for OIM.
Run the projection comparison tests and an HPC pilot for your own inputs
before processing a full scan. The projection bridge uses a
private geometry helper, so kikuchipy is pinned to 0.13.1.

### Using kikuchipy signals on an NVIDIA GPU

The optional adapter does not import or require kikuchipy on the development
computer. On the NVIDIA computer, install this package with its CUDA extra
as above and install kikuchipy separately if you need its file readers.
For existing EBSD or LazyEBSD objects:

```python
from emsoft_di.kikuchipy_gpu import dictionary_indexing_gpu

result = dictionary_indexing_gpu(
    s, dictionary, metric="ncc", keep_n=20,
    pattern_batch=128, dictionary_batch=4096,
    # signal_mask=mask,  # True excludes detector pixels
)
# Host arrays: one row per experimental pattern, best match first.
scores = result.scores
simulation_indices = result.indices
# Retrieve the matching orix rotations without converting Euler conventions:
matched_rotations = dictionary.xmap.rotations[result.indices]
```

This is a separate function, not a modification to kikuchipy's installed
`s.dictionary_indexing()`. It returns `IndexingResult`, not `CrystalMap`;
navigation rows follow `s.data` flattened in C order. It supports NCC
(mean subtraction plus L2 normalization), NDP (L2 normalization), and detector
masks. Navigation masks, custom metrics, and automatic CrystalMap construction
are not implemented. Retain the original signal for scan geometry and metadata.
Lazy data is read in row blocks, although reshaping Dask arrays can require
rechunking. Inputs must have finite values and nonzero norms after preprocessing;
constant patterns are rejected for NCC. Preprocessing and simulation remain
separate from matching. GPU normalization, matrix multiplication and best-match
selection use the existing CuPy backend, with no silent CPU fallback.

If you do not have kikuchipy at all, use `index_patterns()` directly with
NumPy arrays, as shown above; set `center=True` for NCC. You still need both
experimental patterns and a simulated dictionary to index a real scan.
The adapter has been checked against a dense synthetic NumPy reference;
real kikuchipy integration and CUDA execution require validation on the target
machine. Apple Silicon cannot run this CUDA backend.

The package includes a master-pattern projection implementation in
`src/emsoft_di/simulation.py`, adapted from `GenerateEBSDDetector`,
`Lambert2DSquareInverseSingle` and `CalcEBSDPatternSingleFullFast`.

```python
from emsoft_di import detector_rays, generate_patterns
rays = detector_rays(160, 120, pixel_size=50, distance=15000,
                     xpc=0, ypc=0, sample_tilt=70)
# north/south: (energy, x, y) Lambert square arrays.
# rotations: (N,3,3), sample-to-crystal column-vector rotation matrices.
for block in generate_patterns(north, south, rotations, rays,
                               energy_weights=detector_weights):
    patterns_on_gpu = block
    # Save/consume each batch; block.get() transfers a CuPy array to NumPy.
```

This projection is not yet validated against upstream EMsoft outputs. It does
not read OIM master files or automatically convert HDF5 axis order, Euler
angles, EDAX calibration conventions, or Monte Carlo detector weights.
Master maps must have both hemispheres, square odd dimensions, and explicit
energy weights for multiple energy bins. Single-energy maps can also be used.
Detector weights can vary by pixel. The generator yields GPU arrays by default;
transfer batches to host before passing them to the current matcher API.

The package also contains a Python dynamical master generator in
`src/emsoft_di/master/`. It reads crystal and Monte Carlo depth data from an
EMsoft HDF5 file, computes dynamical diffraction with NumPy and CuPy, and writes
a master HDF5 that the HPC dictionary pipeline can load. Run it with
`python -m emsoft_di.master --help`; `hpc/submit_master_then_index.slurm`
places it before dictionary generation and indexing. Crystal parsing, scattering
factor lookup and static table construction run in Python on the CPU. CuPy on
the allocated GPUs handles per-direction reflection selection, dynamical-matrix
assembly, depth propagation and intensity integration in batches.

`python -m emsoft_di.master.volume --monte-carlo INPUT.h5 --output OUTPUT.html`
exports a self-contained interactive heat map from `EMData/MCOpenCL/accum_z`.
It sums counts across x and y, then plots log-scaled electron counts by depth
bin (z, in nm) and incident energy; hovering shows the unscaled count. This
heat-map export is also included in `hpc/submit_master_then_index.slurm`.

The generated master is marked scientifically unvalidated. The Python
implementation must be compared with an independent EMsoft master and checked
against experimental patterns before scientific interpretation. It does not
perform orientation refinement or emit vendor scan maps.

EMsoft applies configurable processing, including high-pass filtering,
adaptive histogram equalization, binning, and masking, before matching.
Provide patterns processed consistently with the dictionary. This package
implements L2 normalization, an optional boolean pixel mask, and optional
mean subtraction (`center=True`, off by default). It does **not** claim
end-to-end equivalence to EMsoft's preprocessing or scientific validation
against experimental scans.

## Validation

```sh
PYTHONPATH=src python -m unittest discover -s tests -v
```

Tests compare batched matching against an independent dense NumPy reference,
check orientation lookup, masks, tie ordering, invalid inputs, and optionally
compare CUDA with CPU. CUDA validation requires an NVIDIA machine.

Upstream attribution and redistribution terms are in LICENSE.
