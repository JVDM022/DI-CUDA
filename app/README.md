# KikuchiCUDA Indexing Studio

Local browser app for single-phase EBSD dictionary indexing. It uses the existing
NCC matcher, kikuchipy 0.13.1 geometry and master reader, and the existing SHT
reconstruction. No remote compute account is required. Existing Kamiak desktop
and HPC commands remain available separately.

## Start on this computer

An isolated environment was installed at `../.venv-kikuchi`:

```bash
cd /Users/johnjin/Documents/Research/HPC-dictionary-indexing-cuda
source ../.venv-kikuchi/bin/activate
bash app/run_app.sh
```

Open http://localhost:8501. The server binds only to localhost. Keep its terminal
open. Input paths refer to files on the server machine. Large datasets are read
from disk rather than uploaded into browser memory.

## Install on another machine

Use Python 3.11 (tested), then from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[app]'
# Optional NVIDIA CUDA 12 driver environment:
python -m pip install -e '.[cuda12]'
# Optional SHT decoder (C++ compiler required):
c++ -std=c++14 -O2 tools/sht/read_sht.cpp -o "$VIRTUAL_ENV/bin/emsoft-di-read-sht"
bash app/run_app.sh
```

On Windows activate `.venv\Scripts\activate` and launch with
`python -m streamlit run app/indexing_app.py --server.address 127.0.0.1`.
SHT needs a separately compiled decoder on PATH; H5 does not.
`validated-environment.txt` records the development machine's installed versions,
not a portable cross-platform lock file.

## Workflow

1. Enter raw UP1/UP2, source ANG, and phase/voltage-matched master H5 or SHT paths.
   ANG alone contains no diffraction patterns. EMsoft and project-native master
   H5 are accepted, not arbitrary HDF5 files.
2. Select the UP acquisition order explicitly. `raster` assumes a rectangular
   UP raster including unused slots on hex rows; `ang-order` means exactly one
   UP pattern per source ANG row. Coordinate checks cannot prove acquisition
   ordering; verify with the acquisition software.
3. Inspect inputs, detector calibration, scan coordinates and raw/preprocessed
   patterns. Select binning that divides both detector dimensions. Sigma is in
   binned pixels. Zero disables Gaussian background subtraction. Defaults match
   the existing pilot workflow: 3° sampling, 2x binning, sigma 3, 128 patterns.
4. Start. The app samples the master phase's fundamental orientation space,
   estimates dictionary disk requirements, generates a disk-backed dictionary,
   and matches batches. Automatic mode chooses CUDA when available, otherwise CPU.
   CPU projection uses kikuchipy directly. CUDA projection is compared against
   kikuchipy before matching. Matching checks sampled scores against CPU float64.
5. Pause after a batch or restart after an error. Select identical inputs,
   settings and run folder to resume. Input SHA256 hashes reject changed files.
   Device can change between resumes; other configuration changes need a new
   folder. Checksums read entire input files and can take time. A stale
   `.workflow.lock` must be removed manually only after confirming the prior
   process stopped. Resume supports process interruption; it is not a backup
   against disk failure or corrupted checkpoint files.
6. Review NCC, score gap, validity, scan preview and experimental/best-match
   patterns. For a full scan choose limit 0 and a new output folder. Finer
   angular sampling grows the dictionary substantially; 3° is a coarse pilot.

## Outputs

- `indexed.h5`: compatible **orix CrystalMap**; open with `orix.io.load()`.
  This is not a vendor raw-pattern acquisition H5 or EMsoft DI schema.
- `indexed.ang`: source detector/scan header and coordinates, master phase
  metadata, Bunge radians in EDAX TSL frame. Invalid/unprocessed rows have phase
  0 and zero angles. IQ holds preprocessed pattern standard deviation; CI holds
  the best-minus-second NCC gap, not a calibrated confidence index. ANG export
  requires source Phase and GRID header blocks.
- `quality.csv`: coordinates, UP index, inclusion, validity, NCC, score gap.
- `diagnostics.json`, `state.json`: quality summary, settings, hashes, device,
  energy integration and reference-check errors.
- `checkpoint.h5`: all retained scores/indices, validity and pattern standard
  deviations in UP order. `dictionary.npy` is retained for preview and reuse on
  resume. Do not change intermediate files.

SHT is reconstructed to Lambert maps in memory using the existing checksum-aware
reader; no standalone converted EMsoft H5 is produced. Energy metadata records
this mode. MC-global H5 weighting remains a spatially integrated approximation,
not detector-dependent weighting. Already weighted masters require highest-bin
mode. No orientation refinement, multi-phase app workflow, remote scheduler,
automatic out-of-memory recovery, or calibrated scientific confidence is claimed.
The app uses one CUDA device; the separate existing HPC workflow supports more.
Validate calibration, reference frames and experimental indexing accuracy before
using the output scientifically. Zero-contrast patterns remain unindexed.

## Tests

```bash
python -m pytest tests/test_workflow.py -q
python -m unittest discover -s tests -v
```

Workflow tests use a small synthetic Lambert master and UP/ANG fixture, exercise
pause/resume, blank patterns, settings rejection, locks, validation, and reopen
ANG/H5 with orix. UI checks render the app using Streamlit AppTest.
The synthetic workflow verifies execution and interchange, not experimental
orientation accuracy. GPU tests require an NVIDIA machine.

## Source provenance

Existing private repository: https://github.com/JVDM022/KikuchiCUDA.git
Fetched origin successfully during implementation. Local HEAD:
`aae49e51281c4caa6fb55fe5689e34518b0c618e`; fetched origin/main:
`10e2deb3838b4b5de0d445929fa171616e3fbbd7`. No reset, merge, push or replacement
of the user's existing local edits was performed. The older extracted
`KikuchiCUDA-main` was inspected; no `KikuchiCUDA-main.zip` was found locally.
The SHT reader and database were already present; the decoder was compiled into
the isolated environment. Dependencies were installed from the Python package
registry, retaining the project's pinned kikuchipy version.

## Validation results

The completed-results UI rendered with metrics and pattern preview. Six new
workflow/UI tests passed; the existing unittest suite passed 29 tests with one
CUDA test skipped. NVIDIA execution and real UP1/ANG/EMsoft Ni H5 inputs were not
available for end-to-end validation.

SHT files carry symmetry through the decoder but no metric lattice in this
adapter. When the master has the default unit lattice, export uses the source
ANG's LatticeConstants. Supply ANG calibration/phase metadata matching the master.
Double-click `Start Indexing Studio.command` on macOS to use the installed local
environment without typing activation commands.
