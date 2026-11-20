# KikuchiCUDA — Kamiak DI

Kamiak DI uploads EBSD inputs to Kamiak and submits GPU dictionary-indexing jobs.
Indexing runs on the cluster. The desktop app supports single-phase scans or a
two-phase comparison using ANG phase assignments and local SHT masters.

## Run Kamiak DI

- **macOS:** double-click `Kamiak DI.command` in this folder, or open `desktop/Kamiak DI.app` while keeping it in the desktop folder.
- **Windows:** double-click `Kamiak DI.cmd` in this folder.
- **Development fallback:** `python3 desktop/kamiak_di.py` opens the older Tk GUI.

The React interface opens in a native desktop window. The new Apple-silicon
Mac build bundles Electron and Python under `dist/native/`; it needs no Terminal,
browser or separate Python installation. OpenSSH is required. Use **Log in to Kamiak** in the app to authenticate with your configured opkssh
installation. Kamiak must already have the project's HPC environment
and scratch-enabled launcher installed under `~/kikuchipyCUDA`.

Enter your Kamiak username, select ANG and UP1 files, assign the SHT master for
Phase 1, and optionally add a second master to compare two phases. Leaving
Phase 2 empty runs a single phase. Review NPA and phase settings, then choose
**Review & submit**.
Keep the app open until it reports job IDs. Set the sampling angle, GPU count and hour limit, then monitor jobs and download
completed results from **Jobs & results**. Save a run receipt to reconnect later.
See the [Kamiak guide](docs/HPC_GUIDE.md) for cluster details.

See [desktop setup and usage](desktop/README.md) for details.

## Project folders

| Folder | Contents |
| --- | --- |
| [desktop/](desktop/README.md) | Kamiak DI interface, launchers and submission helpers |
| [docs/](docs/README.md) | HPC setup, indexing workflow, API reference and reference figures |
| [examples/](examples/README.md) | Aluminium comparison image, crystal inputs and matching example |
| [dist/](dist/README.md) | Existing local Kamiak packages, grouped by platform; excluded from Git |
| `src/emsoft_di/` | GPU matching, projection, master readers and HPC pipeline |
| `tests/` | Numerical and HPC workflow tests |
| `hpc/` | Cluster job scripts |
| `tools/` | SHT reader, preparation and merge utilities |
| `SHTdatabase-master/` | Bundled master database and attribution |
| `notebooks/` | MTEX results notebook |
| `native/` | Existing local native/Fortran experiment |

Keep experimental inputs separate from generated results. Use the scan paths
reported by Kamiak DI when monitoring jobs or retrieving outputs. Sampling
spacing and match scores do not establish indexing accuracy; validate calibration,
phase assignments and orientation conventions for each scan.
