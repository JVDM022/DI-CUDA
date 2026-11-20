# Kamiak DI

Kamiak DI uploads ANG, UP1 and phase-matched SHT files over SSH/SCP, then submits
GPU dictionary-indexing jobs to Kamiak. GPU computation runs on the cluster.

## Before launching

1. Install Python 3 with Tk support and the operating system's OpenSSH client (`ssh` and `scp`).
2. Configure your Kamiak login, then run `opkssh login kamiak` in a terminal. The app uses noninteractive SSH; authentication must already work.
3. Install the project's editable HPC environment and scratch-enabled launcher on Kamiak under `~/kikuchipyCUDA`, following the [HPC guide](../docs/HPC_GUIDE.md).

## Launch

- macOS: open `Kamiak DI.app` here (keep it in this folder), or double-click `Kamiak DI.command` in the repository root.
- Windows: double-click `Kamiak DI.cmd` in the repository root.
- Terminal, from the repository root: `python3 desktop/kamiak_di.py`.

Existing portable packages in `dist/` bundle Python/Tk and the SHT database.
Extract the full Windows ZIP before opening its EXE. The portable Mac app may
be copied from its disk image to Applications. SSH authentication and the
remote cluster installation are still required.

## Submit a scan

1. Enter your Kamiak username and browse to the ANG and UP1 files.
2. Assign a different SHT master to each selected ANG phase. Match beam voltage and phase.
3. For a single-phase ANG, clear **Compare two phases**. A multiphase ANG requires the two-phase comparison.
4. Review **Neighbor pattern averaging (NPA)**; it changes the effective interaction area and may reduce spatial resolution.
5. Choose **Upload and submit**, then review the confirmation showing one GPU and two hours per phase at 1° sampling.
6. Keep the app open through upload and submission. Record the remote run folder and job IDs from the log.

The current desktop app reports upload/submission activity. Job monitoring and
result download use the commands in the HPC guide. Two-phase runs submit a
dependent merge job; wait for it before using combined results.

## Files

- `kamiak_di.py`: desktop interface and submission logic.
- `Kamiak DI.app`, `Kamiak DI.command`, `Kamiak DI.cmd`: platform launchers.
- `queue_run.sh`, `setup.slurm`: cluster preparation helpers.
- `build_portable.py`: existing portable-package builder.
