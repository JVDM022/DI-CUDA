# KikuchiCUDA

EBSD dictionary indexing with a local app and a separate Kamiak/HPC workflow.

## Start here

**Local app on this Mac:** double-click `Start Indexing Studio.command`, or follow
[the app setup guide](app/README.md). The app imports UP/ANG and master patterns,
previews preprocessing, indexes batches with pause/resume, and exports results.

**Kamiak/HPC:** follow [the complete HPC guide](docs/HPC_GUIDE.md). The existing
Kamiak desktop launcher submits work to the cluster; it is separate from the
local Indexing Studio app.

## Where things belong

| Folder / file | Contents |
| --- | --- |
| `Start Indexing Studio.command` | Double-click launcher for the local app on macOS |
| [app/](app/README.md) | Local app, demo generator, setup and workflow instructions |
| [desktop/](desktop/README.md) | Existing Kamiak desktop launchers and submission helpers |
| [examples/](examples/README.md) | Example images, crystal inputs and synthetic matching example |
| [docs/](docs/README.md) | Detailed HPC guide, API reference and reference figures |
| [dist/](dist/README.md) | Existing packaged Kamiak downloads, grouped by platform |
| `src/emsoft_di/` | Indexing, projection, master readers and workflow code |
| `tests/` | Numerical, workflow and UI tests |
| `hpc/` | Cluster job scripts |
| `tools/` | SHT reader and preparation/merge utilities |
| `SHTdatabase-master/` | Bundled third-party master database and its attribution |
| `notebooks/` | MTEX results notebook |
| `native/` | Existing local native/Fortran experiment |
| `pyproject.toml`, `environment.yml` | Installation and environment definitions |

Use a dedicated run folder for each indexing configuration. Experimental inputs,
dictionaries and generated results are excluded from Git. Keep source inputs
separate from generated results.

The local app was checked on CPU with synthetic Ni data. See the app guide for
validation details and limitations; experimental accuracy and Windows/CUDA
execution have not been established by those checks.
