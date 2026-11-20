# Kamiak DI — native React desktop app

Double-click **Kamiak DI.app** in this folder on macOS. It opens a native Electron desktop window containing the React GUI,
without a browser tab or Terminal. Keep this source app in `desktop/` so it
can find the Python service and bundled interface. GPU indexing runs on Kamiak.

The built Apple-silicon Mac app is in
`dist/native/Kamiak DI-darwin-arm64/Kamiak DI.app`. It bundles Electron, Python,
Tk, the interface and SHT database, and can be copied to Applications. OpenSSH
and your Kamiak authentication/configuration are still required. No Node or
separate Python installation is needed to run this packaged app.

The small source-checkout app in `desktop/` forwards to this build when present.
Otherwise it can use an installed development Electron runtime. Windows source
launching requires installing `desktop/shell` dependencies and a suitable Python
runtime; the new Windows desktop package has not been built or validated.

## Connect and select inputs

1. Click **Log in to Kamiak** in the connection card (or Setup & help). The app
   runs `opkssh login kamiak` without opening Terminal. Complete the sign-in in
   the browser that opkssh opens, then return to the app. opkssh must already be
   installed and configured with the `kamiak` provider. The button waits up to
   five minutes, reports missing tools, failed sign-ins and timeouts, and never
   writes authentication output or credentials into the activity log.
2. Enter your Kamiak username and choose **Check connection**. The app verifies
   SSH, Slurm, the remote environment at `~/kikuchipyCUDA/KikuchiPyCUDA` and the
   launcher under `~/kikuchipyCUDA/dictionary-indexing-cuda`. It does not install
   or update the remote backend. See the [HPC guide](../docs/HPC_GUIDE.md).
3. Browse for ANG and UP1 inputs. You can also paste full local paths. Use
   **Read phases from ANG header** after entering an ANG path manually.
4. Assign the first ANG phase and its phase- and voltage-matched SHT master.
   **Phase 2 is optional:** leave its master empty to run one phase, or choose a
   second, different SHT master and distinct ANG phase to compare two phases.
   Clear the second master using its × button to switch back to one phase.

A one-phase run can use a multiphase source ANG. The app uploads a temporary ANG
copy retaining calibration and coordinates but containing only the selected
phase, with every scan row assigned that phase. All patterns are indexed against
the selected master. The source file stays unchanged. This avoids the remote
export's automatic majority-phase label. Two-phase runs upload the original ANG
and merge per-phase matches using the selected original phase IDs.

## Resources and submission

Set **Sampling angle** (0.1–30°), **GPUs per phase** (1 or 2), and **Time limit**
(1–168 whole hours per phase). Defaults are 1°, one GPU and two hours. Smaller
angles create denser dictionaries; they are not a guarantee of angular accuracy.
Kamiak partition/allocation limits can reject requests. Two-phase runs request
these resources separately for each phase job. The dependent CPU merge job
retains its own 30-minute limit.

Review NPA: averaging neighboring patterns may improve signal quality while
reducing spatial resolution. **Review & submit** validates the selected inputs,
then shows the phase assignments and resource request for confirmation. Accepting
**Upload & submit** transfers the inputs and queues full-scan jobs. Keep the app
open through upload. Accepted job IDs remain visible even if a later step fails;
check them before retrying to avoid duplicate jobs.

## Jobs, receipts and results

- **Jobs & results → Refresh status** displays Slurm states, elapsed time and exit
  codes. This is scheduler status, not a numerical indexing progress percentage.
- **Save receipt** downloads a JSON file containing the run folder, job IDs and
  resource settings, without credentials. **Open receipt** restores a run; saved
  completion states are not trusted.
- Choose a local folder and **Download results** after every recorded phase and
  merge job succeeds. The app rechecks remote completion before downloading.
  Results go into a new run folder; existing folders are never overwritten.
- Partial downloads remain on disk on failure. Choose a fresh destination to retry.
- **Activity log → Save log** exports troubleshooting details.
- Close the native window or use **Quit app** to stop the local service. The
  window refuses to close during transfers. Submitted cluster jobs continue
  independently. Opening the native app again focuses the existing window.

## Startup and development

The native package includes a pinned standalone Python runtime instead of using
Apple's old system Python. Startup logs are written to `$TMPDIR/Kamiak-DI-launch.log`.
A per-user temporary session file (permissions 0600 on macOS) supports reopening.
The local service listens on a random port on `127.0.0.1`; authenticated requests
use a per-session token and foreign web origins are rejected. Assets are bundled
and load without internet; an SSH connection is required for HPC operations.

For developers, from the repository root:

```bash
cd desktop/web
npm ci
npm run build
cd ../..
python3 desktop/kamiak_web.py
```

For the native shell, install its dependencies and run it in development:

```bash
npm ci --prefix desktop/shell
desktop/shell/node_modules/.bin/electron desktop/shell/main.cjs
```

Build the Apple-silicon native Mac package with the project's checksum-pinned
CPython archive:

```bash
node desktop/build_desktop.mjs /path/to/di-aarch64-apple-darwin.tar.gz
```

This packages the desktop shell, Python backend/runtime, compiled React interface,
merge helper and SHT database. The existing `build_portable.py` remains the legacy
Tk-package builder. It is separate from the new native React package.

React source is in `desktop/web/`; compiled `desktop/web-assets/` must be committed
alongside source changes. The previous Tk interface remains available as a
fallback with `python3 desktop/kamiak_di.py`.

The native source-checkout Mac launcher is built with:

```bash
cc -O2 -arch arm64 -arch x86_64 desktop/macos_launcher.c -o 'desktop/Kamiak DI.app/Contents/MacOS/KamiakDI'
```

The older Kamiak portable packages have not been rebuilt. The new React desktop
package is separate under `dist/native/`. Native launcher changes require re-signing the
source app locally. Development checks cover mocked SSH/Slurm commands, local
API access checks, phase selection and exports. Live Kamiak jobs and Windows
execution still need validation on their respective systems.
