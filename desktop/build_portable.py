"""Build a portable Kamiak DI archive from an official, pinned CPython runtime.

Usage: python3 desktop/build_portable.py mac|windows RUNTIME.tar.gz OUTPUT_DIR
Build the macOS target on an Apple silicon Mac with Xcode command line tools.
"""
import hashlib
import io
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIGEST = {
    'mac': '2f11b8aa3726bdbd87dd9eb76568412faddf4f47e26913fa45d6a2e424972acd',
    'windows': '355ee61b8207b91574491b1c6dc9b0ebb4408fc56bfafe498f3ef439314b52dc',
}


def build(target, archive, output):
    if hashlib.sha256(archive.read_bytes()).hexdigest() != DIGEST[target]:
        raise ValueError('Unexpected runtime checksum')
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        if target == 'mac':
            app = stage/'Kamiak DI.app'
            contents = app/'Contents'
            base = contents/'Resources'
        else:
            base = stage/'Kamiak DI'
        base.mkdir(parents=True)
        with tarfile.open(archive) as tar:
            tar.extractall(base, filter='data')
        for folder, source in [('desktop','kamiak_di.py'), ('tools','merge_cobalt.py')]:
            (base/folder).mkdir()
            shutil.copy2(ROOT/folder/source, base/folder/source)
        for name in ('setup.slurm', 'queue_run.sh'):
            shutil.copy2(ROOT/'desktop'/name, base/'desktop'/name)
        for name in ('pyproject.toml', 'README.md'):
            shutil.copy2(ROOT/name, base/name)
        shutil.copytree(ROOT/'src', base/'src', ignore=shutil.ignore_patterns('__pycache__','*.egg-info','*.pyc'))
        shutil.copytree(ROOT/'tools/sht', base/'tools/sht')
        (base/'hpc').mkdir()
        shutil.copy2(ROOT/'hpc/submit_two_gpu.slurm', base/'hpc/submit_two_gpu.slurm')
        shutil.copytree(ROOT/'SHTdatabase-master', base/'SHTdatabase-master')
        shutil.copy2(ROOT/'LICENSE', base/'LICENSE')
        if target == 'windows':
            # distlib's Windows GUI launcher resolves <launcher_dir> at startup.
            stub = base/'python/Lib/site-packages/pip/_vendor/distlib/w64.exe'
            script = io.BytesIO()
            with zipfile.ZipFile(script, 'w') as z:
                z.writestr('__main__.py', 'import sys, runpy\nfrom pathlib import Path\n'
                           'base = Path(sys.argv[0]).resolve().parent\n'
                           'runpy.run_path(str(base/"desktop/kamiak_di.py"), run_name="__main__")\n')
            (base/'Kamiak DI.exe').write_bytes(stub.read_bytes() +
                b'#!<launcher_dir>\\python\\pythonw.exe\n' + script.getvalue())
            shutil.make_archive(str(output/'Kamiak-DI-Windows-x64'), 'zip', stage, base.name)
        else:
            plist = plistlib.loads((ROOT/'desktop/Kamiak DI.app/Contents/Info.plist').read_bytes())
            (contents/'Info.plist').write_bytes(plistlib.dumps(plist))
            (contents/'MacOS').mkdir()
            launcher = contents/'MacOS/KamiakDI'
            launcher.write_text('#!/bin/sh\nbase="$(cd "$(dirname "$0")/../Resources" && pwd)" || exit 1\n'
                                'unset PYTHONHOME PYTHONPATH\n'
                                'exec "$base/python/bin/python3" -I "$base/desktop/kamiak_di.py"\n')
            launcher.chmod(0o755)
            subprocess.run(['codesign','--force','--sign','-',str(app)],check=True)
            subprocess.run(['/usr/bin/ditto','-c','-k','--keepParent',str(app),
                            str(output/'Kamiak-DI-macOS-arm64.zip')],check=True)


if __name__ == '__main__':
    build(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]).resolve())
