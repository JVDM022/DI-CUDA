"""Resumable, single-process desktop workflow. Existing HPC commands are untouched."""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import os
import shutil
import numpy as np
from .hpc import read_up, ang_header, preprocess, load_master
from . import index_patterns

SCHEMA = 1

@dataclass
class Settings:
    patterns: str
    ang: str
    master: str
    output: str
    resolution: float = 3.0
    binning: int = 2
    sigma: float = 3.0
    batch: int = 32
    dictionary_batch: int = 2048
    keep: int = 2
    limit: int = 128
    device: str = 'auto'
    energy: str = 'mc-global'
    mapping: str = 'raster'


def hardware():
    try:
        import cupy as cp
        devices = []
        for i in range(cp.cuda.runtime.getDeviceCount()):
            p = cp.cuda.runtime.getDeviceProperties(i)
            name = p['name']
            devices.append(name.decode() if isinstance(name, bytes) else str(name))
        return dict(devices=devices, reason='' if devices else 'No CUDA devices visible')
    except Exception as exc:
        return dict(devices=[], reason=f'{type(exc).__name__}: {exc}')


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(temp, path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def mapping(path, calibration, count, mode):
    xy = np.loadtxt(path, comments='#', usecols=(3, 4), ndmin=2)
    if not np.isfinite(xy).all() or len(np.unique(xy, axis=0)) != len(xy):
        raise ValueError('ANG coordinates must be finite and unique')
    dx, dy = calibration.get('XSTEP', 0), calibration.get('YSTEP', 0)
    if not np.isfinite([dx, dy]).all() or min(dx, dy) <= 0:
        raise ValueError('ANG needs positive XSTEP and YSTEP')
    y = (xy[:, 1]-xy[:, 1].min())/dy
    row = np.rint(y).astype(int)
    if not np.allclose(row, y, atol=1e-3):
        raise ValueError('ANG Y coordinates disagree with YSTEP')
    if mode == 'ang-order':
        if count != len(xy):
            raise ValueError('ANG-order mapping requires exactly one UP pattern per ANG row')
        return xy, np.arange(count)
    if mode != 'raster':
        raise ValueError('Unknown mapping mode')
    x = (xy[:, 0]-xy[:, 0].min())/dx
    if not np.allclose(x, np.rint(x), atol=1e-3):
        x = x - .5*(row % 2)
    col = np.rint(x).astype(int)
    nrows = row.max()+1
    if not np.allclose(x, col, atol=1e-3) or count % nrows:
        raise ValueError('Scan is not a supported rectangular UP raster; choose ANG row order only if acquisition uses it')
    width = count//nrows
    ix = row*width + col
    if np.any(col < 0) or np.any(col >= width) or len(np.unique(ix)) != len(ix):
        raise ValueError('Invalid ANG-to-UP mapping')
    return xy, ix


def inspect(s):
    import h5py
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in
           (s.binning, s.batch, s.dictionary_batch, s.keep)) or s.limit < 0:
        raise ValueError('Binning, batch sizes and keep must be positive; limit must be nonnegative')
    if not np.isfinite([s.resolution, s.sigma]).all() or s.resolution <= 0 or s.sigma < 0:
        raise ValueError('Resolution must be positive and sigma nonnegative')
    if s.device not in ('auto', 'cuda', 'cpu') or s.energy not in ('mc-global', 'highest-bin'):
        raise ValueError('Invalid device or energy mode')
    raw = read_up(s.patterns)
    cal = ang_header(s.ang)
    if not np.isfinite(list(cal.values())).all() or cal['z-star'] <= 0:
        raise ValueError('Detector metadata must be finite with positive z-star')
    if raw.shape[1] % s.binning or raw.shape[2] % s.binning:
        raise ValueError('Binning must divide both detector dimensions')
    xy, ix = mapping(s.ang, cal, len(raw), s.mapping)
    if Path(s.master).suffix.lower() == '.sht':
        if not shutil.which('emsoft-di-read-sht'):
            raise ValueError('SHT reader is not installed. Build emsoft-di-read-sht (README), or supply an EMsoft master H5')
    else:
        with h5py.File(s.master, 'r') as f:
            if 'EMData/EBSDmaster/mLPNH' not in f and 'kikuchicuda_master_schema' not in f.attrs:
                raise ValueError('Not a supported EMsoft or KikuchiCUDA master H5')
    return raw, cal, xy, ix


def run(s, progress=lambda stage, done, total: None, cancelled=lambda: False):
    """Flush data before checkpoint advancement; resume replays any uncommitted batch."""
    import h5py
    from orix import sampling
    import kikuchipy as kp
    from .projection import project
    raw, cal, xy, ix = inspect(s)
    out = Path(s.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = out/'.workflow.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError('Run is already locked. If its process crashed, remove .workflow.lock after confirming it stopped')
    os.close(fd)
    try:
        progress('Hashing inputs', 0, 1)
        identity = asdict(s)
        identity.pop('output')
        # Device may change when resuming on a CPU-only machine.
        identity.pop('device')
        identity['schema'] = SCHEMA
        identity['inputs_sha256'] = {k: digest(getattr(s, k)) for k in ('patterns', 'ang', 'master')}
        state_path = out/'state.json'
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state['identity'] != identity:
                raise ValueError('Inputs or settings changed. Select a new output folder')
        else:
            if any(p.name != lock.name for p in out.iterdir()):
                raise ValueError('Choose an empty output folder or an existing app run')
            state = dict(identity=identity, projected=0, matched=0, status='preparing')
            atomic_json(state_path, state)
        device = s.device
        if device == 'auto':
            device = 'cuda' if hardware()['devices'] else 'cpu'
        if device == 'cuda' and not hardware()['devices']:
            raise RuntimeError('CUDA unavailable. Select CPU or automatic device')
        state['device'] = device
        master, energy = load_master(s.master, s.energy)
        rotations = sampling.get_sample_fundamental(method='cubochoric', resolution=s.resolution,
                                                    point_group=master.phase.point_group)
        detector = kp.detectors.EBSDDetector(shape=(raw.shape[1]//s.binning, raw.shape[2]//s.binning),
            pc=[cal[k] for k in ('x-star', 'y-star', 'z-star')], convention='tsl',
            sample_tilt=cal['SampleTiltAngle'], tilt=cal['CameraElevationAngle'],
            azimuthal=cal.get('CameraAzimuthalAngle', 0))
        shape = (rotations.size, *detector.shape)
        count = min(s.limit, len(raw)) if s.limit else len(raw)
        k = min(s.keep, rotations.size)
        state.update(dictionary_count=rotations.size, pattern_count=count, energy=energy,
                     orientation_convention='orix scalar-first quaternion; Bunge radians; EDAX TSL sample frame')
        path = out/'dictionary.npy'
        if not path.exists():
            if shutil.disk_usage(out).free < np.prod(shape)*4 + count*k*24 + 100_000_000:
                raise RuntimeError('Insufficient disk space for dictionary and results; increase binning or sampling degrees')
            d = np.lib.format.open_memmap(path, mode='w+', dtype='float32', shape=shape)
            d.flush()
        else:
            d = np.load(path, mmap_mode='r+')
            if d.shape != shape:
                raise ValueError('Checkpoint dictionary shape changed')
        # Every CUDA run validates the custom projector against kikuchipy.
        if device == 'cuda':
            check = np.unique(np.linspace(0, rotations.size-1, min(6, rotations.size), dtype=int))
            reference = master.get_patterns(rotations[check], detector, compute=True,
                dtype_out='float32', show_progressbar=False).data
            import cupy as cp
            actual = np.concatenate([cp.asnumpy(b) for b in project(master, rotations[check], detector,
                                                                   batch_size=s.batch)])
            scale = max(float(np.max(np.abs(reference))), 1e-12)
            np.testing.assert_allclose(actual/scale, reference/scale, rtol=2e-4, atol=2e-5)
            state['projection_reference_error'] = float(np.max(np.abs(actual-reference))/scale)
        state['status'] = 'projecting'
        atomic_json(state_path, state)
        for a in range(state['projected'], rotations.size, s.batch):
            if cancelled():
                state['status'] = 'paused'; atomic_json(state_path, state); return
            b = min(a+s.batch, rotations.size)
            if device == 'cpu':
                block = master.get_patterns(rotations[a:b], detector, compute=True,
                    dtype_out='float32', show_progressbar=False).data
            else:
                block = cp.asnumpy(next(project(master, rotations[a:b], detector, batch_size=s.batch)))
            d[a:b] = preprocess(block, 1, s.sigma)
            d.flush()
            state['projected'] = b
            atomic_json(state_path, state)
            progress('Building dictionary', b, rotations.size)
        result_path = out/'checkpoint.h5'
        with h5py.File(result_path, 'a') as f:
            if 'scores' not in f:
                f.create_dataset('scores', (count, k), dtype='f4', fillvalue=np.nan)
                f.create_dataset('simulation_indices', (count, k), dtype='i8', fillvalue=-1)
                f.create_dataset('valid', (count,), dtype='?')
                f.create_dataset('pattern_std', (count,), dtype='f4')
            for a in range(state['matched'], count, s.batch):
                if cancelled():
                    state['status'] = 'paused'; atomic_json(state_path, state); return
                b = min(a+s.batch, count)
                block = preprocess(raw[a:b], s.binning, s.sigma)
                std = block.std(axis=(1, 2))
                valid = np.isfinite(block).all(axis=(1, 2)) & (std > 0)
                ids = np.full((b-a, k), -1, dtype='i8')
                scores = np.full((b-a, k), np.nan, dtype='f4')
                if valid.any():
                    result = index_patterns(block[valid], d, center=True, top_k=k, device=device,
                        pattern_batch=s.batch, dictionary_batch=s.dictionary_batch)
                    ids[valid], scores[valid] = result.indices, result.scores
                    if a == 0:
                        ref = reference_scores(block[valid][:3], d, k, s.dictionary_batch)
                        error = float(np.max(np.abs(ref-scores[valid][:3])))
                        if error > 5e-6:
                            raise RuntimeError('Matching failed CPU float64 score reference check')
                        state['matching_reference_error'] = error
                f['simulation_indices'][a:b], f['scores'][a:b] = ids, scores
                f['valid'][a:b], f['pattern_std'][a:b] = valid, std
                f.flush()
                state.update(matched=b, status='indexing')
                atomic_json(state_path, state)
                progress('Indexing patterns', b, count)
        progress('Exporting results', 0, 1)
        export(out, s, master.phase, rotations, xy, ix)
        state['status'] = 'complete'
        state.pop('error', None)
        atomic_json(state_path, state)
        progress('Complete', count, count)
    except Exception as exc:
        if 'state' in locals():
            state.update(status='error', error=f'{type(exc).__name__}: {exc}')
            atomic_json(out/'state.json', state)
        raise
    finally:
        lock.unlink(missing_ok=True)


def export(out, s, phase, rotations, xy, mapping_indices):
    import h5py
    from orix.crystal_map import CrystalMap, PhaseList
    from orix.quaternion import Rotation
    from orix import io
    from copy import deepcopy
    phase = deepcopy(phase)
    # SHT reconstruction carries symmetry but not a metric lattice. In that case
    # retain the acquisition phase's lattice instead of exporting an invented unit cell.
    if np.allclose(phase.structure.lattice.abcABG(), [1, 1, 1, 90, 90, 90]):
        from diffpy.structure import Lattice
        for line in Path(s.ang).read_text().splitlines():
            words = line.lstrip('# ').split()
            if words and words[0] == 'LatticeConstants' and len(words) == 7:
                values = np.array(words[1:], dtype=float)
                if not np.isfinite(values).all() or np.any(values <= 0):
                    raise ValueError('Invalid source ANG lattice constants')
                phase.structure.lattice = Lattice(*values)
                break
    with h5py.File(out/'checkpoint.h5', 'r') as f:
        ids, scores, valid = f['simulation_indices'][...], f['scores'][...], f['valid'][...]
        std = f['pattern_std'][...]
    included = mapping_indices < len(ids)
    ok = np.zeros(len(xy), dtype=bool)
    ok[included] = valid[mapping_indices[included]]
    quats = np.tile([1., 0, 0, 0], (len(xy), 1))
    quats[ok] = rotations.data[ids[mapping_indices[ok], 0]]
    best = np.full(len(xy), np.nan)
    best[ok] = scores[mapping_indices[ok], 0]
    gap = np.full(len(xy), np.nan)
    if scores.shape[1] > 1:
        gap[ok] = scores[mapping_indices[ok], 0]-scores[mapping_indices[ok], 1]
    xmap = CrystalMap(rotations=Rotation(quats), phase_id=np.where(ok, 0, -1),
                     phase_list=PhaseList(phase), x=xy[:, 0], y=xy[:, 1],
                     prop={'ncc': best, 'score_gap': gap}, scan_unit='um')
    io.save(out/'indexed.partial.h5', xmap, overwrite=True)
    os.replace(out/'indexed.partial.h5', out/'indexed.h5')
    # Preserve source scan geometry and phase header; replace indexed row values.
    lines = Path(s.ang).read_text().splitlines()
    header = [line for line in lines if line.startswith('#')]
    import re
    header = [re.sub(r'^(#\s*MaterialName\s+).*', lambda m: m[1]+phase.name, line) for line in header]
    # Use the master symmetry/lattice metadata from orix, preserving source grid/calibration.
    temp = out/'phase-template.ang'
    try:
        # ANG writer requires rectangular maps; use a tiny template for phase metadata.
        template = CrystalMap(rotations=Rotation.identity(4), phase_list=PhaseList(phase),
                              x=np.array([0, 1, 0, 1]), y=np.array([0, 0, 1, 1]))
        io.save(temp, template, overwrite=True)
        phase_lines = temp.read_text().splitlines()
        start = next(i for i, line in enumerate(phase_lines) if re.match(r'#\s*Phase\s+1', line))
        end = next((i for i in range(start+1, len(phase_lines)) if re.match(r'#\s*GRID:', phase_lines[i])), len(phase_lines))
        old = next((i for i, line in enumerate(header) if re.match(r'#\s*Phase\s+\d+', line)), None)
        grid = next((i for i, line in enumerate(header) if re.match(r'#\s*GRID:', line)), None)
        if old is None or grid is None:
            raise ValueError('ANG header must contain Phase and GRID blocks for compatible export')
        header = header[:old] + phase_lines[start:end] + header[grid:]
    finally:
        temp.unlink(missing_ok=True)
    euler = Rotation(quats).to_euler()
    euler[~ok] = 0
    iq = np.zeros(len(xy)); iq[included] = std[mapping_indices[included]]
    table = np.column_stack((euler, xy, iq, np.where(ok, np.nan_to_num(gap), -1.), ok.astype(int), np.zeros((len(xy), 2))))
    np.savetxt(out/'indexed.partial.ang', table, header='\n'.join(header), comments='', fmt='%.9g')
    os.replace(out/'indexed.partial.ang', out/'indexed.ang')
    np.savetxt(out/'quality.csv', np.column_stack((xy, mapping_indices, included, ok, best, gap)),
               delimiter=',', header='x_um,y_um,up_index,included,valid,ncc,score_gap', comments='')
    atomic_json(out/'diagnostics.json', dict(patterns=len(ids), valid=int(valid.sum()),
        mapped_rows=len(xy), mapped_valid=int(ok.sum()), mean_ncc=float(np.nanmean(best)) if ok.any() else None,
        metric='NCC; score gap is not calibrated confidence; IQ column is preprocessed standard deviation',
        h5_format='orix CrystalMap (not vendor acquisition H5)',
        limitations='No orientation refinement. Validate detector calibration and TSL sample frame experimentally.'))


def reference_scores(patterns, dictionary, k, batch):
    """Independent streamed float64 NCC reference; only sampled experimental rows."""
    def normalize(a):
        a = np.asarray(a, dtype='float64').reshape(len(a), -1)
        a = a-a.mean(axis=1, keepdims=True)
        return a/np.linalg.norm(a, axis=1, keepdims=True)
    p = normalize(patterns)
    best = np.empty((len(p), 0))
    for a in range(0, len(dictionary), batch):
        scores = p @ normalize(dictionary[a:a+batch]).T
        best = np.sort(np.concatenate([best, scores], axis=1), axis=1)[:, -k:]
    return best[:, ::-1]
