"""Headless, disk-backed kikuchipy dictionary generation and CUDA matching.

Run with python -m emsoft_di.hpc --help. Output is HDF5, not vendor ANG.
"""
import argparse
import hashlib
import json
import multiprocessing as multiprocessing
import os
from pathlib import Path
import struct

import numpy as np


def read_up(path):
    path = Path(path)
    with path.open('rb') as stream:
        header = stream.read(16)
    if len(header) != 16:
        raise ValueError('Truncated UP header')
    version, width, height, offset = struct.unpack('<4i', header)
    if version not in (1, 2, 3, 4) or not (0 < width <= 5000 and 0 < height <= 5000) or offset < 16:
        raise ValueError('Unsupported UP header')
    dtype = np.dtype('u1' if path.suffix.lower() == '.up1' else '<u2')
    stride = width * height * dtype.itemsize
    size = path.stat().st_size - offset
    if size <= 0 or size % stride:
        raise ValueError('UP data length does not match header')
    return np.memmap(path, mode='r', offset=offset, dtype=dtype,
                     shape=(size // stride, height, width))


def ang_header(path):
    values = {}
    with open(path) as stream:
        for line in stream:
            if not line.startswith('#'):
                break
            words = line[1:].replace(':', ' ').split()
            if len(words) >= 2:
                try:
                    values[words[0]] = float(words[1])
                except ValueError:
                    pass
    required = ('x-star', 'y-star', 'z-star', 'SampleTiltAngle', 'CameraElevationAngle')
    for key in required:
        if key not in values:
            raise ValueError(f'Missing detector metadata: {key}')
    return values


def preprocess(data, binning, sigma):
    from scipy.ndimage import gaussian_filter
    data = np.asarray(data, dtype=np.float32)
    n, h, w = data.shape
    if h % binning or w % binning:
        raise ValueError('Detector dimensions must be divisible by binning')
    data = data.reshape(n, h // binning, binning, w // binning, binning).mean(axis=(2, 4))
    if sigma:
        data = data - gaussian_filter(data, sigma=(0, sigma, sigma), mode='reflect')
    return np.ascontiguousarray(data, dtype=np.float32)


def scan_layout(ang_path, calibration, pattern_count):
    """Map ANG scan coordinates onto UP pattern indices and hex-grid slots."""
    rows = np.loadtxt(ang_path, comments='#', usecols=(3, 4), ndmin=2)
    x, y = rows.T
    dx, dy = calibration['XSTEP'], calibration['YSTEP']
    row = np.rint((y-y.min())/dy).astype(int)
    col_float = (x-x.min())/dx - 0.5*(row % 2)
    col = np.rint(col_float).astype(int)
    if not np.allclose(col_float, col, atol=1e-3):
        raise ValueError('ANG coordinates do not match expected alternating hex rows')
    nrows = row.max()+1
    if pattern_count % nrows:
        raise ValueError('UP count is not a whole number of raster rows')
    width = pattern_count // nrows
    ix = row*width + col
    if np.any(col < 0) or np.any(col >= width) or np.any(ix >= pattern_count) or len(np.unique(ix)) != len(ix):
        raise ValueError('ANG-to-UP mapping has invalid or duplicate indices')
    return rows, row, col, ix, nrows, width


def build_npa_neighbors(row, col, pattern_index, nrows, width, pattern_count,
                        x_step, y_step):
    """Return center + six nearest hex-scan neighbors for each UP pattern."""
    slots = np.full((nrows, width), -1, dtype=np.int64)
    slots[row, col] = pattern_index
    neighbors = np.full((pattern_count, 7), -1, dtype=np.int64)
    # Patterns absent from the ANG coordinate table stay unchanged.
    neighbors[:, 0] = np.arange(pattern_count)
    for r, c, center in zip(row, col, pattern_index):
        found = [int(center)]
        x_center = c + 0.5 * (r % 2)
        for rr in range(max(0, r-1), min(nrows, r+2)):
            for cc in range(max(0, c-1), min(width, c+2)):
                other = int(slots[rr, cc])
                if other < 0 or other == center:
                    continue
                delta_x = (cc + 0.5 * (rr % 2) - x_center) * x_step
                delta_y = (rr-r) * y_step
                if delta_x*delta_x + delta_y*delta_y <= x_step*x_step*1.001:
                    found.append(other)
        neighbors[center, :len(found)] = found
    return neighbors


def preprocess_with_npa(raw, indices, binning, sigma, neighbor_indices):
    """Preprocess patterns, then average each with its mapped scan neighbors."""
    neighbors = neighbor_indices[indices]
    present = neighbors >= 0
    safe = np.where(present, neighbors, indices[:, None])
    prepared = preprocess(raw[safe.reshape(-1)], binning, sigma)
    prepared = prepared.reshape(len(indices), neighbors.shape[1], *prepared.shape[1:])
    weights = present[:, :, None, None]
    return np.ascontiguousarray(
        (prepared * weights).sum(axis=1) / present.sum(axis=1)[:, None, None],
        dtype=np.float32)


def load_master(path, mode):
    """Use kikuchipy's reader to retain master axis conventions and phase."""
    import h5py
    import kikuchipy as kp
    with h5py.File(path, 'r') as f:
        is_python_master = 'kikuchicuda_master_schema' in f.attrs
    if is_python_master:
        from .master.reader import load_python_master
        return load_python_master(path, mode)
    with h5py.File(path, 'r') as f:
        energies = f['EMData/EBSDmaster/EkeVs'][...].reshape(-1)
        weighted = bool(f['NMLparameters/EBSDMasterNameList/useEnergyWeighting'][0])
        voltage = float(f['NMLparameters/MCCLNameList/EkeV'][0])
        if mode == 'mc-global':
            if weighted:
                raise ValueError('Master already energy-weighted; select highest-bin mode to avoid double weighting')
            ds = f['EMData/MCOpenCL/accum_e']
            if ds.ndim != 3 or ds.shape[-1] != len(energies):
                raise ValueError('MC energy bins do not match master')
            weights = np.array([np.sum(ds[:, :, i], dtype=np.float64) for i in range(len(energies))])
            if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
                raise ValueError('Invalid MC weights')
            weights /= weights.sum()
        else:
            weights = np.zeros(len(energies))
            weights[np.argmax(energies)] = 1
    master = None
    total = None
    for energy, weight in zip(energies, weights):
        if weight == 0:
            continue
        part = kp.load(str(path), projection='lambert', hemisphere='both', energy=float(energy))
        if master is None:
            master = part
            total = np.zeros_like(part.data, dtype=np.float32)
        total += np.asarray(part.data, dtype=np.float32) * np.float32(weight)
    master.data = total
    return master, dict(mode=mode, energies_keV=energies.tolist(), weights=weights.tolist(),
                        voltage_kV=voltage, approximation='Spatially integrated MC weights, not detector-dependent weights')


def build_dictionary(args, output, raw, calibration):
    import dask
    import kikuchipy as kp
    from orix import sampling
    master, energy_info = load_master(args.master, args.energy_mode)
    rotations = sampling.get_sample_fundamental(
        method='cubochoric', resolution=args.resolution, point_group=master.phase.point_group)
    detector = kp.detectors.EBSDDetector(
        shape=(raw.shape[1] // args.binning, raw.shape[2] // args.binning),
        pc=[calibration[k] for k in ('x-star', 'y-star', 'z-star')], convention='tsl',
        sample_tilt=calibration['SampleTiltAngle'], tilt=calibration['CameraElevationAngle'],
        azimuthal=calibration.get('CameraAzimuthalAngle', 0))
    count = rotations.size
    shape = (count, *detector.shape)
    print(f'Dictionary: {count:,} orientations, {shape[1:]} pixels, {np.prod(shape)*4/1e9:.2f} GB on disk', flush=True)
    data = np.lib.format.open_memmap(output / 'dictionary.npy', mode='w+', dtype='float32', shape=shape)
    data.flush()
    del data
    np.save(output / 'quaternions.npy', rotations.data)
    np.save(output / 'eulers.npy', rotations.to_euler())
    from .projection import projection_inputs
    north, south, rays = projection_inputs(master, detector)
    for name, values in (('north', north), ('south', south), ('rays', rays)):
        np.save(output / f'{name}.npy', values)
    check_indices = np.unique(np.linspace(0, count-1, min(6, count), dtype=int))
    with dask.config.set(scheduler='threads', num_workers=args.cpu_threads):
        reference = master.get_patterns(rotations[check_indices], detector,
            compute=True, dtype_out='float32', show_progressbar=False).data
    np.save(output / 'projection_check_indices.npy', check_indices)
    np.save(output / 'projection_reference.npy', reference)
    options = dict(output=str(output), projection_batch=args.projection_batch,
                   background_sigma=args.background_sigma)
    context = multiprocessing.get_context('spawn')
    workers = [context.Process(target=project_worker, args=(options, i,
               count*i//args.devices, count*(i+1)//args.devices)) for i in range(args.devices)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    if any(worker.exitcode != 0 for worker in workers):
        raise RuntimeError('GPU projection or reference validation failed; no matching started')
    return dict(dictionary_count=count, energy=energy_info, detector=str(detector),
                projection_backend='CUDA on all requested GPUs; checked against kikuchipy 0.13.1 CPU',
                quaternion_convention='orix scalar-first; EDAX TSL sample frame',
                euler_convention='orix Rotation.to_euler(), Bunge radians')


def project_worker(options, device_id, start, stop):
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter
    from orix.quaternion import Rotation
    from .simulation import generate_patterns
    output = Path(options['output'])
    north, south, rays = (np.load(output/f'{name}.npy') for name in ('north', 'south', 'rays'))
    rotations = Rotation(np.load(output/'quaternions.npy'))
    with cp.cuda.Device(device_id):
        check_indices = np.load(output/'projection_check_indices.npy')
        reference = np.load(output/'projection_reference.npy')
        check = cp.asnumpy(next(generate_patterns(north, south,
            rotations[check_indices].to_matrix(), rays, batch_size=len(check_indices), geometry_dtype='float64')))
        scale = max(float(np.max(np.abs(reference))), 1e-12)
        np.testing.assert_allclose(check/scale, reference/scale, rtol=2e-4, atol=2e-5,
                                   err_msg='CUDA projection differs from kikuchipy CPU')
        error = float(np.max(np.abs(check-reference))/scale)
        (output/f'projection-check-gpu-{device_id}.json').write_text(json.dumps(
            dict(max_absolute_error_over_peak=error, passed=True)))
        print(f'GPU {device_id}: projection reference PASSED (relative peak error {error:.3g})', flush=True)
        data = np.load(output/'dictionary.npy', mmap_mode='r+')
        generated = generate_patterns(north, south, rotations[start:stop].to_matrix(),
            rays, batch_size=options['projection_batch'], geometry_dtype='float64')
        a = start
        for block in generated:
            if options['background_sigma']:
                block -= gaussian_filter(block, sigma=(0, options['background_sigma'], options['background_sigma']), mode='reflect')
            flat = block.reshape(len(block), -1)
            norm = cp.linalg.norm(flat-flat.mean(axis=1, keepdims=True), axis=1)
            if not bool(cp.isfinite(block).all()) or bool((norm == 0).any()):
                raise ValueError('Invalid or constant dictionary patterns')
            b = a + len(block)
            data[a:b] = cp.asnumpy(block)
            data.flush()
            a = b
            print(f'GPU {device_id} projected {a-start:,}/{stop-start:,}', flush=True)
        if a != stop:
            raise RuntimeError('Incomplete projection')


def match_worker(options, device_id, start, stop):
    import cupy as cp
    import h5py
    from . import index_patterns
    output = Path(options['output'])
    raw = read_up(options['patterns'])
    dictionary = np.load(output / 'dictionary.npy', mmap_mode='r')
    neighbor_indices = (np.load(output / 'npa_neighbors.npy', mmap_mode='r')
                       if options['npa'] else None)
    k = min(options['keep_n'], len(dictionary))
    shard = output / f'gpu-{device_id}.h5'
    with cp.cuda.Device(device_id), h5py.File(shard, 'x') as f:
        # Keep the complete simulation dictionary resident for this worker's
        # entire scan range. The two GPU workers each make their own device copy.
        required_bytes = int(np.prod(dictionary.shape)) * np.dtype(np.float32).itemsize
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        if required_bytes > int(free_bytes * 0.8):
            raise RuntimeError(
                f'GPU {device_id} has {free_bytes / 2**30:.2f} GiB free, but the '
                f'dictionary needs {required_bytes / 2**30:.2f} GiB; refusing to '
                'start matching without keeping the dictionary in VRAM'
            )
        dictionary = cp.asarray(dictionary, dtype=cp.float32)
        cp.cuda.get_current_stream().synchronize()
        cached_free, _ = cp.cuda.runtime.memGetInfo()
        print(
            f'GPU {device_id}: dictionary resident in VRAM '
            f'({required_bytes / 2**30:.2f} GiB; '
            f'{(total_bytes - cached_free) / 2**30:.2f} GiB allocated on device)',
            flush=True,
        )
        f.attrs.update(start=start, stop=stop, complete=False, run_id=options['run_id'])
        indices = f.create_dataset('simulation_indices', (stop-start, k), dtype='i8', fillvalue=-1)
        scores = f.create_dataset('scores', (stop-start, k), dtype='f4', fillvalue=np.nan)
        valid_ds = f.create_dataset('valid', (stop-start,), dtype='?')
        for a in range(start, stop, options['pattern_batch']):
            b = min(a + options['pattern_batch'], stop)
            if neighbor_indices is None:
                block = preprocess(raw[a:b], options['binning'], options['background_sigma'])
            else:
                block = preprocess_with_npa(raw, np.arange(a, b), options['binning'],
                    options['background_sigma'], neighbor_indices)
            flat = block.reshape(len(block), -1)
            centered = flat - flat.mean(axis=1, keepdims=True)
            valid = np.isfinite(flat).all(1) & (np.linalg.norm(centered, axis=1) > 0)
            valid_ds[a-start:b-start] = valid
            if valid.any():
                result = index_patterns(block[valid], dictionary, top_k=k, center=True,
                    dictionary_batch=options['dictionary_batch'], pattern_batch=options['pattern_batch'])
                ix = np.full((b-a, k), -1, dtype=np.int64)
                sc = np.full((b-a, k), np.nan, dtype=np.float32)
                ix[valid], sc[valid] = result.indices, result.scores
                indices[a-start:b-start], scores[a-start:b-start] = ix, sc
            f.flush()
            print(f'GPU {device_id}: {b-start:,}/{stop-start:,}', flush=True)
        f.attrs['complete'] = True


def add_scan_mapping(f, args, count, calibration):
    # This dataset uses the previously validated OIM row-stride mapping.
    full_count = read_up(args.patterns).shape[0]
    rows, row, col, ix, _, _ = scan_layout(args.ang, calibration, full_count)
    group = f.create_group('ang_mapping')
    group.create_dataset('pattern_index', data=ix)
    group.create_dataset('xy_um', data=rows)
    group.create_dataset('included_in_run', data=ix < count)
    group.attrs['status'] = 'Coordinate mapping; orientation conventions require experimental validation'


def merge(args, ranges, count, metadata, calibration):
    import h5py
    output = Path(args.output)
    k = min(args.keep_n, metadata['dictionary_count'])
    rotations = np.load(output / 'quaternions.npy')
    eulers = np.load(output / 'eulers.npy')
    partial = output / 'indexed.partial.h5'
    with h5py.File(partial, 'x') as out:
        out.attrs['complete'] = False
        out.attrs['metadata_json'] = json.dumps(metadata)
        ids = out.create_dataset('simulation_indices', (count, k), dtype='i8')
        scores = out.create_dataset('scores', (count, k), dtype='f4')
        valid = out.create_dataset('valid', (count,), dtype='?')
        best_q = out.create_dataset('best_quaternion', (count, 4), dtype='f8', fillvalue=np.nan)
        best_e = out.create_dataset('best_euler_bunge_radians', (count, 3), dtype='f8', fillvalue=np.nan)
        out.create_dataset('dictionary_quaternions', data=rotations)
        out.create_dataset('dictionary_euler_bunge_radians', data=eulers)
        for device, (start, stop) in enumerate(ranges):
            with h5py.File(output / f'gpu-{device}.h5', 'r') as f:
                if not f.attrs['complete'] or f.attrs['start'] != start or f.attrs['stop'] != stop or f.attrs['run_id'] != metadata['run_id']:
                    raise ValueError('Invalid or incomplete worker output')
                for a in range(start, stop, 4096):
                    b = min(a+4096, stop)
                    sl = slice(a-start, b-start)
                    ix, ok = f['simulation_indices'][sl], f['valid'][sl]
                    if np.any(ix[ok] < 0) or np.any(ix[ok] >= len(rotations)):
                        raise ValueError('Worker returned invalid dictionary indices')
                    ids[a:b], scores[a:b], valid[a:b] = ix, f['scores'][sl], ok
                    q, e = np.full((b-a, 4), np.nan), np.full((b-a, 3), np.nan)
                    q[ok], e[ok] = rotations[ix[ok, 0]], eulers[ix[ok, 0]]
                    best_q[a:b], best_e[a:b] = q, e
        add_scan_mapping(out, args, count, calibration)
        out.attrs['complete'] = True
    os.link(partial, output / 'indexed.h5')  # Refuse to overwrite any existing result.
    partial.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(os.environ.get('KIKUCHIPY_CUDA_ROOT', str(Path.home() / 'kikuchipyCUDA')))
    parser.add_argument('--patterns', type=Path, default=base/'patterns.up1')
    parser.add_argument('--master', type=Path, default=base/'master.h5')
    parser.add_argument('--ang', type=Path, default=base/'scan.ang')
    parser.add_argument('--output', type=Path, required=True, help='New directory, must not already exist')
    parser.add_argument('--devices', type=int, default=2, choices=(1, 2))
    parser.add_argument('--limit', type=int, default=128, help='Patterns to index; 0 means full scan')
    parser.add_argument('--resolution', type=float, default=3.0, help='Orientation sampling degrees; 3 is a coarse pilot')
    parser.add_argument('--binning', type=int, default=2)
    parser.add_argument('--background-sigma', type=float, default=3., help='Gaussian subtraction sigma in binned pixels; 0 disables')
    parser.add_argument('--npa', action='store_true',
                        help='Average each experimental pattern with its six nearest hex-grid scan neighbors')
    parser.add_argument('--energy-mode', choices=('mc-global', 'highest-bin'), default='mc-global')
    parser.add_argument('--keep-n', type=int, default=20)
    parser.add_argument('--pattern-batch', type=int, default=256)
    parser.add_argument('--dictionary-batch', type=int, default=16384)
    parser.add_argument('--projection-batch', type=int, default=128)
    parser.add_argument('--cpu-threads', type=int, default=8)
    parser.add_argument('--dry-run', action='store_true', help='Read inputs and plan; do not create dictionary or require CUDA')
    args = parser.parse_args()
    for key in ('patterns', 'master', 'ang'):
        setattr(args, key, getattr(args, key).resolve(strict=True))
    args.output = args.output.resolve()
    if args.output.exists():
        parser.error('Output directory already exists; select a new directory')
    if min(args.binning, args.keep_n, args.pattern_batch, args.dictionary_batch, args.projection_batch, args.cpu_threads) < 1 or args.resolution <= 0 or args.limit < 0 or args.background_sigma < 0:
        parser.error('Invalid batch, sampling, binning, or limit setting')
    raw = read_up(args.patterns)
    calibration = ang_header(args.ang)
    if raw.shape[1] % args.binning or raw.shape[2] % args.binning:
        parser.error('Binning must divide the pattern dimensions')
    count = min(args.limit, len(raw)) if args.limit else len(raw)
    print(f'UP data: {raw.shape}; run: {count:,} patterns; detector: {calibration}', flush=True)
    if args.dry_run:
        import h5py
        with h5py.File(args.master) as f:
            print('Master:', f['EMData/EBSDmaster/mLPNH'].shape,
                  'energies:', f['EMData/EBSDmaster/EkeVs'][...], flush=True)
        return
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() < args.devices:
        raise RuntimeError('Requested GPUs are not visible; run inside a Slurm GPU allocation')
    # Do not create a CUDA context before spawning GPU workers.
    args.output.mkdir(parents=True)
    options = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    options['run_id'] = hashlib.sha256(os.urandom(32)).hexdigest()
    if args.npa:
        _, row, col, ix, nrows, width = scan_layout(args.ang, calibration, len(raw))
        neighbors = build_npa_neighbors(row, col, ix, nrows, width, len(raw),
                                        calibration['XSTEP'], calibration['YSTEP'])
        np.save(args.output/'npa_neighbors.npy', neighbors)
        print('NPA enabled: equal-weight average of each pattern and its available six nearest hex-grid neighbors', flush=True)
    metadata = dict(options)
    metadata.update(build_dictionary(args, args.output, raw, calibration))
    (args.output/'run.json').write_text(json.dumps(metadata, indent=2))
    ranges = [(count*i//args.devices, count*(i+1)//args.devices) for i in range(args.devices)]
    context = multiprocessing.get_context('spawn')
    processes = [context.Process(target=match_worker, args=(options, i, a, b))
                 for i, (a, b) in enumerate(ranges)]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    if any(p.exitcode != 0 for p in processes):
        raise RuntimeError('GPU worker failed; shards preserved, no final result published')
    merge(args, ranges, count, metadata, calibration)
    print(f'Completed: {args.output / "indexed.h5"}', flush=True)


if __name__ == '__main__':
    main()
