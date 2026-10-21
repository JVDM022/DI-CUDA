"""Generate a dynamical master in Python/CuPy from existing EMsoft MC data."""
import argparse
import json
import multiprocessing
import os
from pathlib import Path
import numpy as np
import h5py

from .physics import read_crystal, Lookup, simulate_batch
from .model import lambert_directions, symmetry_reduce, read_depth_profiles, corrected_weights


def worker(options, device, start, stop):
    folder = Path(options['work'])
    if options['backend'] == 'cuda':
        import cupy as xp
        xp.cuda.Device(device).use()
    else:
        xp = np
    with h5py.File(options['monte_carlo']) as f:
        cell = read_crystal(f['CrystalData'])
        energies, dz, thickness, profiles = read_depth_profiles(f)
    indices = options['energy_indices']
    rays = np.load(folder/'directions.npy', mmap_mode='r')
    output = np.lib.format.open_memmap(folder/f'worker-{device}.npy', mode='w+',
        dtype='float32', shape=(len(cell.positions), len(indices), stop-start))
    stats = dict(max_strong=0, max_weak=0, directions=stop-start, complete=False)
    for out_energy, index in enumerate(indices):
        lookup = Lookup(cell, float(energies[index]), options['dmin'])
        weights = corrected_weights(profiles[index], lookup, dz)
        print(f'Master worker {device}: energy {energies[index]:.4f} keV, {len(lookup.hkl)} candidate beams', flush=True)
        for a in range(start, stop, options['direction_batch']):
            b = min(a+options['direction_batch'], stop)
            value, beam_stats = simulate_batch(lookup, rays[a:b], dz, weights, float(thickness[index]),
                xp=xp, max_beams=options['max_beams'], bethe=options['bethe'])
            output[:, out_energy, a-start:b-start] = value.T
            stats['max_strong'] = max(stats['max_strong'], *(s['strong'] for s in beam_stats))
            stats['max_weak'] = max(stats['max_weak'], *(s['weak'] for s in beam_stats))
            output.flush()
            if a == start or b == stop or (a-start) % (100*options['direction_batch']) == 0:
                print(f'Master worker {device}: {b-start:,}/{stop-start:,} rays, energy {out_energy+1}/{len(indices)}', flush=True)
    stats['complete'] = True
    stats['start'], stats['stop'] = start, stop
    (folder/f'worker-{device}.json').write_text(json.dumps(stats, indent=2))


def write_master(options, ranges, inverse):
    target = Path(options['output'])
    partial = target.with_name(target.name+'.partial')
    folder = Path(options['work'])
    indices = options['energy_indices']
    with h5py.File(options['monte_carlo']) as source, h5py.File(partial, 'x') as out:
        out.attrs.update(kikuchicuda_master_schema=1, complete=False,
            generator='KikuchiCUDA Python/CuPy dynamical master',
            scientifically_validated=False, settings_json=json.dumps(options))
        source.copy('CrystalData', out)
        nml = out.create_group('NMLparameters')
        source.copy(source['NMLparameters/MCCLNameList'], nml, name='MCCLNameList')
        group = nml.create_group('EBSDMasterNameList')
        group['npx'] = [options['npx']]
        group['dmin'] = [options['dmin']]
        group['useEnergyWeighting'] = [0]
        group['combinesites'] = [0]
        mc = out.create_group('EMData').create_group('MCOpenCL')
        # Only copy Monte Carlo results, never a pre-existing master pattern.
        for name, dataset in source['EMData/MCOpenCL'].items():
            if name in ('accum_e', 'accum_z', 'accumSP'):
                data = np.take(dataset[...], indices, axis=-1)
                mc.create_dataset(name, data=data, compression='gzip', shuffle=True)
            elif name == 'numEbins':
                mc[name] = [len(indices)]
            else:
                source.copy(dataset, mc, name=name)
        energies, _, _, _ = read_depth_profiles(source)
        selected = energies[indices]
        if len(indices) == 1:
            nml['MCCLNameList/Ehistmin'][...] = selected[0]
        master = out['EMData'].create_group('EBSDmaster')
        master['EkeVs'] = selected.astype('float32')
        master['numEbins'] = [len(selected)]
        master['BetheParameters'] = np.asarray(options['bethe'], dtype='float32')
        nsites = len(source['CrystalData/Atomtypes'])
        master['numset'] = [nsites]
        shape = (nsites, len(indices), 2*options['npx']+1, 2*options['npx']+1)
        north = master.create_dataset('mLPNH', shape, dtype='float32', compression='gzip', shuffle=True)
        south = master.create_dataset('mLPSH', shape, dtype='float32', compression='gzip', shuffle=True)
        shards = []
        for device, (a, b) in enumerate(ranges):
            stats = json.loads((folder/f'worker-{device}.json').read_text())
            if not stats['complete'] or (stats['start'], stats['stop']) != (a, b):
                raise ValueError('Incomplete master shard')
            data = np.load(folder/f'worker-{device}.npy', mmap_mode='r')
            if data.shape != (nsites, len(indices), b-a) or not np.isfinite(data).all():
                raise ValueError('Invalid master shard shape or intensity')
            shards.append(data)
        for energy in range(len(indices)):
            values = np.concatenate([data[:, energy] for data in shards], axis=1)
            north[:, energy] = values[:, inverse[0]]
            south[:, energy] = values[:, inverse[1]]
        # Truthful provenance: do not impersonate EMEBSDmaster.f90 to bypass
        # kikuchipy's generator-name check. Our reader uses the schema marker.
        out.create_group('PythonMaster')['ProgramName'] = np.asarray(['KikuchiCUDA Python'], dtype=h5py.string_dtype())
        out.attrs['complete'] = True
    os.link(partial, target)
    partial.unlink()
    print(f'Master complete: {target}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--monte-carlo', type=Path, required=True,
                        help='Existing EMsoft MC HDF5 with CrystalData and accum_z; no master required')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=('cuda', 'cpu-reference'), default='cuda')
    parser.add_argument('--devices', type=int, choices=(1, 2), default=2)
    parser.add_argument('--npx', type=int, default=16, help='Half width; master side = 2*npx+1 (pilot default)')
    parser.add_argument('--dmin', type=float, default=.05, help='Smallest plane spacing in nm')
    parser.add_argument('--energy-index', type=int, help='Pilot: simulate one zero-based MC energy bin; default all')
    parser.add_argument('--direction-batch', type=int, default=4)
    parser.add_argument('--max-beams', type=int, default=512, help='Fail above this strong-beam count; never truncate silently')
    parser.add_argument('--bethe', type=float, nargs=4, default=[4.,8.,50.,1.], metavar=('C1','C2','C3','SGCUTOFF'))
    parser.add_argument('--no-symmetry', action='store_true', help='Compute all sampled rays independently')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if min(args.npx, args.direction_batch, args.max_beams) < 1 or not np.isfinite(args.dmin) or args.dmin <= 0:
        parser.error('Positive finite grid, batch, beam and dmin values required')
    if not np.isfinite(args.bethe).all() or not (0 < args.bethe[0] <= args.bethe[1] <= args.bethe[2]) or args.bethe[3] <= 0:
        parser.error('Require 0 < c1 <= c2 <= c3 and positive double-diffraction cutoff')
    source = args.monte_carlo.resolve(strict=True)
    target = args.output.resolve()
    if target.exists() or target.with_name(target.name+'.partial').exists():
        parser.error('Output or partial file already exists; choose a new output path')
    with h5py.File(source) as f:
        cell = read_crystal(f['CrystalData'])
        energies, dz, _, _ = read_depth_profiles(f)
    indices = list(range(len(energies))) if args.energy_index is None else [args.energy_index]
    if min(indices) < 0 or max(indices) >= len(energies):
        parser.error('Energy index is outside the Monte Carlo bins')
    rays = lambert_directions(args.npx)
    symmetry = np.eye(3)[None] if args.no_symmetry else cell.symmetry
    directions, inverse = symmetry_reduce(rays, symmetry)
    print(f'Python dynamical master: {2*args.npx+1} square pixels, both hemispheres; '
          f'{len(directions):,} unique rays; energies {energies[indices]} keV; dz={dz} nm', flush=True)
    if args.dry_run:
        return
    if args.backend == 'cuda':
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() < args.devices:
            raise RuntimeError('Requested GPUs are not visible; allocate a GPU node')
    work = target.with_name(target.name+'.work')
    work.mkdir(parents=True, exist_ok=False)
    np.save(work/'directions.npy', directions)
    np.save(work/'inverse.npy', inverse)
    options = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    options.update(monte_carlo=str(source), output=str(target), work=str(work), energy_indices=indices)
    (work/'settings.json').write_text(json.dumps(options, indent=2))
    count = min(args.devices, len(directions))
    ranges = [(len(directions)*i//count, len(directions)*(i+1)//count) for i in range(count)]
    context = multiprocessing.get_context('spawn')
    processes = [context.Process(target=worker, args=(options, device, a, b))
                 for device, (a, b) in enumerate(ranges)]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    if any(process.exitcode != 0 for process in processes):
        raise RuntimeError('Python master worker failed; partial work preserved; no final master published')
    write_master(options, ranges, inverse)


if __name__ == '__main__':
    main()
