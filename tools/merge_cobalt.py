"""Combine completed S11 cobalt runs in the source ANG row order."""
import json
import sys
from pathlib import Path
import tempfile

import h5py
import numpy as np


def select(alpha, beta):
    a = np.isfinite(alpha)
    b = np.isfinite(beta)
    phase = np.full(alpha.shape, -1, dtype=np.int16)
    phase[a & (~b | (alpha > beta))] = 1
    phase[b & (~a | (beta > alpha))] = 2
    return phase


def main(alpha_path, beta_path, output, pair=None):
    if output.exists():
        raise FileExistsError(output)
    with h5py.File(alpha_path) as a, h5py.File(beta_path) as b:
        meta = []
        for source, expected in ((a, 'alpha-cobalt'), (b, 'beta-cobalt')):
            info = json.loads(source.attrs['metadata_json'])
            if not source.attrs['complete']:
                raise ValueError('Incomplete phase result')
            if pair is None and (expected not in Path(info['master']).name or info['energy']['voltage_kV'] != 20):
                raise ValueError(f'Expected completed 20 kV {expected} run')
            meta.append(info)
        ids = (1, 2) if pair is None else tuple(int(value) for value in pair[:2])
        if pair is not None:
            if len(set(ids)) != 2 or min(ids) < 1:
                raise ValueError('Expected two distinct positive ANG phase IDs')
            for info, master in zip(meta, pair[2:]):
                if Path(info['master']).resolve() != Path(master).resolve():
                    raise ValueError('Unexpected master in phase result')
            if meta[0]['energy']['voltage_kV'] != meta[1]['energy']['voltage_kV']:
                raise ValueError('Phase master voltages differ')
        for key in ('patterns', 'ang', 'resolution', 'binning', 'npa', 'background_sigma', 'limit'):
            if meta[0][key] != meta[1][key]:
                raise ValueError(f'Mismatched {key}')
        for key in ('pattern_index', 'xy_um', 'included_in_run'):
            if not np.array_equal(a['ang_mapping'][key][:], b['ang_mapping'][key][:]):
                raise ValueError(f'Mismatched ANG mapping: {key}')
        ix = a['ang_mapping/pattern_index'][:]
        if not a['ang_mapping/included_in_run'][:].all():
            raise ValueError('Expected full scans')
        scores = [np.where(src['valid'][:], src['scores'][:, 0], np.nan)[ix] for src in (a, b)]
        phase = select(*scores)
        source_ang = Path(meta[0]['ang'])
        data = np.loadtxt(source_ang, comments='#', ndmin=2)
        if not np.allclose(data[:, 3:5], a['ang_mapping/xy_um'][:]):
            raise ValueError('ANG coordinates changed')
        data[:, :3] = 0
        for pid, src in enumerate((a, b), 1):
            mask = phase == pid
            data[mask, :3] = src['best_euler_bunge_radians'][:][ix[mask]]
        data[:, 6] = 0
        phase = np.select([phase == 1, phase == 2], ids, default=-1)
        data[:, 7] = phase
        if data.shape[1] >= 10:
            data[:, 9] = -1
        with tempfile.TemporaryDirectory(dir=output.parent) as stage:
            ready = Path(stage) / 'results'
            ready.mkdir()
            with source_ang.open() as original, (ready/'indexed.ang').open('w') as out:
                for line in original:
                    if not line.startswith('#'):
                        break
                    out.write(line)
                np.savetxt(out, data, fmt=['%.10g']*data.shape[1])
            with h5py.File(ready/'phase_scores.h5', 'w') as out:
                out.attrs['row_order'] = 'source ANG'
                out.attrs['alpha_source'] = str(alpha_path)
                out.attrs['beta_source'] = str(beta_path)
                labels = ('alpha_ncc', 'beta_ncc') if pair is None else tuple(f'phase_{pid}_ncc' for pid in ids)
                for pid, info in zip(ids, meta):
                    out.attrs[f'phase_{pid}_master'] = info['master']
                for key, value in zip((*labels, 'phase', 'pattern_index'), (*scores, phase, ix)):
                    out.create_dataset(key, data=value)
            ready.rename(output)
    print(f'Combined phase map: {output}/indexed.ang')


if __name__ == '__main__':
    if len(sys.argv) not in (4, 8):
        sys.exit('Usage: merge_cobalt.py FIRST.h5 SECOND.h5 OUTPUT [ID1 ID2 MASTER1 MASTER2]')
    main(*(Path(arg).resolve() for arg in sys.argv[1:4]), pair=sys.argv[4:] or None)
