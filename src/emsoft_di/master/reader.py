"""Read truthfully identified Python-generated masters into kikuchipy."""
import h5py
import numpy as np


def load_python_master(path, mode):
    import kikuchipy as kp
    from orix.crystal_map import Phase
    if mode not in ('mc-global', 'highest-bin'):
        raise ValueError('Unknown master energy mode')
    with h5py.File(path) as f:
        if f.attrs.get('kikuchicuda_master_schema') != 1 or not f.attrs.get('complete', False):
            raise ValueError('Unsupported or incomplete Python master')
        master = f['EMData/EBSDmaster']
        energies = master['EkeVs'][...]
        weights = np.zeros(len(energies))
        if mode == 'highest-bin':
            weights[np.argmax(energies)] = 1
        else:
            mc = f['EMData/MCOpenCL/accum_e']
            if mc.shape[-1] != len(energies):
                raise ValueError('MC/master energy bins do not match')
            weights = np.array([np.sum(mc[..., i], dtype=float) for i in range(len(energies))])
            if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
                raise ValueError('Invalid MC energy weights')
            weights /= weights.sum()
        maps = []
        for name in ('mLPNH', 'mLPSH'):
            dataset = master[name]
            if dataset.ndim != 4 or dataset.shape[1] != len(energies):
                raise ValueError('Invalid master shape')
            combined = np.zeros(dataset.shape[-2:], dtype=np.float64)
            for i, weight in enumerate(weights):
                if weight:
                    combined += dataset[:, i].sum(axis=0, dtype=float)*weight
            maps.append(combined.astype('float32'))
        phase = Phase(name='phase', space_group=int(np.asarray(f['CrystalData/SpaceGroupNumber']).item()))
        voltage = float(np.asarray(f['NMLparameters/MCCLNameList/EkeV']).item())
    signal = kp.signals.EBSDMasterPattern(np.stack(maps), projection='lambert', hemisphere='both', phase=phase)
    return signal, dict(mode=mode, energies_keV=energies.tolist(), weights=weights.tolist(),
        voltage_kV=voltage, generator='KikuchiCUDA Python/CuPy',
        approximation='Spatially integrated MC weights, not detector-dependent weights')
