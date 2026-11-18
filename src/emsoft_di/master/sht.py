"""Reconstruct an EBSD SHT master on two square Lambert grids."""
import subprocess
from pathlib import Path

import numpy as np


def lambert_maps(coefficients):
    # Each concentric square samples one latitude at equally spaced azimuths.
    bw = len(coefficients)
    radius = bw
    rings = np.arange(radius + 1)
    north_z = 1 - (rings / radius) ** 2
    z = np.stack([north_z, -north_z])
    sine = np.sqrt(1 - z*z)
    diagonal = np.full_like(z, 1 / np.sqrt(4*np.pi))
    spectrum = np.zeros((2, radius + 1, bw), dtype=complex)
    for m, row in enumerate(coefficients):
        if m:
            diagonal *= -np.sqrt((2*m+1)/(2*m)) * sine
        if not np.any(row):
            continue
        # Normalized associated Legendre functions, including Condon–Shortley phase.
        previous = np.zeros_like(z)
        current = diagonal
        for degree in range(m, bw):
            spectrum[:, :, m] += row[degree] * current
            n = degree + 1
            a = np.sqrt((4*n*n-1)/(n*n-m*m))
            b = np.sqrt(((n-1)**2-m*m)/(4*(n-1)**2-1)) if degree > m else 0
            previous, current = current, a*(z*current-b*previous)

    maps = np.empty((2, 2*radius+1, 2*radius+1), dtype=np.float32)
    maps[:, radius, radius] = spectrum[:, 0, 0].real
    for r in range(1, radius+1):
        # Walk counterclockwise from +x. Fold high orders onto sampled frequencies.
        k = np.arange(8*r)
        # Lambert square coordinates vary linearly along each edge.
        x = np.interp(k, r*np.arange(9), r*np.array([1,1,0,-1,-1,-1,0,1,1])).round().astype(int)
        y = np.roll(x, 2*r)
        modes = spectrum[:, r, :].copy()
        modes[:, 1:] *= 2
        folded = np.zeros((2, 8*r), dtype=complex)
        for hemisphere in range(2):
            np.add.at(folded[hemisphere], np.arange(bw) % (8*r), modes[hemisphere])
        maps[:, radius+y, radius+x] = (np.fft.ifft(folded)*len(k)).real
    return maps


def load_sht_master(path):
    import kikuchipy as kp
    from orix.crystal_map import Phase
    try:
        result = subprocess.run(['emsoft-di-read-sht', str(path)], capture_output=True, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError('Build emsoft-di-read-sht as described in the README') from exc
    except subprocess.CalledProcessError as exc:
        raise ValueError(exc.stderr.decode(errors='replace').strip()) from exc
    header, payload = result.stdout.split(b'\n', 1)
    bandwidth, group, voltage = header.split()
    bw = int(bandwidth)
    coefficients = np.frombuffer(payload, dtype=np.complex128).reshape(bw, bw)
    maps = lambert_maps(coefficients)
    if not np.isfinite(maps).all():
        raise ValueError('Nonfinite reconstructed SHT master')
    phase = Phase(name=Path(path).stem, space_group=int(group))
    master = kp.signals.EBSDMasterPattern(maps, projection='lambert', hemisphere='both', phase=phase)
    return master, dict(mode='sht', voltage_kV=float(voltage), bandwidth=bw, source=str(path))
