"""Inputs, Lambert directions and Monte Carlo depth weighting."""
import numpy as np


def lambert_directions(npx):
    if not isinstance(npx, int) or npx < 1:
        raise ValueError('npx must be a positive integer')
    u, v = np.meshgrid(np.linspace(-1, 1, 2*npx+1), np.linspace(-1, 1, 2*npx+1))
    major = abs(u) >= abs(v)
    divisor = np.where(major, u, v)
    ratio = np.divide(np.where(major, v, u), divisor, out=np.zeros_like(u), where=divisor!=0)
    angle = np.pi/4*ratio
    radius = divisor*np.sqrt(np.maximum(0, 2-divisor**2))
    x = np.where(major, radius*np.cos(angle), radius*np.sin(angle))
    y = np.where(major, radius*np.sin(angle), radius*np.cos(angle))
    z = 1-divisor**2
    north = np.stack((x, y, z), axis=-1)
    south = north.copy()
    south[..., 2] *= -1
    return np.stack((north, south))


def symmetry_reduce(directions, operators):
    """Group only exactly symmetry-related sampled rays; no interpolation."""
    flat = np.asarray(directions).reshape(-1, 3)
    unique = []
    seen = {}
    inverse = np.empty(len(flat), dtype=np.int64)
    for start in range(0, len(flat), 2048):
        equivalents = np.einsum('sij,bj->bsi', operators, flat[start:start+2048])
        # A canonical orbit member, selecting z, then y, then x after rounding.
        rounded = np.round(equivalents, 9)
        order = np.lexsort((rounded[..., 0], rounded[..., 1], rounded[..., 2]), axis=1)
        canonical = rounded[np.arange(len(rounded)), order[:, -1]]
        for offset, key_array in enumerate(canonical):
            key = tuple(key_array)
            if key not in seen:
                seen[key] = len(unique)
                unique.append(flat[start+offset])
            inverse[start+offset] = seen[key]
    return np.asarray(unique), inverse.reshape(directions.shape[:-1])


def read_depth_profiles(file):
    """Mirror EMEBSDmaster's 95% depth cutoff and lambdaE normalization.

    EMsoft's NumPy HDF5 layout is (x,y,depth,energy). The scalar totnum_el
    is used exactly as in the host. Absorption correction is applied later
    when wavelength and the energy-specific potential are available.
    """
    mc = file['EMData/MCOpenCL']
    nml = file['NMLparameters/MCCLNameList']
    dz = float(np.asarray(nml['depthstep'][...]).item())
    total = float(np.asarray(mc['totnum_el'][...]).item())
    bins = int(np.asarray(mc['numEbins'][...]).item())
    energies = (float(np.asarray(nml['Ehistmin'][...]).item()) +
                np.arange(bins)*float(np.asarray(nml['Ebinsize'][...]).item()))
    ds = mc['accum_z']
    if ds.ndim != 4 or ds.shape[-1] != bins or min(dz, total) <= 0 or np.any(energies <= 0):
        raise ValueError('Invalid Monte Carlo depth/energy metadata')
    hist = []
    thickness = []
    cutoff = 1
    # EMEBSDmaster determines the thickness from the central 20% of the
    # Monte Carlo x/y field, while lambdaE below uses the full field.
    center_x, center_y = ds.shape[0] // 2, ds.shape[1] // 2
    radius_x, radius_y = ds.shape[0] // 10, ds.shape[1] // 10
    central = ds[
        center_x - radius_x:center_x + radius_x + 1,
        center_y - radius_y:center_y + radius_y + 1,
        :, :,
    ]
    for energy in range(bins):
        values = np.asarray(central[..., energy], dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError('Invalid Monte Carlo depth counts')
        sums = values.sum(axis=2)
        thresholds = sums*.95
        reaches = np.cumsum(values, axis=2) >= thresholds[..., None]
        cutoff = max(cutoff, int(np.argmax(reaches, axis=2).max())+1)
        thickness.append(cutoff*dz)
        # EMsoft's lambdaE numerator integrates each depth plane over the
        # entire x/y grid, normalized by all counts in all energy/depth bins.
        full_values = np.asarray(ds[..., energy], dtype=np.float64)
        hist.append(full_values.sum(axis=(0, 1))/total)
    if np.sum(hist) <= 0:
        raise ValueError('Monte Carlo depth histogram has no counts')
    # The original host uses max(thick)/depthstep for every lambdaE row.
    profiles = np.asarray(hist)[:, :cutoff]
    return energies, dz, np.asarray(thickness), profiles


def corrected_weights(profile, lookup, depth_step):
    # EMsoft uses depths (iz-1)*dz for mean-absorption correction, while
    # propagating the wave to iz*dz before integration.
    exponent = 2*np.pi*lookup.wavelength*lookup.absorption0*depth_step*np.arange(len(profile))
    with np.errstate(over='raise', invalid='raise'):
        return profile*np.exp(exponent)
