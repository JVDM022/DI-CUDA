"""EMsoft-style dynamical diffraction in Python/NumPy and CuPy.

GetDynMat and the depth-integration mathematics are adapted from EMsoft's
MBmodule.f90 (Marc De Graef Research Group/CMU; repository LICENSE).
Propagation uses a Padé exponential rather than the Fortran eigensolver.
"""
from dataclasses import dataclass
import itertools
import numpy as np
from ._vendor.scattering import scatter_factor_wk_array


@dataclass
class Crystal:
    lattice_nm: np.ndarray
    positions: list
    atomic_numbers: np.ndarray
    occupancy: np.ndarray
    b_iso_nm2: np.ndarray
    space_group: int
    symmetry: np.ndarray

    @property
    def reciprocal(self):
        return np.linalg.inv(self.lattice_nm).T

    @property
    def volume(self):
        return abs(np.linalg.det(self.lattice_nm))


def read_crystal(group):
    from diffpy.structure.spacegroups import GetSpaceGroup
    from diffpy.structure.symmetryutilities import expandPosition
    params = np.asarray(group['LatticeParameters'][...], dtype=float)
    z = np.asarray(group['Atomtypes'][...], dtype=int).reshape(-1)
    atoms = np.asarray(group['AtomData'][...], dtype=float)
    if atoms.shape != (5, len(z)):
        raise ValueError('Expected EMsoft AtomData shape (5, number of asymmetric sites)')
    sg = int(np.asarray(group['SpaceGroupNumber'][...]).item())
    setting = int(np.asarray(group['SpaceGroupSetting'][...]).item())
    if setting != 1:
        raise ValueError('Only standard space-group setting 1 is supported; alternate origins require explicit conversion')
    if params.shape != (6,) or np.any(params[:3] <= 0) or not np.isfinite(params).all():
        raise ValueError('Invalid cell parameters (nm and degrees required)')
    if np.any((z < 1) | (z > 98)) or np.any(atoms[3] < 0) or np.any(atoms[3] > 1) or np.any(atoms[4] < 0) or not np.isfinite(atoms).all():
        raise ValueError('Invalid atomic number, occupation or isotropic B factor')
    # EMsoft CalcMatrices convention: a || X, c* || Z.
    a, b, c, alpha, beta, gamma = params
    ca, cb, cg = np.cos(np.deg2rad([alpha, beta, gamma]))
    sgamma = np.sin(np.deg2rad(gamma))
    volume2 = 1-ca*ca-cb*cb-cg*cg+2*ca*cb*cg
    if volume2 <= 0 or abs(sgamma) < 1e-12:
        raise ValueError('Degenerate crystal metric')
    lattice = np.array([[a, 0, 0], [b*cg, b*sgamma, 0],
        [c*cb, c*(ca-cb*cg)/sgamma, c*np.sqrt(volume2)/sgamma]])
    group_ops = GetSpaceGroup(sg)
    positions = [np.asarray(expandPosition(group_ops, a)[0]) for a in atoms[:3].T]
    # x' = R*x in fractional column coordinates; A's rows are direct vectors.
    matrices = [lattice.T @ op.R @ np.linalg.inv(lattice.T) for op in group_ops.iter_symops()]
    symmetry = np.unique(np.round(matrices, 10), axis=0)
    for matrix in symmetry:
        np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-8)
    return Crystal(lattice, positions, z, atoms[3], atoms[4], sg, symmetry)


def centering_allowed(h, symbol):
    a, b, c = h.T
    letter = symbol[0]
    if letter == 'F':
        return ((a-b) % 2 == 0) & ((b-c) % 2 == 0)
    if letter == 'I':
        return (a+b+c) % 2 == 0
    if letter == 'A':
        return (b+c) % 2 == 0
    if letter == 'B':
        return (a+c) % 2 == 0
    if letter == 'C':
        return (a+b) % 2 == 0
    if letter == 'R':
        return (-a+b+c) % 3 == 0
    return np.ones(len(h), dtype=bool)


class Lookup:
    """Voltage-dependent WK potential and per-site Sgh on difference vectors."""
    def __init__(self, crystal, energy_keV, dmin_nm, *, max_candidates=20000):
        from diffpy.structure.spacegroups import GetSpaceGroup
        if energy_keV <= 0 or dmin_nm <= 0:
            raise ValueError('Positive energy and dmin required')
        self.crystal = crystal
        self.limits = np.ceil(np.linalg.norm(crystal.lattice_nm, axis=1)/dmin_nm).astype(int)
        if np.prod(2*self.limits+1) > 2000000:
            raise ValueError('Reflection cube too large; increase dmin or use a smaller cell')
        h = np.array(list(itertools.product(*(range(-a, a+1) for a in self.limits))), dtype=int)
        length = np.linalg.norm(h @ crystal.reciprocal, axis=1)
        keep = (length > 0) & (length < 1/dmin_nm)
        keep &= centering_allowed(h, GetSpaceGroup(crystal.space_group).short_name)
        self.hkl = np.vstack((np.zeros((1, 3), dtype=int), h[keep]))
        if len(self.hkl) > max_candidates:
            raise ValueError('Candidate count exceeds safety limit; no reflection truncation was performed')
        self.g = self.hkl @ crystal.reciprocal
        self.shape = tuple(4*self.limits+1)
        grid = np.indices(self.shape).reshape(3, -1).T - 2*self.limits
        self.potential = np.zeros(len(grid), dtype=np.complex128)
        self.sgh = np.zeros((len(crystal.positions), len(grid)), dtype=np.complex128)
        g2 = np.sum((grid @ crystal.reciprocal)**2, axis=1)
        # Cubic length degeneracies are evaluated once, without altering hkl phases.
        unique, inverse = np.unique(np.round(g2, 12), return_inverse=True)
        prefactor = 0.04787801/crystal.volume
        for site, (z, occ, biso, positions) in enumerate(zip(
                crystal.atomic_numbers, crystal.occupancy, crystal.b_iso_nm2, crystal.positions)):
            sigma_A = 10*np.sqrt(biso/(8*np.pi**2))
            # WK's zero-temperature phonon formula has a removable singularity.
            re, im = scatter_factor_wk_array(.1*2*np.pi*np.sqrt(np.maximum(unique, 0)),
                int(z), float(max(sigma_A, 1e-7)), float(energy_keV))
            factors = (re+1j*im)[inverse]
            for a in range(0, len(grid), 2048):
                b = min(a+2048, len(grid))
                phase = np.exp(-2j*np.pi*(grid[a:b] @ positions.T)).sum(axis=1)
                self.potential[a:b] += occ*phase*factors[a:b]*prefactor*.664840340614319
                # For delta = row hkl - column hkl, Sgh uses the opposite delta.
                self.sgh[site, a:b] = z*z*occ*np.exp(-biso*.25*g2[a:b])*phase
        origin = self.index(np.zeros((1, 3), dtype=int))[0]
        self.absorption0 = float(self.potential[origin].imag)
        potential_v0 = abs(self.potential[origin].real)/.664840340614319
        psi = energy_keV*1000*(1+.00097847561649*energy_keV) + potential_v0
        self.wavelength = 1.22642584554/np.sqrt(psi)
        self.potential[origin] = 0
        if not np.isfinite(self.potential).all() or not np.isfinite(self.sgh).all() or self.absorption0 < 0:
            raise ValueError('Scattering calculation produced invalid values')
        self._device_cache = {}

    def index(self, hkl):
        hkl = np.asarray(hkl, dtype=int)
        shifted = hkl + 2*self.limits
        if np.any(shifted < 0) or np.any(shifted >= np.array(self.shape)):
            raise ValueError('Requested reciprocal vector exceeds the lookup range')
        return np.ravel_multi_index(shifted.T, self.shape)

    def _cuda_arrays(self, xp, batch_size):
        """Upload fixed crystal tables and cache pair potentials on one GPU."""
        device = int(xp.cuda.runtime.getDevice())
        cached = self._device_cache.get(device)
        if cached is not None:
            return cached

        hkl = xp.asarray(self.hkl, dtype=xp.int32)
        g = xp.asarray(self.g, dtype=xp.float64)
        potential = xp.asarray(self.potential, dtype=xp.complex128)
        sgh = xp.asarray(self.sgh, dtype=xp.complex128)
        count = len(self.hkl)
        pair_bytes = count * count * np.dtype(np.complex128).itemsize
        classification_bytes = (batch_size + 2) * count * count * np.dtype(np.float64).itemsize
        free_bytes, _ = xp.cuda.runtime.memGetInfo()
        if pair_bytes + classification_bytes > free_bytes * 0.55:
            raise MemoryError(
                f'GPU reflection tables and selection workspace need about '
                f'{(pair_bytes + classification_bytes) / 1e9:.2f} GB; '
                'reduce direction_batch or increase dmin'
            )

        pair_potential = xp.empty((count, count), dtype=xp.complex128)
        limits = xp.asarray(self.limits, dtype=xp.int32)
        strides = (self.shape[1] * self.shape[2], self.shape[2], 1)
        rows_per_chunk = max(1, min(128, int(2**20 // max(count, 1))))
        columns = hkl[None, :, :]
        for start in range(0, count, rows_per_chunk):
            stop = min(start + rows_per_chunk, count)
            difference = hkl[start:stop, None, :] - columns
            shifted = difference + 2 * limits
            flat = (shifted[..., 0] * strides[0] +
                    shifted[..., 1] * strides[1] + shifted[..., 2])
            pair_potential[start:stop] = potential[flat]

        structure_potential = potential[xp.asarray(self.index(self.hkl), dtype=xp.int64)]
        cached = dict(hkl=hkl, g=g, potential=potential, sgh=sgh,
                      structure_potential=structure_potential,
                      pair_potential=pair_potential)
        self._device_cache[device] = cached
        return cached

    def matrices_batch(self, directions, *, xp, max_beams=512,
                       bethe=(4., 8., 50., 1.)):
        """Build batched dynamical matrices for a direction batch on the GPU.

        Directions have variable strong/weak beam counts. The strong beams
        are packed to the largest strong-beam count in this batch. The weak-beam
        Bethe correction is reduced over the full candidate list on-device.
        """
        table = self._cuda_arrays(xp, len(directions))
        hkl, g = table['hkl'], table['g']
        pair = table['pair_potential']
        structure = table['structure_potential']
        sgh = table['sgh']
        rays = xp.asarray(directions, dtype=xp.float64)
        rays /= xp.linalg.norm(rays, axis=1, keepdims=True)
        k = rays / self.wavelength
        denominator = 2 * xp.sum((k[:, None, :] + g[None, :, :]) * rays[:, None, :], axis=2)
        numerator = -xp.sum(g[None, :, :] * (2 * k[:, None, :] + g[None, :, :]), axis=2)
        excitation = xp.divide(numerator, denominator,
            out=xp.full_like(numerator, xp.inf), where=xp.abs(denominator) > 1e-12)
        excitation[:, 0] = 0

        zero_potential = xp.abs(structure) <= 1e-10
        ratio = xp.divide(xp.abs(excitation) / self.wavelength, xp.abs(structure)[None, :],
            out=xp.full_like(excitation, xp.inf), where=(~zero_potential)[None, :])
        selected = xp.where(zero_potential[None, :],
                            xp.abs(excitation) <= bethe[3], ratio <= bethe[2])
        selected[:, 0] = True

        pair_abs = xp.abs(pair)
        beam_ratio = xp.divide(
            (xp.abs(excitation) / self.wavelength)[:, :, None], pair_abs[None, :, :],
            out=xp.full((len(rays), len(hkl), len(hkl)), xp.inf, dtype=xp.float64),
            where=pair_abs[None, :, :] > 1e-10,
        )
        xp.copyto(beam_ratio, xp.inf, where=~selected[:, None, :])
        minimum_ratio = xp.min(beam_ratio, axis=2)
        strong = selected & (minimum_ratio <= bethe[0])
        strong[:, 0] = True
        weak = selected & (minimum_ratio > bethe[0]) & (minimum_ratio <= bethe[1])
        strong_count = xp.sum(strong, axis=1).astype(xp.int32)
        weak_count = xp.sum(weak, axis=1).astype(xp.int32)
        maximum = int(xp.max(strong_count).get())
        if maximum > max_beams:
            raise ValueError(
                f'{maximum} strong beams exceeds max_beams={max_beams}; '
                'increase the limit explicitly; no beams were discarded'
            )

        count = len(hkl)
        row = xp.arange(count, dtype=xp.int32)[None, :]
        # The incident (000) beam must occupy matrix row zero for propagation.
        order = xp.argsort(
            xp.where(strong & (row != 0), row + 1, count + 1), axis=1
        )[:, :maximum - 1]
        order = xp.concatenate((xp.zeros((len(rays), 1), dtype=xp.int32), order), axis=1)
        width = maximum
        packed_count = xp.minimum(strong_count, width)
        active = xp.arange(width)[None, :] < packed_count[:, None]
        beam_hkl = hkl[order]
        beam_excitation = excitation[xp.arange(len(rays))[:, None], order]
        active_pair = active[:, :, None] & active[:, None, :]
        dynamic = xp.where(active_pair, pair[order[:, :, None], order[:, None, :]], 0)
        diagonal = 2 * beam_excitation / self.wavelength + 1j * self.absorption0
        dynamic[:, xp.arange(width), xp.arange(width)] = xp.where(active, diagonal, 0)

        all_beams = xp.arange(count, dtype=xp.int32)
        left = pair[order[:, :, None], all_beams[None, None, :]]
        right = pair[all_beams[None, :, None], order[:, None, :]]
        if bool(xp.any(weak & (xp.abs(excitation) < 1e-14))):
            raise ValueError('Zero excitation error in weak-beam denominator')
        inv_excitation = xp.where(weak, 1 / xp.where(weak, excitation, 1), 0)
        correction = 0.5 * self.wavelength * (
            (left * inv_excitation[:, None, :]) @ right
        )
        diagonal_correction = 0.5 * self.wavelength * xp.sum(
            xp.where(weak[:, None, :], xp.abs(left) ** 2 /
                     xp.where(weak[:, None, :], excitation[:, None, :], 1), 0), axis=2)
        correction[:, xp.arange(width)] = diagonal_correction
        correction *= active_pair
        dynamic -= correction

        difference = beam_hkl[:, :, None, :] - beam_hkl[:, None, :, :]
        shifted = difference + 2 * xp.asarray(self.limits, dtype=xp.int32)
        flat = (shifted[..., 0] * self.shape[1] * self.shape[2] +
                shifted[..., 1] * self.shape[2] + shifted[..., 2])
        site = sgh[:, flat].transpose(1, 0, 2, 3)
        site *= active[:, None, :, None] & active[:, None, None, :]
        return dynamic, site, strong_count, weak_count

    def matrices(self, direction, *, c1=4., c2=8., c3=50., sg_cutoff=1., max_beams=512):
        """Reflection selection, Bethe corrections and Sgh (EMsoft conventions)."""
        ray = np.asarray(direction, dtype=float)
        ray = ray/np.linalg.norm(ray)
        k = ray/self.wavelength
        denominator = 2*((k+self.g) @ ray)
        numerator = -np.sum(self.g*(2*k+self.g), axis=1)
        sg = np.divide(numerator, denominator, out=np.full(len(self.g), np.inf), where=abs(denominator)>1e-12)
        sg[0] = 0
        potentials = self.potential[self.index(self.hkl)]
        db = abs(potentials) <= 1e-10
        ratio = np.divide(abs(sg)/self.wavelength, abs(potentials),
                          out=np.full(len(sg), np.inf), where=~db)
        selected = np.where(db, abs(sg) <= sg_cutoff, ratio <= c3)
        selected[0] = True
        h, s = self.hkl[selected], sg[selected]
        difference = h[:, None, :] - h[None, :, :]
        coupling = abs(self.potential[self.index(difference.reshape(-1, 3))]).reshape(len(h), len(h))
        test = np.divide(abs(s)[:, None]/self.wavelength, coupling,
                         out=np.full_like(coupling, np.inf), where=coupling>1e-10).min(axis=1)
        strong = test <= c1
        strong[0] = True
        weak = (test > c1) & (test <= c2)
        if strong.sum() > max_beams:
            raise ValueError(f'{strong.sum()} strong beams exceeds max_beams={max_beams}; no beams discarded')
        hs, ss, hw, sw = h[strong], s[strong], h[weak], s[weak]
        n = len(hs)
        delta = hs[:, None, :] - hs[None, :, :]
        ix = self.index(delta.reshape(-1, 3)).reshape(n, n)
        dynamic = self.potential[ix].copy()
        np.fill_diagonal(dynamic, 2*ss/self.wavelength + 1j*self.absorption0)
        if len(hw):
            if np.any(abs(sw) < 1e-14):
                raise ValueError('Zero excitation error in weak-beam denominator')
            left_ix = self.index((hs[:, None, :]-hw[None, :, :]).reshape(-1, 3)).reshape(n, len(hw))
            right_ix = self.index((hw[:, None, :]-hs[None, :, :]).reshape(-1, 3)).reshape(len(hw), n)
            left, right = self.potential[left_ix], self.potential[right_ix]
            correction = .5*self.wavelength*((left/sw[None, :]) @ right)
            np.fill_diagonal(correction, .5*self.wavelength*np.sum(abs(left)**2/sw[None, :], axis=1))
            dynamic -= correction
        return dynamic, self.sgh[:, ix], dict(strong=n, weak=len(hw))


def batched_expm(a, xp=np):
    """Scaling-and-squaring Padé(13) exponential, CPU/CUDA, no eigensolver."""
    a = xp.asarray(a, dtype=xp.complex128)
    if a.ndim != 3 or a.shape[-1] != a.shape[-2]:
        raise ValueError('Expected (batch, n, n) matrices')
    coefficients = (64764752532480000.,32382376266240000.,7771770303897600.,
        1187353796428800.,129060195264000.,10559470521600.,670442572800.,
        33522128640.,1323241920.,40840800.,960960.,16380.,182.,1.)
    b = coefficients
    norm = xp.max(xp.sum(abs(a), axis=1), axis=1)
    squarings = xp.maximum(0, xp.ceil(xp.log2(xp.maximum(norm/5.371920351148152, 1)))).astype(int)
    a = a/xp.exp2(squarings)[:, None, None]
    eye = xp.eye(a.shape[-1], dtype=xp.complex128)[None]
    a2 = a@a
    a4 = a2@a2
    a6 = a4@a2
    u = a@(a6@(b[13]*a6+b[11]*a4+b[9]*a2)+b[7]*a6+b[5]*a4+b[3]*a2+b[1]*eye)
    v = a6@(b[12]*a6+b[10]*a4+b[8]*a2)+b[6]*a6+b[4]*a4+b[2]*a2+b[0]*eye
    result = xp.linalg.solve(v-u, v+u)
    for step in range(int(squarings.max())):
        result = xp.where((squarings>step)[:, None, None], result@result, result)
    return result


def integrate_lgh(matrices, wavelength, depth_step, weights, thickness, *, xp=np):
    """Discrete depth integral of conjugate(psi_g)*psi_h, as CalcLgh.

    The incident beam is row zero. A state is propagated before each depth
    weight is applied (EMsoft samples depth indices 1..izz).
    """
    d = xp.asarray(matrices, dtype=xp.complex128)
    propagator = batched_expm(1j*np.pi*wavelength*depth_step*d, xp)
    psi = xp.zeros(d.shape[:2], dtype=xp.complex128)
    psi[:, 0] = 1
    result = xp.zeros_like(d)
    for weight in weights:
        psi = (propagator @ psi[:, :, None])[:, :, 0]
        result += float(weight)*psi.conj()[:, :, None]*psi[:, None, :]
    return result*(depth_step/thickness)


def simulate_batch(lookup, directions, depth_step, weights, thickness, *, xp=np, max_beams=512,
                   bethe=(4., 8., 50., 1.)):
    if xp is not np:
        dynamic, site, strong, weak = lookup.matrices_batch(
            directions, xp=xp, max_beams=max_beams, bethe=bethe
        )
        lgh = integrate_lgh(dynamic, lookup.wavelength, depth_step, weights, thickness, xp=xp)
        value = xp.real(xp.sum(lgh[:, None, :, :] * site, axis=(-2, -1)))
        value /= len(lookup.crystal.positions)
        value = xp.asnumpy(value)
        if not np.isfinite(value).all() or value.min() < -1e-8*max(1., float(abs(value).max())):
            raise ValueError('Dynamical propagation produced nonfinite or negative intensity')
        counts = xp.asnumpy(xp.stack((strong, weak), axis=1))
        return np.maximum(value, 0), [
            dict(strong=int(row[0]), weak=int(row[1])) for row in counts
        ]

    prepared = [lookup.matrices(ray, max_beams=max_beams, c1=bethe[0], c2=bethe[1],
                 c3=bethe[2], sg_cutoff=bethe[3]) for ray in directions]
    n = max(len(a) for a, _, _ in prepared)
    dynamic = np.zeros((len(prepared), n, n), dtype=np.complex128)
    site = np.zeros((len(prepared), len(lookup.crystal.positions), n, n), dtype=np.complex128)
    for i, (d, s, _) in enumerate(prepared):
        dynamic[i, :len(d), :len(d)] = d
        site[i, :, :len(d), :len(d)] = s
    lgh = integrate_lgh(dynamic, lookup.wavelength, depth_step, weights, thickness, xp=xp)
    value = xp.real(xp.sum(lgh[:, None]*xp.asarray(site), axis=(-2, -1)))
    value /= sum(len(p) for p in lookup.crystal.positions)
    value = np.asarray(value) if xp is np else xp.asnumpy(value)
    if not np.isfinite(value).all() or value.min() < -1e-8*max(1., float(abs(value).max())):
        raise ValueError('Dynamical propagation produced nonfinite or negative intensity')
    return np.maximum(value, 0), [stats for _, _, stats in prepared]
