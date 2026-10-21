"""Master-pattern projection adapted from EMsoft (see LICENSE).

Array convention: hemisphere maps are (energy, x, y), rays (height,width,3).
Rotations explicitly map sample-frame column vectors to crystal-frame vectors.
"""
import numpy as np


def detector_rays(width, height, *, xpc=0., ypc=0., pixel_size,
                  distance, sample_tilt=70., camera_tilt=0., omega=0.):
    """Generate sample-frame unit rays using EMsoft GenerateEBSDDetector.

    xpc/ypc use EMsoft pixel offsets, NOT EDAX x-star/y-star. Distance and
    pixel_size must use the same length unit. Angles are degrees. Output
    includes EMsoft's vertical reversal and detector-view x convention.
    """
    if width < 1 or height < 1 or pixel_size <= 0 or distance <= 0:
        raise ValueError("Positive dimensions, pixel_size and distance required")
    x = (xpc + (1-width)*.5 + np.arange(width))*pixel_size
    y = (ypc - (1-height)*.5 - np.arange(height))*pixel_size
    x, y = np.meshgrid(x, y)
    alpha = np.pi/2 - np.deg2rad(sample_tilt-camera_tilt)
    ca, sa = np.cos(alpha), np.sin(alpha)
    cw, sw = np.cos(np.deg2rad(omega)), np.sin(np.deg2rad(omega))
    ls, lc = -sw*x + distance*cw, cw*x + distance*sw
    rays = np.stack((y*ca+sa*ls, lc, -sa*y+ca*ls), axis=-1)
    return (rays / np.linalg.norm(rays, axis=-1, keepdims=True))[::-1].copy()


def generate_patterns(north, south, rotations, rays, *, energy_weights=None,
                      batch_size=32, device="cuda", geometry_dtype="float32"):
    """Yield batches of simulated patterns, as NumPy or CuPy arrays.

    Maps: (E, S, S), odd S >= 3; 2D maps are treated as one energy.
    Weights: (E,) or (E,H,W), already interpolated onto the detector.
    No implicit energy integration, solid-angle correction or preprocessing.
    Caller supplies consistent energy bins, weighting and rotation convention.
    geometry_dtype='float64' matches kikuchipy's double-precision geometry.
    """
    if device == "cuda":
        try:
            import cupy as xp
        except ImportError as exc:
            raise RuntimeError("Install a CUDA extra to generate GPU patterns") from exc
    elif device == "cpu":
        xp = np
    else:
        raise ValueError("device must be cuda or cpu")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be positive")
    nh, sh = xp.asarray(north, dtype=xp.float32), xp.asarray(south, dtype=xp.float32)
    if nh.ndim == 2:
        nh = nh[None]
    if sh.ndim == 2:
        sh = sh[None]
    if (nh.ndim != 3 or nh.shape != sh.shape or nh.shape[1] != nh.shape[2]
            or nh.shape[1] < 3 or nh.shape[1] % 2 != 1):
        raise ValueError("Hemisphere maps need matching (E,odd S,odd S) shape")
    r = np.asarray(rotations, dtype=np.float64)
    if r.ndim != 3 or r.shape[1:] != (3, 3) or not np.isfinite(r).all():
        raise ValueError("rotations must have shape (N,3,3) and be finite")
    if not np.allclose(r @ r.transpose(0, 2, 1), np.eye(3), atol=1e-5) or not np.allclose(np.linalg.det(r), 1, atol=1e-5):
        raise ValueError("rotations must be proper orthogonal matrices")
    if geometry_dtype not in ('float32', 'float64'):
        raise ValueError('geometry_dtype must be float32 or float64')
    ray = xp.asarray(rays, dtype=geometry_dtype)
    if ray.ndim != 3 or ray.shape[-1] != 3:
        raise ValueError("rays must have shape (H,W,3)")
    norm = xp.linalg.norm(ray, axis=-1, keepdims=True)
    if not bool(xp.isfinite(ray).all()) or bool((norm == 0).any()):
        raise ValueError("rays must be finite and nonzero")
    ray = ray / norm
    e, s, _ = nh.shape
    if energy_weights is None:
        if e != 1:
            raise ValueError("Multiple energy bins require explicit energy_weights")
        weights = xp.ones((1, 1, 1), dtype=xp.float32)
    else:
        weights = xp.asarray(energy_weights, dtype=xp.float32)
        if weights.shape == (e,):
            weights = weights[:, None, None]
        if weights.shape not in ((e, 1, 1), (e, *ray.shape[:2])):
            raise ValueError("energy_weights must have shape (E,) or (E,H,W)")
    if not all(bool(xp.isfinite(a).all()) for a in (nh, sh, weights)):
        raise ValueError("Maps and weights must be finite")
    if bool((weights < 0).any()):
        raise ValueError("Energy weights must be nonnegative")
    radius = (s-1)//2
    for start in range(0, len(r), batch_size):
        dc = xp.einsum('bij,hwj->bhwi', xp.asarray(r[start:start+batch_size], dtype=geometry_dtype), ray)
        dc /= xp.linalg.norm(dc, axis=-1, keepdims=True)
        x, y, z = dc[..., 0], dc[..., 1], dc[..., 2]
        # Rosca-Lambert inverse, algebraically simplified from Lambert.f90.
        major_x = xp.abs(y) <= xp.abs(x)
        safe_x = xp.where(x == 0, 1, x)
        safe_y = xp.where(y == 0, 1, y)
        scale = xp.sqrt(xp.maximum(0, 1-xp.abs(z)))
        u = xp.where(major_x, xp.sign(x)*scale,
                     xp.sign(y)*scale*(4/np.pi)*xp.arctan(x/safe_y))
        v = xp.where(major_x, xp.sign(x)*scale*(4/np.pi)*xp.arctan(y/safe_x),
                     xp.sign(y)*scale)
        u, v = xp.clip((u+1)*radius, 0, s-1), xp.clip((v+1)*radius, 0, s-1)
        i, j = xp.floor(u).astype(xp.int64), xp.floor(v).astype(xp.int64)
        ip, jp = xp.minimum(i+1, s-1), xp.minimum(j+1, s-1)
        a, b = u-i, v-j
        output = xp.zeros(dc.shape[:-1], dtype=xp.float32)
        for k in range(e):
            def interp(mp):
                return mp[i,j]*(1-a)*(1-b)+mp[ip,j]*a*(1-b)+mp[i,jp]*(1-a)*b+mp[ip,jp]*a*b
            output += weights[k]*xp.where(z >= 0, interp(nh[k]), interp(sh[k]))
        yield output
