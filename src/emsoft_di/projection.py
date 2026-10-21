"""Bridge kikuchipy geometry to the EMsoft-derived NumPy/CuPy projector."""
import numpy as np
from .simulation import generate_patterns


def projection_inputs(master, detector):
    # Private API: keep kikuchipy pinned to 0.13.1 and validate against get_patterns.
    from kikuchipy.signals.util._master_pattern import _get_direction_cosines_from_detector
    north, south = master._get_master_pattern_arrays_from_energy(None)
    rays = _get_direction_cosines_from_detector(detector).reshape(*detector.shape, 3)
    # The original EMsoft-derived projector uses (x,y), kikuchipy uses (row,y;col,x).
    return np.asarray(north).T.copy(), np.asarray(south).T.copy(), rays


def project(master, rotations, detector, *, device='cuda', batch_size=32):
    north, south, rays = projection_inputs(master, detector)
    return generate_patterns(north, south, rotations.to_matrix(), rays,
                             batch_size=batch_size, device=device, geometry_dtype='float64')
