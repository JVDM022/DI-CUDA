import unittest
import numpy as np
from emsoft_di import detector_rays, generate_patterns, index_patterns


class SimulationTests(unittest.TestCase):
    def test_axes_and_hemispheres(self):
        mp = np.arange(25).reshape(5, 5).astype(float)
        rays = np.array([[[0,0,1], [1,0,0], [0,1,0], [0,0,-1]]], float)
        p = next(generate_patterns(mp, mp+100, np.eye(3)[None], rays, device="cpu"))
        np.testing.assert_allclose(p[0,0], [12,22,14,112])

    def test_energy_and_batches(self):
        rays = detector_rays(6, 4, pixel_size=1, distance=8)
        np.testing.assert_allclose(np.linalg.norm(rays, axis=-1), 1)
        maps = np.stack([np.ones((5,5))*2, np.ones((5,5))*3])
        blocks = list(generate_patterns(maps, maps, np.repeat(np.eye(3)[None], 3, axis=0),
                      rays, energy_weights=[.25,.75], batch_size=2, device="cpu"))
        self.assertEqual([len(b) for b in blocks], [2,1])
        np.testing.assert_allclose(np.concatenate(blocks), 2.75)

    def test_simulate_then_match(self):
        rng = np.random.default_rng(9)
        mp = rng.random((33,33))
        rays = detector_rays(16,12,pixel_size=1,distance=20)
        rotations = np.array([np.eye(3), [[0,-1,0],[1,0,0],[0,0,1]]])
        patterns = next(generate_patterns(mp, mp, rotations, rays, device="cpu"))
        result = index_patterns(patterns[::-1], patterns, device="cpu")
        np.testing.assert_array_equal(result.indices[:,0], [1,0])

    def test_invalid_rotation(self):
        with self.assertRaises(ValueError):
            next(generate_patterns(np.ones((5,5)), np.ones((5,5)),
                 np.zeros((1,3,3)), np.ones((2,2,3)), device="cpu"))
