import unittest
import numpy as np
from emsoft_di import index_patterns


class MatchingTests(unittest.TestCase):
    def test_matches_dense_reference_across_batches(self):
        rng = np.random.default_rng(18)
        d = rng.normal(size=(19, 4, 5)).astype(np.float32)
        p = rng.normal(size=(7, 4, 5)).astype(np.float32)
        a, b = p.reshape(7, -1), d.reshape(19, -1)
        reference = (a / np.linalg.norm(a, axis=1, keepdims=True)) @ (
            b / np.linalg.norm(b, axis=1, keepdims=True)).T
        expected = np.argsort(-reference, axis=1)[:, :5]
        angles = rng.random((19, 3))
        r = index_patterns(p, d, orientations=angles, top_k=5,
                           pattern_batch=3, dictionary_batch=2, device="cpu")
        np.testing.assert_array_equal(r.indices, expected)
        np.testing.assert_allclose(r.scores, np.take_along_axis(reference, expected, 1), atol=1e-6)
        np.testing.assert_array_equal(r.orientations, angles[expected])

    def test_ties(self):
        r = index_patterns(np.ones((2, 4)), np.ones((9, 4)), top_k=4,
                           dictionary_batch=2, device="cpu")
        np.testing.assert_array_equal(r.indices, [[0, 1, 2, 3]] * 2)

    def test_mask_center_and_invalid(self):
        p = np.array([[1, 2, 8], [2, 1, 0]], dtype=float)
        r = index_patterns(p, p, mask=[True, True, False], center=True, device="cpu")
        np.testing.assert_array_equal(r.indices[:, 0], [0, 1])
        for invalid in (np.zeros((1, 3)), np.full((1, 3), np.nan)):
            with self.assertRaises(ValueError):
                index_patterns(invalid, p, device="cpu")

    def test_cuda_reference_when_available(self):
        try:
            import cupy as cp
            available = cp.cuda.runtime.getDeviceCount() > 0
        except Exception:
            available = False
        if not available:
            self.skipTest("NVIDIA CUDA runtime unavailable")
        rng = np.random.default_rng(4)
        p, d = rng.normal(size=(7, 32)), rng.normal(size=(17, 32))
        cpu = index_patterns(p, d, top_k=3, device="cpu", dictionary_batch=4)
        for device_id in range(cp.cuda.runtime.getDeviceCount()):
            with self.subTest(device=device_id), cp.cuda.Device(device_id):
                gpu = index_patterns(p, d, top_k=3, device="cuda", dictionary_batch=4)
                np.testing.assert_array_equal(cpu.indices, gpu.indices)
                np.testing.assert_allclose(cpu.scores, gpu.scores, atol=1e-6)
                tied = index_patterns(np.ones((2, 4)), np.ones((9, 4)),
                    top_k=4, dictionary_batch=2, device="cuda")
                np.testing.assert_array_equal(tied.indices, [[0, 1, 2, 3]] * 2)


if __name__ == "__main__":
    unittest.main()
