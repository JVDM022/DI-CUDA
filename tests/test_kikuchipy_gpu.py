import unittest
from types import SimpleNamespace
import numpy as np
from emsoft_di.kikuchipy_gpu import dictionary_indexing_gpu


class AdapterTests(unittest.TestCase):
    def test_ncc_mask_navigation_and_top_matches(self):
        rng = np.random.default_rng(42)
        d = rng.normal(size=(11, 3, 4))
        p = rng.normal(size=(2, 3, 3, 4))
        mask = np.zeros((3, 4), dtype=bool)
        mask[0] = True
        a, b = p.reshape(6, 12)[:, ~mask.ravel()], d.reshape(11, 12)[:, ~mask.ravel()]
        a -= a.mean(1, keepdims=True)
        b -= b.mean(1, keepdims=True)
        dense = (a / np.linalg.norm(a, axis=1, keepdims=True)) @ (b / np.linalg.norm(b, axis=1, keepdims=True)).T
        order = np.argsort(-dense, axis=1)[:, :4]
        result = dictionary_indexing_gpu(SimpleNamespace(data=p), SimpleNamespace(data=d),
            keep_n=4, signal_mask=mask, pattern_batch=2, dictionary_batch=3, device="cpu")
        np.testing.assert_array_equal(result.indices, order)
        np.testing.assert_allclose(result.scores, np.take_along_axis(dense, order, 1), atol=1e-6)

    def test_small_dictionary_and_invalid_input(self):
        s = SimpleNamespace(data=np.arange(12).reshape(1, 3, 4))
        self.assertEqual(dictionary_indexing_gpu(s, s, device="cpu").indices.shape, (1, 1))
        for kwargs in ({"metric": "unknown"}, {"signal_mask": np.zeros((3, 4))}, {"keep_n": 0}):
            with self.assertRaises(ValueError):
                dictionary_indexing_gpu(s, s, device="cpu", **kwargs)
