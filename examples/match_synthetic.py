"""Small synthetic matcher example; explicitly select CPU or CUDA."""
import argparse
import numpy as np
from emsoft_di import index_patterns

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--device', required=True, choices=('cpu', 'cuda'))
args = parser.parse_args()
rng = np.random.default_rng(42)
dictionary = rng.normal(size=(100, 16, 16)).astype(np.float32)
expected = np.array([3, 17, 82])
patterns = dictionary[expected] + rng.normal(0, .01, size=(3, 16, 16))
result = index_patterns(patterns, dictionary, top_k=3, center=True,
                        dictionary_batch=32, device=args.device)
np.testing.assert_array_equal(result.indices[:, 0], expected)
print('Best dictionary indices:', result.indices[:, 0])
print('Best NCC scores:', result.scores[:, 0])
