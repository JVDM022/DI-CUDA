import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

from emsoft_di.master.volume import read_mc_volume, depth_energy_counts


@unittest.skipUnless(h5py, 'h5py unavailable')
class MonteCarloVolumeTests(unittest.TestCase):
    def test_reads_opencl_counts_and_sums_xy_by_depth_and_energy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mc.h5'
            expected = np.arange(2*3*4*2, dtype=np.int32).reshape(2, 3, 4, 2)
            with h5py.File(path, 'w') as file:
                file['EMData/MCOpenCL/accum_z'] = expected
                file['EMData/MCOpenCL/numEbins'] = [2]
                nml = file.create_group('NMLparameters/MCCLNameList')
                nml['Ehistmin'] = [5.0]
                nml['Ebinsize'] = [2.5]
                nml['depthstep'] = [0.25]

            counts, energies, depth_step = read_mc_volume(path)
            np.testing.assert_array_equal(counts, expected)
            np.testing.assert_allclose(energies, [5.0, 7.5])
            self.assertEqual(depth_step, 0.25)
            np.testing.assert_array_equal(
                depth_energy_counts(counts), expected.sum(axis=(0, 1)))


if __name__ == '__main__':
    unittest.main()
