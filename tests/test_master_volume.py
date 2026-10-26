import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

from emsoft_di.master.volume import (
    read_mc_volume, depth_energy_counts, read_interaction_map, _write_heatmap_svg,
)


@unittest.skipUnless(h5py, 'h5py unavailable')
class MonteCarloVolumeTests(unittest.TestCase):
    def test_writes_static_svg_heatmap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'penetration.svg'
            _write_heatmap_svg(path, np.asarray([[0, 1], [4, 16]]),
                               np.asarray([-1.0, 1.0]), np.asarray([0.5, 1.5]),
                               'Interaction map', 'Lateral (nm)', 'Depth (nm)')
            text = path.read_text(encoding='utf-8')
            self.assertTrue(text.startswith('<svg '))
            self.assertIn('Interaction map', text)
            self.assertIn('Lateral (nm)', text)
            self.assertIn('<rect', text)

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

    def test_reads_interaction_cross_section_and_axis_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mc.h5'
            expected = np.arange(15, dtype=np.int32).reshape(3, 5)
            with h5py.File(path, 'w') as file:
                ds = file.create_dataset('EMData/MCOpenCL/accum_path', data=expected)
                ds.attrs['lateral_min_nm'] = -1.0
                ds.attrs['lateral_step_nm'] = 1.0
                ds.attrs['depth_step_nm'] = 0.5

            counts, lateral_nm, depth_nm = read_interaction_map(path)
            np.testing.assert_array_equal(counts, expected)
            np.testing.assert_allclose(lateral_nm, [-0.5, 0.5, 1.5])
            np.testing.assert_allclose(depth_nm, [0.25, 0.75, 1.25, 1.75, 2.25])

    def test_legacy_mc_file_explains_missing_trajectory_map(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy.h5'
            with h5py.File(path, 'w'):
                pass
            with self.assertRaisesRegex(ValueError, 'Rerun the Python Monte Carlo'):
                read_interaction_map(path)


if __name__ == '__main__':
    unittest.main()
