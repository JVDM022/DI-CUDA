import importlib.util
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from emsoft_di.hpc import (read_up, preprocess, preprocess_with_npa,
    build_npa_neighbors, scan_layout, add_scan_mapping, export_indexed_ang, merge)


class HPCTests(unittest.TestCase):
    def test_up_reader_and_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'patterns.up1'
            expected = np.arange(48, dtype='u1').reshape(3, 4, 4)
            path.write_bytes(struct.pack('<4i', 4, 4, 4, 16)+expected.tobytes())
            np.testing.assert_array_equal(read_up(path), expected)
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                read_up(path)

    @unittest.skipUnless(importlib.util.find_spec('scipy'), 'scipy unavailable')
    def test_preprocessing_does_not_mix_patterns(self):
        data = np.arange(32).reshape(2, 4, 4)
        together = preprocess(data, 2, 1)
        separate = np.concatenate([preprocess(data[i:i+1], 2, 1) for i in range(2)])
        np.testing.assert_array_equal(together, separate)

    def test_npa_uses_center_and_six_nearest_hex_neighbors(self):
        row, col = np.indices((3, 3))
        row, col = row.ravel(), col.ravel()
        pattern_index = row * 3 + col
        neighbors = build_npa_neighbors(row, col, pattern_index, 3, 3, 9, 1., np.sqrt(3)/2)
        self.assertEqual(np.count_nonzero(neighbors[4] >= 0), 7)
        self.assertEqual(np.count_nonzero(neighbors[0] >= 0), 3)
        data = np.arange(9, dtype=np.uint8).reshape(9, 1, 1)
        averaged = preprocess_with_npa(data, np.array([4]), 1, 0, neighbors)
        np.testing.assert_allclose(averaged[0, 0, 0], 30/7)

    def test_square_ang_layout_and_npa_use_eight_neighbors(self):
        with tempfile.TemporaryDirectory() as directory:
            ang = Path(directory)/'square.ang'
            coords = [(c*1.5, r*1.5) for r in range(3) for c in range(4)]
            np.savetxt(ang, [[0, 0, 0, x, y] for x, y in coords])
            calibration = {'XSTEP': 1.5, 'YSTEP': 1.5}
            _, row, col, ix, nrows, width = scan_layout(ang, calibration, len(coords))
            np.testing.assert_array_equal(row, np.repeat(np.arange(3), 4))
            np.testing.assert_array_equal(col, np.tile(np.arange(4), 3))
            neighbors = build_npa_neighbors(row, col, ix, nrows, width,
                                            len(coords), 1.5, 1.5)
            self.assertEqual(neighbors.shape, (12, 9))
            self.assertEqual(np.count_nonzero(neighbors[5] >= 0), 9)
            self.assertEqual(np.count_nonzero(neighbors[0] >= 0), 4)

    @unittest.skipUnless(importlib.util.find_spec('h5py'), 'h5py unavailable')
    def test_merge_retains_blank_rows_and_mapping_and_rejects_overwrite(self):
        import h5py
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            up = root/'patterns.up1'
            up.write_bytes(struct.pack('<4i', 4, 2, 2, 16)+bytes(24))
            ang = root/'scan.ang'
            ang_rows = np.array([
                [0,0,0,x,y,123,0,0,456,0]
                for x,y in [(0,0),(1,0),(2,0),(.5,1),(1.5,1)]
            ], dtype=float)
            with ang.open('w') as stream:
                stream.write('# Phase 1\n# COLUMN_COUNT: 13\n')
                stream.write('# COLUMN_HEADERS: a, b, c, d, e, f, g, h, i, j, k, l, m\n')
                np.savetxt(stream, ang_rows)
            rotations = np.array([[1.,0,0,0], [0.,1,0,0]])
            np.save(root/'quaternions.npy', rotations)
            eulers = np.array([[.1,.2,.3], [.4,.5,.6]])
            np.save(root/'eulers.npy', eulers)
            assigned_by_gpu = ([0,2,4], [1,3,5])
            for device, assigned_rows in enumerate(assigned_by_gpu):
                with h5py.File(root/f'gpu-{device}.h5','w') as f:
                    f.attrs.update(count=6,complete=True,run_id='test')
                    indices = np.full((6,1), -1, dtype=np.int64)
                    scores = np.full((6,1), np.nan, dtype=np.float32)
                    valid = np.zeros(6, dtype=bool)
                    assigned = np.zeros(6, dtype=bool)
                    for row in assigned_rows:
                        assigned[row] = True
                        valid[row] = row != 1
                        if valid[row]:
                            indices[row,0] = 0 if row == 0 else 1
                            scores[row,0] = 1.0 if row == 0 else .9
                    f['simulation_indices'] = indices
                    f['scores'] = scores
                    f['valid'] = valid
                    f['assigned'] = assigned
            args = SimpleNamespace(output=root,keep_n=1,patterns=up,ang=ang)
            merge(args,2,6,dict(dictionary_count=2,run_id='test'),dict(XSTEP=1,YSTEP=1))
            with h5py.File(root/'indexed.h5') as f:
                self.assertTrue(f.attrs['complete'])
                np.testing.assert_array_equal(f['ang_mapping/pattern_index'],[0,1,2,3,4])
                self.assertTrue(np.isnan(f['best_quaternion'][1]).all())
                np.testing.assert_array_equal(f['best_quaternion'][2],rotations[1])
            exported = np.loadtxt(root/'indexed.ang', comments='#', ndmin=2)
            header = (root/'indexed.ang').read_text().splitlines()
            self.assertIn('# COLUMN_COUNT: 10', header)
            np.testing.assert_allclose(exported[0,:3], eulers[0])
            np.testing.assert_array_equal(exported[1,:3], [0,0,0])
            np.testing.assert_array_equal(exported[2,:3], eulers[1])
            self.assertEqual(exported[0,7], 0)
            self.assertEqual(exported[1,7], -1)
            self.assertEqual(exported[0,6], 0)
            self.assertEqual(exported[0,9], -1)
            with self.assertRaises(FileExistsError):
                merge(args,2,6,dict(dictionary_count=2,run_id='test'),dict(XSTEP=1,YSTEP=1))

    @unittest.skipUnless(importlib.util.find_spec('kikuchipy'), 'kikuchipy unavailable')
    def test_projection_matches_kikuchipy(self):
        import kikuchipy as kp
        from orix.crystal_map import Phase
        from orix.quaternion import Rotation
        from emsoft_di.projection import project
        rng = np.random.default_rng(85)
        # Asymmetric maps expose transposition and hemisphere mistakes.
        master = kp.signals.EBSDMasterPattern(rng.uniform(1,2,size=(2,41,41)).astype('float32'),
            hemisphere='both',projection='lambert',phase=Phase(space_group=225))
        detector = kp.detectors.EBSDDetector(shape=(8,10),pc=[.48,.61,.66],
            convention='tsl',sample_tilt=69.9,tilt=2)
        rotations = Rotation.from_euler([[.1,.2,.3],[2,1,3],[5,2,1]])
        expected = master.get_patterns(rotations,detector,compute=True,show_progressbar=False).data
        actual = np.concatenate(list(project(master,rotations,detector,device='cpu',batch_size=2)))
        np.testing.assert_allclose(actual,expected,rtol=2e-4,atol=2e-5)
        if importlib.util.find_spec('cupy'):
            import cupy as cp
            try:
                count = cp.cuda.runtime.getDeviceCount()
            except cp.cuda.runtime.CUDARuntimeError:
                count = 0
            for device_id in range(count):
                with self.subTest(device=device_id), cp.cuda.Device(device_id):
                    actual = np.concatenate([cp.asnumpy(a) for a in project(master,rotations,detector,batch_size=2)])
                    np.testing.assert_allclose(actual,expected,rtol=2e-4,atol=2e-5)
