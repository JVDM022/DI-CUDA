import importlib.util
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from emsoft_di.hpc import (read_up, preprocess, preprocess_with_npa,
    build_npa_neighbors, add_scan_mapping, merge)


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

    @unittest.skipUnless(importlib.util.find_spec('h5py'), 'h5py unavailable')
    def test_merge_retains_blank_rows_and_mapping_and_rejects_overwrite(self):
        import h5py
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            up = root/'patterns.up1'
            up.write_bytes(struct.pack('<4i', 4, 2, 2, 16)+bytes(24))
            ang = root/'scan.ang'
            np.savetxt(ang, [[0,0,0,x,y] for x,y in [(0,0),(1,0),(2,0),(.5,1),(1.5,1)]])
            rotations = np.array([[1.,0,0,0], [0.,1,0,0]])
            np.save(root/'quaternions.npy', rotations)
            np.save(root/'eulers.npy', np.zeros((2,3)))
            ranges = [(0,3),(3,6)]
            for device,(a,b) in enumerate(ranges):
                with h5py.File(root/f'gpu-{device}.h5','w') as f:
                    f.attrs.update(start=a,stop=b,complete=True,run_id='test')
                    f['simulation_indices'] = [[0],[-1],[1]]
                    f['scores'] = [[1.],[np.nan],[.9]]
                    f['valid'] = [True,False,True]
            args = SimpleNamespace(output=root,keep_n=1,patterns=up,ang=ang)
            merge(args,ranges,6,dict(dictionary_count=2,run_id='test'),dict(XSTEP=1,YSTEP=1))
            with h5py.File(root/'indexed.h5') as f:
                self.assertTrue(f.attrs['complete'])
                np.testing.assert_array_equal(f['ang_mapping/pattern_index'],[0,1,2,3,4])
                self.assertTrue(np.isnan(f['best_quaternion'][1]).all())
                np.testing.assert_array_equal(f['best_quaternion'][2],rotations[1])
            with self.assertRaises(FileExistsError):
                merge(args,ranges,6,dict(dictionary_count=2,run_id='test'),dict(XSTEP=1,YSTEP=1))

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
