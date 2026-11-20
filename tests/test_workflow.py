import json
import struct
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import h5py
import pytest
from emsoft_di.workflow import Settings, inspect, run, mapping


def fixture(root):
    import kikuchipy as kp
    from orix.crystal_map import Phase
    yy, xx = np.mgrid[-1:1:17j, -1:1:17j]
    north = np.sin(xx*9)+np.cos(yy*7)+xx*yy*2
    master = kp.signals.EBSDMasterPattern(np.stack([north, north[::-1]]).astype('float32'),
        projection='lambert', hemisphere='both', phase=Phase(name='Nickel', space_group=225))
    raw = np.random.default_rng(3).integers(1, 255, (4, 8, 8), dtype='uint8')
    raw[1] = 0
    (root/'patterns.up1').write_bytes(struct.pack('<4i', 1, 8, 8, 16)+raw.tobytes())
    header = '''# x-star 0.5
# y-star 0.5
# z-star 0.6
# SampleTiltAngle 70
# CameraElevationAngle 0
# Phase 1
# MaterialName Nickel
# Symmetry 43
# LatticeConstants 3.52 3.52 3.52 90 90 90
# GRID: SqrGrid
# XSTEP: 1
# YSTEP: 1
# NCOLS_ODD: 2
# NCOLS_EVEN: 2
# NROWS: 2
'''
    np.savetxt(root/'scan.ang', [[0,0,0,x,y,0,0,1,0,0] for y in range(2) for x in range(2)],
               header=header.rstrip(), comments='')
    with h5py.File(root/'master.h5', 'w') as f:
        f.create_dataset('EMData/EBSDmaster/mLPNH', data=north)
    return Settings(str(root/'patterns.up1'), str(root/'scan.ang'), str(root/'master.h5'),
                    str(root/'run'), resolution=30., binning=2, sigma=0., batch=2, limit=0, device='cpu'), master


def test_full_run_resume_exports_and_input_guard(tmp_path):
    from orix import io
    s, master = fixture(tmp_path)
    events = []
    with patch('emsoft_di.workflow.load_master', return_value=(master, {'mode':'synthetic'})):
        run(s, lambda *args: events.append(args), lambda: len(events) >= 2)
        state = json.loads((tmp_path/'run/state.json').read_text())
        assert state['status'] == 'paused'
        projected = state['projected']
        run(s)
        state = json.loads((tmp_path/'run/state.json').read_text())
        assert state['status'] == 'complete'
        assert state['projected'] >= projected
        for filename in ['indexed.h5', 'indexed.ang']:
            xmap = io.load(tmp_path/'run'/filename)
            assert xmap.size == 4
            assert np.sum(xmap.phase_id == -1) == 1
        with h5py.File(tmp_path/'run/checkpoint.h5') as f:
            assert f['simulation_indices'][1,0] == -1
            assert np.isnan(f['scores'][1,0])
        run(s)  # complete-run retry is safe
        s.sigma = 1.
        with pytest.raises(ValueError, match='changed'):
            run(s)


def test_validation_and_lock(tmp_path):
    s, _ = fixture(tmp_path)
    s.binning = 3
    with pytest.raises(ValueError, match='divide'):
        inspect(s)
    s.binning = 2
    out = Path(s.output); out.mkdir(); (out/'.workflow.lock').touch()
    with pytest.raises(RuntimeError, match='locked'):
        run(s)


def test_mapping_and_truncation(tmp_path):
    s, _ = fixture(tmp_path)
    raw, cal, xy, ix = inspect(s)
    assert ix.tolist() == [0,1,2,3]
    with pytest.raises(ValueError):
        mapping(s.ang, cal, 5, 'ang-order')
    Path(s.patterns).write_bytes(b'broken')
    with pytest.raises(ValueError, match='Truncated'):
        inspect(s)


def test_app_initial_render():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path(__file__).parents[1]/'app/indexing_app.py')).run(timeout=30)
    assert not app.exception
    assert app.title[0].value == 'From patterns to orientations'


def test_matching_pause_and_partial_export(tmp_path):
    s, master = fixture(tmp_path)
    s.limit = 2
    paused = []
    def progress(stage, done, total):
        if stage == 'Indexing patterns': paused.append(done)
    with patch('emsoft_di.workflow.load_master', return_value=(master, {})):
        run(s, progress)
    from orix import io
    result = io.load(tmp_path/'run/indexed.h5')
    assert np.sum(result.phase_id == -1) == 3
    table = np.loadtxt(tmp_path/'run/indexed.ang', comments='#')
    np.testing.assert_allclose(table[:,3:5], [[0,0],[1,0],[0,1],[1,1]])
    assert np.all(table[[1,2,3],6] == -1)


def test_hex_mapping_and_invalid_y(tmp_path):
    path = tmp_path/'hex.ang'
    rows = np.array([[0,0,0,0,0], [0,0,0,1,0], [0,0,0,.5,.866], [0,0,0,1.5,.866]])
    np.savetxt(path, rows)
    cal = {'XSTEP':1.,'YSTEP':.866}
    xy, ix = mapping(path, cal, 4, 'raster')
    assert ix.tolist() == [0,1,2,3]
    rows[-1,4] = .9
    np.savetxt(path, rows)
    with pytest.raises(ValueError, match='Y coordinates'):
        mapping(path, cal, 4, 'raster')
