"""Create a tiny reproducible Ni scan using the bundled real SHT master."""
import argparse
import struct
from pathlib import Path
import h5py
import numpy as np
from emsoft_di.hpc import load_master
from orix import sampling
import kikuchipy as kp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path, help='New folder; never overwrites inputs')
    args = parser.parse_args()
    root = args.destination.resolve()
    if root.exists():
        parser.error('Destination exists; choose a new folder')
    source = Path(__file__).resolve().parents[1]/'SHTdatabase-master/EBSD/Cubic/Ni (nickel) [A1]/Ni (nickel) [A1] {20kV}.sht'
    master, metadata = load_master(source, 'highest-bin')
    root.mkdir(parents=True)
    # Store reconstructed maps in the project's explicitly identified master schema.
    with h5py.File(root/'Ni-demo-master.h5', 'w') as f:
        f.attrs.update(kikuchicuda_master_schema=1, complete=True)
        f.create_dataset('CrystalData/SpaceGroupNumber', data=225)
        f.create_dataset('EMData/EBSDmaster/EkeVs', data=[20.])
        for name, array in zip(['mLPNH','mLPSH'], master.data):
            f.create_dataset('EMData/EBSDmaster/'+name, data=array[None,None])
        f.create_dataset('EMData/MCOpenCL/accum_e', data=np.ones((1,1,1)))
        f.create_dataset('NMLparameters/MCCLNameList/EkeV', data=20.)
    rotations = sampling.get_sample_fundamental(method='cubochoric', resolution=30., point_group=master.phase.point_group)
    detector = kp.detectors.EBSDDetector(shape=(16,16), pc=[.5,.5,.6], convention='tsl', sample_tilt=70, tilt=0)
    data = master.get_patterns(rotations[:4], detector, compute=True, dtype_out='float32', show_progressbar=False).data
    lo = data.min(axis=(1,2), keepdims=True); hi = data.max(axis=(1,2), keepdims=True)
    raw = np.rint(10+235*(data-lo)/(hi-lo)).astype('uint8')
    (root/'scan.up1').write_bytes(struct.pack('<4i',1,16,16,16)+raw.tobytes())
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
# NROWS: 2'''
    np.savetxt(root/'scan.ang', [[0,0,0,x,y,0,0,1,0,0] for y in range(2) for x in range(2)], header=header, comments='')
    np.save(root/'ground-truth-quaternions.npy', rotations[:4].data)
    print(f'Demo created: {root}\nUse sampling 30°, binning 1, sigma 0, limit 0; master: Ni-demo-master.h5')

if __name__ == '__main__':
    main()
