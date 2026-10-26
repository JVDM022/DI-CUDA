"""GPU-first Python port of EMsoft EMMCOpenCL full-EBSD Monte Carlo mode.

The trajectory equations follow EMsoft's OpenCL EMMC.cl and its host-side
binning in EMMCOpenCL.f90. The upstream copyright and BSD-style terms are
retained in the repository LICENSE. The CUDA implementation writes only the
HDF5 Monte Carlo datasets consumed by KikuchiCUDA; it does not claim to
reproduce every optional EMsoft visualization dataset.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np


AVOGADRO = 6.02214076e23
ATOMIC_WEIGHTS = np.asarray([
    1.00794, 4.002602, 6.941, 9.012182, 10.811, 12.0107, 14.0067,
    15.9994, 18.9984032, 20.1797, 22.98976928, 24.3050, 26.9815386,
    28.0855, 30.973762, 32.065, 35.453, 39.948, 39.0983, 40.078,
    44.955912, 47.867, 50.9415, 51.9961, 54.938045, 55.845,
    58.933195, 58.6934, 63.546, 65.38, 69.723, 72.64, 74.92160,
    78.96, 79.904, 83.798, 85.4678, 87.62, 88.90585, 91.224,
    92.90638, 95.96, 98.9062, 101.07, 102.90550, 106.42, 107.8682,
    112.411, 114.818, 118.710, 121.760, 127.60, 126.90447, 131.293,
    132.9054519, 137.327, 138.90547, 140.116, 140.90765, 144.242,
    145.0, 150.36, 151.964, 157.25, 158.92535, 162.500, 164.93032,
    167.259, 168.93421, 173.054, 174.9668, 178.49, 180.94788, 183.84,
    186.207, 190.23, 192.217, 195.084, 196.966569, 200.59, 204.3833,
    207.2, 208.98040, 209.0, 210.0, 222.0, 223.0, 226.0, 227.0,
    232.03806, 231.03588, 238.02891, 237.0, 244.0, 243.0, 247.0,
    251.0, 252.0,
], dtype=np.float64)


@dataclass(frozen=True)
class MCSettings:
    electrons: int
    energy_kev: float = 30.0
    energy_min_kev: float = 15.0
    energy_bin_kev: float = 1.0
    depth_max_nm: float = 100.0
    depth_step_nm: float = 1.0
    lateral_max_nm: float | None = None
    detector_size: int = 501
    sample_tilt_deg: float = 70.0
    omega_deg: float = 0.0
    steps: int = 300
    seed: int = 20261022
    batch_electrons: int = 262144

    @property
    def energy_bins(self):
        return int((self.energy_kev - self.energy_min_kev) / self.energy_bin_kev) + 1

    @property
    def depth_bins(self):
        return int(self.depth_max_nm / self.depth_step_nm) + 1

    @property
    def lateral_limit_nm(self):
        return self.depth_max_nm if self.lateral_max_nm is None else self.lateral_max_nm

    @property
    def lateral_radius(self):
        return math.ceil(self.lateral_limit_nm / self.depth_step_nm)


def material_properties(crystal):
    """Calculate EMsoft CalcDensity's average Z, atomic weight and density."""
    multiplicities = np.asarray([len(p) for p in crystal.positions], dtype=np.float64)
    z = np.asarray(crystal.atomic_numbers, dtype=np.int64)
    occupancy = np.asarray(crystal.occupancy, dtype=np.float64)
    if (len(z) == 0 or np.any(z > len(ATOMIC_WEIGHTS)) or
            np.any(multiplicities <= 0) or np.any(occupancy < 0)):
        raise ValueError('Crystal has unsupported elements or invalid occupied sites')
    atom_count = multiplicities.sum()
    occupied_count = np.sum(multiplicities * occupancy)
    average_z = float(np.sum(multiplicities * z) / atom_count)
    total_weight = float(np.sum(multiplicities * ATOMIC_WEIGHTS[z - 1] * occupancy))
    if occupied_count <= 0:
        raise ValueError('Crystal has no occupied atomic sites')
    average_weight = total_weight / occupied_count
    density = total_weight / (crystal.volume * 1e-21 * AVOGADRO)
    if not np.isfinite([average_z, average_weight, density]).all() or density <= 0:
        raise ValueError('Could not derive valid material properties from CrystalData')
    return average_z, average_weight, density


def _round_fortran(value):
    """Fortran NINT: nearest integer, with half values away from zero."""
    return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)


def _lambert_xy(x, y, z):
    norm = math.sqrt(x*x + y*y + z*z)
    x, y, z = x/norm, y/norm, z/norm
    if abs(z) == 1.0:
        return 0.0, 0.0
    scale_x, scale_y = 0.886226925452758, 1.253314137315500
    if abs(y) <= abs(x) and x != 0.0:
        q = math.copysign(math.sqrt(2.0*(1.0+z)), x)
        lx, ly = q*scale_x, q*math.atan(y/x)/scale_x
    elif y != 0.0:
        q = math.copysign(math.sqrt(2.0*(1.0+z)), y)
        lx, ly = q*math.atan(x/y)/scale_x, q*scale_x
    else:
        lx, ly = 0.0, 0.0
    return lx/scale_y, ly/scale_y


def _histogram_exit(accum_e, accum_z, lam_x, lam_y, depth, energy, settings):
    nx = (settings.detector_size - 1) // 2
    ix = _round_fortran(nx * lam_y)
    iy = _round_fortran(-nx * lam_x)
    if max(abs(ix), abs(iy)) > nx:
        return
    if not (energy > settings.energy_min_kev and energy <= settings.energy_kev):
        return
    ie = _round_fortran((energy-settings.energy_min_kev)/settings.energy_bin_kev)
    if not 0 <= ie < settings.energy_bins:
        return
    accum_e[ix+nx, iy+nx, ie] += 1
    iz = _round_fortran(abs(depth)/settings.depth_step_nm)
    coarse_radius = nx // 10
    px, py = _round_fortran(ix/10.0), _round_fortran(iy/10.0)
    if 0 <= iz < settings.depth_bins and abs(px) <= coarse_radius and abs(py) <= coarse_radius:
        accum_z[px+coarse_radius, py+coarse_radius, iz, ie] += 1


def _path_histogram_bin(x, zpos, settings):
    """Bin a scatter point by lateral distance from the incident beam and depth."""
    radius = settings.lateral_radius
    lateral_min = -radius * settings.depth_step_nm
    lateral = x - zpos * math.tan(math.radians(settings.sample_tilt_deg))
    ix = _round_fortran((lateral - lateral_min) / settings.depth_step_nm)
    iz = _round_fortran(zpos / settings.depth_step_nm)
    width = 2 * radius + 1
    return (ix, iz) if 0 <= ix < width and 0 <= iz < settings.depth_bins else None


def simulate_cpu(crystal, settings, include_path=False):
    """Small, slow, seeded reference implementation for tests and smoke runs."""
    zmean, atomic_weight, density = material_properties(crystal)
    size, nenergy = settings.detector_size, settings.energy_bins
    nz, nx = settings.depth_bins, (size-1)//2
    accum_e = np.zeros((size, size, nenergy), dtype=np.int64)
    coarse = 2*(nx//10)+1
    accum_z = np.zeros((coarse, coarse, nz, nenergy), dtype=np.int64)
    path_radius = settings.lateral_radius
    accum_path = np.zeros((2*path_radius+1, nz), dtype=np.int64) if include_path else None
    rng = np.random.default_rng(settings.seed)
    incident = math.radians(settings.sample_tilt_deg)
    omega = math.radians(settings.omega_deg)
    start_direction = (math.cos(omega)*math.sin(incident),
                       math.sin(omega)*math.sin(incident), math.cos(incident))
    for _ in range(settings.electrons):
        x = y = zpos = 0.0
        cx, cy, cz = start_direction
        energy = settings.energy_kev
        alpha = 3.4e-3*zmean**(2/3)/energy
        sige = 5.21*602.2*zmean*zmean/energy**2*4*math.pi/alpha/(1+alpha)*((energy+511)/(energy+1024))**2
        mfp = atomic_weight*1e7/(density*sige)
        step = -mfp*math.log(max(float(rng.random()), np.finfo(float).tiny))
        x += step*cx; y += step*cy; zpos += step*cz
        jmean = (9.76*zmean + 58.5*zmean**(-0.19))*1e-3
        dedx = -0.00785*(zmean/(atomic_weight*energy))*math.log(1.166*energy/jmean+0.9911)
        energy += step*density*dedx
        if accum_path is not None and zpos > 0.0:
            path_bin = _path_histogram_bin(x, zpos, settings)
            if path_bin is not None:
                accum_path[path_bin] += 1
        escaped = False
        escape_depth = 0.0
        for _step_index in range(settings.steps):
            if energy <= 0.0:
                break
            alpha = 3.4e-3*zmean**(2/3)/energy
            sige = 5.21*602.2*zmean*zmean/energy**2*4*math.pi/alpha/(1+alpha)*((energy+511)/(energy+1024))**2
            mfp = atomic_weight*1e7/(density*sige)
            distance = -mfp*math.log(max(float(rng.random()), np.finfo(float).tiny))
            dedx = -0.00785*(zmean/(atomic_weight*energy))*math.log(1.166*energy/jmean+0.9911)
            u = float(rng.random())
            phi = math.acos(np.clip(1.0-(2.0*alpha*u)/(1.0+alpha-u), -1.0, 1.0))
            psi = 2.0*math.pi*float(rng.random())
            if abs(cz) >= 0.99999:
                cnew = (math.sin(phi)*math.cos(psi), math.sin(phi)*math.sin(psi),
                        math.copysign(math.cos(phi), cz))
            else:
                dsq = math.sqrt(max(0.0, 1.0-cz*cz))
                cnew = (math.sin(phi)*(cx*cz*math.cos(psi)-cy*math.sin(psi))/dsq+cx*math.cos(phi),
                        math.sin(phi)*(cy*cz*math.cos(psi)+cx*math.sin(psi))/dsq+cy*math.cos(phi),
                        -math.sin(phi)*math.cos(psi)*dsq+cz*math.cos(phi))
            if abs(cnew[2]) > 1e-5:
                escape_depth = zpos/cnew[2]
            x += distance*cnew[0]; y += distance*cnew[1]; zpos += distance*cnew[2]
            energy += distance*density*dedx
            if accum_path is not None and zpos > 0.0:
                path_bin = _path_histogram_bin(x, zpos, settings)
                if path_bin is not None:
                    accum_path[path_bin] += 1
            cx, cy, cz = cnew
            if zpos <= 0.0:
                lam_x, lam_y = _lambert_xy(cx, cy, cz)
                _histogram_exit(accum_e, accum_z, lam_x, lam_y,
                                escape_depth, energy, settings)
                escaped = True
                break
        # Keep this assignment explicit: no exit event means no histogram count.
        del escaped
    if include_path:
        return accum_e, accum_z, accum_path
    return accum_e, accum_z


_CUDA_SOURCE = r'''// Keep the NVRTC kernel self-contained: NVRTC does not reliably search
// the host compiler's standard C header directories on HPC compute nodes.
typedef unsigned int uint32_t;
typedef int int32_t;

__device__ __forceinline__ uint32_t mix32(uint32_t x) {
  x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15;
  x *= 0x846ca68bU; x ^= x >> 16; return x;
}
__device__ __forceinline__ float uniform_lfsr(uint32_t *a, uint32_t *b,
                                                uint32_t *c, uint32_t *d) {
  uint32_t t;
  t = (((*a << 6) ^ *a) >> 13); *a = ((*a & 0xfffffffeU) << 18) ^ t;
  t = (((*b << 2) ^ *b) >> 27); *b = ((*b & 0xfffffff8U) << 2) ^ t;
  t = (((*c << 13) ^ *c) >> 21); *c = ((*c & 0xfffffff0U) << 7) ^ t;
  t = (((*d << 3) ^ *d) >> 12); *d = ((*d & 0xffffff80U) << 13) ^ t;
  int32_t v = (int32_t)(*a ^ *b ^ *c ^ *d);
  float u = fabsf((float)v / 2147483647.0f);
  return fminf(fmaxf(u, 1.0e-12f), 0.99999994f);
}
__device__ __forceinline__ void lambert(float x, float y, float z,
                                         float *lx, float *ly) {
  float m = rsqrtf(x*x + y*y + z*z); x*=m; y*=m; z*=m;
  if (fabsf(z) == 1.0f) { *lx=0.0f; *ly=0.0f; return; }
  const float p2=0.886226925452758f, pio2=1.253314137315500f;
  float q;
  if (fabsf(y)<=fabsf(x) && x!=0.0f) {
    q=copysignf(sqrtf(2.0f*(1.0f+z)),x);
    *lx=q*p2/pio2; *ly=q*atanf(y/x)/p2/pio2;
  } else if (y!=0.0f) {
    q=copysignf(sqrtf(2.0f*(1.0f+z)),y);
    *lx=q*atanf(x/y)/p2/pio2; *ly=q*p2/pio2;
  } else { *lx=0.0f; *ly=0.0f; }
}
__device__ __forceinline__ int nintf_away(float x) {
  return x >= 0.0f ? (int)floorf(x+0.5f) : (int)ceilf(x-0.5f);
}
extern "C" __global__ void simulate_mc(
  unsigned long long *hist_e, unsigned long long *hist_z,
  unsigned long long *hist_path,
  const long long offset, const long long count, const unsigned int seed,
  const float zmean, const float aw, const float rho, const float energy0,
  const float emin, const float ebin, const float depthstep,
  const float depthmax, const float lateralmax, const float pathstep, const int npath,
  const int numsx, const int ne, const int nz, const float sig, const float omega,
  const int steps) {
  long long tid=(long long)blockDim.x*blockIdx.x+threadIdx.x;
  if (tid>=count) return;
  uint32_t h=mix32(seed+(uint32_t)(offset+tid));
  uint32_t a=(mix32(h+1U)|2U), b=(mix32(h+2U)|8U);
  uint32_t c=(mix32(h+3U)|16U), d=(mix32(h+4U)|128U);
  const float pi=3.14159f;
  float x=0.0f,y=0.0f,zpos=0.0f;
  float cx=cosf(omega)*sinf(sig), cy=sinf(omega)*sinf(sig), cz=cosf(sig);
  float energy=energy0;
  float alpha=3.4e-3f*powf(zmean,0.66667f)/energy;
  float sige=(5.21f*602.2f)*zmean*zmean/(energy*energy)*4.0f*pi/alpha/(1.0f+alpha)*
    powf(energy+511.0f,2.0f)/powf(energy+1024.0f,2.0f);
  float mfp=aw*1.0e7f/(rho*sige);
  float dist=-mfp*logf(uniform_lfsr(&a,&b,&c,&d));
  x+=dist*cx; y+=dist*cy; zpos+=dist*cz;
  float jmean=(9.76f*zmean+58.5f*powf(zmean,-0.19f))*1.0e-3f;
  float dedx=-0.00785f*(zmean/(aw*energy))*logf(1.166f*energy/jmean+0.9911f);
  energy+=dist*rho*dedx;
  if (zpos>0.0f && zpos<=depthmax) {
    float lateral=x-zpos*tanf(sig);
    int ilat=nintf_away((lateral+ceilf(lateralmax/pathstep)*pathstep)/pathstep);
    int izpath=nintf_away(zpos/pathstep);
    if (ilat>=0 && ilat<npath && izpath>=0 && izpath<nz) {
      atomicAdd(hist_path+((long long)ilat*nz+izpath),1ULL);
    }
  }
  float escape_depth=0.0f;
  for (int k=0;k<steps && energy>0.0f;k++) {
    alpha=3.4e-3f*powf(zmean,0.66667f)/energy;
    sige=(5.21f*602.2f)*zmean*zmean/(energy*energy)*4.0f*pi/alpha/(1.0f+alpha)*
      powf(energy+511.0f,2.0f)/powf(energy+1024.0f,2.0f);
    mfp=aw*1.0e7f/(rho*sige);
    dist=-mfp*logf(uniform_lfsr(&a,&b,&c,&d));
    dedx=-0.00785f*(zmean/(aw*energy))*logf(1.166f*energy/jmean+0.9911f);
    float u=uniform_lfsr(&a,&b,&c,&d);
    float phi=acosf(fminf(1.0f,fmaxf(-1.0f,1.0f-(2.0f*alpha*u)/(1.0f+alpha-u))));
    float psi=2.0f*pi*uniform_lfsr(&a,&b,&c,&d);
    float sn=sinf(phi), cs=cosf(phi), cnx,cny,cnz;
    if (cz>=0.99999f || cz<=-0.99999f) {
      cnx=sn*cosf(psi); cny=sn*sinf(psi); cnz=copysignf(cs,cz);
    } else {
      float dsq=sqrtf(fmaxf(0.0f,1.0f-cz*cz));
      float inv=1.0f/dsq;
      cnx=sn*(cx*cz*cosf(psi)-cy*sinf(psi))*inv+cx*cs;
      cny=sn*(cy*cz*cosf(psi)+cx*sinf(psi))*inv+cy*cs;
      cnz=-sn*cosf(psi)*dsq+cz*cs;
    }
    if (fabsf(cnz)>1.0e-5f) escape_depth=zpos/cnz;
    x+=dist*cnx; y+=dist*cny; zpos+=dist*cnz;
    energy+=dist*rho*dedx;
    if (zpos>0.0f && zpos<=depthmax) {
      float lateral=x-zpos*tanf(sig);
      int ilat=nintf_away((lateral+ceilf(lateralmax/pathstep)*pathstep)/pathstep);
      int izpath=nintf_away(zpos/pathstep);
      if (ilat>=0 && ilat<npath && izpath>=0 && izpath<nz) {
        atomicAdd(hist_path+((long long)ilat*nz+izpath),1ULL);
      }
    }
    cx=cnx;cy=cny;cz=cnz;
    if (zpos<=0.0f) {
      float lx,ly; lambert(cx,cy,cz,&lx,&ly);
      int nx=(numsx-1)/2;
      int ix=nintf_away(nx*ly), iy=nintf_away(-nx*lx);
      if (abs(ix)<=nx && abs(iy)<=nx && energy>emin && energy<=energy0) {
        int ie=nintf_away((energy-emin)/ebin);
        if (ie>=0 && ie<ne) {
          long long ei=((long long)(ix+nx)*numsx+(iy+nx))*ne+ie;
          atomicAdd(hist_e+ei,1ULL);
          int cr=nx/10, px=nintf_away(ix/10.0f), py=nintf_away(iy/10.0f);
          int iz=nintf_away(fabsf(escape_depth)/depthstep);
          if (iz>=0 && iz<nz && abs(px)<=cr && abs(py)<=cr) {
            int csiz=2*cr+1;
            long long zi=(((long long)(px+cr)*csiz+(py+cr))*nz+iz)*ne+ie;
            atomicAdd(hist_z+zi,1ULL);
          }
        }
      }
      break;
    }
  }
}
'''


def _simulate_cuda_device(device, options, output_dir):
    import cupy as cp
    settings = MCSettings(**options['settings'])
    count = options['counts'][device]
    nx = (settings.detector_size-1)//2
    coarse = 2*(nx//10)+1
    with cp.cuda.Device(device):
        hist_e = cp.zeros((settings.detector_size, settings.detector_size,
                           settings.energy_bins), dtype=cp.uint64)
        hist_z = cp.zeros((coarse, coarse, settings.depth_bins,
                           settings.energy_bins), dtype=cp.uint64)
        path_radius = settings.lateral_radius
        path_bins = 2*path_radius+1
        hist_path = cp.zeros((path_bins, settings.depth_bins), dtype=cp.uint64)
        kernel = cp.RawKernel(_CUDA_SOURCE, 'simulate_mc', options=('--std=c++11',))
        zmean, aw, rho = options['material']
        block = 256
        sig = np.deg2rad(settings.sample_tilt_deg).item()
        omega = np.deg2rad(settings.omega_deg).item()
        started = 0
        while started < count:
            n = min(settings.batch_electrons, count-started)
            kernel(((n+block-1)//block,), (block,), (
                hist_e, hist_z, hist_path, np.int64(started), np.int64(n),
                np.uint32(settings.seed + device*0x9e3779b9), np.float32(zmean),
                np.float32(aw), np.float32(rho), np.float32(settings.energy_kev),
                np.float32(settings.energy_min_kev), np.float32(settings.energy_bin_kev),
                np.float32(settings.depth_step_nm), np.float32(settings.depth_max_nm),
                np.float32(settings.lateral_limit_nm),
                np.float32(settings.depth_step_nm), np.int32(path_bins),
                np.int32(settings.detector_size),
                np.int32(settings.energy_bins), np.int32(settings.depth_bins),
                np.float32(sig), np.float32(omega), np.int32(settings.steps)))
            started += n
            if started == count or started % max(settings.batch_electrons*10, 1) == 0:
                print(f'Monte Carlo GPU {device}: {started:,}/{count:,} trajectories', flush=True)
        np.savez_compressed(Path(output_dir)/f'mc-gpu-{device}.npz',
                            accum_e=cp.asnumpy(hist_e), accum_z=cp.asnumpy(hist_z),
                            accum_path=cp.asnumpy(hist_path))


def _write_hdf5(crystal_source, target, settings, accum_e, accum_z, accum_path, material, elapsed):
    import h5py
    from .master.physics import read_crystal
    from .master.model import read_depth_profiles

    target = Path(target)
    partial = target.with_name(target.name + '.partial')
    if target.exists() or partial.exists():
        raise FileExistsError(f'Output or partial file already exists: {target}')
    date = datetime.now(timezone.utc)
    with h5py.File(crystal_source, 'r') as source:
        if 'CrystalData' not in source:
            raise ValueError('Crystal input must be an EMsoft .xtal or HDF5 containing CrystalData')
        crystal = read_crystal(source['CrystalData'])
        zmean, aw, rho = material
        if accum_e.dtype != np.int32 or accum_z.dtype != np.int32:
            raise ValueError('Monte Carlo histogram counts exceed the EMsoft int32 HDF5 format')
        nml = {
            'stdout': 1, 'numsx': settings.detector_size, 'globalworkgrpsz': 256,
            'num_el': 1, 'totnum_el': settings.electrons, 'multiplier': 1,
            'devid': 0, 'platid': 0, 'ivolx': 1001, 'ivoly': 1001, 'ivolz': 101,
            'sig': settings.sample_tilt_deg, 'omega': settings.omega_deg,
            'EkeV': settings.energy_kev, 'Ehistmin': settings.energy_min_kev,
            'Ebinsize': settings.energy_bin_kev, 'depthmax': settings.depth_max_nm,
            'depthstep': settings.depth_step_nm, 'MCmode': 'full',
            'xtalname': str(crystal_source), 'dataname': str(target), 'mode': 'full',
        }
        nml_text = '&MCCLdata\n' + ''.join(f' {k} = {v!r},\n' for k,v in nml.items()) + '/\n'
        with h5py.File(partial, 'x') as out:
            out.attrs.update(kikuchicuda_mc_schema=1, complete=False,
                generator='KikuchiCUDA Python/CuPy Monte Carlo', scientifically_validated=False,
                settings_json=json.dumps({**asdict(settings), 'material': {
                    'average_atomic_number': zmean, 'average_atomic_weight': aw,
                    'density_g_cm3': rho}}, sort_keys=True))
            source.copy('CrystalData', out)
            nmlfiles = out.create_group('NMLfiles')
            nmlfiles.create_dataset('MCOpenCLNML', data=np.asarray([nml_text], dtype=h5py.string_dtype()))
            params = out.create_group('NMLparameters').create_group('MCCLNameList')
            for key, value in nml.items():
                if isinstance(value, str):
                    params.create_dataset(key, data=np.asarray([value], dtype=h5py.string_dtype()))
                elif key in ('sig','omega','EkeV','Ehistmin','Ebinsize','depthmax','depthstep'):
                    params[key] = np.asarray([value], dtype=np.float64)
                else:
                    params[key] = np.asarray([value], dtype=np.int32)
            header = out.create_group('EMheader').create_group('MCOpenCL')
            for key, value in {
                'Date': date.strftime('%Y-%m-%d'), 'StartTime': date.strftime('%H:%M:%S UTC'),
                'StopTime': datetime.now(timezone.utc).strftime('%H:%M:%S UTC'),
                'Duration': float(elapsed), 'ProgramName': 'KikuchiCUDA Python Monte Carlo',
                'Version': '1.0', 'HostName': os.uname().nodename,
            }.items():
                header.create_dataset(key, data=np.asarray([value], dtype=h5py.string_dtype()
                    if isinstance(value, str) else None))
            mc = out.create_group('EMData').create_group('MCOpenCL')
            mc.attrs['HDF_FileVersion'] = '4.0'
            mc['numzbins'] = [settings.depth_bins]
            mc['numEbins'] = [settings.energy_bins]
            mc['totnum_el'] = [settings.electrons]
            mc['multiplier'] = [1]
            mc.create_dataset('accum_e', data=accum_e, compression='gzip', shuffle=True)
            mc.create_dataset('accum_z', data=accum_z, compression='gzip', shuffle=True)
            path_dataset = mc.create_dataset('accum_path', data=accum_path,
                                             compression='gzip', shuffle=True)
            path_radius = settings.lateral_radius
            path_dataset.attrs['axis_order'] = 'lateral_x,depth_z'
            path_dataset.attrs['lateral_min_nm'] = -path_radius * settings.depth_step_nm
            path_dataset.attrs['lateral_max_nm'] = path_radius * settings.depth_step_nm
            path_dataset.attrs['lateral_step_nm'] = settings.depth_step_nm
            path_dataset.attrs['depth_step_nm'] = settings.depth_step_nm
            path_dataset.attrs['quantity'] = 'electron scattering-event counts'
            path_dataset.attrs['lateral_coordinate'] = 'x relative to the incident beam axis'
            out.attrs['complete'] = True
    os.replace(partial, target)
    print(f'Monte Carlo HDF5 written: {target}', flush=True)


def run(settings, crystal_source, output, devices=(0, 1), backend='cuda'):
    """Simulate electron trajectories and write a downstream-compatible MC HDF5."""
    import h5py
    from .master.physics import read_crystal
    with h5py.File(crystal_source, 'r') as source:
        if 'CrystalData' not in source:
            raise ValueError('Crystal input must be an EMsoft .xtal or HDF5 containing CrystalData')
        crystal = read_crystal(source['CrystalData'])
    material = material_properties(crystal)
    if settings.detector_size < 3 or settings.detector_size % 2 != 1:
        raise ValueError('detector-size must be an odd integer >= 3')
    if min(settings.electrons, settings.steps, settings.batch_electrons) < 1:
        raise ValueError('electrons, steps and batch-electrons must be positive')
    numeric_settings = (settings.energy_kev, settings.energy_min_kev,
        settings.energy_bin_kev, settings.depth_max_nm, settings.depth_step_nm,
        settings.lateral_limit_nm, settings.sample_tilt_deg, settings.omega_deg)
    if not np.isfinite(numeric_settings).all():
        raise ValueError('Monte Carlo settings must be finite')
    if (settings.energy_kev <= settings.energy_min_kev or settings.energy_bin_kev <= 0 or
            settings.depth_max_nm <= 0 or settings.depth_step_nm <= 0 or
            settings.lateral_limit_nm <= 0):
        raise ValueError('Invalid energy or depth histogram settings')
    if settings.seed < 0 or settings.seed > np.iinfo(np.uint32).max:
        raise ValueError('seed must be an unsigned 32-bit integer')
    if backend == 'cpu-reference':
        accum_e64, accum_z64, accum_path64 = simulate_cpu(crystal, settings, include_path=True)
        elapsed = 0.0
    elif backend == 'cuda':
        import cupy as cp
        devices = tuple(devices)
        visible = cp.cuda.runtime.getDeviceCount()
        if not devices or len(set(devices)) != len(devices) or any(d < 0 or d >= visible for d in devices):
            raise RuntimeError(f'Requested CUDA devices {devices}; only {visible} GPUs are visible')
        counts = [settings.electrons//len(devices)]*len(devices)
        for i in range(settings.electrons%len(devices)):
            counts[i] += 1
        counts_by_device = {device: counts[i] for i, device in enumerate(devices)}
        work = Path(output).with_name(Path(output).name+'.mcwork')
        work.mkdir(parents=True, exist_ok=False)
        options = {'settings': asdict(settings), 'material': material,
                   'counts': counts_by_device}
        start = time.perf_counter()
        context = multiprocessing.get_context('spawn')
        processes = [context.Process(target=_simulate_cuda_device, args=(device, options, work))
                     for device in devices if counts_by_device[device] > 0]
        for process in processes: process.start()
        for process in processes: process.join()
        if any(process.exitcode != 0 for process in processes):
            raise RuntimeError('A CUDA Monte Carlo worker failed; partial work is preserved')
        accum_e64 = np.zeros((settings.detector_size, settings.detector_size,
                              settings.energy_bins), dtype=np.uint64)
        nx = (settings.detector_size-1)//2
        accum_z64 = np.zeros((2*(nx//10)+1, 2*(nx//10)+1,
                              settings.depth_bins, settings.energy_bins), dtype=np.uint64)
        path_radius = math.ceil(settings.depth_max_nm / settings.depth_step_nm)
        accum_path64 = np.zeros((2*path_radius+1, settings.depth_bins), dtype=np.uint64)
        for device in devices:
            shard = work/f'mc-gpu-{device}.npz'
            if shard.exists():
                with np.load(shard) as f:
                    accum_e64 += f['accum_e']
                    accum_z64 += f['accum_z']
                    accum_path64 += f['accum_path']
        elapsed = time.perf_counter()-start
        if (np.any(accum_e64 > np.iinfo(np.int32).max) or
                np.any(accum_z64 > np.iinfo(np.int32).max) or
                np.any(accum_path64 > np.iinfo(np.int32).max)):
            raise OverflowError('Counts exceed EMsoft int32 HDF5 limits; reduce the electron count')
        # Keep shards and settings if HDF5 writing fails, for diagnosis.
    else:
        raise ValueError("backend must be 'cuda' or 'cpu-reference'")
    if (np.any(accum_e64 > np.iinfo(np.int32).max) or
            np.any(accum_z64 > np.iinfo(np.int32).max) or
            np.any(accum_path64 > np.iinfo(np.int32).max)):
        raise OverflowError('Counts exceed EMsoft int32 HDF5 limits; reduce the electron count')
    accum_e = np.asarray(accum_e64, dtype=np.int32)
    accum_z = np.asarray(accum_z64, dtype=np.int32)
    accum_path = np.asarray(accum_path64, dtype=np.int32)
    _write_hdf5(crystal_source, output, settings, accum_e, accum_z,
                accum_path, material, elapsed)
    if backend == 'cuda':
        import shutil
        shutil.rmtree(work)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--crystal', type=Path, required=True,
                        help='EMsoft .xtal or HDF5 containing CrystalData; MC data are not read')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--electrons', type=int, required=True,
                        help='Number of simulated incident electrons; choose explicitly')
    parser.add_argument('--energy-kev', type=float, default=30.0)
    parser.add_argument('--energy-min-kev', type=float, default=15.0)
    parser.add_argument('--energy-bin-kev', type=float, default=1.0)
    parser.add_argument('--depth-max-nm', type=float, default=100.0)
    parser.add_argument('--depth-step-nm', type=float, default=1.0)
    parser.add_argument('--lateral-max-nm', type=float,
                        help='Maximum absolute lateral distance recorded; defaults to depth-max-nm')
    parser.add_argument('--detector-size', type=int, default=501)
    parser.add_argument('--sample-tilt-deg', type=float, default=70.0)
    parser.add_argument('--omega-deg', type=float, default=0.0)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--seed', type=int, default=20261022)
    parser.add_argument('--batch-electrons', type=int, default=262144)
    parser.add_argument('--backend', choices=('cuda','cpu-reference'), default='cuda')
    parser.add_argument('--devices', type=int, nargs='+', default=[0,1])
    args = parser.parse_args()
    settings = MCSettings(electrons=args.electrons, energy_kev=args.energy_kev,
        energy_min_kev=args.energy_min_kev, energy_bin_kev=args.energy_bin_kev,
        depth_max_nm=args.depth_max_nm, depth_step_nm=args.depth_step_nm,
        lateral_max_nm=args.lateral_max_nm,
        detector_size=args.detector_size, sample_tilt_deg=args.sample_tilt_deg,
        omega_deg=args.omega_deg, steps=args.steps, seed=args.seed,
        batch_electrons=args.batch_electrons)
    run(settings, args.crystal.resolve(strict=True), args.output.resolve(),
        tuple(args.devices), args.backend)


if __name__ == '__main__':
    main()
