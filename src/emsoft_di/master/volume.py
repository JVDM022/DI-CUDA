"""Export an interactive electron-penetration heat map from EMsoft OpenCL MC data."""
import argparse
from pathlib import Path

import h5py
import numpy as np


def read_mc_volume(path):
    """Read OpenCL electron counts in (x, y, depth, energy) order."""
    with h5py.File(path, 'r') as source:
        try:
            dataset = source['EMData/MCOpenCL/accum_z']
            mc = source['EMData/MCOpenCL']
            nml = source['NMLparameters/MCCLNameList']
            bins = int(np.asarray(mc['numEbins'][...]).item())
            start = float(np.asarray(nml['Ehistmin'][...]).item())
            step = float(np.asarray(nml['Ebinsize'][...]).item())
            depth_step = float(np.asarray(nml['depthstep'][...]).item())
        except KeyError as exc:
            raise ValueError(f'Missing required OpenCL Monte Carlo dataset or metadata: {exc}') from exc
        if dataset.ndim != 4 or dataset.shape[-1] != bins or min(dataset.shape[:3]) < 1:
            raise ValueError('accum_z must have shape (x, y, depth, numEbins)')
        if (bins < 1 or not np.isfinite([start, step, depth_step]).all() or
                start <= 0 or step < 0 or depth_step <= 0):
            raise ValueError('Invalid Monte Carlo energy or depth metadata')
        counts = np.asarray(dataset[...])
    if not np.issubdtype(counts.dtype, np.number) or not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError('accum_z contains invalid electron counts')
    energies = start + np.arange(bins, dtype=np.float64) * step
    return counts, energies, depth_step


def depth_energy_counts(counts):
    """Sum x/y positions, retaining electron counts for every z and energy bin."""
    return np.sum(counts, axis=(0, 1), dtype=np.float64)


def export_penetration_heatmap(monte_carlo, output):
    """Write a self-contained heat map of integrated electron counts by z/energy."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError("Install the HPC extra with `python -m pip install -e '.[hpc]'`") from exc
    counts, energies, depth_step = read_mc_volume(monte_carlo)
    profile = depth_energy_counts(counts)
    if profile.max() <= 0:
        raise ValueError('accum_z contains no electron counts to display')
    z_nm = (np.arange(profile.shape[0], dtype=np.float64) + .5) * depth_step
    log_profile = np.log10(profile.T + 1)
    heatmap = go.Heatmap(x=z_nm, y=energies, z=log_profile,
        customdata=profile.T, colorscale='Magma',
        colorbar=dict(title='log10(electrons + 1)'),
        hovertemplate='Energy: %{y:g} keV<br>Depth z: %{x:.4g} nm'
                      '<br>Electrons (summed over x-y): %{customdata:.0f}<extra></extra>')
    figure = go.Figure(data=[heatmap])
    figure.update_layout(title='Electron penetration by Monte Carlo depth and energy',
        xaxis_title='Monte Carlo depth z (nm)', yaxis_title='Incident energy (keV)',
        margin=dict(l=75, r=35, t=65, b=65),
        annotations=[dict(text='Color is log-scaled; hover shows the summed electron count for each depth/energy bin.',
                           x=0, y=-.22, xref='paper', yref='paper', showarrow=False,
                           xanchor='left')])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    figure.write_html(output, full_html=True, include_plotlyjs=True, auto_play=False)
    print(f'Interactive electron-penetration heat map: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--monte-carlo', type=Path, required=True,
                        help='EMsoft HDF5 containing EMData/MCOpenCL/accum_z')
    parser.add_argument('--output', type=Path, required=True,
                        help='New self-contained interactive HTML output')
    args = parser.parse_args()
    export_penetration_heatmap(args.monte_carlo.resolve(strict=True), args.output.resolve())


if __name__ == '__main__':
    main()
