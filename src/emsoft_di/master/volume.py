"""Export an interactive electron-penetration heat map from EMsoft OpenCL MC data."""
import argparse
from html import escape
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


def read_interaction_map(path):
    """Read trajectory scatter-event counts as lateral position by depth."""
    with h5py.File(path, 'r') as source:
        key = 'EMData/MCOpenCL/accum_path'
        if key not in source:
            raise ValueError(
                'This HDF5 has no trajectory-position map. Rerun the Python Monte Carlo '
                'with the updated code; legacy EMsoft MC files only store exit/depth histograms.')
        dataset = source[key]
        if dataset.ndim != 2 or min(dataset.shape) < 1:
            raise ValueError('accum_path must have shape (lateral_x, depth_z)')
        try:
            lateral_min = float(dataset.attrs['lateral_min_nm'])
            lateral_step = float(dataset.attrs['lateral_step_nm'])
            depth_step = float(dataset.attrs['depth_step_nm'])
        except KeyError as exc:
            raise ValueError(f'Missing trajectory-map axis metadata: {exc}') from exc
        counts = np.asarray(dataset[...])
    if (not np.issubdtype(counts.dtype, np.number) or not np.isfinite(counts).all()
            or np.any(counts < 0) or not np.isfinite([lateral_min, lateral_step, depth_step]).all()
            or lateral_step <= 0 or depth_step <= 0):
        raise ValueError('Invalid trajectory-position counts or axis metadata')
    lateral = lateral_min + (np.arange(counts.shape[0]) + .5) * lateral_step
    depth = (np.arange(counts.shape[1]) + .5) * depth_step
    return counts, lateral, depth


def _write_heatmap_svg(output, values, x, y, title, x_label, y_label):
    """Write a dependency-free vector heat map with axes and a color scale."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape != (len(y), len(x)):
        raise ValueError('SVG heat map values must have shape (len(y), len(x))')
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('SVG heat map values must be finite and nonnegative')
    logged = np.log10(values + 1.0)
    maximum = float(logged.max())
    normalized = logged / maximum if maximum > 0 else np.zeros_like(logged)
    palette = [
        (0.00, (0, 0, 4)), (0.20, (66, 10, 104)),
        (0.40, (147, 38, 103)), (0.60, (221, 81, 58)),
        (0.80, (244, 153, 31)), (1.00, (252, 255, 164)),
    ]

    def color(value):
        for (left, c0), (right, c1) in zip(palette, palette[1:]):
            if value <= right:
                fraction = (value - left) / (right - left)
                rgb = [round(a + fraction * (b - a)) for a, b in zip(c0, c1)]
                return '#%02x%02x%02x' % tuple(rgb)
        return '#fcffa4'

    width, height = 1000, 700
    left, top, plot_width, plot_height = 115, 75, 720, 520
    cell_width, cell_height = plot_width / len(x), plot_height / len(y)
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2:g}" y="36" text-anchor="middle" font-family="Arial,sans-serif" font-size="22">{escape(title)}</text>',
        '<g shape-rendering="crispEdges">',
    ]
    for iy in range(len(y)):
        for ix in range(len(x)):
            fill = color(float(normalized[iy, ix]))
            px = left + ix * cell_width
            py = top + iy * cell_height
            pieces.append(f'<rect x="{px:.3f}" y="{py:.3f}" width="{cell_width+0.03:.3f}" height="{cell_height+0.03:.3f}" fill="{fill}"/>')
    pieces.append('</g>')
    pieces.append(f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#222" stroke-width="1"/>')

    def tick_text(value):
        return f'{value:.4g}'

    for tick in range(5):
        fraction = tick / 4
        xv = float(x[0] + fraction * (x[-1] - x[0])) if len(x) > 1 else float(x[0])
        px = left + fraction * plot_width
        pieces.append(f'<line x1="{px:.2f}" y1="{top+plot_height}" x2="{px:.2f}" y2="{top+plot_height+6}" stroke="#222"/>')
        pieces.append(f'<text x="{px:.2f}" y="{top+plot_height+24}" text-anchor="middle" font-family="Arial,sans-serif" font-size="13">{tick_text(xv)}</text>')
        yv = float(y[0] + fraction * (y[-1] - y[0])) if len(y) > 1 else float(y[0])
        py = top + fraction * plot_height
        pieces.append(f'<line x1="{left-6}" y1="{py:.2f}" x2="{left}" y2="{py:.2f}" stroke="#222"/>')
        pieces.append(f'<text x="{left-10}" y="{py+4:.2f}" text-anchor="end" font-family="Arial,sans-serif" font-size="13">{tick_text(yv)}</text>')
    pieces.append(f'<text x="{left+plot_width/2}" y="{top+plot_height+55}" text-anchor="middle" font-family="Arial,sans-serif" font-size="16">{escape(x_label)}</text>')
    pieces.append(f'<text transform="translate(28 {top+plot_height/2}) rotate(-90)" text-anchor="middle" font-family="Arial,sans-serif" font-size="16">{escape(y_label)}</text>')

    cb_x, cb_y, cb_w, cb_h = 875, top, 24, plot_height
    for band in range(100):
        fraction = 1.0 - band / 99
        py = cb_y + band * cb_h / 100
        pieces.append(f'<rect x="{cb_x}" y="{py:.2f}" width="{cb_w}" height="{cb_h/100+0.1:.2f}" fill="{color(fraction)}"/>')
    pieces.append(f'<rect x="{cb_x}" y="{cb_y}" width="{cb_w}" height="{cb_h}" fill="none" stroke="#222"/>')
    for fraction in (0, .5, 1):
        py = cb_y + (1 - fraction) * cb_h
        label = f'{fraction*maximum:.3g}'
        pieces.append(f'<text x="{cb_x+cb_w+8}" y="{py+4:.2f}" font-family="Arial,sans-serif" font-size="12">{label}</text>')
    pieces.append(f'<text transform="translate(970 {top+plot_height/2}) rotate(-90)" text-anchor="middle" font-family="Arial,sans-serif" font-size="13">log10(count + 1)</text>')
    pieces.append('</svg>')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    output.write_text('\n'.join(pieces), encoding='utf-8')


def export_penetration_heatmap(monte_carlo, output, view='auto', svg_output=None):
    """Write a self-contained interactive electron-penetration heat map."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError("Install the HPC extra with `python -m pip install -e '.[hpc]'`") from exc
    if view == 'auto':
        with h5py.File(monte_carlo, 'r') as source:
            view = ('interaction' if 'EMData/MCOpenCL/accum_path' in source
                    else 'depth-energy')
        print(f'Volume view selected: {view}', flush=True)
    if view == 'interaction':
        profile, lateral_nm, depth_nm = read_interaction_map(monte_carlo)
        if profile.max() <= 0:
            raise ValueError('accum_path contains no scattering events to display')
        logged = np.log10(profile.T + 1)
        svg_values, svg_x, svg_y = profile.T, lateral_nm, depth_nm
        svg_title = 'Electron interaction volume (Monte Carlo scatter events)'
        svg_x_label = 'Lateral distance from incident beam axis (nm)'
        svg_y_label = 'Depth below sample surface (nm)'
        heatmap = go.Heatmap(x=lateral_nm, y=depth_nm, z=logged,
            customdata=profile.T, colorscale='Inferno',
            colorbar=dict(title='log10(scatter events + 1)'),
            hovertemplate='Depth: %{y:.4g} nm<br>Lateral distance: %{x:.4g} nm'
                          '<br>Scatter events: %{customdata:.0f}<extra></extra>')
        figure = go.Figure(data=[heatmap])
        figure.update_layout(title='Electron interaction volume (Monte Carlo scatter events)',
            xaxis_title='Lateral distance from incident beam axis (nm)',
            yaxis_title='Depth below sample surface (nm)',
            yaxis=dict(autorange='reversed'),
            margin=dict(l=85, r=35, t=65, b=75),
            annotations=[dict(text='Cross-section in the beam-tilt plane; logarithmic color scale; hover shows event counts.',
                               x=0, y=-.25, xref='paper', yref='paper', showarrow=False,
                               xanchor='left')])
    elif view == 'depth-energy':
        counts, energies, depth_step = read_mc_volume(monte_carlo)
        profile = depth_energy_counts(counts)
        if profile.max() <= 0:
            raise ValueError('accum_z contains no electron counts to display')
        depth_nm = (np.arange(profile.shape[0], dtype=np.float64) + .5) * depth_step
        svg_values, svg_x, svg_y = profile.T, depth_nm, energies
        svg_title = 'Electron penetration by Monte Carlo depth and energy'
        svg_x_label = 'Monte Carlo depth z (nm)'
        svg_y_label = 'Incident energy (keV)'
        heatmap = go.Heatmap(x=depth_nm, y=energies, z=np.log10(profile.T + 1),
            customdata=profile.T, colorscale='Magma',
            colorbar=dict(title='log10(electrons + 1)'),
            hovertemplate='Energy: %{y:g} keV<br>Depth z: %{x:.4g} nm'
                          '<br>Electrons (summed over x-y): %{customdata:.0f}<extra></extra>')
        figure = go.Figure(data=[heatmap])
        figure.update_layout(title='Electron penetration by Monte Carlo depth and energy',
            xaxis_title='Monte Carlo depth z (nm)', yaxis_title='Incident energy (keV)',
            margin=dict(l=75, r=35, t=65, b=65))
    else:
        raise ValueError("view must be 'interaction' or 'depth-energy'")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    figure.write_html(output, full_html=True, include_plotlyjs=True, auto_play=False)
    print(f'Interactive electron-penetration heat map: {output}', flush=True)
    if svg_output is not None:
        svg_output = Path(svg_output)
        _write_heatmap_svg(svg_output, svg_values, svg_x, svg_y,
                           svg_title, svg_x_label, svg_y_label)
        print(f'Static SVG heat map: {svg_output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--monte-carlo', type=Path, required=True,
                        help='EMsoft HDF5 containing EMData/MCOpenCL/accum_z')
    parser.add_argument('--output', type=Path, required=True,
                        help='New self-contained interactive HTML output')
    parser.add_argument('--view', choices=('auto', 'interaction', 'depth-energy'), default='auto',
                        help='Auto-select trajectory cross-section when available; otherwise legacy depth-vs-energy')
    parser.add_argument('--svg', type=Path,
                        help='Optional path for an additional static vector SVG export')
    args = parser.parse_args()
    export_penetration_heatmap(args.monte_carlo.resolve(strict=True), args.output.resolve(),
                               args.view, args.svg.resolve() if args.svg else None)


if __name__ == '__main__':
    main()
