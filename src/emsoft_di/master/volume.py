"""Export an interactive electron-penetration heat map from EMsoft OpenCL MC data."""
import argparse
from html import escape
import json
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


def read_sample_tilt(path):
    """Read the simulated sample tilt angle stored by the Python MC writer."""
    with h5py.File(path, 'r') as source:
        metadata = source.attrs.get('settings_json')
    if metadata is None:
        return 0.0
    if isinstance(metadata, bytes):
        metadata = metadata.decode('utf-8')
    try:
        angle = float(json.loads(metadata).get('sample_tilt_deg', 0.0))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError('Invalid sample_tilt_deg metadata in Monte Carlo HDF5') from exc
    if not np.isfinite(angle) or abs(angle) >= 90:
        raise ValueError('Sample tilt must be finite and between -90 and 90 degrees')
    return angle


def tilt_interaction_map(profile, lateral_nm, depth_nm, tilt_deg, max_bins=900):
    """Rotate beam-centered sample coordinates into the lab frame.

    The sample surface is tilted by ``tilt_deg`` from horizontal; lab Z/beam
    remains vertical. The input lateral coordinate is the offset from the
    incident beam axis and depth is measured along the sample normal.
    """
    values = np.asarray(profile, dtype=np.float64).T  # depth by lateral
    lateral_nm = np.asarray(lateral_nm, dtype=np.float64)
    depth_nm = np.asarray(depth_nm, dtype=np.float64)
    if values.shape != (len(depth_nm), len(lateral_nm)):
        raise ValueError('Interaction map shape does not match its axes')
    if values.size == 0 or max_bins < 2:
        raise ValueError('Interaction map and display resolution must be nonempty')

    # Bound browser/SVG size while retaining total scatter counts in each display bin.
    def reduce_axis(array, axis, coordinates):
        count = array.shape[axis]
        bins = min(count, max_bins)
        edges = np.linspace(0, count, bins + 1, dtype=np.int64)
        reduced = np.add.reduceat(array, edges[:-1], axis=axis)
        centers = (edges[:-1] + edges[1:] - 1) / 2
        step = (coordinates[1] - coordinates[0]) if len(coordinates) > 1 else 1.0
        coords = coordinates[0] + centers * step
        return reduced, coords

    values, depth_nm = reduce_axis(values, 0, depth_nm)
    values, lateral_nm = reduce_axis(values, 1, lateral_nm)
    theta = np.deg2rad(tilt_deg)
    cosine, sine = np.cos(theta), np.sin(theta)
    x_min, x_max = lateral_nm[0] * cosine, lateral_nm[-1] * cosine
    dx0 = lateral_nm[1] - lateral_nm[0] if len(lateral_nm) > 1 else 1.0
    dx = max(dx0 * cosine, (x_max - x_min) / (max_bins - 1))
    lab_x = np.arange(x_min, x_max + dx * 0.5, dx)

    z_min = lateral_nm[0] * sine + depth_nm[0] / cosine
    z_max = lateral_nm[-1] * sine + depth_nm[-1] / cosine
    dz0 = depth_nm[1] - depth_nm[0] if len(depth_nm) > 1 else 1.0
    dz = max(min(dx0 * abs(sine), dz0 / cosine), (z_max - z_min) / (max_bins - 1))
    lab_z = np.arange(z_min, z_max + dz * 0.5, dz)

    # Inverse rotation: lab (X,Z) -> beam-centered lateral offset and sample depth.
    lateral_query = lab_x / cosine
    depth_query = (lab_z[:, None] - lateral_query[None, :] * sine) * cosine
    ix1 = np.searchsorted(lateral_nm, lateral_query, side='right')
    iz1 = np.searchsorted(depth_nm, depth_query, side='right')
    valid = ((lateral_query[None, :] >= lateral_nm[0]) &
             (lateral_query[None, :] <= lateral_nm[-1]) &
             (depth_query >= depth_nm[0]) & (depth_query <= depth_nm[-1]))
    ix1 = np.clip(ix1, 1, len(lateral_nm) - 1)
    iz1 = np.clip(iz1, 1, len(depth_nm) - 1)
    ix0, iz0 = ix1 - 1, iz1 - 1
    fx = np.clip((lateral_query - lateral_nm[ix0]) /
                 (lateral_nm[ix1] - lateral_nm[ix0]), 0, 1)
    fz = np.clip((depth_query - depth_nm[iz0]) /
                 (depth_nm[iz1] - depth_nm[iz0]), 0, 1)
    lower = values[iz0, ix0] * (1 - fx)[None, :] + values[iz0, ix1] * fx[None, :]
    upper = values[iz1, ix0] * (1 - fx)[None, :] + values[iz1, ix1] * fx[None, :]
    rotated = np.where(valid, lower * (1 - fz) + upper * fz, 0.0)

    # The beam is vertical in lab Z; the unrotated surface is a line at theta.
    beam_line = (np.zeros(2), np.asarray([depth_nm[0] / cosine,
                                          depth_nm[-1] / cosine]))
    surface_x = np.asarray([lateral_nm[0] * cosine, lateral_nm[-1] * cosine])
    surface_line = (surface_x, surface_x * np.tan(theta))
    return rotated.T, lab_x, lab_z, beam_line, surface_line


def _write_heatmap_svg(output, values, x, y, title, x_label, y_label,
                       *, beam_line=None, surface_line=None, equal_aspect=False):
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
    left, top = 115, 75
    plot_width, plot_height = 720, 520
    if equal_aspect and len(x) > 1 and len(y) > 1:
        x_range, y_range = float(x[-1] - x[0]), float(y[-1] - y[0])
        scale = min(720 / x_range, 520 / y_range)
        plot_width, plot_height = x_range * scale, y_range * scale
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
    for line, line_color, dash, line_width in (
        (surface_line, '#8eeaff', '', 2.5),
        (beam_line, 'white', '7 5', 2.0),
    ):
        if line is not None:
            line_x, line_y = line
            points = ' '.join(
                f'{left + (float(lx) - x[0]) / (x[-1] - x[0]) * plot_width:.2f},'
                f'{top + (float(ly) - y[0]) / (y[-1] - y[0]) * plot_height:.2f}'
                for lx, ly in zip(line_x, line_y)
            )
            dash_attr = f' stroke-dasharray="{dash}"' if dash else ''
            pieces.append(f'<polyline points="{points}" fill="none" stroke="{line_color}" stroke-width="{line_width}"{dash_attr}/>')
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


def export_penetration_heatmap(monte_carlo, output, view='auto', svg_output=None,
                               tilt_deg=None, fit_aspect=False):
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
        angle = read_sample_tilt(monte_carlo) if tilt_deg is None else float(tilt_deg)
        if not np.isfinite(angle) or abs(angle) >= 90:
            raise ValueError('Tilt angle must be finite and between -90 and 90 degrees')
        profile, lateral_nm, depth_nm, beam_line, surface_line = tilt_interaction_map(
            profile, lateral_nm, depth_nm, angle)
        logged = np.log10(profile.T + 1)
        svg_values, svg_x, svg_y = profile.T, lateral_nm, depth_nm
        aspect_note = ' (display fitted; angle not to scale)' if fit_aspect else ''
        svg_title = f'Electron interaction volume (lab frame; sample surface {angle:g}°){aspect_note}'
        svg_x_label = 'Lab X (nm)'
        svg_y_label = 'Lab Z / gravity direction (nm)'
        heatmap = go.Heatmap(x=lateral_nm, y=depth_nm, z=logged,
            customdata=profile.T, colorscale='Inferno',
            colorbar=dict(title='log10(scatter events + 1)'),
            hovertemplate='Depth: %{y:.4g} nm<br>Lateral distance: %{x:.4g} nm'
                          '<br>Scatter events: %{customdata:.0f}<extra></extra>')
        beam = go.Scatter(x=beam_line[0], y=beam_line[1], mode='lines', name='Vertical beam / lab Z',
                          line=dict(color='white', dash='dash', width=2), hoverinfo='skip')
        surface = go.Scatter(x=surface_line[0], y=surface_line[1], mode='lines',
                             name='Sample surface',
                             line=dict(color='#8eeaff', width=2), hoverinfo='skip')
        figure = go.Figure(data=[heatmap, surface, beam])
        figure.update_layout(title=svg_title,
            xaxis_title='Lab X (nm)',
            yaxis_title='Lab Z / gravity direction (nm)',
            xaxis=(dict() if fit_aspect else dict(scaleanchor='y', scaleratio=1)),
            yaxis=dict(autorange='reversed'),
            margin=dict(l=85, r=35, t=65, b=75),
            annotations=[dict(text=f'Lab-frame view; sample surface at {angle:g}° to horizontal; dashed line is vertical beam / gravity Z; logarithmic colors show binned scatter-event counts.' + (' Display fitted to panel; apparent angle is not to scale.' if fit_aspect else ''),
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
                           svg_title, svg_x_label, svg_y_label,
                           beam_line=beam_line if view == 'interaction' else None,
                           surface_line=surface_line if view == 'interaction' else None,
                           equal_aspect=view == 'interaction' and not fit_aspect)
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
    parser.add_argument('--tilt-deg', type=float,
                        help='Sample-frame display tilt in degrees; defaults to MC metadata')
    parser.add_argument('--fit-aspect', action='store_true',
                        help='Fill the plot panel; apparent geometry is not to scale')
    args = parser.parse_args()
    export_penetration_heatmap(args.monte_carlo.resolve(strict=True), args.output.resolve(),
                               args.view, args.svg.resolve() if args.svg else None,
                               args.tilt_deg, args.fit_aspect)


if __name__ == '__main__':
    main()
