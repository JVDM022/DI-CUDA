"""Launch with streamlit run app/indexing_app.py."""
from pathlib import Path
import json
import threading
import queue
import traceback
import numpy as np
import streamlit as st
from emsoft_di.workflow import Settings, inspect, hardware, run
from emsoft_di.hpc import preprocess

st.set_page_config(page_title='KikuchiCUDA · Indexing Studio', page_icon='◈', layout='wide')
st.markdown('''<style>.stApp {background: #f6f8fc} h1 {letter-spacing:-1.5px}
[data-testid="stMetric"] {background:white;padding:18px;border-radius:14px;border:1px solid #e2e8f0}
.stButton>button {border-radius:9px} </style>''', unsafe_allow_html=True)
st.caption('KIKUCHICUDA / INDEXING STUDIO')
st.title('From patterns to orientations')
st.write('Inspect your scan, run a pilot, then index the full dataset. Work stays on this computer.')
if 'jobs' not in st.session_state:
    st.session_state.jobs = None
job = st.session_state.jobs
busy = job is not None and job['thread'].is_alive()
with st.sidebar:
    st.subheader('Compute')
    hw = hardware()
    st.success('CUDA: '+', '.join(hw['devices'])) if hw['devices'] else st.info('CPU ready · no CUDA GPU detected')
    if not hw['devices']:
        st.caption(hw['reason'])
    st.markdown('**First run**\n\n1. Select files by local path.\n2. Inspect inputs and preview.\n3. Start the 128-pattern pilot.\n4. Review quality before a full scan.')
    st.caption('ANG supplies detector calibration and coordinates. The master supplies crystal symmetry. Use a new run folder when changing settings.')
left, right = st.columns([1.1, 1])
with left:
    st.subheader('1 · Import & configure')
    st.caption('Use paths on the machine running this app; large UP files are streamed from disk.')
    patterns = st.text_input('Experimental patterns (.up1 / .up2)', disabled=busy)
    ang = st.text_input('Scan and detector calibration (.ang)', disabled=busy)
    master = st.text_input('Master pattern (.h5 / supported .sht)', disabled=busy)
    output = st.text_input('Run folder', value=str(Path.cwd()/'runs'/'pilot-001'), disabled=busy)
    with st.expander('Indexing settings', expanded=True):
        a, b = st.columns(2)
        resolution = a.number_input('Sampling (degrees)', .1, 30., 3., disabled=busy)
        binning = b.selectbox('Detector binning', [1, 2, 4, 8], index=1, disabled=busy)
        sigma = a.number_input('Background sigma (binned pixels)', 0., 100., 3., disabled=busy)
        limit = b.number_input('Pattern limit (0 = all)', 0, value=128, disabled=busy)
        device = a.selectbox('Device', ['auto', 'cpu', 'cuda'], disabled=busy)
        energy = b.selectbox('Energy weighting', ['mc-global', 'highest-bin'], disabled=busy)
        scan_mapping = st.selectbox('UP acquisition order', ['raster', 'ang-order'], disabled=busy,
            help='Raster includes rectangular padding slots for hex scans. ANG-order requires one UP pattern per ANG row. Confirm against acquisition settings.')
    with st.expander('Memory & batching'):
        batch = st.number_input('Patterns / projection batch', 1, value=32, disabled=busy)
        dbatch = st.number_input('Dictionary batch', 1, value=2048, disabled=busy)
        keep = st.number_input('Top matches to retain', 2, value=2, disabled=busy)
    s = Settings(patterns, ang, master, output, resolution, binning, sigma, batch, dbatch,
                 keep, limit, device, energy, scan_mapping)
with right:
    st.subheader('2 · Inspect & preview')
    if st.button('Inspect inputs', disabled=busy, use_container_width=True):
        try:
            raw, cal, xy, ix = inspect(s)
            st.success(f'{len(raw):,} patterns · {raw.shape[2]} × {raw.shape[1]} pixels · {len(xy):,} scan points')
            st.json(cal, expanded=False)
            # Sample nonblank patterns, retaining exact numerical preprocessing.
            sample = min(len(raw)-1, int(np.argmax(np.std(raw[:min(32, len(raw))], axis=(1, 2)))))
            images = [np.asarray(raw[sample], dtype=float), preprocess(raw[sample:sample+1], binning, sigma)[0]]
            cols = st.columns(2)
            for col, image, title in zip(cols, images, ['Raw pattern', 'Preprocessed']):
                lo, hi = np.nanpercentile(image, [1, 99])
                col.image(np.clip((image-lo)/max(hi-lo, 1e-12), 0, 1), caption=title, clamp=True)
            st.caption('Display contrast uses 1st–99th percentiles; indexing uses the original preprocessed values.')
            st.scatter_chart({'x': xy[:10000, 0], 'y': -xy[:10000, 1]}, x='x', y='y')
        except Exception as exc:
            st.error(str(exc))
    st.info('Default: 128-pattern pilot, 3° sampling, 2× binning, Gaussian background subtraction and NCC. Full indexing can require substantial time and disk space.')
    st.subheader('3 · Run & review')
    if st.button('Start / resume indexing', type='primary', disabled=busy, use_container_width=True):
        try:
            inspect(s)
            messages, stop = queue.Queue(), threading.Event()
            def worker(settings, events, stop_event):
                try:
                    run(settings, lambda *args: events.put(('progress', args)), stop_event.is_set)
                    events.put(('finished', None))
                except Exception:
                    events.put(('error', traceback.format_exc()))
            thread = threading.Thread(target=worker, args=(s, messages, stop), daemon=True)
            st.session_state.jobs = dict(thread=thread, messages=messages, stop=stop, output=output, last=None, error=None)
            thread.start()
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

@st.fragment(run_every='1s')
def monitor():
    job = st.session_state.jobs
    if job:
        while not job['messages'].empty():
            kind, payload = job['messages'].get_nowait()
            if kind == 'progress': job['last'] = payload
            if kind == 'error': job['error'] = payload
        if job['last']:
            stage, done, total = job['last']
            st.progress(min(done/max(total, 1), 1.), text=f'{stage} · {done:,} / {total:,}')
        if job['thread'].is_alive():
            if st.button('Pause after current batch'):
                job['stop'].set()
            st.caption('You can resume using the same inputs, settings and run folder. Closing the server stops the process; completed batches remain saved.')
        if not job['thread'].is_alive() and not job.get('ui_done'):
            job['ui_done'] = True
            st.rerun()
        if job['error']:
            st.error('Run stopped. Completed batches are preserved. Resolve the error and resume.')
            st.code(job['error'])
            Path(job['output']).mkdir(parents=True, exist_ok=True)
            (Path(job['output'])/'error.log').write_text(job['error'])
    folder = Path(job['output'] if job else output)
    state_path = folder/'state.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
        st.caption('Run status: '+state['status'])
        if state['status'] == 'complete':
            report = json.loads((folder/'diagnostics.json').read_text())
            cols = st.columns(3)
            cols[0].metric('Indexed patterns', f"{report['valid']:,}")
            cols[1].metric('Mapped valid points', f"{report['mapped_valid']:,}")
            cols[2].metric('Mean NCC', '—' if report['mean_ncc'] is None else f"{report['mean_ncc']:.3f}")
            import pandas as pd
            quality = pd.read_csv(folder/'quality.csv')
            st.scatter_chart(quality.assign(y_view=-quality.y_um), x='x_um', y='y_view', color='ncc')
            st.caption('NCC and best–second score gaps indicate match quality; neither is a calibrated confidence probability.')
            row = st.number_input('Preview UP pattern index', 0, state['pattern_count']-1, value=0)
            try:
                import h5py
                cfg = Settings(**{k: v for k,v in state['identity'].items() if k in Settings.__dataclass_fields__}, output=str(folder))
                from emsoft_di.hpc import read_up
                with h5py.File(folder/'checkpoint.h5') as f:
                    idx = int(f['simulation_indices'][row, 0])
                if idx >= 0:
                    exp = preprocess(read_up(cfg.patterns)[row:row+1], cfg.binning, cfg.sigma)[0]
                    sim = np.load(folder/'dictionary.npy', mmap_mode='r')[idx]
                    for col, image, caption in zip(st.columns(2), [exp, sim], ['Experimental', 'Best dictionary match']):
                        lo, hi = np.percentile(image, [1, 99])
                        col.image(np.clip((image-lo)/max(hi-lo, 1e-12), 0, 1), caption=caption)
                else: st.info('Blank or invalid pattern; no orientation assigned.')
            except Exception as exc: st.error(f'Preview unavailable: {exc}')
            for name in ['indexed.h5', 'indexed.ang', 'quality.csv', 'diagnostics.json', 'state.json']:
                with (folder/name).open('rb') as f:
                    st.download_button('Download '+name, f, file_name=name, key=name)
            st.caption('H5 is an orix CrystalMap. ANG retains scan geometry; phase comes from the master. Unprocessed pilot rows are unindexed. IQ is pattern standard deviation; CI is score gap.')
monitor()
