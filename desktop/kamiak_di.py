"""Desktop launcher for one- and two-phase Kamiak dictionary indexing."""
import os
import json
import math
import shutil
import time
import tempfile
import queue
import re
import shlex
import subprocess
import threading
import uuid
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

ROOT = Path(__file__).resolve().parents[1]


def phases(path):
    found = {}
    current = None
    with path.open(encoding='utf-8', errors='replace') as stream:
        for line in stream:
            if not line.startswith('#'):
                break
            match = re.match(r'#\s*Phase\s+(\d+)\s*$', line)
            if match:
                current = int(match[1])
                found[current] = f'Phase {current}'
            name = re.match(r'#\s*MaterialName\s+(.+)', line)
            if current is not None and name:
                found[current] = name[1].strip()
    return found


def indexing_settings(resolution=1., gpus=1, hours=2):
    try:
        resolution = float(resolution)
        gpu_value, hour_value = float(gpus), float(hours)
        if not math.isfinite(resolution) or not .1 <= resolution <= 30:
            raise ValueError('Sampling angle must be between 0.1° and 30°.')
        if gpu_value not in (1, 2):
            raise ValueError('Choose one or two GPUs per phase.')
        if not math.isfinite(hour_value) or not hour_value.is_integer() or not 1 <= hour_value <= 168:
            raise ValueError('Time limit must be a whole number of hours from 1 to 168.')
    except (TypeError, OverflowError):
        raise ValueError('Enter a numeric sampling angle, GPU count and hour limit.') from None
    return resolution, int(gpu_value), int(hour_value)


def batch_script(count, ids, npa, *, resolution=1., gpus=1, hours=2):
    resolution, gpus, hours = indexing_settings(resolution, gpus, hours)
    days, remainder = divmod(hours, 24)
    time_limit = (f'{days}-' if days else '') + f'{remainder:02d}:00:00'
    lines = ['set -euo pipefail', 'cd "$HOME/kikuchipyCUDA/dictionary-indexing-cuda"',
             'root="$HOME/kikuchipyCUDA"', 'run="$HOME/$1"', 'jobs=()']
    for i in range(count):
        lines.extend([
            f'id=$(sbatch --parsable --gres=gpu:{gpus} --time={time_limit} hpc/submit_two_gpu.slurm --patterns "$run/scan.up1" --ang "$run/scan.ang" --master "$run/master{i+1}.sht" --devices {gpus} --limit 0 --resolution {resolution:g}' + (' --npa' if npa else '') + ')',
            'jobs+=("${id%%;*}")', f'echo "Phase {ids[i]} job: ${{id%%;*}}"'])
    if count == 2:
        lines.extend([
            "printf -v cmd '%q ' " + '"$root/KikuchiPyCUDA/bin/python" "$run/merge_cobalt.py" "$root/dictionary-run-${jobs[0]}/indexed.h5" "$root/dictionary-run-${jobs[1]}/indexed.h5" "$run/combined" ' + f'{ids[0]} {ids[1]} "$run/master1.sht" "$run/master2.sht"',
            'merge=$(sbatch --parsable --partition=kamiak --cpus-per-task=2 --mem=8G --time=00:30:00 --dependency="afterok:${jobs[0]}:${jobs[1]}" --job-name=phase-merge --wrap="$cmd")',
            'echo "Merge job: ${merge%%;*}"',
            'echo "Combined results: $run/combined"'])
    return '\n'.join(lines) + '\n'


def single_phase_ang(source, destination, phase_id):
    """Retain coordinates/calibration and label every point with the chosen master phase.

    A temporary upload copy avoids the remote export's majority-phase heuristic.
    The original ANG is never modified. All patterns are indexed against this master.
    """
    if phase_id not in phases(source): raise ValueError('Selected phase is absent from ANG header.')
    active=None
    global_keys={'GRID','XSTEP','YSTEP','NCOLS_ODD','NCOLS_EVEN','NROWS','OPERATOR','SAMPLEID','SCANID',
                 'x-star','y-star','z-star','SampleTiltAngle','CameraElevationAngle','CameraAzimuthalAngle'}
    with Path(source).open() as original, Path(destination).open('x') as out:
        for line in original:
            if line.startswith('#'):
                match=re.match(r'#\s*Phase\s+(\d+)\s*$',line)
                if match: active=int(match[1])
                key=line.lstrip('# ').split(':',1)[0].split()
                if key and key[0] in global_keys: active=None
                if active is None or active==phase_id: out.write(line)
            elif line.strip():
                fields=line.split()
                if len(fields)<8: raise ValueError('ANG rows require at least eight columns for phase export.')
                fields[7]=str(phase_id)
                out.write(' '.join(fields)+'\n')


def submit(user, files, ids, npa, log, *, resolution=1., gpus=1, hours=2):
    # Preserve an explicit single-phase assignment even with a multiphase source ANG.
    if len(ids)==1:
        with tempfile.TemporaryDirectory(prefix='kamiak-single-phase-') as temporary:
            prepared=[]
            for source,name in files:
                if name=='scan.ang':
                    selected=Path(temporary)/'scan.ang'
                    single_phase_ang(source,selected,ids[0]); source=selected
                prepared.append((source,name))
            return _submit_files(user,prepared,ids,npa,log,resolution=resolution,gpus=gpus,hours=hours)
    return _submit_files(user,files,ids,npa,log,resolution=resolution,gpus=gpus,hours=hours)


def _submit_files(user, files, ids, npa, log, *, resolution=1., gpus=1, hours=2):
    resolution, gpus, hours = indexing_settings(resolution, gpus, hours)
    host = user + '@kamiak.wsu.edu'
    folder = 'kikuchipyCUDA/scans/di-' + uuid.uuid4().hex[:12]
    remote = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', host]
    def run(args, **kwargs):
        result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), **kwargs)
        output = result.stdout.decode("utf-8", errors="replace").strip()
        if output:
            for line in output.splitlines():
                log(line)
        if result.returncode:
            raise RuntimeError(f'{args[0]} exited with status {result.returncode}')
    run(remote + [f'mkdir -p {folder}'])
    log(f'Run folder: ~/{folder}')
    for source, name in files:
        log(f'Uploading {source.name} as {name} …')
        run(['scp', '-B', '-o', 'ConnectTimeout=12', str(source), f'{host}:{folder}/{name}'])
        run(remote + [f'test "$(stat -c %s {folder}/{name})" = {source.stat().st_size}'])
    run(remote + [f'bash -s -- {folder}'], input=batch_script(len(ids), ids, npa, resolution=resolution, gpus=gpus, hours=hours).encode("utf-8"))



def valid_user(user):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', user):
        raise ValueError('Enter your Kamiak username using letters, numbers, dots, underscores or hyphens.')
    return user


def ssh_command(user, command, timeout=40):
    """Run a bounded, noninteractive SSH check; never handle passwords in the app."""
    valid_user(user)
    if not shutil.which('ssh'):
        raise RuntimeError('OpenSSH is missing. Install the SSH client before connecting.')
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12',
                             user+'@kamiak.wsu.edu', command], capture_output=True,
                            text=True, timeout=timeout,
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or 'Kamiak connection failed.')
    return result.stdout.strip()


def check_connection(user):
    if not shutil.which('scp'):
        raise RuntimeError('SCP is missing. Install the OpenSSH client before uploading.')
    return ssh_command(user, 'set -e; command -v sbatch >/dev/null; '
        'test -x "$HOME/kikuchipyCUDA/KikuchiPyCUDA/bin/python"; '
        'test -f "$HOME/kikuchipyCUDA/dictionary-indexing-cuda/hpc/submit_two_gpu.slurm"; '
        'printf "Kamiak connection and indexing installation are ready.\\n"')


def job_status(user, jobs):
    ids = list(jobs)
    if not ids or any(not re.fullmatch(r'[0-9]+', x) for x in ids):
        raise ValueError('No valid job IDs to check.')
    joined = ','.join(ids)
    output = ssh_command(user, f'sacct -n -P -j {joined} --format=JobIDRaw,State,Elapsed,ExitCode')
    found = {}
    for line in output.splitlines():
        fields = line.split('|')
        if len(fields) >= 4 and fields[0] in ids:
            found[fields[0]] = fields[1:4]
    return found


def inspect_inputs(ang, up, master_paths, phase_ids):
    """Validate files before uploading without importing the GPU/scientific stack."""
    import struct
    names = phases(ang)
    if not names or len(names) > 2:
        raise ValueError('The ANG header must describe one or two phases for this launcher.')
    if len(set(phase_ids)) != len(phase_ids) or not set(phase_ids).issubset(names) or (len(phase_ids)==2 and set(phase_ids)!=set(names)):
        raise ValueError('Assign a valid ANG phase to each selected master; two-phase comparison needs distinct phases.')
    if len(master_paths) != len(phase_ids) or len({p.resolve() for p in master_paths}) != len(master_paths):
        raise ValueError('Choose a different SHT master for each phase.')
    for p in [ang, up, *master_paths]:
        if not p.is_file() or not p.stat().st_size:
            raise ValueError(f'Choose an existing, nonempty file: {p}')
    if up.suffix.lower() != '.up1' or ang.suffix.lower() != '.ang' or any(p.suffix.lower() != '.sht' for p in master_paths):
        raise ValueError('This Kamiak launcher accepts ANG, UP1 and SHT files.')
    with up.open('rb') as f:
        header = f.read(16)
    if len(header) != 16:
        raise ValueError('The UP1 header is truncated.')
    version, width, height, offset = struct.unpack('<4i', header)
    if version not in (1, 2, 3, 4) or not (0 < width <= 5000 and 0 < height <= 5000) or offset < 16:
        raise ValueError('Unsupported UP1 header.')
    payload = up.stat().st_size-offset
    if payload <= 0 or payload % (width*height):
        raise ValueError('UP1 file size does not match its pattern dimensions.')
    return f'{payload//(width*height):,} patterns · {width} × {height} pixels · {len(names)} phase(s)'


class KamiakApp:
    """Desktop interface; worker threads deliver results through a UI event queue."""
    def __init__(self, root):
        self.root = root
        self.root.title('Kamiak DI — GPU dictionary indexing')
        self.root.geometry('1100x800')
        self.root.minsize(940, 690)
        self.root.configure(bg='#edf2f7')
        self.events = queue.Queue()
        self.busy = False
        self.connection_user = None
        self.session_user = None
        self.remote_folder = None
        self.jobs = {}
        self.submission_complete = False
        self.run_settings = {}
        self.inputs = {}
        self.controls = []
        self.user = tk.StringVar()
        self.two = tk.BooleanVar(value=True)
        self.npa = tk.BooleanVar(value=True)
        self.resolution = tk.StringVar(value='1')
        self.gpus = tk.StringVar(value='1')
        self.hours = tk.StringVar(value='2')
        self.message = tk.StringVar(value='Start by checking your Kamiak connection.')
        self.connection = tk.StringVar(value='Not checked')
        self.summary = tk.StringVar(value='Choose your ANG scan and UP1 patterns to begin.')
        self.job_summary = tk.StringVar(value='Submitted jobs will appear here. Your inputs stay on Kamiak after upload.')
        self.names = {}
        self.phase_vars = [tk.StringVar(), tk.StringVar()]
        self.phase_boxes = []
        self.second_controls = []
        self.style()
        self.build()
        self.user.trace_add('write', self.username_changed)
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.root.after(100, self.poll)

    def style(self):
        s = ttk.Style(self.root)
        s.theme_use('clam')
        s.configure('.', font=('Helvetica', 12), background='#edf2f7', foreground='#172b4d')
        s.configure('TFrame', background='#edf2f7')
        s.configure('Card.TFrame', background='white')
        s.configure('Card.TLabel', background='white')
        s.configure('Title.TLabel', font=('Helvetica', 25, 'bold'), background='#edf2f7')
        s.configure('Section.TLabel', font=('Helvetica', 15, 'bold'), background='white')
        s.configure('Muted.TLabel', foreground='#53657d', background='white')
        s.configure('TEntry', padding=8, fieldbackground='white')
        s.configure('TButton', padding=(14, 9), background='#e3eaf4')
        s.configure('Primary.TButton', background='#2463eb', foreground='white')
        s.map('Primary.TButton', background=[('active', '#174bc4'), ('disabled', '#9eb3da')])
        s.configure('TCheckbutton', background='white')
        s.configure('TNotebook.Tab', padding=(20, 12))
        s.configure('Treeview', rowheight=32, fieldbackground='white', background='white')
        s.configure('Treeview.Heading', font=('Helvetica', 12, 'bold'), padding=8)

    def button(self, parent, text, command, primary=False):
        b = ttk.Button(parent, text=text, command=command, style='Primary.TButton' if primary else 'TButton')
        self.controls.append((b, 'normal'))
        return b

    def card(self, parent, title, subtitle):
        frame = ttk.Frame(parent, padding=18, style='Card.TFrame')
        frame.pack(fill='x', pady=(0, 14))
        ttk.Label(frame, text=title, style='Section.TLabel').pack(anchor='w')
        ttk.Label(frame, text=subtitle, style='Muted.TLabel', wraplength=840).pack(anchor='w', pady=(4, 12))
        return frame

    def build(self):
        header = ttk.Frame(self.root, padding=(26, 20))
        header.pack(fill='x')
        ttk.Label(header, text='Kamiak DI', style='Title.TLabel').pack(anchor='w')
        ttk.Label(header, text='GPU dictionary indexing on Kamiak  /  Connect → Select scan → Submit → Retrieve results').pack(anchor='w', pady=(5, 0))
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill='both', expand=True, padx=24)
        form = ttk.Frame(self.tabs)
        jobs = ttk.Frame(self.tabs, padding=18)
        activity = ttk.Frame(self.tabs, padding=12)
        self.tabs.add(form, text='New scan')
        self.tabs.add(jobs, text='Jobs & results')
        self.tabs.add(activity, text='Activity log')
        self.jobs_tab = jobs
        canvas = tk.Canvas(form, bg='#edf2f7', highlightthickness=0)
        scroll = ttk.Scrollbar(form, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y'); canvas.pack(side='left', fill='both', expand=True)
        inner = ttk.Frame(canvas, padding=(0, 12, 10, 0))
        window = canvas.create_window((0, 0), window=inner, anchor='nw')
        inner.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>', lambda e: canvas.itemconfigure(window, width=e.width))
        def wheel(event):
            if event.delta:
                canvas.yview_scroll(-1 if event.delta > 0 else 1, 'units')
        canvas.bind_all('<MouseWheel>', wheel)
        connect = self.card(inner, '01  Connect to Kamiak', 'Use your existing SSH login. In a terminal, run opkssh login kamiak before checking the connection.')
        row = ttk.Frame(connect, style='Card.TFrame'); row.pack(fill='x')
        ttk.Label(row, text='Kamiak username', style='Card.TLabel').pack(side='left')
        entry = ttk.Entry(row, textvariable=self.user, width=25); entry.pack(side='left', padx=12)
        self.controls.append((entry, 'normal'))
        self.button(row, 'Check connection', self.connect).pack(side='left')
        ttk.Label(row, textvariable=self.connection, style='Muted.TLabel').pack(side='left', padx=12)
        scan = self.card(inner, '02  Select your scan', 'ANG provides phase assignments and calibration. UP1 contains the experimental diffraction patterns.')
        self.file_row(scan, 'ANG scan', 'ang')
        self.file_row(scan, 'UP1 patterns', 'up1')
        ttk.Label(scan, textvariable=self.summary, style='Muted.TLabel', wraplength=840).pack(anchor='w', pady=(10, 0))
        phase = self.card(inner, '03  Match phases & masters', 'Choose an SHT master matching each ANG phase and the scan beam voltage. GPU indexing uses the existing Kamiak pipeline.')
        check = ttk.Checkbutton(phase, text='Compare two phases', variable=self.two, command=self.toggle)
        check.pack(anchor='w', pady=(0, 10)); self.controls.append((check, 'normal'))
        for i in range(2):
            row = ttk.Frame(phase, style='Card.TFrame'); row.pack(fill='x', pady=(0, 6))
            ttk.Label(row, text=f'Phase {i+1}', style='Card.TLabel', width=15).pack(side='left')
            box = ttk.Combobox(row, textvariable=self.phase_vars[i], state='readonly', width=36)
            box.pack(side='left', fill='x', expand=True)
            self.phase_boxes.append(box); self.controls.append((box, 'readonly'))
            e, b = self.file_row(phase, f'Master {i+1}', 'sht')
            if i: self.second_controls.extend([box, e, b])
        options = self.card(inner, '04  Review & submit', 'Choose orientation sampling and compute resources for each phase. Smaller angles produce larger dictionaries. Kamiak partition and allocation limits still apply.')
        resource_row = ttk.Frame(options, style='Card.TFrame'); resource_row.pack(fill='x', pady=(0, 14))
        for column, (title, variable, low, high) in enumerate([
            ('Sampling angle (degrees)', self.resolution, .1, 30),
            ('GPUs per phase', self.gpus, 1, 2),
            ('Time limit (hours per phase)', self.hours, 1, 168)]):
            cell = ttk.Frame(resource_row, style='Card.TFrame'); cell.pack(side='left', expand=True, fill='x', padx=(0, 14))
            ttk.Label(cell, text=title, style='Card.TLabel').pack(anchor='w', pady=(0, 6))
            if variable is self.gpus:
                field = ttk.Combobox(cell, textvariable=variable, values=('1', '2'), state='readonly', width=12)
                state = 'readonly'
            else:
                field = ttk.Spinbox(cell, textvariable=variable, from_=low, to=high,
                                    increment=.1 if variable is self.resolution else 1, width=14)
                state = 'normal'
            field.pack(anchor='w'); self.controls.append((field, state))
        check = ttk.Checkbutton(options, text='Neighbor pattern averaging (NPA)', variable=self.npa)
        check.pack(anchor='w'); self.controls.append((check, 'normal'))
        ttk.Label(options, text='NPA may improve signal quality while reducing spatial resolution.', style='Muted.TLabel').pack(anchor='w', pady=(4, 12))
        row = ttk.Frame(options, style='Card.TFrame'); row.pack(fill='x')
        self.button(row, 'Validate inputs', self.validate).pack(side='left')
        self.button(row, 'Upload & submit to Kamiak', self.start, primary=True).pack(side='right')
        ttk.Label(jobs, textvariable=self.job_summary, wraplength=970).pack(anchor='w', pady=(0, 12))
        self.table = ttk.Treeview(jobs, columns=('role','state','elapsed','exit'), show='tree headings', height=8)
        self.table.heading('#0', text='Job ID'); self.table.column('#0', width=130)
        for col, title in [('role','Task'),('state','State'),('elapsed','Elapsed'),('exit','Exit code')]:
            self.table.heading(col, text=title); self.table.column(col, width=170)
        self.table.pack(fill='both', expand=True)
        actions = ttk.Frame(jobs); actions.pack(fill='x', pady=12)
        self.button(actions, 'Refresh job status', self.refresh).pack(side='left')
        self.button(actions, 'Copy result paths', self.copy_paths).pack(side='left', padx=8)
        self.button(actions, 'Download results…', self.download).pack(side='right')
        self.button(actions, 'Save run receipt…', self.save_receipt).pack(side='right', padx=8)
        ttk.Label(jobs, text='Refresh after submission to see queue/running/completion states. Download is enabled by successful completion of every recorded job. A receipt lets you reconnect after restarting.', wraplength=950).pack(anchor='w')
        self.button(jobs, 'Open saved receipt…', self.load_receipt).pack(anchor='w', pady=(12, 0))
        self.log = ScrolledText(activity, height=12, wrap='word', font=('Menlo', 11), bg='#11213a', fg='#dce7f6', insertbackground='white', relief='flat', padx=12, pady=12)
        self.log.pack(fill='both', expand=True); self.log.configure(state='disabled')
        self.button(activity, 'Save activity log…', self.save_log).pack(anchor='e', pady=(8, 0))
        footer = ttk.Frame(self.root, padding=(24, 12)); footer.pack(fill='x')
        self.progress = ttk.Progressbar(footer, mode='indeterminate', length=130)
        self.progress.pack(side='right', padx=(10, 0))
        ttk.Label(footer, textvariable=self.message, wraplength=820).pack(side='left')
        self.toggle()

    def file_row(self, parent, label, ext):
        row = ttk.Frame(parent, style='Card.TFrame'); row.pack(fill='x', pady=4)
        ttk.Label(row, text=label, style='Card.TLabel', width=15).pack(side='left')
        var = tk.StringVar(); self.inputs[label] = var
        entry = ttk.Entry(row, textvariable=var, state='readonly')
        entry.pack(side='left', fill='x', expand=True, padx=(0, 8)); self.controls.append((entry, 'readonly'))
        b = self.button(row, 'Browse…', lambda: self.pick(label, ext)); b.pack(side='right')
        return entry, b

    def username_changed(self, *args):
        if self.user.get().strip() != self.connection_user:
            self.connection_user = None; self.connection.set('Not checked')

    def toggle(self):
        for widget in self.second_controls:
            desired = 'readonly' if isinstance(widget, (ttk.Entry, ttk.Combobox)) else 'normal'
            widget.configure(state=desired if self.two.get() and not self.busy else 'disabled')

    def pick(self, label, ext):
        options = dict(filetypes=[(ext.upper()+' files', '*.'+ext)])
        if ext == 'sht' and (ROOT/'SHTdatabase-master/EBSD').exists():
            options['initialdir'] = str(ROOT/'SHTdatabase-master/EBSD')
        path = filedialog.askopenfilename(parent=self.root, **options)
        if not path: return
        if ext == 'ang':
            try:
                names = phases(Path(path))
                if not names or len(names) > 2: raise ValueError('Choose an ANG scan containing one or two phases.')
            except (OSError, ValueError) as exc:
                messagebox.showerror('ANG scan', str(exc), parent=self.root); return
            self.names = names
            labels = [f'{n} · {names[n]}' for n in sorted(names)]
            for i, box in enumerate(self.phase_boxes):
                box.configure(values=labels); self.phase_vars[i].set(labels[i] if i < len(labels) else '')
            self.two.set(len(names)==2); self.toggle()
            self.summary.set('Detected phases: '+', '.join(names.values()))
        self.inputs[label].set(path)

    def append_log(self, message):
        self.log.configure(state='normal'); self.log.insert('end', time.strftime('%H:%M:%S')+'  '+message+'\n')
        self.log.see('end'); self.log.configure(state='disabled')
        if message.startswith('Run folder: ~/'):
            self.remote_folder = message.split('~/', 1)[1]
        match = re.search(r'^(Phase (\d+) job|Merge job): (\d+)$', message)
        if match:
            job_id = match[3]
            self.jobs[job_id] = {'role': 'Merge' if match[1]=='Merge job' else 'Phase '+match[2], 'state':'Submitted', 'elapsed':'—', 'exit':'—'}
            self.render_jobs()

    def task(self, label, function, done=None):
        if self.busy: return
        self.busy = True; self.message.set(label); self.progress.start(12)
        for widget, _ in self.controls: widget.configure(state='disabled')
        def worker():
            try: self.events.put(('done', function(), done))
            except Exception as exc: self.events.put(('error', str(exc), None))
        threading.Thread(target=worker, daemon=True).start()

    def poll(self):
        while not self.events.empty():
            kind, payload, callback = self.events.get()
            if kind == 'log':
                self.append_log(payload); self.message.set(payload[:120]); continue
            self.busy = False; self.progress.stop()
            for widget, state in self.controls: widget.configure(state=state)
            self.toggle()
            if kind == 'error':
                self.message.set('Action stopped. See Activity log for details.')
                self.append_log(payload)
                messagebox.showerror('Kamiak DI', payload+'\n\nIf SSH authentication failed, run opkssh login kamiak in a terminal and check the connection again. Accepted jobs remain on Kamiak; refresh their status before submitting again.', parent=self.root)
            else:
                self.message.set('Ready')
                if callback: callback(payload)
        self.root.after(100, self.poll)

    def connect(self):
        user = self.user.get().strip()
        def done(text):
            self.connection_user = user; self.connection.set('Connected · ready'); self.append_log(text)
            self.message.set('Kamiak is ready. Select and validate your scan.')
        self.task('Checking SSH and remote indexing installation…', lambda: check_connection(user), done)

    def selection(self):
        user = valid_user(self.user.get().strip())
        count = 2 if self.two.get() else 1
        try: ids = [int(v.get().split(' · ')[0]) for v in self.phase_vars[:count]]
        except ValueError: raise ValueError('Select the ANG phase for each master.') from None
        ang, up = Path(self.inputs['ANG scan'].get()), Path(self.inputs['UP1 patterns'].get())
        masters = [Path(self.inputs[f'Master {i+1}'].get()) for i in range(count)]
        indexing_settings(self.resolution.get(), self.gpus.get(), self.hours.get())
        summary = inspect_inputs(ang, up, masters, ids)
        return user, ids, ang, up, masters, summary

    def validate(self):
        try:
            *_, summary = self.selection(); self.summary.set(summary); self.message.set('Inputs validated. Review phase assignments and submit.')
        except (ValueError, OSError) as exc: messagebox.showerror('Check inputs', str(exc), parent=self.root)

    def start(self):
        try: user, ids, ang, up, masters, summary = self.selection()
        except (ValueError, OSError) as exc:
            messagebox.showerror('Check inputs', str(exc), parent=self.root); return
        if self.connection_user != user:
            messagebox.showinfo('Check connection first', 'Use Check connection before submitting this scan.', parent=self.root); return
        if self.jobs and not messagebox.askyesno('Start another scan?', 'Save the current run receipt first if you need to reconnect to these jobs later. Continue with a new submission?', parent=self.root): return
        resolution, gpus, hours = indexing_settings(self.resolution.get(), self.gpus.get(), self.hours.get())
        detail = '\n'.join(f'ANG phase {i}: {p.name}' for i, p in zip(ids, masters))
        if not messagebox.askokcancel('Submit to Kamiak?', summary+'\n\n'+detail+f'\n\nFull scan · {gpus} GPU(s) and {hours} hours per phase · {resolution:g}° sampling\nNPA: '+('On' if self.npa.get() else 'Off'), parent=self.root): return
        files = [(ang,'scan.ang'),(up,'scan.up1')]+[(p,f'master{i+1}.sht') for i,p in enumerate(masters)]
        if len(ids)==2: files.append((ROOT/'tools/merge_cobalt.py','merge_cobalt.py'))
        self.jobs = {}; self.submission_complete = False; self.remote_folder = None; self.session_user = user; self.render_jobs()
        npa = self.npa.get()
        self.run_settings = dict(resolution=resolution, gpus=gpus, hours=hours, npa=npa)
        def done(_):
            self.submission_complete = True
            self.render_jobs(); self.tabs.select(self.jobs_tab)
            self.message.set('Submission accepted. Save your receipt, then refresh job status.')
        self.task('Uploading scan and submitting jobs…', lambda: submit(user, files, ids, npa, lambda text: self.events.put(('log', text, None)), resolution=resolution, gpus=gpus, hours=hours), done)

    def render_jobs(self):
        for row in self.table.get_children(): self.table.delete(row)
        for job, info in self.jobs.items(): self.table.insert('', 'end', text=job, values=(info['role'],info['state'],info['elapsed'],info['exit']))
        if not self.remote_folder: self.job_summary.set('No remote run selected. Submit a scan or open a receipt.')
        if self.remote_folder: self.job_summary.set(f'{self.session_user}@kamiak.wsu.edu  ·  ~/{self.remote_folder}')

    def refresh(self):
        if not self.jobs:
            messagebox.showinfo('No jobs yet', 'Submit a scan or open a saved receipt first.', parent=self.root); return
        user, ids = self.session_user, list(self.jobs)
        def done(status):
            for job in ids:
                values = status.get(job, ['Awaiting accounting','—','—'])
                self.jobs[job].update(state=values[0], elapsed=values[1], exit=values[2])
            self.render_jobs(); self.message.set('Job status refreshed. Elapsed time is from the scheduler, not indexing progress.')
        self.task('Checking Slurm job states…', lambda: job_status(user, ids), done)

    def result_paths(self):
        paths = [f'kikuchipyCUDA/dictionary-run-{job}' for job, info in self.jobs.items() if info['role'].startswith('Phase ')]
        if any(info['role']=='Merge' for info in self.jobs.values()) and self.remote_folder:
            paths.append(self.remote_folder+'/combined')
        return paths

    def copy_paths(self):
        if not self.jobs: return
        text = '\n'.join(f'{self.session_user}@kamiak.wsu.edu:~/{path}' for path in self.result_paths())
        self.root.clipboard_clear(); self.root.clipboard_append(text); self.message.set('Result paths copied.')

    def download(self):
        if not self.submission_complete or not self.jobs or any(j['state'] != 'COMPLETED' or j['exit'] != '0:0' for j in self.jobs.values()):
            messagebox.showinfo('Wait for completion', 'Refresh job status first. The full submission must succeed and every recorded phase and merge job must finish successfully before downloading.', parent=self.root); return
        destination = filedialog.askdirectory(parent=self.root, title='Choose a folder for this run')
        if not destination: return
        target = Path(destination)/Path(self.remote_folder).name
        if target.exists():
            messagebox.showerror('Folder already exists', 'Choose another destination to avoid overwriting results.', parent=self.root); return
        user, paths = self.session_user, self.result_paths()
        def work():
            # Recheck remotely immediately before copying.
            statuses = job_status(user, self.jobs)
            if len(statuses)!=len(self.jobs) or any(v[0]!='COMPLETED' or v[2]!='0:0' for v in statuses.values()):
                raise RuntimeError('Jobs are not all complete. Refresh their status before downloading.')
            target.mkdir()
            for path in paths:
                self.events.put(('log', 'Downloading ~/'+path, None))
                result = subprocess.run(['scp','-B','-o','ConnectTimeout=12','-r',user+'@kamiak.wsu.edu:'+path,str(target)], capture_output=True, text=True, creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                if result.returncode: raise RuntimeError('Download incomplete; files retained in '+str(target)+'\n'+result.stderr)
            return str(target)
        self.task('Downloading completed results…', work, lambda path: self.message.set('Results downloaded to '+path))

    def save_receipt(self):
        if not self.jobs: return
        path = filedialog.asksaveasfilename(parent=self.root, defaultextension='.json', initialfile='kamiak-run.json', filetypes=[('Run receipt','*.json')])
        if path:
            try: Path(path).write_text(json.dumps({'schema':1,'user':self.session_user,'folder':self.remote_folder,'jobs':self.jobs,'submission_complete':self.submission_complete,'settings':self.run_settings},indent=2))
            except OSError as exc: messagebox.showerror('Save receipt',str(exc),parent=self.root)

    def load_receipt(self):
        path = filedialog.askopenfilename(parent=self.root, filetypes=[('Run receipt','*.json')])
        if not path: return
        try:
            data = json.loads(Path(path).read_text())
            valid_user(data['user'])
            if data['schema'] != 1 or not re.fullmatch(r'kikuchipyCUDA/scans/di-[a-f0-9]{12}', data['folder']): raise ValueError('Invalid run receipt.')
            jobs = data['jobs']
            if not jobs or len(jobs)>3: raise ValueError('Invalid receipt job count.')
            for job, info in jobs.items():
                if not re.fullmatch(r'[0-9]+',job) or not re.fullmatch(r'(?:Phase [0-9]+|Merge)', info['role']): raise ValueError('Invalid receipt job.')
                for key in ('state','elapsed','exit'):
                    if not isinstance(info[key],str): raise ValueError('Invalid receipt state.')
            if not isinstance(data.get('submission_complete'), bool): raise ValueError('Invalid submission receipt.')
            self.session_user = data['user']; self.remote_folder = data['folder']; self.jobs = jobs
            self.submission_complete = data['submission_complete']
            self.run_settings = data.get('settings', {})
            # Never trust saved completion states; require fresh accounting before download.
            for info in self.jobs.values(): info.update(state='Needs refresh', elapsed='—', exit='—')
            self.render_jobs(); self.message.set('Receipt loaded. Refresh job status to reconnect.')
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            messagebox.showerror('Open receipt', str(exc), parent=self.root)

    def save_log(self):
        path = filedialog.asksaveasfilename(parent=self.root, defaultextension='.log', initialfile='kamiak-di.log')
        if path:
            try: Path(path).write_text(self.log.get('1.0','end'))
            except OSError as exc: messagebox.showerror('Save log',str(exc),parent=self.root)

    def close(self):
        if self.busy:
            messagebox.showinfo('Action in progress', 'Wait for the upload, check or download to finish before closing. Accepted cluster jobs continue independently.', parent=self.root); return
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        KamiakApp(root)
        root.mainloop()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        messagebox.showerror('Kamiak DI could not start',
            str(exc)+'\n\nStartup details are in Kamiak-DI-launch.log in your temporary folder.', parent=root)
        root.destroy()
        raise


if __name__ == '__main__':
    main()
