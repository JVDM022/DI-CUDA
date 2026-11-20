"""Desktop launcher for one- and two-phase Kamiak dictionary indexing."""
import os
import queue
import re
import shlex
import subprocess
import threading
import uuid
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

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


def batch_script(count, ids, npa):
    lines = ['set -euo pipefail', 'cd "$HOME/kikuchipyCUDA/dictionary-indexing-cuda"',
             'root="$HOME/kikuchipyCUDA"', 'run="$HOME/$1"', 'jobs=()']
    for i in range(count):
        lines.extend([
            f'id=$(sbatch --parsable --gres=gpu:1 --time=02:00:00 hpc/submit_two_gpu.slurm --patterns "$run/scan.up1" --ang "$run/scan.ang" --master "$run/master{i+1}.sht" --devices 1 --limit 0 --resolution 1' + (' --npa' if npa else '') + ')',
            'jobs+=("${id%%;*}")', f'echo "Phase {ids[i]} job: ${{id%%;*}}"'])
    if count == 2:
        lines.extend([
            "printf -v cmd '%q ' " + '"$root/KikuchiPyCUDA/bin/python" "$run/merge_cobalt.py" "$root/dictionary-run-${jobs[0]}/indexed.h5" "$root/dictionary-run-${jobs[1]}/indexed.h5" "$run/combined" ' + f'{ids[0]} {ids[1]} "$run/master1.sht" "$run/master2.sht"',
            'sbatch --partition=kamiak --cpus-per-task=2 --mem=8G --time=00:30:00 --dependency="afterok:${jobs[0]}:${jobs[1]}" --job-name=phase-merge --wrap="$cmd"',
            'echo "Combined results: $run/combined"'])
    return '\n'.join(lines) + '\n'


def submit(user, files, ids, npa, log):
    host = user + '@kamiak.wsu.edu'
    folder = 'kikuchipyCUDA/scans/di-' + uuid.uuid4().hex[:12]
    remote = ['ssh', '-o', 'BatchMode=yes', host]
    def run(args, **kwargs):
        result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), **kwargs)
        output = result.stdout.decode("utf-8", errors="replace").strip()
        if output:
            log(output)
        if result.returncode:
            raise RuntimeError(f'{args[0]} exited with status {result.returncode}')
    run(remote + [f'mkdir -p {folder}'])
    log(f'Run folder: ~/{folder}')
    for source, name in files:
        log(f'Uploading {source.name} as {name} …')
        run(['scp', '-B', str(source), f'{host}:{folder}/{name}'])
        run(remote + [f'test "$(stat -c %s {folder}/{name})" = {source.stat().st_size}'])
    run(remote + [f'bash -s -- {folder}'], input=batch_script(len(ids), ids, npa).encode("utf-8"))


def main():
    root = tk.Tk()
    root.title('Kamiak DI')
    frame = ttk.Frame(root, padding=16)
    frame.pack(fill='both', expand=True)
    frame.columnconfigure(1, weight=1)
    user = tk.StringVar()
    ttk.Label(frame, text='Kamiak username').grid(row=0, column=0, sticky='w')
    ttk.Entry(frame, textvariable=user).grid(row=0, column=1, sticky='ew')
    names = {}
    paths = {}
    selections = []
    boxes = []
    logqueue = queue.Queue()
    def browse(label, ext, row):
        var = tk.StringVar()
        paths[label] = var
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky='w')
        ttk.Entry(frame, textvariable=var, width=52, state='readonly').grid(row=row, column=1, sticky='ew')
        def pick():
            options = dict(filetypes=[(ext.upper(), '*.'+ext)])
            if ext == 'sht' and (ROOT/'SHTdatabase-master/EBSD').exists():
                options['initialdir'] = str(ROOT/'SHTdatabase-master/EBSD')
            value = filedialog.askopenfilename(**options)
            if not value:
                return
            var.set(value)
            if ext == 'ang':
                names.clear()
                try:
                    names.update(phases(Path(value)))
                except OSError as exc:
                    messagebox.showerror('ANG file', str(exc))
                labels = [f'{n} · {names[n]}' for n in sorted(names)]
                for i, box in enumerate(boxes):
                    box.configure(values=labels)
                    selections[i].set(labels[i] if len(labels)>i else '')
        btn = ttk.Button(frame, text='Browse…', command=pick)
        btn.grid(row=row, column=2)
        return btn
    browse('ANG scan', 'ang', 1)
    browse('UP1 patterns', 'up1', 2)
    two = tk.BooleanVar(value=True)
    npa = tk.BooleanVar(value=True)
    def toggle():
        boxes[1].configure(state='readonly' if two.get() else 'disabled')
        second.configure(state='normal' if two.get() else 'disabled')
    ttk.Checkbutton(frame,text='Compare two phases',variable=two,command=toggle).grid(row=3,column=0,columnspan=2,sticky='w')
    for i in range(2):
        row = 4+i*2
        ttk.Label(frame,text=f'ANG phase {i+1}').grid(row=row,column=0,sticky='w')
        sel = tk.StringVar()
        selections.append(sel)
        box = ttk.Combobox(frame,state='readonly',textvariable=sel)
        box.grid(row=row,column=1,sticky='ew')
        boxes.append(box)
        second = browse(f'Master {i+1}', 'sht', row+1)
    ttk.Checkbutton(frame,text='Neighbor pattern averaging (NPA)',variable=npa).grid(row=8,column=0,columnspan=2,sticky='w')
    status = tk.Text(frame,width=75,height=8,wrap='word')
    status.grid(row=10,column=0,columnspan=3,sticky='ew')
    def start():
        username = user.get().strip()
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', username):
            return messagebox.showerror('Kamiak DI','Enter your Kamiak username.')
        count = 2 if two.get() else 1
        try:
            ids = [int(s.get().split(' · ')[0]) for s in selections[:count]]
            if len(set(ids)) != count or any(i not in names for i in ids):
                raise ValueError('Select distinct phases from the ANG header.')
            if count == 1 and len(names)>1:
                raise ValueError('For a multiphase header, use the two-phase comparison to preserve ANG phase IDs.')
            files = [(Path(paths[key].get()),name) for key,name in [('ANG scan','scan.ang'),('UP1 patterns','scan.up1')]]
            masters = [Path(paths[f'Master {i+1}'].get()) for i in range(count)]
            if len(set(masters)) != count:
                raise ValueError('Choose a different SHT master for each phase.')
            files += [(p,f'master{i+1}.sht') for i,p in enumerate(masters)]
            if count == 2:
                files.append((ROOT/'tools/merge_cobalt.py','merge_cobalt.py'))
            if any(not p.is_file() or not p.stat().st_size for p,_ in files):
                raise ValueError('Choose each required file.')
        except (ValueError,OSError) as exc:
            return messagebox.showerror('Check inputs',str(exc))
        detail = '\n'.join(f'{names[i]} → {p.name}' for i,p in zip(ids,masters))
        if not messagebox.askokcancel('Submit full scan?',detail+'\n\nOne GPU and two hours per phase, 1° spacing.'):
            return
        enabled = npa.get()
        action.configure(state='disabled')
        def worker():
            try:
                submit(username,files,ids,enabled,logqueue.put)
            except (RuntimeError,OSError) as exc:
                logqueue.put(str(exc))
            finally:
                logqueue.put(None)
        threading.Thread(target=worker,daemon=True).start()
    action = ttk.Button(frame,text='Upload and submit',command=start)
    action.grid(row=9,column=1,sticky='e',pady=8)
    def poll():
        while not logqueue.empty():
            msg = logqueue.get()
            if msg is None:
                action.configure(state='normal')
            else:
                status.insert('end',msg+'\n')
                status.see('end')
        root.after(150,poll)
    poll()
    root.mainloop()


if __name__ == '__main__':
    main()
