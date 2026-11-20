"""Local React interface for Kamiak DI. No web packages or Node needed at runtime."""
import copy
import os
import tempfile
import urllib.request
import json
import mimetypes
from pathlib import Path
import queue
import re
import secrets
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, unquote
import webbrowser
import sys

# Finder launches Python in isolated mode; import the sibling backend explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kamiak_di import (ROOT, phases, valid_user, indexing_settings, inspect_inputs,
                       check_connection, submit, job_status)

ASSETS = Path(__file__).resolve().parent/'web-assets'


def login_kamiak():
    """Delegate browser authentication to the installed opkssh, never capture credentials in logs."""
    executable=shutil.which('opkssh')
    if not executable:
        for path in (Path('/usr/local/bin/opkssh'),Path('/opt/homebrew/bin/opkssh'),
                     Path('/opt/local/bin/opkssh'),Path.home()/'.local/bin/opkssh'):
            if path.is_file() and os.access(path,os.X_OK):
                executable=str(path);break
    if not executable:
        raise RuntimeError('opkssh is not installed or could not be found. Install and configure opkssh for Kamiak, then try again.')
    try:
        result=subprocess.run([executable,'login','kamiak'],stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=300,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    except subprocess.TimeoutExpired:
        raise RuntimeError('Kamiak sign-in timed out after five minutes. Click Log in to Kamiak to try again.') from None
    except OSError:
        raise RuntimeError('Could not start opkssh. Check its installation and try again.') from None
    if result.returncode:
        # opkssh output may contain authentication URLs or token details. Keep it out
        # of frontend state, receipts and saved logs; show a useful generic recovery.
        raise RuntimeError('Kamiak sign-in did not complete. Finish the browser sign-in, check the configured kamiak provider in opkssh, and try again.')
    return 'Sign-in completed. Enter your Kamiak username and click Check connection.'


class Session:
    def __init__(self, dialogs=None):
        self.lock = threading.RLock()
        self.dialogs = dialogs
        self.state = dict(busy=False, operation='', error=None, notice='Ready to connect to Kamiak.',
                          connected_user=None, run=None, logs=[], validation=None, statuses={},chosen=None,phases={},product='Kamiak DI React')

    def snapshot(self):
        with self.lock: return copy.deepcopy(self.state)

    def log(self, text):
        with self.lock:
            self.state['logs'].append(dict(time=time.strftime('%H:%M:%S'), text=text))
            self.state['logs'] = self.state['logs'][-1000:]
            run = self.state['run']
            if not run: return
            if text.startswith('Run folder: ~/'): run['folder'] = text.split('~/',1)[1]
            match = re.fullmatch(r'(Phase (\d+) job|Merge job): (\d+)', text)
            if match:
                run['jobs'][match[3]] = dict(role='Merge' if match[1]=='Merge job' else 'Phase '+match[2])

    def choose(self, kind):
        if not self.dialogs: raise ValueError('Native file chooser unavailable; paste a local path instead.')
        answer = queue.Queue()
        self.dialogs.put((kind, answer))
        value = answer.get()
        if isinstance(value, Exception): raise value
        return value

    def selection(self, data):
        user = valid_user(str(data.get('user','')).strip())
        ids = data.get('ids',[])
        if not isinstance(ids,list) or len(ids) not in (1,2) or any(type(i) is not int for i in ids):
            raise ValueError('Assign one or two ANG phases.')
        masters = [Path(p) for p in data.get('masters',[])]
        ang, up = Path(data.get('ang','')), Path(data.get('up',''))
        resolution,gpus,hours = indexing_settings(data.get('resolution',1),data.get('gpus',1),data.get('hours',2))
        summary = inspect_inputs(ang,up,masters,ids)
        return user,ids,ang,up,masters,dict(resolution=resolution,gpus=gpus,hours=hours,npa=bool(data.get('npa',True))),summary

    def begin(self, action, data):
        if action not in ('login','connect','browse','phases','validate','submit','refresh','download','restore','clear'):
            raise ValueError('Unknown action')
        with self.lock:
            if self.state['busy']: raise ValueError('Wait for the current action to finish.')
            self.state.update(busy=True,operation=action,error=None,notice='')
        def work():
            try:
                result = self.perform(action,data)
                with self.lock:
                    if result: self.state['notice']=result
            except Exception as exc:
                with self.lock: self.state['error']=str(exc)
                self.log(str(exc))
            finally:
                with self.lock: self.state.update(busy=False,operation='')
        threading.Thread(target=work,daemon=True).start()

    def perform(self, action, data):
        if action == 'login':
            self.state['connected_user']=None
            self.log('Starting Kamiak browser sign-in with opkssh.')
            result=login_kamiak()
            self.log('Kamiak sign-in completed. Check the connection to verify access.')
            return result
        if action == 'clear':
            self.state['error']=None; return 'Ready.'
        if action == 'browse':
            kind=data.get('kind')
            if kind not in ('ang','up','master0','master1','destination'): raise ValueError('Unknown file type')
            value=self.choose(kind)
            self.state['chosen']=dict(kind=kind,path=value,stamp=time.time())
            if kind=='ang' and value:
                found=phases(Path(value))
                if not 1 <= len(found) <= 2: raise ValueError('Choose an ANG scan with one or two phases.')
                self.state['phases']=found
            return 'File selected.' if value else 'Selection cancelled.'
        if action == 'phases':
            found=phases(Path(data.get('ang','')))
            if not 1 <= len(found) <= 2: raise ValueError('The ANG header must contain one or two phases.')
            self.state['phases']=found; return 'ANG phases loaded.'
        if action == 'connect':
            user=valid_user(str(data.get('user','')).strip())
            self.state['connected_user']=None
            result=check_connection(user)
            self.state['connected_user']=user; self.log(result); return 'Connected to Kamiak. Indexing installation found.'
        if action in ('validate','submit'):
            user,ids,ang,up,masters,settings,summary=self.selection(data)
            self.state['validation']=summary
            if action=='validate': return summary
            if self.state['connected_user']!=user: raise ValueError('Check this username’s connection before submitting.')
            run=dict(schema=2,user=user,folder=None,jobs={},settings=settings,complete=False)
            self.state.update(run=run,statuses={})
            files=[(ang,'scan.ang'),(up,'scan.up1')]+[(p,f'master{i+1}.sht') for i,p in enumerate(masters)]
            if len(ids)==2: files.append((ROOT/'tools/merge_cobalt.py','merge_cobalt.py'))
            self.log(f'Submitting: {settings["resolution"]:g}°, {settings["gpus"]} GPU(s), {settings["hours"]} hours per phase; NPA {settings["npa"]}')
            submit(user,files,ids,settings['npa'],self.log,resolution=settings['resolution'],gpus=settings['gpus'],hours=settings['hours'])
            if len(run['jobs']) != len(ids)+(1 if len(ids)==2 else 0): raise RuntimeError('Not all job IDs were received. Accepted jobs remain on Kamiak; inspect the activity log before retrying.')
            run['complete']=True
            return 'Jobs submitted. Save your receipt and refresh status to follow this run.'
        if action=='restore':
            receipt=data.get('receipt',{})
            if not isinstance(receipt,dict) or receipt.get('schema')!=2: raise ValueError('Choose a Kamiak DI React run receipt.')
            valid_user(receipt.get('user',''))
            if not re.fullmatch(r'kikuchipyCUDA/scans/di-[a-f0-9]{12}',str(receipt.get('folder',''))): raise ValueError('Invalid receipt run folder.')
            jobs=receipt.get('jobs',{})
            if not isinstance(jobs,dict) or not 1<=len(jobs)<=3 or type(receipt.get('complete')) is not bool: raise ValueError('Invalid receipt job data.')
            roles=[]
            for job,info in jobs.items():
                if not re.fullmatch(r'[0-9]+',job) or not isinstance(info,dict) or not re.fullmatch(r'(Phase [0-9]+|Merge)',str(info.get('role',''))): raise ValueError('Invalid receipt job ID or role.')
                roles.append(info['role'])
            if len(set(roles))!=len(roles): raise ValueError('Duplicate receipt tasks.')
            self.state.update(run=copy.deepcopy(receipt),statuses={})
            return 'Receipt restored. Refresh status before downloading.'
        run=self.state['run']
        if not run or not run['jobs']: raise ValueError('Submit a scan or open a run receipt first.')
        statuses=job_status(run['user'],run['jobs'])
        self.state['statuses']=statuses
        if action=='refresh': return 'Scheduler status refreshed.'
        if action=='download':
            if not run['complete'] or len(statuses)!=len(run['jobs']) or any(v[0]!='COMPLETED' or v[2]!='0:0' for v in statuses.values()): raise ValueError('All phase and merge jobs must complete successfully before download.')
            destination=Path(data.get('destination',''))
            if not data.get('destination') or not destination.is_dir(): raise ValueError('Choose an existing local download folder.')
            target=destination/Path(run['folder']).name
            target.mkdir(exist_ok=False)
            paths=[f'kikuchipyCUDA/dictionary-run-{job}' for job,info in run['jobs'].items() if info['role'].startswith('Phase ')]
            if any(info['role']=='Merge' for info in run['jobs'].values()): paths.append(run['folder']+'/combined')
            for path in paths:
                self.log('Downloading ~/'+path)
                result=subprocess.run(['scp','-B','-o','ConnectTimeout=12','-r',run['user']+'@kamiak.wsu.edu:'+path,str(target)],capture_output=True,text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                if result.returncode: raise RuntimeError('Download incomplete; partial files retained in '+str(target)+'\n'+result.stderr)
            return 'Results saved to '+str(target)


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass
    def reply(self,code,body,mime='application/json'):
        if not isinstance(body,bytes): body=json.dumps(body).encode()
        self.send_response(code)
        self.send_header('Content-Type',mime)
        self.send_header('Content-Length',str(len(body)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Security-Policy',"default-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'")
        self.end_headers(); self.wfile.write(body)
    def allowed(self):
        origin='http://127.0.0.1:'+str(self.server.server_port)
        return self.headers.get('Host')==origin[7:] and self.headers.get('Origin',origin)==origin
    def authorized(self):
        return self.allowed() and secrets.compare_digest(self.headers.get('X-Kamiak-Token',''),self.server.token)
    def do_GET(self):
        if not self.allowed(): self.reply(403,{'error':'Local access only'}); return
        path=urlsplit(self.path).path
        if path=='/api/state':
            if not self.authorized(): self.reply(403,{'error':'Reopen Kamiak DI from the app launcher.'}); return
            self.reply(200,self.server.session.snapshot()); return
        name=unquote(path).lstrip('/') or 'index.html'
        file=(ASSETS/name).resolve()
        if not file.is_relative_to(ASSETS.resolve()) or not file.is_file(): self.reply(404,{'error':'Not found'}); return
        self.reply(200,file.read_bytes(),mimetypes.guess_type(str(file))[0] or 'application/octet-stream')
    def do_POST(self):
        if not self.authorized(): self.reply(403,{'error':'Unauthorized local request'}); return
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=1024*1024: raise ValueError('Invalid request size')
            data=json.loads(self.rfile.read(length))
            if not isinstance(data,dict): raise ValueError('Invalid request')
            if self.path=='/api/shutdown':
                if self.server.session.snapshot()['busy']: raise ValueError('Wait for the current transfer to finish.')
                self.reply(200,{'ok':True}); threading.Thread(target=self.server.shutdown,daemon=True).start(); return
            if self.path!='/api/action': self.reply(404,{'error':'Unknown endpoint'}); return
            self.server.session.begin(data.get('action'),data.get('data',{}))
            self.reply(202,{'ok':True})
        except (ValueError,TypeError,KeyError) as exc: self.reply(400,{'error':str(exc)})


def make_server(session,port=0):
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    server.session=session; server.token=secrets.token_urlsafe(32)
    server.daemon_threads=True
    return server


def main():
    import tkinter as tk
    from tkinter import filedialog
    if not (ASSETS/'index.html').exists(): raise RuntimeError('Built interface missing. Run npm ci && npm run build inside desktop/web.')
    metadata=Path(tempfile.gettempdir())/'Kamiak-DI-session.json'
    try:
        previous=json.loads(metadata.read_text())
        origin=f'http://127.0.0.1:{int(previous["port"])}'
        request=urllib.request.Request(origin+'/api/state',headers={'X-Kamiak-Token':previous['token']})
        with urllib.request.urlopen(request,timeout=2) as response:
            previous_state=json.load(response)
        if '--desktop' not in sys.argv and previous_state.get('product')=='Kamiak DI React':
            webbrowser.open(origin+'/#'+previous['token']);return
    except (OSError,ValueError,KeyError,TypeError): pass
    root=tk.Tk(); root.withdraw()
    dialogs=queue.Queue(); session=Session(dialogs); server=make_server(session)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    def poll():
        if not thread.is_alive(): root.destroy();return
        if not dialogs.empty():
            kind,answer=dialogs.get()
            try:
                root.attributes('-topmost',True)
                if kind=='destination': value=filedialog.askdirectory(parent=root,title='Download completed results to…')
                else:
                    ext={'ang':'ang','up':'up1','master0':'sht','master1':'sht'}[kind]
                    options=dict(parent=root,title='Choose '+ext.upper()+' file',filetypes=[(ext.upper(),'*.'+ext)])
                    if ext=='sht' and (ROOT/'SHTdatabase-master/EBSD').exists(): options['initialdir']=str(ROOT/'SHTdatabase-master/EBSD')
                    value=filedialog.askopenfilename(**options)
                answer.put(value)
            except Exception as exc: answer.put(exc)
            finally: root.attributes('-topmost',False)
        root.after(100,poll)
    root.after(100,poll)
    url=f'http://127.0.0.1:{server.server_port}/#'+server.token
    fd=os.open(metadata,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
    os.fchmod(fd,0o600) if hasattr(os,'fchmod') else None
    with os.fdopen(fd,'w') as f: json.dump(dict(port=server.server_port,token=server.token),f)
    if '--desktop' in sys.argv:
        print(json.dumps(dict(kamiak_ready=True,port=server.server_port,token=server.token)),flush=True)
    else:
        webbrowser.open(url)
        print('Kamiak DI React interface running on localhost; use Quit app to stop.',flush=True)
    try: root.mainloop()
    finally:
        server.shutdown();server.server_close()
        try:
            if json.loads(metadata.read_text()).get('token')==server.token: metadata.unlink()
        except (OSError,ValueError): pass


if __name__=='__main__':main()
