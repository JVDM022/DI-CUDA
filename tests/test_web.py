"""React local service tests; SSH, submissions and native dialogs are mocked."""
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import threading
import unittest
import urllib.request
import urllib.error
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('kamiak_web',Path(__file__).parents[1]/'desktop/kamiak_web.py')
web=importlib.util.module_from_spec(spec);spec.loader.exec_module(web)

class WebTests(unittest.TestCase):
    def inputs(self,p):
        (p/'scan.ang').write_text('# Phase 1\n# MaterialName Alpha\n# Phase 3\n# MaterialName Beta\n# GRID: SqrGrid\n0 0 0 0 0 0 0 1\n')
        (p/'scan.up1').write_bytes(struct.pack('<4i',1,4,4,16)+bytes(16))
        for name in ['a.sht','b.sht']:(p/name).write_bytes(b'master')
        return dict(user='researcher',ang=str(p/'scan.ang'),up=str(p/'scan.up1'),masters=[str(p/'a.sht')],ids=[3],resolution=2.5,gpus=2,hours=6,npa=False)

    def test_login_uses_fixed_command_and_does_not_log_auth_output(self):
        session=web.Session();session.state['connected_user']='researcher'
        result=subprocess.CompletedProcess([],0,'private-auth-output','private-auth-error')
        with patch.object(web.shutil,'which',return_value='/usr/local/bin/opkssh'), patch.object(web.subprocess,'run',return_value=result) as run:
            notice=session.perform('login',{})
        self.assertEqual(run.call_args.args[0],['/usr/local/bin/opkssh','login','kamiak'])
        self.assertEqual(run.call_args.kwargs['timeout'],300)
        self.assertEqual(run.call_args.kwargs['stdin'],subprocess.DEVNULL)
        self.assertIsNone(session.state['connected_user'])
        self.assertIn('Check connection',notice)
        self.assertNotIn('private-auth',json.dumps(session.snapshot()))

    def test_login_missing_tool_failure_and_timeout(self):
        with patch.object(web.shutil,'which',return_value=None),patch.object(web.Path,'is_file',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'not installed'):web.login_kamiak()
        with patch.object(web.shutil,'which',return_value='/usr/local/bin/opkssh'):
            with patch.object(web.subprocess,'run',return_value=subprocess.CompletedProcess([],1,'private-auth-token','private-auth-error')):
                with self.assertRaisesRegex(RuntimeError,'did not complete') as error:web.login_kamiak()
                self.assertNotIn('private-auth',str(error.exception))
            with patch.object(web.subprocess,'run',side_effect=subprocess.TimeoutExpired(['opkssh'],300,output='private-auth-token')):
                with self.assertRaisesRegex(RuntimeError,'five minutes') as error:web.login_kamiak()
                self.assertNotIn('private-auth',str(error.exception))

    def test_single_or_two_phase_selection_with_custom_resources(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);data=self.inputs(p);session=web.Session()
            result=session.selection(data)
            self.assertEqual(result[1],[3]);self.assertEqual(result[5]['gpus'],2)
            data.update(ids=[1,3],masters=[str(p/'a.sht'),str(p/'b.sht')])
            self.assertEqual(session.selection(data)[1],[1,3])
            data['ids']=[1,1]
            with self.assertRaises(ValueError):session.selection(data)

    def test_submission_records_jobs_and_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            data=self.inputs(Path(tmp));session=web.Session();session.state['connected_user']='researcher'
            def submit(user,files,ids,npa,log,**settings):
                self.assertEqual(settings,dict(resolution=2.5,gpus=2,hours=6))
                log('Run folder: ~/kikuchipyCUDA/scans/di-123456abcdef');log('Phase 3 job: 123')
            with patch.object(web,'submit',side_effect=submit): session.perform('submit',data)
            self.assertTrue(session.state['run']['complete'])
            self.assertEqual(session.state['run']['jobs'],{'123':{'role':'Phase 3'}})

    def test_partial_submission_and_untrusted_receipt(self):
        session=web.Session()
        receipt=dict(schema=2,user='researcher',folder='kikuchipyCUDA/scans/di-123456abcdef',complete=True,jobs={'123':dict(role='Phase 3',state='COMPLETED')})
        session.perform('restore',dict(receipt=receipt))
        self.assertEqual(session.state['statuses'],{})
        with patch.object(web,'job_status',return_value={'123':['RUNNING','00:01','0:0']}):
            with self.assertRaisesRegex(ValueError,'complete successfully'):session.perform('download',{})
        receipt['folder']='../../bad'
        with self.assertRaises(ValueError):session.perform('restore',dict(receipt=receipt))

    def test_native_browse_types_are_restricted(self):
        with self.assertRaises(ValueError):web.Session().perform('browse',dict(kind='anything'))

    def test_local_api_requires_session_token_and_rejects_foreign_origins(self):
        session=web.Session();server=web.make_server(session)
        worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
        origin=f'http://127.0.0.1:{server.server_port}'
        def request(path,headers=None,data=None):
            return urllib.request.urlopen(urllib.request.Request(origin+path,headers=headers or {},data=data),timeout=3)
        try:
            with self.assertRaises(urllib.error.HTTPError) as err:request('/api/state')
            self.assertEqual(err.exception.code,403)
            headers={'X-Kamiak-Token':server.token}
            with request('/api/state',headers) as response:self.assertEqual(json.load(response)['product'],'Kamiak DI React')
            with self.assertRaises(urllib.error.HTTPError) as err:request('/api/state',{**headers,'Origin':'https://elsewhere.example'})
            self.assertEqual(err.exception.code,403)
            with request('/') as response:self.assertIn(b'id="root"',response.read())
            with self.assertRaises(urllib.error.HTTPError):request('/api/action',headers,json.dumps({'action':'bogus'}).encode())
        finally:server.shutdown();server.server_close();worker.join()

if __name__=='__main__':unittest.main()
