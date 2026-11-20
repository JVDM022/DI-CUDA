"""Desktop checks use temporary inputs and mocked SSH; never submit cluster jobs."""
import importlib.util
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('kamiak_di', Path(__file__).parents[1]/'desktop/kamiak_di.py')
app = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(app)


class DesktopTests(unittest.TestCase):
    def test_batch_preserves_indexing_settings_and_records_merge(self):
        script = app.batch_script(2, [1, 3], True)
        subprocess.run(['bash', '-n'], input=script, text=True, check=True)
        self.assertEqual(script.count('--limit 0 --resolution 1 --npa'), 2)
        self.assertIn('--dependency="afterok:${jobs[0]}:${jobs[1]}"', script)
        self.assertIn('echo "Merge job: ${merge%%;*}"', script)
        self.assertIn('Phase 3 job:', script)
        self.assertNotIn('merge=', app.batch_script(1, [1], False))

    def test_custom_angle_gpu_and_walltime_reach_slurm_and_indexer(self):
        script = app.batch_script(2, [1, 2], False, resolution=2.5, gpus=2, hours=30)
        subprocess.run(['bash', '-n'], input=script, text=True, check=True)
        self.assertEqual(script.count('--gres=gpu:2 --time=1-06:00:00'), 2)
        self.assertEqual(script.count('--devices 2 --limit 0 --resolution 2.5'), 2)
        self.assertNotIn('--npa', script)
        for values in [(0,1,2), (float('nan'),1,2), (1,3,2), (1,1,0), (1,1,2.5), (1,1,169), ('1; bad',1,2)]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                app.indexing_settings(*values)

    def test_input_inspection_and_phase_assignments(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp); ang=p/'scan.ang'; up=p/'scan.up1'; master=p/'Ni.sht'
            ang.write_text('# Phase 1\n# MaterialName Nickel\n0 0 0 0 0\n')
            up.write_bytes(struct.pack('<4i',1,4,4,16)+bytes(32)); master.write_bytes(b'master')
            self.assertIn('2 patterns', app.inspect_inputs(ang,up,[master],[1]))
            with self.assertRaisesRegex(ValueError,'valid ANG phase'):
                app.inspect_inputs(ang,up,[master],[2])
            up.write_bytes(b'truncated')
            with self.assertRaisesRegex(ValueError,'truncated'):
                app.inspect_inputs(ang,up,[master],[1])

    def test_connection_and_status_are_bounded_and_validate_arguments(self):
        with patch.object(app.shutil,'which',return_value='/usr/bin/ssh'), patch.object(app.subprocess,'run') as run:
            run.return_value=subprocess.CompletedProcess([],0,'123|COMPLETED|00:12|0:0|\n123.batch|COMPLETED|00:12|0:0|\n','')
            states=app.job_status('researcher',['123'])
            self.assertEqual(states,{'123':['COMPLETED','00:12','0:0']})
            self.assertEqual(run.call_args.kwargs['timeout'],40)
            self.assertIn('BatchMode=yes',run.call_args.args[0])
            with self.assertRaises(ValueError): app.job_status('researcher',['123; rm'])
            with self.assertRaises(ValueError): app.check_connection('user;bad')

    def test_upload_emits_each_job_line_and_stops_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'scan.ang';p.write_text('# Phase 1\n# MaterialName Alpha\n# GRID: SqrGrid\n0 0 0 0 0 0 0 1\n')
            outputs=[b'',b'',b'',b'Phase 1 job: 123\nMerge job: 456\n']
            responses=[subprocess.CompletedProcess([],0,o) for o in outputs]
            log=[]
            with patch.object(app.subprocess,'run',side_effect=responses) as mocked:
                app.submit('researcher',[(p,'scan.ang')],[1],False,log.append, resolution=2.5, gpus=2, hours=6)
                script = mocked.call_args.kwargs['input'].decode()
                self.assertIn('--gres=gpu:2 --time=06:00:00', script)
                self.assertIn('--devices 2 --limit 0 --resolution 2.5', script)
            self.assertIn('Phase 1 job: 123',log)
            self.assertIn('Merge job: 456',log)
            with patch.object(app.subprocess,'run',return_value=subprocess.CompletedProcess([],1,b'permission denied')):
                with self.assertRaises(RuntimeError): app.submit('researcher',[],[1],False,lambda _:None)

    def test_saved_job_status_is_not_trusted_and_partial_jobs_are_visible(self):
        # Exercise UI callbacks with stand-ins, without needing a display.
        from types import SimpleNamespace
        class Log:
            def configure(self,**kwargs):pass
            def insert(self,*args):pass
            def see(self,*args):pass
        view=SimpleNamespace(log=Log(),jobs={},remote_folder=None,render_jobs=lambda:None)
        app.KamiakApp.append_log(view,'Phase 3 job: 321')
        self.assertEqual(view.jobs['321']['role'],'Phase 3')
        app.KamiakApp.append_log(view,'Merge job: 322')
        view.remote_folder='kikuchipyCUDA/scans/di-123456abcdef'
        paths=app.KamiakApp.result_paths(view)
        self.assertEqual(paths,['kikuchipyCUDA/dictionary-run-321',view.remote_folder+'/combined'])


    def test_receipt_restores_jobs_but_requires_fresh_status(self):
        import json
        from types import SimpleNamespace
        class Value:
            def __init__(self): self.value=''
            def set(self,value): self.value=value
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'receipt.json'
            p.write_text(json.dumps(dict(schema=1,user='researcher',folder='kikuchipyCUDA/scans/di-123456abcdef',
                jobs={'123':dict(role='Phase 3',state='COMPLETED',elapsed='00:01',exit='0:0')},
                submission_complete=True,settings=dict(resolution=2.5,gpus=2,hours=6))))
            view=SimpleNamespace(root=None,render_jobs=lambda:None,message=Value())
            with patch.object(app.filedialog,'askopenfilename',return_value=str(p)), patch.object(app.messagebox,'showerror') as error:
                app.KamiakApp.load_receipt(view)
                error.assert_not_called()
            self.assertEqual(view.jobs['123']['state'],'Needs refresh')
            self.assertEqual(view.run_settings['gpus'],2)
            self.assertTrue(view.submission_complete)

    def test_partial_submission_cannot_download(self):
        from types import SimpleNamespace
        view=SimpleNamespace(root=None,submission_complete=False,jobs={'123':dict(state='COMPLETED',exit='0:0')})
        with patch.object(app.messagebox,'showinfo') as notice, patch.object(app.filedialog,'askdirectory') as picker:
            app.KamiakApp.download(view)
            notice.assert_called_once(); picker.assert_not_called()


    def test_single_phase_copy_preserves_selected_phase_and_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); source=p/'source.ang'; dest=p/'upload.ang'
            content='# x-star 0.5\n# Phase 1\n# MaterialName Alpha\n# Phase 3\n# MaterialName Beta\n# GRID: SqrGrid\n# XSTEP: 1\n0 0 0 12.5 8.5 4 0.9 1 0 0\n'
            source.write_text(content)
            app.single_phase_ang(source,dest,3)
            prepared=dest.read_text()
            self.assertEqual(source.read_text(),content)
            self.assertEqual(app.phases(dest),{3:'Beta'})
            self.assertIn('# GRID: SqrGrid',prepared)
            self.assertIn('# x-star 0.5',prepared)
            row=next(line for line in prepared.splitlines() if not line.startswith('#')).split()
            self.assertEqual(row[3:5],['12.5','8.5'])
            self.assertEqual(row[7],'3')


if __name__=='__main__': unittest.main()
