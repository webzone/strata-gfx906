import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.gfx906_prefill_validate import Pipe, fixture, idle_smi, parse_done, store, verify_model, estimate_evidence
from tools.gfx906_model import REVISION, REPO


class BytesTokenizer:
    def encode(self, rendered, parse_special):
        return list(rendered.encode())


class Template:
    def render(self, history, enable_thinking):
        return ''.join('<'+m['role']+'>'+m['content']+'</end>' for m in history)+'<assistant>'


class ValidationTests(unittest.TestCase):
    def test_peak_estimate_counts_repeated_protocol_and_product_traces(self):
        args=([65536],['code'],[(2048,0,24,0)],1,1)
        base=estimate_evidence(*args)
        self.assertGreaterEqual(base,65536*8)
        trace=estimate_evidence(*args,trace_mmq=True)
        self.assertEqual(trace-base,32*48*32*1024)
        self.assertGreater(estimate_evidence(*args,profile_experts=True),base)
        fixed=64*65536
        self.assertEqual(estimate_evidence([65536],['code'],[(2048,0,24,0)],2,1),fixed+2*(base-fixed))
        self.assertGreater(estimate_evidence([65536],['code'],[(2048,0,24,0)],2,2),base)
        self.assertLess(estimate_evidence([65536],['code'],[(4096,0,24,0)],1,1,trace_mmq=True),trace)
        for targets,kinds,arms,repeats,models in [([],['code'],[(2048,0,24,0)],1,1),([1],[],[(2048,0,24,0)],1,1),([1],['code'],[],1,1),([1],['code'],[(2048,0,24,0)],0,1)]:
            with self.assertRaises(ValueError):estimate_evidence(targets,kinds,arms,repeats,models)

    def test_idle_requires_two_unused_cards_and_no_pid(self):
        text='No KFD PIDs currently running\n'
        text+='GPU[0]: VRAM Total Used Memory (B): 10866688\nGPU[1]: VRAM Total Used Memory (B): 10866688\n'
        text+='GPU[0]: GPU use (%): 0\nGPU[1]: GPU use (%): 0\n'
        self.assertTrue(idle_smi(text))
        self.assertFalse(idle_smi(text.replace('running','missing')))
        self.assertFalse(idle_smi(text.replace('10866688','10000000000')))
        self.assertFalse(idle_smi(text.replace('(%): 0','(%): 1')))
        self.assertFalse(idle_smi(text.replace('GPU[1]: VRAM Total Used Memory (B): 10866688\n','')))

    def test_done_is_complete_and_finite(self):
        line='DONE 1 12 1.5 0.5 stop 0 1 0 2 2 0 0 0 12'
        self.assertEqual(parse_done(line)['prompt_read'],12)
        with self.assertRaises(ValueError):parse_done('DONE 1 12 1.5 0.5 stop')
        with self.assertRaises(ValueError):parse_done(line.replace('1.5','nan'))
        with self.assertRaises(ValueError):parse_done(line.replace('1.5','-1'))

    def test_evidence_is_private_and_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'data.json';store(path,dict(ids=[1,2]))
            self.assertEqual(path.stat().st_mode&0o777,0o600)
            with self.assertRaises(FileExistsError):store(path,dict(ids=[3]))
            self.assertEqual(json.loads(path.read_text())['ids'],[1,2])

    def test_budget_is_actual_count_and_role_suffix_preserved(self):
        for kind in ['code','chinese','chat']:
            f=fixture(BytesTokenizer(),Template(),kind,2048)
            self.assertLessEqual(f['tokens'],2048)
            self.assertGreaterEqual(f['tokens'],2032)
            self.assertEqual(len(f['ids']),f['tokens'])
            self.assertTrue(f['rendered'].endswith('<assistant>'))
            self.assertEqual(f['messages'][-1]['role'],'user')

    def test_unverified_cache_cannot_bypass_full_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);path=tmp/'fixture.gguf';path.write_bytes(b'abc')
            from tools.gfx906_prefill_validate import fingerprint
            cfg=dict(args=['--native',str(path)])
            cache=dict(repo=REPO,revision=REVISION,full_sha256=False,files=[dict(path=str(path),sha256='expected',fingerprint=fingerprint(path))])
            with patch('tools.gfx906_prefill_validate.MODEL_FILES',{'IQ2_XS':[(path.name,3,'expected')]}),patch('tools.gfx906_prefill_validate.digest',return_value='bad') as hashed:
                with self.assertRaises(ValueError):verify_model(cfg,'IQ2_XS',tmp/'identity.json',cache)
                self.assertEqual(hashed.call_count,1)
                self.assertFalse((tmp/'identity.json').exists())

    def fake(self, tmp, extra=''):
        exe=Path(tmp)/'fake.py'
        exe.write_text('''#!/usr/bin/env python3
import sys
print('INFO engine=unit-fixture',flush=True)
print('READY 262144 stop',flush=True)
for line in sys.stdin:
 if line.startswith('QUIT'):break
 if line.startswith('GEN '):
  assert len(line.split())==5, 'The actual pipe protocol requires one comma-separated ID field'
  n=len(line.split()[-1].split(','))
  print('T 7',flush=True)
  print(f'DONE 1 {n} 1.5 0.5 stop 0 1 REUSE 0 0 0 0 0 {n}',flush=True)
'''.replace('REUSE',extra or '0'))
        exe.chmod(0o700);return exe

    def test_complete_pipe_raw_startup_request_output_and_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe=self.fake(tmp);out=Path(tmp)/'out';out.mkdir()
            pipe=Pipe(exe,[],tmp,dict(os.environ),out,timeout=5)
            try:
                result=pipe.generate([11,12,13],3,5)
                self.assertEqual(result['ids'],[7])
                self.assertEqual(result['engine']['reused'],0)
            finally:rc=pipe.close()
            self.assertEqual(rc,0)
            stdout=(out/'engine.stdout.raw').read_bytes()
            self.assertTrue(stdout.startswith(b'INFO engine=unit-fixture\nREADY'))
            self.assertIn(b'T 7\nDONE 1 3',stdout)
            from serve.server import StrataEngine
            keys=StrataEngine.sampling_keys(dict(temperature=0,top_k=1,top_p=1,min_p=0,seed=42))
            # Compare to the actual frontend contract, not just a fake reproducing our own assumptions.
            expected=('GEN 3'+keys+' 11,12,13\nQUIT\n').encode()
            self.assertEqual((out/'engine.stdin.raw').read_bytes(),expected)
            for f in out.glob('*.raw'):self.assertEqual(f.stat().st_mode&0o777,0o600)

    def test_reused_sample_refused_and_own_child_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe=self.fake(tmp,'1');out=Path(tmp)/'out';out.mkdir()
            pipe=Pipe(exe,[],tmp,dict(os.environ),out,timeout=5)
            try:
                with self.assertRaisesRegex(ValueError,'reused'):pipe.generate([1,2],3,5)
            finally:self.assertEqual(pipe.close(),0)


if __name__=='__main__':unittest.main()
