"""Offline artifact gates only; synthetic files are not GPU/model validation."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from tools.gfx906_model import MODEL_FILES,REPO,REVISION
from tools.gfx906_prefill_compare import report,validate_identity
from tools.gfx906_prefill_validate import parse_done


def identity(model='IQ2_XS'):
    return dict(model=model,repo=REPO,revision=REVISION,full_sha256=True,files=[
        dict(path='/offline/'+name,sha256=sha,fingerprint=dict(device=1,inode=i+1,bytes=size,mtime_ns=1))
        for i,(name,size,sha) in enumerate(MODEL_FILES[model])])


class AuditTests(unittest.TestCase):
    def test_published_model_attestation_and_complete_shards(self):
        valid=identity();self.assertEqual(validate_identity('IQ2_XS',valid),valid)
        for mutation in ['revision','bool','hash','size','name','missing']:
            bad=copy.deepcopy(valid)
            if mutation=='revision':bad['revision']='a'*40
            if mutation=='bool':bad['full_sha256']=1
            if mutation=='hash':bad['files'][0]['sha256']='b'*64
            if mutation=='size':bad['files'][0]['fingerprint']['bytes']-=1
            if mutation=='name':bad['files'][0]['path']='wrong.gguf'
            if mutation=='missing':bad['files'].pop()
            with self.assertRaises(ValueError):validate_identity('IQ2_XS',bad)

    def evidence(self,root):
        def save(path,value):
            path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
        ids=[1]*1024;sha=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
        settings=dict(targets='1024',kinds='code',configs=['offline-IQ2_XS'],arms='2048:0:24:0',repeats=1)
        save(root/'candidate.json',dict(sha256='a'*64,application_source_ref='b'*40,settings=settings))
        save(root/'candidate-after.json',dict(sha256='a'*64))
        for name in ['deployment-before.json','deployment-after.json']:save(root/name,{'offline':'unchanged'})
        save(root/'IQ2_XS-identity.json',identity())
        save(root/'fixtures/code-1024.json',dict(ids=ids,input_ids_sha256=sha,tokens=1024,kind='code',target=1024))
        arm=root/'IQ2_XS-r0-c2048-m0-s24-j0';arm.mkdir()
        line='DONE 2 1024 10 2 length 0 0 0 0 0 0 0 0 1024'
        save(arm/'case-0.json',dict(ids=[2,3],engine=parse_done(line),kind='code',target=1024,input_ids_sha256=sha,ttft_s=.01,wall_s=.012))
        save(arm/'exit.json',dict(exit_code=0))
        save(arm/'profiles.json',dict(request_profiles=[dict(request=dict(prompt_tokens=1024,reused=0,read_from=0,cancelled=False,prefill_rows=1023),stages=[dict(device=0,tokens=1023),dict(device=1,tokens=1023)])]))
        save(arm/'memory-peaks.json',dict(samples=1,sample_interval_s=1.0,rss_bytes=1,minimum_available_ram_bytes=1,vram_bytes={'offline0':1,'offline1':1}))
        (arm/'engine.stdin.raw').write_text('GEN 2 top_k=1 seed=42 '+','.join(map(str,ids))+'\nQUIT\n')
        (arm/'engine.stdout.raw').write_text('INFO offline\nREADY\nT 2\nT 3\n'+line+'\n')
        for name in ['engine.stderr.raw','memory.samples.jsonl']:(arm/name).write_text('offline\n')
        return arm

    def test_complete_audit_rejects_failure_or_substituted_repetition(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);arm=self.evidence(root)
            self.assertTrue(report(root)['deployment_unchanged'])
            (root/'failure.json').write_text('{}')
            with self.assertRaises(ValueError):report(root)
            (root/'failure.json').unlink();arm.rename(root/arm.name.replace('-r0-','-r7-'))
            with self.assertRaises(ValueError):report(root)

    def test_memory_and_profile_cold_coverage_are_required(self):
        for mutation in ['samples','card','stage_rows','reuse']:
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);arm=self.evidence(root)
                mp=arm/'memory-peaks.json';mem=json.loads(mp.read_text())
                pp=arm/'profiles.json';profiles=json.loads(pp.read_text())
                if mutation=='samples':mem['samples']=0
                if mutation=='card':mem['vram_bytes'].pop('offline1')
                if mutation=='stage_rows':profiles['request_profiles'][0]['stages'][0]['tokens']=1024
                if mutation=='reuse':profiles['request_profiles'][0]['request']['reused']=1
                mp.write_text(json.dumps(mem));pp.write_text(json.dumps(profiles))
                with self.assertRaises(ValueError):report(root)

    def test_done_token_count_must_match_original_submitted_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);arm=self.evidence(root)
            path=arm/'case-0.json';case=json.loads(path.read_text())
            case['engine']['prompt_tokens']=1025;case['engine']['prompt_read']=1025;path.write_text(json.dumps(case))
            stdout=arm/'engine.stdout.raw';stdout.write_text(stdout.read_text().replace('DONE 2 1024','DONE 2 1025').replace('0 0 1024\n','0 0 1025\n'))
            with self.assertRaises(ValueError):report(root)
