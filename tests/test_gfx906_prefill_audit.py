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
from tools.gfx906_prefill_profile import read_records,summarize


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
        settings=dict(targets='1024',kinds='code',configs=['offline-IQ2_XS'],arms='2048:0:24:0',repeats=1,
                      max_new=2,profile_experts=False,trace_mmq=False)
        save(root/'candidate.json',dict(engine='/offline/strata',sha256='a'*64,application_source_ref='b'*40,settings=settings))
        save(root/'candidate-after.json',dict(sha256='a'*64))
        for name in ['deployment-before.json','deployment-after.json']:save(root/name,{'offline':'unchanged'})
        save(root/'IQ2_XS-identity.json',identity())
        save(root/'fixtures/code-1024.json',dict(ids=ids,input_ids_sha256=sha,tokens=1024,kind='code',target=1024))
        arm=root/'IQ2_XS-r0-c2048-m0-s24-j0';arm.mkdir()
        args=['--native',identity()['files'][0]['path'],'--ple-gguf',identity()['files'][1]['path'],
              '--prefill','2048','--prompt-cache','0','--short-read','0','--kv','int8']
        save(arm/'config.private.json',dict(exe='/offline/strata',backend='hip',experimental_gfx906=True,gpu=[0,1],
            layer_split='24',args=args,env=dict(STRATA_PREFILL_TIMING='1',STRATA_GFX906_PREFILL_ATTN='0',
                STRATA_GFX906_MTP_BATCH='0',STRATA_MTP_BATCH='1',STRATA_GFX906_MMQ_J='0',
                STRATA_PREFILL_EXPERT_PROFILE='0',STRATA_GFX906_MMQ_TRACE='0')))
        save(arm/'process.json',dict(pid=1,command=['/offline/strata','--serve',*args,'--layer-split','24']))
        info='INFO context=262144 kv=int8 spec=4 engine=offline'
        save(arm/'startup.json',dict(info=[info],load_s=1.0))
        line='DONE 2 1024 10 2 length 0 0 0 0 0 0 0 0 1024'
        save(arm/'case-0.json',dict(ids=[2,3],engine=parse_done(line),kind='code',target=1024,input_ids_sha256=sha,ttft_s=.01,wall_s=.012))
        save(arm/'exit.json',dict(exit_code=0))
        raw=''
        for device,begin,end in [(0,0,24),(1,24,48)]:
            raw+='strata prefill profile: '+json.dumps(dict(schema=1,device=device,layer_begin=begin,layer_end=end,
                pos0=0,tokens=1023,chunk=2048,ms_wall=1.0,ms_gpu_timeline=1.0,phase_ms={}))+'\n'
        raw+='strata prefill draft: '+json.dumps(dict(schema=1,device=1,batched=False,ms_wall=1.0))+'\n'
        raw+='strata prefill request: '+json.dumps(dict(schema=1,prompt_tokens=1024,reused=0,read_from=0,
            cancelled=False,prefill_rows=1023,refilled_slots=0,ms_wall=10.0))+'\n'
        save(arm/'profiles.json',summarize(read_records(raw)))
        (arm/'engine.stderr.raw').write_text(raw)
        memory=dict(samples=1,sample_interval_s=1.0,rss_bytes=1,minimum_available_ram_bytes=1,vram_bytes={'offline0':1,'offline1':1})
        save(arm/'memory-peaks.json',memory)
        sample=dict(monotonic_s=1.0,rss_bytes=1,available_ram_bytes=1,vram_bytes=memory['vram_bytes'])
        (arm/'memory.samples.jsonl').write_text(json.dumps(sample)+'\n')
        (arm/'engine.stdin.raw').write_text('GEN 2 top_k=1 seed=42 '+','.join(map(str,ids))+'\nQUIT\n')
        (arm/'engine.stdout.raw').write_text(info+'\nREADY 248320\nT 2\nT 3\n'+line+'\n')
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
        for mutation in ['samples','card','interval','stage_rows','reuse']:
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);arm=self.evidence(root)
                mp=arm/'memory-peaks.json';mem=json.loads(mp.read_text())
                pp=arm/'profiles.json';profiles=json.loads(pp.read_text())
                if mutation=='samples':mem['samples']=0
                if mutation=='card':mem['vram_bytes'].pop('offline1')
                if mutation=='interval':mem['sample_interval_s']=True
                if mutation=='stage_rows':profiles['request_profiles'][0]['stages'][0]['tokens']=1024
                if mutation=='reuse':profiles['request_profiles'][0]['request']['reused']=1
                mp.write_text(json.dumps(mem));pp.write_text(json.dumps(profiles))
                with self.assertRaises(ValueError):report(root)

    def test_launch_and_startup_metadata_cannot_substitute_controls(self):
        for mutation in ['chunk','duplicate','split','tile','gpu','model','override','binary','info']:
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);arm=self.evidence(root)
                cfg=json.loads((arm/'config.private.json').read_text())
                if mutation=='chunk':cfg['args'][cfg['args'].index('--prefill')+1]='4096'
                if mutation=='duplicate':cfg['args'].append('--prefill=4096')
                if mutation=='split':cfg['args']+=['--layer-split','23']
                if mutation=='tile':cfg['env']['STRATA_GFX906_MMQ_J']='auto'
                if mutation=='gpu':cfg['gpu']=[False,True]
                if mutation=='model':cfg['args'][cfg['args'].index('--native')+1]='/offline/wrong.gguf'
                if mutation=='override':cfg['env']['HSA_OVERRIDE_GFX_VERSION']='9.0.6'
                (arm/'config.private.json').write_text(json.dumps(cfg))
                args=cfg['args'] if '--layer-split' in cfg['args'] else cfg['args']+['--layer-split','24']
                command=['/offline/wrong-strata' if mutation=='binary' else '/offline/strata','--serve',*args]
                (arm/'process.json').write_text(json.dumps(dict(pid=1,command=command)))
                if mutation=='info':(arm/'startup.json').write_text(json.dumps(dict(info=['INFO changed'],load_s=1.0)))
                with self.assertRaises(ValueError):report(root)

    def test_actual_batch_path_is_required_not_just_requested_switch(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);arm=self.evidence(root)
            candidate=json.loads((root/'candidate.json').read_text());candidate['settings']['arms']='2048:1:24:0'
            (root/'candidate.json').write_text(json.dumps(candidate))
            cfg=json.loads((arm/'config.private.json').read_text());cfg['env']['STRATA_GFX906_MTP_BATCH']='1'
            (arm/'config.private.json').write_text(json.dumps(cfg));arm=arm.rename(root/arm.name.replace('-m0-','-m1-'))
            with self.assertRaisesRegex(ValueError,'Actual last-stage MTP'):report(root)

    def test_raw_sampling_parameters_and_profiles_are_authoritative(self):
        for mutation in ['seed','greedy','limit','profile']:
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);arm=self.evidence(root)
                path=arm/'engine.stdin.raw';raw=path.read_text()
                if mutation=='seed':raw=raw.replace('seed=42','seed=43')
                if mutation=='greedy':raw=raw.replace('top_k=1','top_k=2')
                if mutation=='limit':raw=raw.replace('GEN 2 ','GEN 3 ')
                path.write_text(raw)
                if mutation=='profile':
                    path=arm/'engine.stderr.raw';path.write_text(path.read_text().replace('"ms_gpu_timeline": 1.0','"ms_gpu_timeline": 2.0'))
                with self.assertRaises(ValueError):report(root)

    def test_memory_summary_is_recomputed_from_all_raw_samples(self):
        for mutation in ['peak','missing','card','backwards']:
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);arm=self.evidence(root);path=arm/'memory.samples.jsonl'
                sample=json.loads(path.read_text());lines=[sample]
                if mutation=='peak':sample['rss_bytes']=2
                if mutation=='missing':lines=[]
                if mutation=='card':sample['vram_bytes'].pop('offline1')
                if mutation=='backwards':lines.append(copy.deepcopy(sample))
                path.write_text(''.join(json.dumps(row)+'\n' for row in lines))
                with self.assertRaises(ValueError):report(root)

    def test_requested_layer_profiles_and_positive_request_timing_are_required(self):
        for mutation in ['layers','ttft','wall']:
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);arm=self.evidence(root)
                if mutation=='layers':
                    candidate=json.loads((root/'candidate.json').read_text());candidate['settings']['profile_experts']=True
                    (root/'candidate.json').write_text(json.dumps(candidate))
                    cfg=json.loads((arm/'config.private.json').read_text());cfg['env']['STRATA_PREFILL_EXPERT_PROFILE']='1'
                    (arm/'config.private.json').write_text(json.dumps(cfg))
                else:
                    case=json.loads((arm/'case-0.json').read_text())
                    if mutation=='ttft':case['ttft_s']=float('nan')
                    if mutation=='wall':case['wall_s']=case['ttft_s']/2
                    (arm/'case-0.json').write_text(json.dumps(case))
                with self.assertRaises(ValueError):report(root)

    def test_generated_count_cannot_exceed_raw_request_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);arm=self.evidence(root)
            path=arm/'case-0.json';case=json.loads(path.read_text());case['ids'].append(4);case['engine']['generated']=3
            path.write_text(json.dumps(case));path=arm/'engine.stdout.raw'
            path.write_text(path.read_text().replace('T 3\n','T 3\nT 4\n').replace('DONE 2 ','DONE 3 '))
            with self.assertRaisesRegex(ValueError,'bounded cold generation'):report(root)

    def test_done_token_count_must_match_original_submitted_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);arm=self.evidence(root)
            path=arm/'case-0.json';case=json.loads(path.read_text())
            case['engine']['prompt_tokens']=1025;case['engine']['prompt_read']=1025;path.write_text(json.dumps(case))
            stdout=arm/'engine.stdout.raw';stdout.write_text(stdout.read_text().replace('DONE 2 1024','DONE 2 1025').replace('0 0 1024\n','0 0 1025\n'))
            with self.assertRaises(ValueError):report(root)
