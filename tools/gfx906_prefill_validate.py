#!/usr/bin/env python3
"""Serial, lease-protected, pipe-only gfx906 long-prefill validation.

No service operations, downloads, system changes or deployment writes. All output
is private, new and auditable. Raw protocol bytes are retained, including startup.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import resource
import shutil
import signal
import subprocess
import threading
import time

from tools.gfx906_model import MODEL_FILES, REPO, REVISION, digest
from tools.gfx906_prefill_profile import replace_arg, read_records, summarize

FLOOR = 4 * 2**30


def store(path, value):
    """New immutable evidence; never overwrite even a failed request."""
    with Path(path).open('xb') as file:
        file.write((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode())
    Path(path).chmod(0o600)


def parse_arms(value):
    result = []
    for item in value.split(','):
        fields = item.split(':')
        if len(fields) == 3:
            fields.append('0')
        if len(fields) != 4 or fields[3] not in ('0','16','32','48','64','auto'):
            raise ValueError('Arms are chunk:mtp:split[:0|16|32|48|64|auto]')
        chunk, mtp, split = map(int, fields[:3])
        tile = -1 if fields[3] == 'auto' else int(fields[3])
        if chunk not in (2048,3072,4096) or mtp not in (0,1) or split not in (23,24,25):
            raise ValueError('Conservative chunk/split bounds required')
        result.append((chunk, mtp, split, tile))
    if len(result) != len(set(result)):
        raise ValueError('Duplicate arms would overwrite evidence')
    return result


def fingerprint(path):
    st = Path(path).stat()
    return dict(device=st.st_dev, inode=st.st_ino, bytes=st.st_size, mtime_ns=st.st_mtime_ns)


def idle_smi(text):
    used = [int(x) for x in re.findall(r'VRAM Total Used Memory \(B\):\s*(\d+)', text)]
    utilization = [int(x) for x in re.findall(r'GPU use \(%\):\s*(\d+)', text)]
    return ('No KFD PIDs currently running' in text and len(used)==2 and max(used)<256*2**20
            and len(utilization)==2 and max(utilization)==0)


def preflight(root, evidence, minimum_ram_gib=64, drain=False):
    deadline = time.monotonic() + (60 if drain else 0)
    with Path(evidence).open('xb') as log:
        while True:
            commands = [['/opt/rocm/bin/rocm-smi','--showuse','--showmeminfo','vram','--showpids'],
                        ['ss','-lntp'], ['docker','ps','--format','{{.Names}} {{.Status}}'],
                        ['free','-m'], ['df','-m',str(root)]]
            outputs = []
            for cmd in commands:
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
                log.write(('COMMAND '+repr(cmd)+'\n').encode()+result.stdout);log.flush()
                outputs.append(result.stdout.decode('utf-8', 'replace'))
            if re.search(r'(?m)^LISTEN.*:8082\b', outputs[1]):
                raise RuntimeError('Owner API listener exists; refusing another engine')
            if 'No KFD PIDs currently running' not in outputs[0]:
                raise RuntimeError('GPU workload exists; refusing another engine')
            if idle_smi(outputs[0]): break
            if not drain or time.monotonic()>=deadline:
                raise RuntimeError('Both cards must be idle; deferred VRAM cleanup did not drain')
            time.sleep(2)
    available = int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines()
                         if l.startswith('MemAvailable:')))*1024
    if available < minimum_ram_gib*2**30 or shutil.disk_usage(root).free<FLOOR:
        raise RuntimeError('RAM/disk reserve insufficient; no engine started')


def verify_model(config, model, output, identity_cache=None):
    native = Path(config['args'][config['args'].index('--native')+1])
    entries = []
    for index, (name, size, sha) in enumerate(MODEL_FILES[model]):
        path = native if index==0 else Path(config['args'][config['args'].index('--ple-gguf')+1])
        if path.name!=name:raise ValueError('Configured shard filename does not match pinned identity')
        before=fingerprint(path)
        if before['bytes']!=size: raise ValueError('Wrong model shard size: '+str(path))
        trusted = (identity_cache or {}).get('full_sha256') is True and (identity_cache or {}).get('revision')==REVISION and (identity_cache or {}).get('repo')==REPO
        cached = next((e for e in (identity_cache or {}).get('files', []) if e.get('path')==str(path)), None)
        if trusted and cached and cached.get('fingerprint')==before and cached.get('sha256')==sha:
            actual=sha
        else:
            actual=digest(path)
        if actual!=sha or fingerprint(path)!=before:
            raise ValueError('Pinned source identity changed; never overwrite/download: '+str(path))
        entries.append(dict(path=str(path),sha256=actual,fingerprint=before))
    result=dict(repo=REPO,revision=REVISION,model=model,full_sha256=True,files=entries)
    store(output,result);return result


def parse_done(line):
    f=line.split()
    if len(f)<15 or f[0]!='DONE':raise ValueError('Missing complete DONE metrics')
    result=dict(generated=int(f[1]),prompt_tokens=int(f[2]),prompt_ms=float(f[3]),decode_ms=float(f[4]),finish=f[5],
                drafts_accepted=int(f[6]),drafts_offered=int(f[7]),reused=int(f[8]),hits=int(f[9]),lookups=int(f[10]),
                ram_blobs=int(f[11]),file_blobs=int(f[12]),file_mb=float(f[13]),prompt_read=int(f[14]))
    if any(v<0 for v in result.values() if type(v) is int) or result['drafts_accepted']>result['drafts_offered'] or result['hits']>result['lookups']:
        raise ValueError('Invalid DONE counters')
    if any(not math.isfinite(result[k]) or result[k]<0 for k in ['prompt_ms','decode_ms','file_mb']):
        raise ValueError('Invalid DONE timing')
    return result


class MemorySampler:
    """Read-only sysfs/proc sampling; no repeated GPU management subprocesses."""
    def __init__(self, pid, output):
        self.pid=pid;self.output=Path(output);self.stop=threading.Event();self.error=None
        self.paths=sorted(Path('/sys/class/drm').glob('card[0-9]*/device/mem_info_vram_used'))
        if len(self.paths)!=2:raise RuntimeError('Two GPU VRAM sysfs counters required for peak evidence')
        self.file=(self.output/'memory.samples.jsonl').open('xb');os.fchmod(self.file.fileno(),0o600)
        self.peaks=dict(rss_bytes=0,vram_bytes={},minimum_available_ram_bytes=None,sample_interval_s=1.0,samples=0)
        self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start()

    def _run(self):
        try:
            while not self.stop.is_set():
                rss=0
                try:
                    status=Path(f'/proc/{self.pid}/status').read_text()
                    rss=int(next(l.split()[1] for l in status.splitlines() if l.startswith('VmRSS:')))*1024
                except (FileNotFoundError,StopIteration):pass
                available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
                vram={p.parent.resolve().name:int(p.read_text()) for p in self.paths}
                self.peaks['rss_bytes']=max(self.peaks['rss_bytes'],rss)
                for bus,value in vram.items():self.peaks['vram_bytes'][bus]=max(self.peaks['vram_bytes'].get(bus,0),value)
                previous=self.peaks['minimum_available_ram_bytes'];self.peaks['minimum_available_ram_bytes']=min(previous,available) if previous is not None else available
                self.peaks['samples']+=1
                self.file.write((json.dumps(dict(monotonic_s=time.monotonic(),rss_bytes=rss,available_ram_bytes=available,vram_bytes=vram))+'\n').encode());self.file.flush()
                if shutil.disk_usage(self.output).free<FLOOR:raise RuntimeError('Disk reserve fell below 4 GiB')
                self.stop.wait(1)
        except BaseException as error:self.error=error

    def close(self):
        self.stop.set();self.thread.join(timeout=5);self.file.close()
        store(self.output/'memory-peaks.json',self.peaks)
        if self.error:raise self.error


class Pipe:
    def __init__(self, exe, args, cwd, env, output, timeout=600, monitor=False):
        self.output=Path(output);self.timeout=timeout;self.lines=queue.Queue()
        self.stderr=(self.output/'engine.stderr.raw').open('xb')
        self.stdout=(self.output/'engine.stdout.raw').open('xb')
        self.stdin=(self.output/'engine.stdin.raw').open('xb')
        for f in (self.stdin,self.stdout,self.stderr):os.fchmod(f.fileno(),0o600)
        self.proc=None;self.sampler=None
        try:
            self.proc=subprocess.Popen([str(exe),'--serve',*args],cwd=cwd,env=env,
                                       stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr)
            self.thread=threading.Thread(target=self._pump,daemon=True);self.thread.start()
            if monitor:self.sampler=MemorySampler(self.proc.pid,self.output)
            store(self.output/'process.json',dict(pid=self.proc.pid,command=[str(exe),'--serve',*args]))
            start=time.monotonic();self.info=[]
            while True:
                line=self.next(max(0.01,timeout-(time.monotonic()-start)))
                if line.startswith('INFO '):self.info.append(line)
                elif line.startswith('READY '):break
                elif line.startswith('ERR'):raise RuntimeError(line)
            self.load_s=time.monotonic()-start
        except BaseException:
            self.close();raise

    def _pump(self):
        try:
            for raw in iter(self.proc.stdout.readline,b''):
                self.stdout.write(raw);self.stdout.flush();self.lines.put(raw)
        finally:self.lines.put(None)

    def next(self, timeout):
        deadline=time.monotonic()+timeout
        while True:
            if self.sampler and self.sampler.error:raise self.sampler.error
            remaining=deadline-time.monotonic()
            if remaining<=0:raise TimeoutError('Own engine exceeded bounded progress window')
            try:raw=self.lines.get(timeout=min(1,remaining))
            except queue.Empty:continue
            if raw is None:raise RuntimeError('Own engine exited before complete protocol response')
            return raw.decode('utf-8', 'strict').rstrip('\r\n')

    def send(self, command):
        raw=command.encode();self.stdin.write(raw);self.stdin.flush()
        self.proc.stdin.write(raw);self.proc.stdin.flush()

    def generate(self, ids, max_new, timeout):
        started=time.monotonic();first=None;generated=[]
        self.send('GEN '+str(max_new)+' top_k=1 seed=42 '+ ','.join(map(str,ids))+'\n')
        while True:
            line=self.next(max(0.01,timeout-(time.monotonic()-started)))
            if line.startswith('T '):
                if first is None:first=time.monotonic()-started
                generated.append(int(line.split()[1]))
            elif line.startswith('DONE '):
                metrics=parse_done(line)
                if len(generated)!=metrics['generated'] or metrics['prompt_tokens']!=len(ids):
                    raise ValueError('Protocol/input/generated counts disagree')
                if metrics['reused']!=0 or metrics['prompt_read']!=len(ids):
                    raise ValueError('Cold sample reused tokens; refuse comparison')
                if not generated:raise ValueError('No real generation')
                return dict(ids=generated,engine=metrics,ttft_s=first,wall_s=time.monotonic()-started)
            elif line.startswith('ERR'):raise RuntimeError(line)

    def close(self):
        rc=None
        if self.proc is not None:
            if self.proc.poll() is None:
                try:
                    self.send('QUIT\n');self.proc.stdin.close();self.proc.wait(timeout=25)
                except (OSError,ValueError,subprocess.TimeoutExpired):
                    self.proc.terminate()
                    try:self.proc.wait(timeout=20)
                    except subprocess.TimeoutExpired:self.proc.kill();self.proc.wait(timeout=20)
            rc=self.proc.returncode
            if hasattr(self,'thread'):self.thread.join(timeout=5)
            self.proc.stdout.close()
            if not self.proc.stdin.closed:self.proc.stdin.close()
        for f in (self.stdin,self.stdout,self.stderr):f.close()
        if self.sampler:self.sampler.close();self.sampler=None
        return rc


def messages(kind, n):
    if kind=='code':
        body='\n'.join(f'export function rule{i}(x) {{ return x === {i} ? x + {i+1} : x - {i}; }}' for i in range(n))
        return [dict(role='user',content='Review this source and describe its behavior precisely in a paragraph.\n'+body)]
    if kind=='chinese':
        body='\n'.join(f'第{i}段：生产线记录温度、湿度与设备状态。阈值为25摄氏度，异常持续三分钟应报告，不能擅自停机。记录需要核验来源，区分推测和事实。' for i in range(n))
        return [dict(role='user',content='阅读以下维护记录，用中文概括共同规则和应注意的边界。\n'+body)]
    if kind=='chat':
        history=[]
        for turn in range(4):
            body='\n'.join(f'Record {turn}-{i}: the service owns its layers and caches; synchronize the producer before consumption; never assume peer access.' for i in range(n))
            history += [dict(role='user',content=f'Archive part {turn}.\n'+body),
                        dict(role='assistant',content='Acknowledged. I will preserve ownership, synchronization and recorded evidence.')]
        history.append(dict(role='user',content='Summarize the archived rules in one precise paragraph.'))
        return history
    raise ValueError('Unknown workload kind')


def fixture(tokenizer, template, kind, target):
    def encode(n, pad=0):
        history=messages(kind,n)
        history[-1]['content']+='\n'+' x'*pad
        rendered=template.render(history,enable_thinking=False)
        ids=tokenizer.encode(rendered,parse_special=True)
        return dict(messages=history,rendered=rendered,ids=ids)
    low,high=0,1
    while len(encode(high)['ids'])<=target:low,high=high,high*2
    while high-low>1:
        mid=(low+high)//2
        if len(encode(mid)['ids'])<=target:low=mid
        else:high=mid
    best=encode(low); gap=target-len(best['ids'])
    # Pad only the final user's content. Search actual token counts, never assume one token per character.
    p_low,p_high=0,max(1,gap+16)
    while len(encode(low,p_high)['ids'])<=target:
        p_low,p_high=p_high,p_high*2
        if p_high>target*4:raise ValueError('Tokenizer padding cannot reach budget')
    while p_high-p_low>1:
        mid=(p_low+p_high)//2
        candidate=encode(low,mid)
        if len(candidate['ids'])<=target:p_low=mid;best=candidate
        else:p_high=mid
    candidate=encode(low,p_low)
    if len(candidate['ids'])>len(best['ids']):best=candidate
    if not target-16<=len(best['ids'])<=target:raise ValueError('Tokenizer budget padding did not converge')
    best.update(kind=kind,target=target,tokens=len(best['ids']),input_ids_sha256=hashlib.sha256(json.dumps(best['ids'],separators=(',',':')).encode()).hexdigest())
    return best


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--engine',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--source-ref',required=True,help='Verified compiled application source commit, not runner revision')
    parser.add_argument('--configs',nargs='+',required=True)
    parser.add_argument('--targets',default='65536,131072,196608')
    parser.add_argument('--kinds',default='code,chinese,chat')
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--arms',default='2048:0:24,3072:1:24,4096:1:24',help='chunk:mtp:first-stage-end[:mmq_j]; QSA always off')
    parser.add_argument('--profile-experts',action='store_true',help='Per-layer timers and exact aggregate routing histograms; no extra GPU synchronization')
    parser.add_argument('--trace-mmq',action='store_true',help='Record grouped product geometry/selected tiles; separate diagnostic run')
    parser.add_argument('--max-new',type=int,default=64)
    parser.add_argument('--request-timeout',type=int,default=1200)
    parser.add_argument('--identity-cache',type=Path)
    args=parser.parse_args()
    root=args.root.resolve();exe=args.engine.resolve();out=args.output_dir
    if out.exists() or not exe.is_file() or exe==root/'build-hip/strata':raise ValueError('New output/isolated candidate required')
    if args.repeats<1 or args.max_new<1 or args.request_timeout<1:raise ValueError('Positive bounds required')
    if not re.fullmatch('[0-9a-f]{40}',args.source_ref):raise ValueError('Full verified application commit required')
    targets=[int(x) for x in args.targets.split(',')];kinds=args.kinds.split(',')
    arms=parse_arms(args.arms)
    if any(n<1024 or n>196608 for n in targets):raise ValueError('Conservative context bounds required')
    peak_new_bytes=64*sum(targets)*len(kinds)+8*2**20*len(arms)*args.repeats*len(args.configs)
    if shutil.disk_usage(out.parent).free<FLOOR+peak_new_bytes:raise ValueError('Peak evidence plus 4-GiB reserve insufficient')
    out.mkdir(mode=0o700,parents=True);resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    from serve.server import engine_args,child_env
    from serve.frontend import ChatTemplate
    from tools.strata_tokenizer import Tokenizer
    def interrupted(sig,frame):raise InterruptedError('Own bounded validation interrupted')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    with (root/'logs/model-launch.lock').open('a+b') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        preflight(root,out/'preflight.raw')
        deployed=[root/p for p in ['build-hip/strata','run-iq2-xs.json','run-iq3-s.json','run-iq2-xs.sh','run-iq3-s.sh','tools/gfx906_launch.sh']]
        before={str(p):digest(p) for p in deployed};store(out/'deployment-before.json',before)
        store(out/'candidate.json',dict(engine=str(exe),sha256=digest(exe),application_source_ref=args.source_ref,estimated_peak_evidence_bytes=peak_new_bytes,settings=vars(args)|dict(root=str(root),engine=str(exe),output_dir=str(out),identity_cache=str(args.identity_cache))))
        cache=json.loads(args.identity_cache.read_text()) if args.identity_cache else None
        references={};fixtures=None
        try:
            for config_path in args.configs:
                config=json.loads(Path(config_path).read_bytes())
                if config.get('backend')!='hip' or config.get('experimental_gfx906') is not True or config.get('gpu')!=[0,1]:raise ValueError('Explicit gfx906 HIP two-card config required')
                native=Path(config['args'][config['args'].index('--native')+1])
                model=next((m for m in MODEL_FILES if f'-{m}-' in native.name),None)
                if not model:raise ValueError('Only pinned acceptance/model identities supported')
                identity=verify_model(config,model,out/(model+'-identity.json'),cache)
                tk=Tokenizer.from_gguf(native);pack=Path(config['args'][config['args'].index('--pack')+1]);tpl=ChatTemplate(pack/'tokenizer/chat_template.jinja')
                if fixtures is None:
                    fixtures=[];(out/'fixtures').mkdir(mode=0o700)
                    for target in targets:
                        for kind in kinds:
                            f=fixture(tk,tpl,kind,target);fixtures.append(f);store(out/'fixtures'/f'{kind}-{target}.json',f)
                else:
                    for f in fixtures:
                        if tk.encode(tpl.render(f['messages'],enable_thinking=False),parse_special=True)!=f['ids']:raise ValueError('Models/templates do not tokenize the same fixtures')
                for repeat in range(args.repeats):
                    # Rotate order across repeats; retain the control reference even if it runs later.
                    for chunk,mtp,split,mmq_j in arms[repeat%len(arms):]+arms[:repeat%len(arms)]:
                        tile_value='auto' if mmq_j == -1 else str(mmq_j)
                        label=f'{model}-r{repeat}-c{chunk}-m{mtp}-s{split}-j{tile_value}';arm=out/label;arm.mkdir(mode=0o700)
                        preflight(root,arm/'preflight.raw')
                        for file in identity['files']:
                            if fingerprint(file['path'])!=file['fingerprint']:raise ValueError('Verified model changed before arm')
                        cfg=copy.deepcopy(config);cfg['exe']=str(exe);cfg['layer_split']=str(split)
                        for flag,value in [('--prefill',chunk),('--prompt-cache',0),('--short-read',0)]:cfg['args']=replace_arg(cfg['args'],flag,value)
                        cfg.setdefault('env',{}).update(STRATA_PREFILL_TIMING='1',STRATA_GFX906_PREFILL_ATTN='0',STRATA_GFX906_MTP_BATCH=str(mtp),STRATA_MTP_BATCH='1',STRATA_GFX906_MMQ_J=tile_value,
                            STRATA_PREFILL_EXPERT_PROFILE='1' if args.profile_experts else '0',STRATA_GFX906_MMQ_TRACE='1' if args.trace_mmq else '0')
                        if 'HSA_OVERRIDE_GFX_VERSION' in cfg['env']:raise ValueError('GPU impersonation forbidden')
                        env=child_env(cfg);env.pop('HSA_OVERRIDE_GFX_VERSION',None);env.pop('STRATA_API_KEY',None)
                        store(arm/'config.private.json',cfg);pipe=None;cases=[]
                        try:
                            pipe=Pipe(exe,engine_args(cfg),root,env,arm,monitor=True)
                            store(arm/'startup.json',dict(info=pipe.info,load_s=pipe.load_s))
                            for index,f in enumerate(fixtures):
                                result=pipe.generate(f['ids'],args.max_new,args.request_timeout)
                                if any(t<0 or t>=len(tk.tokens) for t in result['ids']):raise ValueError('Invalid generated token ID')
                                result.update(kind=f['kind'],target=f['target'],input_ids_sha256=f['input_ids_sha256'],text=tk.decode(result['ids']))
                                cases.append(result);store(arm/f'case-{index}.json',result)
                                print(json.dumps(dict(arm=label,case=index,tokens=f['tokens'],prompt_ms=result['engine']['prompt_ms'],ttft_s=result['ttft_s']),ensure_ascii=False),flush=True)
                        finally:
                            rc=pipe.close() if pipe else None
                            store(arm/'exit.json',dict(exit_code=rc))
                        if rc!=0:raise RuntimeError('Own candidate did not exit cleanly')
                        preflight(root,arm/'cleanup.raw',drain=True)
                        profiles=summarize(read_records((arm/'engine.stderr.raw').read_text(errors='replace')))
                        store(arm/'profiles.json',profiles)
                        if len(profiles['request_profiles'])!=len(fixtures) or not all(p['request']['cold'] for p in profiles['request_profiles']):
                            raise ValueError('Missing/ambiguous cold request profiles')
                        if not profiles['stages'] or {s['device'] for s in profiles['stages']}!={0,1}:
                            raise ValueError('Both real stage timelines required')
                        if mtp and not any(d['batched'] for d in profiles['draft']):raise ValueError('Requested MTP batch path never exercised')
                        if not mtp and any(d['batched'] for d in profiles['draft']):raise ValueError('Control unexpectedly batched MTP')
                        for file in identity['files']:
                            if fingerprint(file['path'])!=file['fingerprint']:raise ValueError('Verified model changed during arm')
                        references[label]=cases
            comparisons=[]
            for label,cases in references.items():
                model=label.split('-r')[0];controls=[cs for name,cs in references.items() if name.startswith(model+'-') and '-c2048-m0-s24-j0' in name]
                comparisons.append(dict(arm=label,generated_ids_equal_controls=bool(controls) and all(all(a['ids']==b['ids'] for a,b in zip(cases,cs)) for cs in controls),
                                        control_repetitions=len(controls)))
            store(out/'comparison.json',comparisons)
        except BaseException as error:
            store(out/'failure.json',dict(error=repr(error)));raise
        finally:
            after={str(p):digest(p) for p in deployed};store(out/'deployment-after.json',after)
            candidate=json.loads((out/'candidate.json').read_bytes());final_engine_sha=digest(exe)
            store(out/'candidate-after.json',dict(sha256=final_engine_sha))
            if before!=after:raise RuntimeError('Owner deployment changed during validation; review, never overwrite')
            if candidate['sha256']!=final_engine_sha:raise RuntimeError('Candidate binary changed during validation; refuse mixed source comparison')


if __name__=='__main__':
    main()
