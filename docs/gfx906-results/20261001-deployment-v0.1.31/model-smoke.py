#!/usr/bin/env python3
"""Bounded real-model smoke of the deployment environment, without opening a listener."""
import argparse
import copy
import json
from pathlib import Path
import sys
import threading
import time

ROOT = Path('/home/chris/dev/strata-gfx906')
sys.path.insert(0, str(ROOT))
from serve.server import StrataEngine, child_env, engine_args
from serve.frontend import ChatTemplate
from tools.strata_tokenizer import Tokenizer
from tools.gfx906_smoke import CASES

ap = argparse.ArgumentParser()
ap.add_argument('--gpus', required=True)
ap.add_argument('--out', type=Path, required=True)
a = ap.parse_args()
if a.out.exists():
    ap.error('refusing to overwrite acceptance evidence')
cfg = copy.deepcopy(json.loads((ROOT / 'run-iq2-xs.json').read_text()))
cfg['gpu'] = [int(i) for i in a.gpus.split(',')]
args = cfg['args']
for flag, value in [('--max-context', '4096'), ('--mtp-window', '4096'), ('--prefill', '128'), ('--prompt-cache', '0')]:
    if flag in args:
        args[args.index(flag) + 1] = value
    else:
        args += [flag, value]
env = child_env(cfg)
assert 'HSA_OVERRIDE_GFX_VERSION' not in env
report = {'status': 'starting', 'model': 'GSQ-RCO IQ2_XS', 'physical_gpus': cfg['gpu'],
          'args': engine_args(cfg), 'runtime_env': {k: env[k] for k in ('GPU_MAX_HW_QUEUES', 'HSA_ENABLE_SDMA', 'HSA_FORCE_FINE_GRAIN_PCIE', 'HIP_VISIBLE_DEVICES')},
          'limitations': ['4K context and short prompts only; not a CPU-logit oracle or 262K acceptance',
                          'requested runtime env only, not an isolated performance A/B'], 'cases': []}

def save():
    tmp = a.out.with_suffix('.tmp')
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(a.out)

save()
engine = None
try:
    shard = ROOT / 'models/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf'
    tk = Tokenizer.from_gguf(shard)
    tpl = ChatTemplate(ROOT / 'packs/iq2_xs/tokenizer/chat_template.jinja')
    started = time.monotonic()
    engine = StrataEngine(cfg['exe'], engine_args(cfg), cwd=str(ROOT), log=str(a.out.with_suffix('.engine.log')), env=env)
    report.update(status='ready', load_seconds=time.monotonic() - started, info=engine.info)
    save()
    for name, prompt, maximum in CASES:
        text = tpl.render([{'role': 'user', 'content': prompt}], enable_thinking=False)
        prompt_ids = tk.encode(text, parse_special=True)
        started = time.monotonic()
        generated = [t for t in engine.generate(prompt_ids, maximum, {'temperature': 0}, threading.Event()) if t is not None]
        if not generated or any(t < 0 or t >= len(tk.tokens) for t in generated):
            raise RuntimeError('empty or invalid token stream')
        case = {'name': name, 'prompt': prompt, 'prompt_ids': prompt_ids, 'max_new': maximum, 'ids': generated,
                'text': tk.decode(generated), 'seconds': time.monotonic() - started, 'engine': dict(engine.last)}
        report['cases'].append(case)
        save()
        print(json.dumps(case, ensure_ascii=False), flush=True)
    report['status'] = 'passed_transport_and_generation'
except BaseException as e:
    report.update(status='failed', error=type(e).__name__ + ': ' + str(e))
    raise
finally:
    if engine is not None:
        engine.unload()
        report['exit_code'] = engine.exit_code()
        if hasattr(engine.log, 'close'):
            engine.log.close()
    save()
if report.get('exit_code') != 0:
    raise SystemExit('engine did not exit cleanly')
