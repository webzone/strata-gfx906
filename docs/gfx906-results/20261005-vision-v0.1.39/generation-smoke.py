"""Bounded pipe-only IQ2_XS image smoke; no API listeners or saved-config changes."""
import hashlib, json, pathlib, shutil, sys, threading, time
root = pathlib.Path('/home/chris/dev/strata-gfx906')
work = pathlib.Path('/data/strata-gfx906/vision-gfx906-20261005')
sys.path.insert(0, str(root))
from serve.server import Vision, StrataEngine, Service, child_env, engine_args
from serve.frontend import ChatTemplate, openai_to_messages
from tools.strata_tokenizer import Tokenizer
model = root / 'models/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf'
ple = model.with_name(model.name.replace('00001', '00002'))
pack = root / 'packs/iq2_xs'
vcfg = dict(exe=str(work/'build/bin/strata-vision'), mmproj=str(work/'mmproj-Qwen3.8-Flash-Next-BF16.gguf'),
            model=str(model), gpu=True, max_tokens=300, threads=6)
cfg = dict(exe='/data/strata-gfx906/eval-v0139/build-fork/strata', backend='hip', gpu=[0,1], layer_split='24',
           env=dict(GPU_MAX_HW_QUEUES='8', HSA_ENABLE_SDMA='0', HSA_FORCE_FINE_GRAIN_PCIE='1'),
           args=['--pack',str(pack),'--native',str(model),'--ple-gguf',str(ple),
                 '--expert-profile',str(root/'data/expert-profile.bin'),'--expert-cache','auto',
                 '--max-context','4096','--kv','int8','--pool-workers','6','--prefill','128',
                 '--spec','2','--mtp',str(root/'mtp/rt'),'--suffix-draft','0','--prompt-cache','0','--adapt-every','0','--pcie-frac','0',
                 '--vram-reserve-mib','1024','--vision'])
report = dict(status='starting', text_config=cfg, vision_config=vcfg, cases=[],
              limits='One real image and short generation at 4K; not logits parity, throughput or long-context acceptance.')
report_path = work/'generation.json'
def save():
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
save()
engine = vision = None
try:
    log = open(work/'generation-vision.stderr.log','w')
    vision = Vision(vcfg, log=log, env=child_env(cfg))
    started = time.monotonic()
    engine = StrataEngine(cfg['exe'], engine_args(cfg), cwd=str(root),
                          log=str(work/'generation-engine.stderr.log'), env=child_env(cfg))
    report.update(load_seconds=time.monotonic()-started, info=engine.info)
    tk = Tokenizer.from_gguf(model)
    tpl = ChatTemplate(pack/'tokenizer/chat_template.jinja')
    svc = Service(engine,tk,tpl,vision=vision)
    photo = work/'reference-newspaper.jpeg'
    shutil.copy2(root/'third_party/llama.cpp/tools/mtmd/test-1.jpeg', photo)
    messages=[dict(role='user',content=[dict(type='image_url',image_url=dict(url=str(photo))),
              dict(type='text',text='Transcribe the largest all-caps headline below the newspaper name. Output only that headline.')])]
    request = dict(messages=messages, chat_template_kwargs=dict(enable_thinking=False), max_tokens=32)
    report['request'] = request
    messages, tools, kwargs = openai_to_messages(request)
    ids, thinking, maximum = svc.prepare(messages,tools,kwargs,max_new=32)
    embedding = pathlib.Path(svc.embeddings.path)
    shutil.copy2(embedding,work/'generation-image.sve')
    streams = [('hip',embedding),('cpu-fp32',work/'accepted-300-gpu0/cpu-300-gpu0-photo-2.sve'),('upstream-cpu-bf16',work/'compare-300-gpu0/cpu-300-gpu0-photo-2.sve')]
    for mode, embeddings in streams:
        output=[t for t in engine.generate(ids,maximum,dict(temperature=0),threading.Event(),embeddings=str(embeddings)) if t is not None]
        text=tk.decode(output)
        ok='MEN WALK ON MOON' in text.upper()
        report['cases'].append(dict(embedding_sha256=hashlib.sha256(embeddings.read_bytes()).hexdigest(),encoder=mode,messages=messages,prompt_ids=ids,ids=output,text=text,
                                    expected='MEN WALK ON MOON',pass_headline=ok,engine=dict(engine.last)))
        save()
    report['token_ids_equal'] = len({tuple(c['ids']) for c in report['cases']}) == 1
    report['status']='passed' if all(c['pass_headline'] for c in report['cases']) else 'failed_headline'
finally:
    if engine is not None:
        engine_proc = engine.proc
        engine.unload()
        report['engine_exit']=engine_proc.returncode
        if hasattr(engine.log,'close'): engine.log.close()
    if vision is not None:
        vision.close()
        report['vision_exit']=vision.proc.poll()
        log.close()
    save()
print(json.dumps(report,ensure_ascii=False,indent=2))
if report['status']!='passed' or report.get('engine_exit')!=0 or report.get('vision_exit')!=0:
    raise SystemExit(1)
