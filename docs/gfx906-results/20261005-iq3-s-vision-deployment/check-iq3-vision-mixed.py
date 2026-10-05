import concurrent.futures,json,pathlib,threading,time,urllib.request,base64,hashlib
root=pathlib.Path('/home/chris/dev/strata-gfx906')
out=root/'logs/deploy-iq3-s-vision-20261005/mixed-check'; out.mkdir(exist_ok=False)
model='Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S'
def req(path,body=None):
    q=urllib.request.Request('http://127.0.0.1:8082'+path,data=json.dumps(body).encode() if body else None,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(q,timeout=240) as r: return r.status,json.loads(r.read())
def call(name,body):
    (out/(name+'.request.json')).write_text(json.dumps(body,indent=2)+'\n')
    start=time.monotonic(); status,result=req('/v1/chat/completions',body)
    (out/(name+'.response.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return dict(name=name,status=status,seconds=time.monotonic()-start,nonempty=bool(result['choices'][0]['message']['content']))
barrier=threading.Barrier(5)
def text(i):
    body=dict(model=model,messages=[dict(role='user',content=f'Concurrent request {i}: list integers from 1 to 150, separated by commas, without explanation.')],max_tokens=192,temperature=0,chat_template_kwargs=dict(enable_thinking=False))
    barrier.wait(10); return call(f'text-{i}',body)
photo=pathlib.Path('/data/strata-gfx906/vision-gfx906-20261005/accepted-1024-limit-gpu0/limit.bmp')
b64=base64.b64encode(photo.read_bytes()).decode()
image=dict(model=model,messages=[dict(role='user',content=[dict(type='image_url',image_url=dict(url='data:image/bmp;base64,'+b64)),dict(type='text',text='Describe the visible colors and geometric pattern in one brief sentence.')])],max_tokens=64,temperature=0,chat_template_kwargs=dict(enable_thinking=False))
samples=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
    futures=[pool.submit(text,i) for i in range(4)]; barrier.wait(10)
    deadline=time.monotonic()+60
    while True:
        _,m=req('/metrics'); samples.append(dict(time=time.time(),live=m['live']))
        if m['live'].get('running',0)==4: break
        assert time.monotonic()<deadline,'four live text requests not observed'
        time.sleep(.05)
    image_future=pool.submit(call,'image',image)
    futures.append(image_future)
    while any(not f.done() for f in futures):
        _,m=req('/metrics'); samples.append(dict(time=time.time(),live=m['live']))
        time.sleep(.1)
    rows=[f.result() for f in futures]
assert all(x['status']==200 and x['nonempty'] for x in rows),rows
(out/'live.json').write_text(json.dumps(samples,indent=2)+'\n')
summary=dict(pass_check=True,cases=rows,image_input_sha256=hashlib.sha256(photo.read_bytes()).hexdigest(),limits='One fresh synthetic 1024-token GPU image submitted while four text requests were active; liveness/status smoke, not semantic quality, fairness or long-soak acceptance.')
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2),flush=True)
