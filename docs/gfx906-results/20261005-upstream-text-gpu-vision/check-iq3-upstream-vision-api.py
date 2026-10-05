import base64,concurrent.futures,datetime,hashlib,json,pathlib,threading,time,urllib.request
root=pathlib.Path('/home/chris/dev/strata-gfx906')
log=pathlib.Path('/home/chris/dev/strata-gfx906/logs/restore-upstream-vision-20261005T053548Z')
out=log/'api-checks'; out.mkdir(exist_ok=False)
base='http://127.0.0.1:8082'
model='Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S'
def request(path,body=None,timeout=240):
    data=json.dumps(body).encode() if body is not None else None
    req=urllib.request.Request(base+path,data=data,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as r: return r.status,json.loads(r.read())
def store(name,path,body=None):
    if body is not None: (out/(name+'.request.json')).write_text(json.dumps(body,indent=2)+'\n')
    start=time.monotonic(); status,result=request(path,body)
    (out/(name+'.response.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return dict(name=name,http_status=status,seconds=time.monotonic()-start,result=result)
report={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checks':[],'limits':'Short API and four-request overlap checks; not full-context, vision quality or long-soak acceptance.'}
for name,path in [('health','/health'),('models','/v1/models'),('status','/v1/status'),('metrics','/metrics')]:
    row=store(name,path); report['checks'].append({k:v for k,v in row.items() if k!='result'})
    if name=='health': assert row['result']['images'] is True
    if name=='models': assert row['result']['data'][0]['id']==model and 'image' in row['result']['data'][0]['architecture']['input_modalities']
    if name=='status': assert row['result']['concurrency']['serving']==4 and row['result']['vision']['enabled'] is True
    if name=='metrics': assert int(row['result']['engine']['batch_slots'])==4
body=dict(model=model,messages=[dict(role='user',content='Compute 17 + 25. Output only the integer.')],max_tokens=16,temperature=0,chat_template_kwargs=dict(enable_thinking=False))
row=store('text','/v1/chat/completions',body)
assert row['result']['choices'][0]['message']['content'].strip()=='42'
report['checks'].append({k:v for k,v in row.items() if k!='result'})
photo=root/'docs/gfx906-results/20261005-vision-v0.1.39/reference-newspaper.jpeg'
image='data:image/jpeg;base64,'+base64.b64encode(photo.read_bytes()).decode()
body=dict(model=model,messages=[dict(role='user',content=[dict(type='image_url',image_url=dict(url=image)),dict(type='text',text='Transcribe the largest all-caps headline below the newspaper name. Output only that headline.')])],max_tokens=32,temperature=0,chat_template_kwargs=dict(enable_thinking=False))
row=store('image','/v1/chat/completions',body)
answer=row['result']['choices'][0]['message']['content']; assert 'MEN WALK ON MOON' in answer.upper(),answer
report['checks'].append(dict(name='image',http_status=row['http_status'],seconds=row['seconds'],answer=answer,input_sha256=hashlib.sha256(photo.read_bytes()).hexdigest()))
# Four simultaneous text generations; sample actual admitted batch-slot states.
barrier=threading.Barrier(5)
def concurrent_call(i):
    b=dict(model=model,messages=[dict(role='user',content=f'Request {i}: count from 1 to 100, one number per line. Start immediately, no explanation.')],max_tokens=128,temperature=0,chat_template_kwargs=dict(enable_thinking=False))
    barrier.wait(10)
    return store(f'parallel-{i}','/v1/chat/completions',b)
samples=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    futures=[pool.submit(concurrent_call,i) for i in range(4)]
    barrier.wait(10)
    while any(not f.done() for f in futures):
        _,metrics=request('/metrics',timeout=15)
        samples.append({'time':time.time(),'live':metrics['live']})
        time.sleep(.1)
    rows=[f.result() for f in futures]
(out/'parallel-live.json').write_text(json.dumps(samples,indent=2)+'\n')
max_running=max((s['live'].get('running',0) for s in samples),default=0)
max_slots=max((sum(x['state']!='idle' for x in s['live'].get('slots',[])) for s in samples),default=0)
assert max_running==4 and max_slots==4,(max_running,max_slots)
assert all(x['http_status']==200 and x['result']['choices'][0]['message']['content'] for x in rows)
report['parallel']={'requested':4,'reported_slots':4,'maximum_live_requests':max_running,'maximum_occupied_slots':max_slots,'responses':[dict(name=x['name'],http_status=x['http_status'],seconds=x['seconds']) for x in rows]}
report['pass']=True
(out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
