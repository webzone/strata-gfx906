import datetime, json, pathlib, time, urllib.request

out = pathlib.Path('/home/chris/dev/strata-gfx906/logs/restore-upstream-vision-20261005T053548Z/decode-check')
out.mkdir(exist_ok=False)
prompts = [
    'Explain the double-slit experiment, interference, and measurement in detail. Write at least 1500 words with concrete examples.',
    'Explain how a relational database implements transactions, indexes, and crash recovery. Write at least 1500 words with concrete examples.',
    'Explain Python asynchronous programming, cancellation, and backpressure. Write at least 1500 words with concrete examples.',
]
rows = []
for i, prompt in enumerate(prompts):
    body = {'model': 'Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S', 'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': 512, 'temperature': 0, 'chat_template_kwargs': {'enable_thinking': False}}
    (out / f'{i}.request.json').write_text(json.dumps(body, indent=2) + '\n')
    req = urllib.request.Request('http://127.0.0.1:8082/v1/chat/completions',
                                 data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=240) as response:
        assert response.status == 200
        result = json.load(response)
    elapsed = time.monotonic() - start
    assert result['choices'][0]['message']['content'] and result['usage']['completion_tokens'] >= 128
    (out / f'{i}.response.json').write_text(json.dumps(result, indent=2) + '\n')
    row = {'case': i, 'wall_seconds': elapsed, 'usage': result['usage'], 'timings': result['timings']}
    rows.append(row)
    print(json.dumps(row), flush=True)
tokens = sum(x['timings']['predicted_n'] for x in rows)
ms = sum(x['timings']['predicted_ms'] for x in rows)
summary = {'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'cases': rows,
           'weighted_decode_tok_s': 1000 * tokens / ms,
           'limits': 'Three sequential short-context, greedy, thinking-disabled, MTP-enabled real generations. GPU vision stays loaded and four batch slots configured. Not a matched before/after, long-context agentic, concurrency throughput or long-soak benchmark.'}
(out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2), flush=True)
