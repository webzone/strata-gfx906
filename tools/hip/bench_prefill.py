#!/usr/bin/env python3
"""Matched fresh/follow-up prefill probe against an otherwise idle local Strata server.

Does not start/stop services. Use a dedicated server and its engine log, restart
between build/configuration arms, and keep model/template/settings identical.
"""
import argparse
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

PATTERN = re.compile(r'prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([0-9.]+) tok/s\), (\d+) generated in (\d+) ms \(([0-9.]+) tok/s\)')
FIELDS = ('prompt_tokens', 'reused', 'fresh', 'prefill_ms', 'prefill_tps', 'generated', 'decode_ms', 'decode_tps')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:8080')
    p.add_argument('--model', required=True)
    p.add_argument('--engine-log', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--rows', default='140,280,140,280', help='Comma-separated source row counts, NOT token counts')
    p.add_argument('--fresh-only', action='store_true', help='Omit follow-up reuse requests for cold-prefill comparisons')
    p.add_argument('--max-tokens', type=int, default=128)
    p.add_argument('--timeout', type=float, default=360)
    args = p.parse_args()
    rows = [int(n) for n in args.rows.split(',')]
    if not rows or any(n <= 0 for n in rows) or args.max_tokens <= 0 or args.timeout <= 0:
        p.error('Rows, max-tokens and timeout must be positive')
    if args.output.exists():
        p.error('Refusing to overwrite existing benchmark results')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output.with_suffix(args.output.suffix + '.raw')
    raw_dir.mkdir(mode=0o700) # Fail before any API request if evidence already exists.
    results = []

    def request(messages, kind, trial):
        offset = args.engine_log.stat().st_size
        body = dict(model=args.model, messages=messages, max_tokens=args.max_tokens,
                    temperature=0, top_k=1, top_p=1, min_p=0, seed=42,
                    reasoning_effort='none')
        headers = {'Content-Type': 'application/json'}
        if os.environ.get('STRATA_API_KEY'):
            headers['Authorization'] = 'Bearer ' + os.environ['STRATA_API_KEY']
        stem = raw_dir / f'{trial:03d}-{kind}'
        encoded = json.dumps(body).encode()
        stem.with_suffix('.request.json').write_bytes(encoded) # Exact submitted bytes; no auth header.
        req = urllib.request.Request(args.url.rstrip('/') + '/v1/chat/completions', data=encoded, headers=headers)
        start = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as response:
                reply_bytes = response.read()
        except urllib.error.HTTPError as error:
            stem.with_suffix('.response.json').write_bytes(error.read())
            raise
        elapsed = time.monotonic() - start
        stem.with_suffix('.response.json').write_bytes(reply_bytes)
        data = json.loads(reply_bytes)
        # The API response may arrive just before stderr's completion line is flushed.
        deadline = time.monotonic() + 2
        while True:
            with args.engine_log.open('rb') as log:
                log.seek(offset)
                raw_log = log.read()
                matches = PATTERN.findall(raw_log.decode(errors='replace'))
            if matches or time.monotonic() >= deadline:
                break
            time.sleep(0.02)
        stem.with_suffix('.engine.log').write_bytes(raw_log)
        if len(matches) != 1:
            raise RuntimeError('Expected exactly one completed engine timing line; check idle server and log path')
        metrics = dict(zip(FIELDS, map(float, matches[0])))
        if kind == 'fresh' and metrics['reused'] != 0:
            raise RuntimeError('Fresh prompt reused cached tokens; restart this arm before comparison')
        choice = data['choices'][0]
        results.append(dict(label=args.label, kind=kind, trial=trial, wall_s=elapsed,
                            metrics=metrics, usage=data.get('usage'),
                            finish_reason=choice.get('finish_reason'), message=choice['message']))
        args.output.write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps(results[-1]), flush=True)
        return choice['message']

    request([{'role': 'user', 'content': 'Reply READY.'}], 'warmup', 0)
    for trial, n in enumerate(rows, 1):
        code = '\n'.join(f'export function rule{i}(x) {{ return x === {i} ? x + {i+1} : x - {i}; }}' for i in range(n))
        messages = [{'role': 'user', 'content': f'CASE {trial}: Review this source and describe its behavior precisely in a paragraph.\n' + code}]
        reply = request(messages, 'fresh', trial)
        if not args.fresh_only:
            messages += [reply, {'role': 'user', 'content': 'Explain the most relevant boundary case and the smallest useful regression test. ' + 'Focus on integer equality, zero, negative input, unexpected types, and caller assumptions. ' * 5}]
            request(messages, 'followup', trial)


if __name__ == '__main__':
    main()
