#!/usr/bin/env python3
"""Offline immutable-pipe-evidence comparison; never launches an engine or contacts a host."""
import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from .gfx906_prefill_validate import parse_done, parse_arms
from .gfx906_model import MODEL_FILES, REPO, REVISION

ARM = re.compile(r'^(IQ2_XS|IQ3_S)-r(\d+)-c(2048|3072|4096)-m([01])-s(23|24|25)-j(0|16|32|48|64|auto)$')


def compare_arms(references):
    parsed = {}
    for name, cases in references.items():
        match = ARM.fullmatch(name)
        if not match or not cases:
            raise ValueError('Recognized nonempty arms required')
        parsed[name] = match.groups()
        for case in cases:
            if not isinstance(case.get('ids'), list) or not case['ids'] or any(type(n) is not int or n < 0 for n in case['ids']):
                raise ValueError('Nonempty real generated token IDs required')
            if not isinstance(case.get('input_ids_sha256'), str) or not re.fullmatch('[0-9a-f]{64}', case['input_ids_sha256']):
                raise ValueError('Exact fixture identity required')
    def against(cases, controls):
        if not controls:
            return None
        for reference in controls:
            if len(cases) != len(reference) or any((a['kind'], a['target'], a['input_ids_sha256']) !=
                                                 (b['kind'], b['target'], b['input_ids_sha256']) for a, b in zip(cases, reference)):
                raise ValueError('Control fixtures missing, reordered or changed')
        return all(all(a['ids'] == b['ids'] for a, b in zip(cases, reference)) for reference in controls)
    rows = []
    for name, cases in references.items():
        model, repeat, chunk, mtp, split, tile = parsed[name]
        canonical = [refs for n, refs in references.items() if parsed[n][0] == model and parsed[n][2:] == ('2048','0','24','0')]
        paired = [refs for n, refs in references.items() if parsed[n][0] == model and parsed[n][2:] == (chunk,'0',split,'0')]
        default_split = [refs for n,refs in references.items() if parsed[n][0] == model and parsed[n][2:] == (chunk,mtp,'24',tile)]
        rows.append(dict(arm=name, canonical_control_repetitions=len(canonical), ids_equal_canonical=against(cases, canonical),
                         same_chunk_control_repetitions=len(paired), ids_equal_same_chunk=against(cases, paired),
                         default_split_control_repetitions=len(default_split), ids_equal_default_split=against(cases,default_split)))
    return rows


def validate_identity(model, identity):
    """Validate stored full-hash attestation against published pin; no weights/host access."""
    if (model not in MODEL_FILES or identity.get('model') != model or identity.get('repo') != REPO
        or identity.get('revision') != REVISION or identity.get('full_sha256') is not True):
        raise ValueError('Pinned full model identity required')
    entries = identity.get('files')
    if not isinstance(entries,list) or len(entries) != len(MODEL_FILES[model]):
        raise ValueError('Complete pinned model shards required')
    for entry,(name,size,sha) in zip(entries,MODEL_FILES[model]):
        fp=entry.get('fingerprint',{})
        if (not isinstance(entry.get('path'),str) or Path(entry['path']).name != name or entry.get('sha256') != sha
            or any(type(fp.get(k)) is not int or fp[k] < 0 for k in ['device','inode','bytes','mtime_ns']) or fp['bytes'] != size):
            raise ValueError('Published source hash/size/fingerprint mismatch')
    return identity


def report(directory):
    directory = Path(directory)
    if (directory/'failure.json').exists():
        raise ValueError('Failed validation cannot be audited as completed')
    candidate = json.loads((directory/'candidate.json').read_text())
    if (not re.fullmatch('[0-9a-f]{64}',candidate.get('sha256',''))
        or not re.fullmatch('[0-9a-f]{40}',candidate.get('application_source_ref',''))):
        raise ValueError('Exact application source/binary identity required')
    after = json.loads((directory/'candidate-after.json').read_text())
    if candidate['sha256'] != after['sha256']:
        raise ValueError('Application changed during validation')
    if json.loads((directory/'deployment-before.json').read_text()) != json.loads((directory/'deployment-after.json').read_text()):
        raise ValueError('Protected deployment changed')
    settings = candidate['settings']
    targets = [int(n) for n in settings['targets'].split(',')]; kinds = settings['kinds'].split(',')
    if (not targets or len(set(targets)) != len(targets) or any(n<1024 or n>196608 for n in targets)
        or not kinds or len(set(kinds)) != len(kinds) or any(k not in ['code','chinese','chat'] for k in kinds)
        or type(settings['repeats']) is not int or settings['repeats'] < 1):
        raise ValueError('Valid unique requested evidence geometry required')
    identities = {model:validate_identity(model,json.loads((directory/(model+'-identity.json')).read_text()))
                  for model in MODEL_FILES if (directory/(model+'-identity.json')).exists()}
    if len(identities) != len(settings['configs']) or not identities:
        raise ValueError('Requested model identity coverage incomplete')
    expected_arms = {f'{model}-r{repeat}-c{chunk}-m{mtp}-s{split}-j{"auto" if tile == -1 else tile}'
                     for model in identities for repeat in range(settings['repeats'])
                     for chunk,mtp,split,tile in parse_arms(settings['arms'])}
    fixtures = []
    for target in targets:
        for kind in kinds:
            fixture = json.loads((directory/'fixtures'/f'{kind}-{target}.json').read_text())
            hashed = hashlib.sha256(json.dumps(fixture['ids'],separators=(',',':')).encode()).hexdigest()
            if fixture['input_ids_sha256'] != hashed or fixture['tokens'] != len(fixture['ids']) or not target-16 <= len(fixture['ids']) <= target:
                raise ValueError('Fixture token identity/budget changed')
            fixtures.append(fixture)
    refs = {}; arms = []
    for arm in sorted(directory.iterdir()):
        if not arm.is_dir() or not ARM.fullmatch(arm.name):
            continue
        if json.loads((arm/'exit.json').read_text())['exit_code'] != 0:
            raise ValueError('Unclean own engine exit')
        files = sorted(arm.glob('case-*.json'), key=lambda f: int(f.stem.split('-')[-1]))
        if [int(f.stem.split('-')[-1]) for f in files] != list(range(len(files))):
            raise ValueError('Missing case index')
        cases = [json.loads(file.read_text()) for file in files]
        if len(cases) != len(fixtures) or any((case['kind'],case['target'],case['input_ids_sha256']) !=
                                            (fixture['kind'],fixture['target'],fixture['input_ids_sha256']) for case,fixture in zip(cases,fixtures)):
            raise ValueError('Missing or changed requested fixture coverage')
        submitted = [line.split(' ',4)[4] for line in (arm/'engine.stdin.raw').read_text().splitlines() if line.startswith('GEN ')]
        if len(submitted) != len(fixtures) or any([int(n) for n in wire.split(',')] != fixture['ids'] for wire,fixture in zip(submitted,fixtures)):
            raise ValueError('Raw submitted tokens differ from fixtures')
        responses = []; pending = []
        for line in (arm/'engine.stdout.raw').read_text().splitlines():
            if line.startswith('T '):pending.append(int(line.split()[1]))
            elif line.startswith('DONE '):responses.append(dict(ids=pending,engine=parse_done(line)));pending=[]
            elif line.startswith('ERR'):raise ValueError('Engine protocol error in raw evidence')
        if pending or len(responses) != len(cases) or any(case['ids'] != response['ids'] or case['engine'] != response['engine']
                                                       for case,response in zip(cases,responses)):
            raise ValueError('Stored results differ from complete raw protocol')
        for case,fixture in zip(cases,fixtures):
            metrics = case['engine']
            if (metrics['reused'] != 0 or metrics['prompt_read'] != metrics['prompt_tokens']
                or metrics['prompt_tokens'] != len(fixture['ids']) or metrics['generated'] != len(case['ids'])):
                raise ValueError('Not a complete cold generation')
            for key in ['prompt_ms','decode_ms']:
                if not math.isfinite(metrics[key]) or metrics[key] < 0:
                    raise ValueError('Invalid real engine timing')
        refs[arm.name] = cases
        profiles = json.loads((arm/'profiles.json').read_text())['request_profiles']
        if len(profiles) != len(cases) or any(len(p['stages']) != 2 or {stage['device'] for stage in p['stages']} != {0,1} for p in profiles):
            raise ValueError('Complete two-device request-scoped profiles required')
        for p,f in zip(profiles,fixtures):
            request=p['request']
            if (request['prompt_tokens'] != len(f['ids']) or request['reused'] != 0 or request['read_from'] != 0
                or request['cancelled'] is not False or request['prefill_rows'] != len(f['ids'])-1
                or any(stage['tokens'] != len(f['ids'])-1 for stage in p['stages'])):
                raise ValueError('Request profile does not attest the actual complete cold rows')
        memory=json.loads((arm/'memory-peaks.json').read_text())
        if (type(memory.get('samples')) is not int or memory['samples'] < 1 or memory.get('sample_interval_s') != 1.0
            or any(type(memory.get(k)) is not int or memory[k] < 0 for k in ['rss_bytes','minimum_available_ram_bytes'])
            or not isinstance(memory.get('vram_bytes'),dict) or len(memory['vram_bytes']) != 2
            or any(type(v) is not int or v < 0 for v in memory['vram_bytes'].values())):
            raise ValueError('Complete two-card one-second observed memory evidence required')
        arms.append(dict(arm=arm.name, cases=cases, request_profiles=profiles,
                         memory=memory,
                         mean_prefill_ms=statistics.mean(c['engine']['prompt_ms'] for c in cases),
                         mean_ttft_s=statistics.mean(c['ttft_s'] for c in cases),
                         raw_sha256={name:hashlib.sha256((arm/name).read_bytes()).hexdigest() for name in
                                     ['engine.stdin.raw','engine.stdout.raw','engine.stderr.raw','memory.samples.jsonl']}))
    if not refs:
        raise ValueError('No completed arms')
    if set(refs) != expected_arms:
        raise ValueError('Requested exact model/arm/repetition coverage incomplete')
    return dict(schema=1, compiled_application=candidate, model_identities=identities, deployment_unchanged=True,
                comparisons=compare_arms(refs), arms=arms,
                limits='Measured submitted fixtures/repetitions only; sampled memory peaks; request/device timelines overlap; no model quality or performance claim from operator timings.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args();data = report(args.directory)
    with args.output.open('x',encoding='utf-8') as file:
        import os
        os.chmod(args.output,0o600)
        json.dump(data,file,ensure_ascii=False,indent=2,allow_nan=False);file.write('\n')
    print(json.dumps(data['comparisons'],indent=2))


if __name__ == '__main__':
    main()
