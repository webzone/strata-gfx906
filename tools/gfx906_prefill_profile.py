#!/usr/bin/env python3
"""Offline gfx906 prefill experiment plans and stage-aware log summaries.

Never executes an engine, contacts an API, or starts/stops a service. Plans clone a
saved config into private NEW files; the source config and deployment stay intact.
Stage wall times and host/GPU subintervals overlap: no cross-device wall-time sum.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil

PREFIXES = {
    "strata prefill profile: ": "stages",
    "strata prefill draft: ": "draft",
    "strata prefill request: ": "requests",
    "strata prefill layer: ": "layers",
    "strata prefill mmq: ": "products",
}


def _constant(value):
    raise ValueError("Nonfinite JSON number: " + value)


def read_records(text):
    records = {name: [] for name in PREFIXES.values()}
    records["segments"] = []
    pending = dict(stages=[], draft=[], layers=[], products=[])
    for number, line in enumerate(text.splitlines(), 1):
        for prefix, kind in PREFIXES.items():
            if not line.startswith(prefix):
                continue
            try:
                record = json.loads(line[len(prefix):], parse_constant=_constant)
                if not isinstance(record, dict) or type(record.get("schema")) is not int or record["schema"] != 1:
                    raise ValueError("Unsupported profile schema")
                for key, value in record.items():
                    if key.startswith("ms_") and (type(value) not in (float, int) or not math.isfinite(value) or value < 0):
                        raise ValueError("Invalid duration " + key)
                if kind == "stages":
                    for key in ("device", "layer_begin", "layer_end", "pos0", "tokens", "chunk"):
                        if type(record.get(key)) is not int or record[key] < 0:
                            raise ValueError("Invalid stage coordinate " + key)
                    if record["layer_end"] < record["layer_begin"] or record["chunk"] == 0:
                        raise ValueError("Invalid stage range/chunk")
                    phases = record.get("phase_ms")
                    if not isinstance(phases, dict) or "ms_gpu_timeline" not in record or "ms_wall" not in record:
                        raise ValueError("Missing stage timers")
                    for key, value in phases.items():
                        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                            raise ValueError("Invalid phase " + key)
                elif kind in ("draft", "requests") and "ms_wall" not in record:
                    raise ValueError("Missing wall timer")
                if kind in ("layers", "products"):
                    coordinates = ("device", "layer", "pos0")
                    counters = ("tokens", "chunk", "row_calls", "routed_rows", "max_rows", "gu_type", "down_type") if kind == "layers" else (
                        "groups", "total_rows", "group_rows", "max_rows", "weight_rows", "weight_cols", "requested_j", "selected_j")
                    for key in coordinates + counters:
                        if type(record.get(key)) is not int or record[key] < 0:
                            raise ValueError("Invalid expert coordinate/counter " + key)
                    if kind == "layers":
                        phases = record.get("phase_ms")
                        if not isinstance(phases, dict) or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in phases.values()):
                            raise ValueError("Invalid layer phase timer")
                        hist = record.get("row_histogram")
                        if not isinstance(hist,list) or len(hist)!=16 or any(type(v) is not int or v<0 for v in hist):
                            raise ValueError("Invalid expert row histogram")
                        if record.get("row_bin_upper") != [0,1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,None]:
                            raise ValueError("Unknown expert histogram binning")
                    elif type(record.get("forced")) is not bool or not isinstance(record.get("type"),str) or record["group_rows"]>record["total_rows"]:
                        raise ValueError("Invalid MMQ product metadata")
                if kind == "requests":
                    for key in ("prompt_tokens", "reused", "read_from", "prefill_rows", "refilled_slots"):
                        if type(record.get(key)) is not int or record[key] < 0:
                            raise ValueError("Invalid request coordinate " + key)
                    if type(record.get("cancelled")) is not bool:
                        raise ValueError("Invalid cancellation marker")
                records[kind].append(record)
                if kind in pending:
                    pending[kind].append(record)
                elif kind == "requests":
                    # Requests are serialized; the envelope is emitted only after all stage futures finish.
                    records["segments"].append(dict(request=record, **pending))
                    pending = dict(stages=[], draft=[], layers=[], products=[])
            except (TypeError, ValueError, KeyError) as error:
                raise ValueError(f"Profile line {number}: {error}") from error
    return records


def summarize(records):
    groups = {}
    for record in records["stages"]:
        key = tuple(record[name] for name in ("device", "layer_begin", "layer_end", "chunk"))
        if key not in groups:
            groups[key] = dict(device=key[0], layer_begin=key[1], layer_end=key[2], chunk=key[3],
                               calls=0, tokens=0, phase_ms={}, ms_gpu_timeline=0, ms_stage_wall_sum=0,
                               host_ms={}, counters={}, fallback_layers=set())
        group = groups[key]
        group["calls"] += 1
        group["tokens"] += record["tokens"]
        group["ms_gpu_timeline"] += record["ms_gpu_timeline"]
        group["ms_stage_wall_sum"] += record["ms_wall"]
        for phase, ms in record["phase_ms"].items():
            group["phase_ms"][phase] = group["phase_ms"].get(phase, 0) + ms
        for name, value in record.items():
            if name.startswith("ms_") and name not in ("ms_gpu_timeline", "ms_wall"):
                group["host_ms"][name] = group["host_ms"].get(name, 0) + value
            elif name.startswith("experts_") or name.endswith("_layer_chunks"):
                if type(value) is not int or value < 0:
                    raise ValueError("Invalid counter " + name)
                group["counters"][name] = group["counters"].get(name, 0) + value
        group["fallback_layers"].update(record.get("fallback_layers", []))
    for group in groups.values():
        group["fallback_layers"] = sorted(group["fallback_layers"])
        total = group["ms_gpu_timeline"]
        group["phase_percent_of_device_timeline"] = {
            phase: 100 * ms / total if total else 0 for phase, ms in group["phase_ms"].items()
        }
    layer_groups = {}
    for record in records.get("layers", []):
        key = tuple(record[k] for k in ("device", "layer", "chunk", "gu_type", "down_type"))
        group = layer_groups.setdefault(key, dict(zip(("device", "layer", "chunk", "gu_type", "down_type"), key),
            calls=0, row_calls=0, routed_rows=0, max_rows=0, phase_ms={}, row_histogram=[0]*16, row_bin_upper=record["row_bin_upper"]))
        group["calls"]+=1;group["row_calls"]+=record["row_calls"];group["routed_rows"]+=record["routed_rows"]
        group["max_rows"]=max(group["max_rows"],record["max_rows"])
        group["row_histogram"]=[a+b for a,b in zip(group["row_histogram"],record["row_histogram"])]
        for phase,value in record["phase_ms"].items():group["phase_ms"][phase]=group["phase_ms"].get(phase,0)+value
    products = {}
    for record in records.get("products", []):
        key = tuple(record[k] for k in ("device", "layer", "type", "weight_rows", "weight_cols", "selected_j", "forced"))
        group = products.setdefault(key,dict(zip(("device", "layer", "type", "weight_rows", "weight_cols", "selected_j", "forced"),key),
            calls=0, group_rows=0, groups=0, max_rows=0))
        for name in ("group_rows", "groups"):group[name]+=record[name]
        group["calls"]+=1;group["max_rows"]=max(group["max_rows"],record["max_rows"])
    requests = [dict(record, cold=(record["reused"] == 0 and record["read_from"] == 0 and not record["cancelled"]))
                for record in records["requests"]]
    request_profiles = []
    for segment in records.get("segments", []):
        profile = summarize(dict(stages=segment["stages"], requests=[segment["request"]], draft=segment["draft"],
                                 layers=segment.get("layers",[]), products=segment.get("products",[])))
        profile["request"] = profile.pop("requests")[0]
        profile.pop("request_profiles")
        request_profiles.append(profile)
    return dict(schema=1, stages=[groups[key] for key in sorted(groups)], requests=requests, draft=records["draft"],
                request_profiles=request_profiles, layers=[layer_groups[k] for k in sorted(layer_groups)],
                products=[products[k] for k in sorted(products)],
                warning="Stage wall times and host/GPU subintervals overlap. Do not add them across devices. "
                        "Request timers precede the first verify window: not TTFT or completed-generation throughput.")


def replace_arg(args, flag, value):
    result = []
    i = 0
    while i < len(args):
        if args[i] == flag:
            if i + 1 == len(args) or args[i + 1].startswith("--"):
                raise ValueError("Missing argument for " + flag)
            i += 2
        elif args[i].startswith(flag + "="):
            i += 1
        else:
            result.append(args[i])
            i += 1
    return result + [flag, str(value)]


def make_plan(config_path, engine, output, chunks, source_ref, test_port=8096):
    config_path, engine, output = map(Path, (config_path, engine, output))
    raw = config_path.read_bytes()
    base = json.loads(raw)
    if not isinstance(base, dict) or not isinstance(base.get("args"), list) or not all(isinstance(x, str) for x in base["args"]):
        raise ValueError("Saved config must have a string-list args field")
    if base.get("backend") != "hip" or base.get("experimental_gfx906") is not True:
        raise ValueError("Plan requires an explicitly opted-in gfx906 HIP config")
    if not isinstance(base.get("env", {}), dict):
        raise ValueError("Saved config env must be an object")
    if not chunks or any(type(x) is not int or x not in (1024, 2048, 3072, 4096) for x in chunks) or len(set(chunks)) != len(chunks):
        raise ValueError("Unique conservative chunks required: 1024, 2048, 3072, 4096; 8192 deliberately deferred")
    if type(test_port) is not int or not 1024 <= test_port <= 65535 or test_port == 8082:
        raise ValueError("Use an unprivileged dedicated test port, not the owner's deployment port 8082")
    if not engine.is_file():
        raise ValueError("Candidate engine must already exist; the planner never builds one")
    # All expensive preparation is outside this tool. Still retain the project's 4-GiB reserve.
    existing = output.parent
    while not existing.exists():
        existing = existing.parent
    if shutil.disk_usage(existing).free < 4 * 2**30:
        raise ValueError("Less than 4 GiB free; no experiment files created")
    if output.exists():
        raise ValueError("Refusing to overwrite an existing experiment directory")
    digest = hashlib.sha256()
    with engine.open("rb") as binary:
        for block in iter(lambda: binary.read(1024 * 1024), b""):
            digest.update(block)
    output.mkdir(parents=True, mode=0o700)
    def private(name, content):
        path = output / name
        with path.open("xb") as target:
            target.write(content)
        path.chmod(0o600)
    private("source-config.json", raw)
    arms = []
    for chunk in chunks:
        for label, attention, mtp in (("control", "0", "0"), ("attention", "1", "0"), ("mtp", "0", "1"), ("both", "1", "1")):
            config = copy.deepcopy(base)
            config["exe"] = str(engine.resolve())
            config["host"], config["port"] = "127.0.0.1", test_port # Private test configs, never the deployed listener.
            config["args"] = replace_arg(replace_arg(config["args"], "--prefill", chunk), "--prompt-cache", 0)
            env = config.setdefault("env", {})
            env.update(STRATA_PREFILL_TIMING="1", STRATA_GFX906_PREFILL_ATTN=attention, STRATA_GFX906_MTP_BATCH=mtp)
            name = f"{chunk}-{label}.json"
            private(name, (json.dumps(config, indent=2) + "\n").encode())
            arms.append(dict(config=name, chunk=chunk, attention=attention, mtp=mtp))
    manifest = dict(schema=1, source_ref=source_ref, source_config_sha256=hashlib.sha256(raw).hexdigest(),
                    engine_sha256=digest.hexdigest(), engine=str(engine.resolve()), arms=arms,
                    note="Offline plan only. No service launched or deployment changed. First compare control chunks; "
                         "then isolate attention/MTP at the best chunk. Restart your dedicated test engine per arm, "
                         "hold model/tokenized prompts/context/sampling/split/environment fixed, require zero reuse, "
                         "preserve raw requests/responses/logs, measure TTFT/refill and later decode separately. "
                         "Source configs may contain secrets: these private files must not be committed.")
    private("plan.json", (json.dumps(manifest, indent=2) + "\n").encode())
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="Clone private experiment configs; does not run them")
    plan.add_argument("--config", type=Path, required=True)
    plan.add_argument("--engine", type=Path, required=True)
    plan.add_argument("--output-dir", type=Path, required=True)
    plan.add_argument("--source-ref", required=True)
    plan.add_argument("--chunks", default="2048,3072,4096")
    plan.add_argument("--test-port", type=int, default=8096)
    report = sub.add_parser("summarize", help="Summarize raw timing records without inventing model speed")
    report.add_argument("log", type=Path)
    args = parser.parse_args()
    if args.command == "plan":
        result = make_plan(args.config, args.engine, args.output_dir, [int(x) for x in args.chunks.split(",")], args.source_ref, args.test_port)
    else:
        result = summarize(read_records(args.log.read_text(errors="replace")))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
