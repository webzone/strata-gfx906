#!/usr/bin/env python3
"""Deterministic A/B harness for strata-mi50 (engine changes and exact switches).

    ab.py VARIANTS.json [--reps N] [--keep]

VARIANTS.json: a list of {"name": str, "env": {"K": "V" | null}, "args": {"--flag": "value" | null},
"engine": "/host/path/to/strata" | null}. Every variant starts from ~/strata-data/config/tuned.json, the
container is recreated by run.sh (with the engine binary bind-mounted when "engine" is set), and a fixed
greedy workload runs without a nonce: an exact change must reproduce the baseline output hashes, so the
decode rates compare speed only. One JSON line per variant goes to ab-results.jsonl. At the end the tuned
config and the stock engine are restored (unless --keep).
"""
import glob, hashlib, json, os, subprocess, sys, time, urllib.request

HOME = os.path.expanduser("~")
CFG = f"{HOME}/strata-data/config/strata-unsloth-ud-q5_k_xl.json"
TUNED = f"{HOME}/strata-data/config/tuned.json"
OUT = f"{HOME}/strata-deploy/ab-results.jsonl"
URL = "http://127.0.0.1:8095"
KEY = open(f"{HOME}/strata-deploy/api_key").read().strip()
HDR = {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}

DOC = ""
for p in sorted(glob.glob(f"{HOME}/strata-gfx906/docs/*.md")):
    DOC += open(p, encoding="utf-8", errors="replace").read()
    if len(DOC) > 60_000:
        break
DOC = DOC[:60_000]

WORK = [
    ("chat", "Объясни простыми словами, чем TCP отличается от UDP, и приведи по два примера использования каждого.", 500),
    ("code", "Write a Python function that parses an ISO-8601 date string and returns the weekday name. Include tests.", 500),
    ("story", "Напиши короткий рассказ (около 300 слов) о маяке на краю света.", 500),
    ("long", DOC + "\n\nName the three main topics of these documents.", 200),
]


def ask(prompt, max_tokens):
    body = {"model": "x", "max_tokens": max_tokens, "temperature": 0,
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), HDR)
    with urllib.request.urlopen(req, timeout=1800) as r:
        m = json.loads(r.read())["choices"][0]["message"]
    out = (m.get("reasoning_content") or "") + "\n---\n" + (m.get("content") or "")
    with urllib.request.urlopen(urllib.request.Request(URL + "/metrics", headers=HDR), timeout=30) as r:
        mt = json.loads(r.read())
    last = max(mt.get("requests", []), key=lambda x: x.get("time", 0))
    return {"dec": last.get("decode_tok_s"), "out": last.get("output_tokens"),
            "prompt": last.get("prompt_tokens"), "prompt_ms": last.get("prompt_ms"),
            "drafts": f'{last.get("drafts_accepted")}/{last.get("drafts_offered")}',
            "hash": hashlib.sha1(out.encode()).hexdigest()[:10]}


def configure(v):
    c = json.load(open(TUNED))
    args = c["args"]
    for k, val in v.get("args", {}).items():
        if k in args:
            i = args.index(k)
            if val is None:
                del args[i:i + 2]
            else:
                args[i + 1] = str(val)
        elif val is not None:
            args += [k, str(val)]
    env = c.setdefault("env", {})
    for k, val in v.get("env", {}).items():
        if val is None:
            env.pop(k, None)
        else:
            env[k] = str(val)
    json.dump(c, open(CFG, "w"), indent=1)


def start(engine=None, extra=""):
    e = dict(os.environ)
    mount = f"-v {engine}:/opt/strata/engine/strata:ro" if engine else ""
    e["STRATA_EXTRA_ENV"] = f"{mount} {extra}".strip()
    subprocess.run([f"{HOME}/strata-deploy/run.sh"], check=True, capture_output=True, env=e)
    t0 = time.time()
    while time.time() - t0 < 900:
        try:
            with urllib.request.urlopen(URL + "/health", timeout=5) as r:
                if json.loads(r.read()).get("loaded"):
                    return round(time.time() - t0)
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError("server did not come up")


def main():
    variants = json.load(open(sys.argv[1]))
    reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 3
    base_hash = None
    try:
        for v in variants:
            rec = {"name": v["name"], "variant": v, "time": time.strftime("%H:%M:%S")}
            try:
                configure(v)
                rec["startup_s"] = start(v.get("engine"), v.get("docker", ""))
                ask("Привет!", 16)
                res = {}
                for name, prompt, mt in WORK:
                    res[name] = [ask(prompt, mt) for _ in range(reps if name != "long" else 1)]
                s = {}
                for name, runs in res.items():
                    decs = [r["dec"] or 0 for r in runs]
                    s[name] = {"dec": round(sum(decs) / len(decs), 2), "min": min(decs), "max": max(decs),
                               "hash": sorted({r["hash"] for r in runs}), "drafts": runs[0]["drafts"]}
                hashes = {k: s[k]["hash"] for k in s}
                if base_hash is None:
                    base_hash = hashes
                s["same_as_first"] = hashes == base_hash
                tot = [s[k]["dec"] for k in ("chat", "code", "story")]
                s["mean3"] = round(sum(tot) / 3, 2)
                rec["results"] = res
                rec["summary"] = s
                print(v["name"], json.dumps(s, ensure_ascii=False), flush=True)
            except Exception as e:
                rec["error"] = repr(e)
                print(v["name"], "ERROR", e, flush=True)
                subprocess.run(f"docker logs --tail 40 strata-mi50 > {HOME}/strata-deploy/ab-fail-{v['name']}.log 2>&1",
                               shell=True)
            with open(OUT, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    finally:
        if "--keep" not in sys.argv:
            configure({})
            start()


if __name__ == "__main__":
    main()
