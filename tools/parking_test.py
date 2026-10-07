#!/usr/bin/env python3
"""Conversation parking test: a follow-up to conversation A must decode the same tokens whether A's state was still
live (reference engine: A, then the follow-up) or came back from the parking cache (second engine: A, then B, which
parks A, then the follow-up).  Greedy; run with --extra "--pcie-frac 0 --adapt-every 1000000" for exactness.

  python3 tools/parking_test.py --exe engine/strata --config strata-<model>.json \
      --extra "--layer-split 12,24,36 --conversation-cache-mib 8192 --conversation-cache-slots 4 --pcie-frac 0"
"""
import argparse, json, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batch_test import Engine, tokenizer  # noqa: E402

DOC = ("The history of the printing press begins with movable type in East Asia and continues with Gutenberg's "
       "press in Mainz around 1450, which spread across Europe within decades, lowered the cost of books, changed "
       "how scholars, merchants and churches shared knowledge, and fed the Reformation and the scientific "
       "revolution. ")


def chat(user):
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def gen(eng, out, ids, n, cancel_prefill=False):
    eng.send(f"GEN {n} " + ",".join(map(str, ids)))
    got = []
    stopped = False
    for line in out:
        if line.startswith("T "):
            got.append(int(line.split()[1]))
        elif cancel_prefill and not stopped and line.startswith("PP "):
            reached, total = map(int, line.split()[1:3])
            if 0 < reached < total:
                eng.send("STOP")
                stopped = True
        elif line.startswith("ERR"):
            raise SystemExit("engine: " + line)
        elif line.startswith("DONE"):
            if cancel_prefill and (not stopped or got or line.split()[5] != "cancel"):
                raise SystemExit("the short request was not cancelled during prefill: " + line)
            return got, line
    raise SystemExit("the engine ended")


def run(a, cfg, tok, with_b):
    # STRATA_SNAPSHOT_VERIFY: the engine reads the restored draft ring back (on the GPU that holds it) after a restore
    eng = Engine(a.exe, cfg, 0, {**cfg.get("env", {}), "STRATA_IQ_MT_MIN": "1", "STRATA_SNAPSHOT_VERIFY": "1"}, a.extra.split())
    commands = []
    send = eng.send
    def record(line):
        commands.append(line)
        send(line)
    eng.send = record
    out = eng.lines()
    pa = tok.encode(chat(DOC * a.repeat + "\nSummarize this text in five sentences."), parse_special=True)
    ans, _ = gen(eng, out, pa, a.max_new)
    interruption = None
    if with_b:
        pb = tok.encode(chat("Write a short poem about the sea, then explain its metaphors." * 50), parse_special=True)
        if a.cancel_b:
            pb = pb[:611] + pb[-7:]   # match the owner's 618-token interrupted admission
        _, interruption = gen(eng, out, pb, a.max_new, cancel_prefill=a.cancel_b)
    end = tok.encode("<|im_end|>\n", parse_special=True)
    follow = pa + ans + ([] if ans and ans[-1] in end else end) + \
        tok.encode(chat("Now give three keywords for that text, with one sentence each.")[0:], parse_special=True)
    t0 = time.time()
    got, done = gen(eng, out, follow, a.max_new)
    dt = time.time() - t0
    if with_b:   # B again: A is parked a second time, from retained K/V (every stage's) - only its new tail copied
        gen(eng, out, pb, 8)
    eng.send("QUIT")
    eng.p.wait(timeout=180)
    log = Path(eng.log_path).read_text()
    restored = re.findall(r"restored \d+ tokens[^\n]*", log)
    if with_b and "SNAPSHOT_VERIFY draft=" not in log:
        print("no SNAPSHOT_VERIFY line: the draft ring was not read back", flush=True)
    reparked = [int(x) for x in re.findall(r"parked \d+ tokens .*reused_kv_bytes=(\d+)", log)]
    return len(pa), len(follow), got, done, dt, restored, reparked, {
        "commands": commands, "tokens": got, "done": done, "log": log,
        "snapshot_verified": "SNAPSHOT_VERIFY draft=" in log, "interruption": interruption}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--repeat", type=int, default=60, help="copies of the paragraph in conversation A")
    ap.add_argument("--max-new", type=int, default=120)
    ap.add_argument("--cancel-b", action="store_true", help="cancel a 618-token B during prefill, before any output")
    ap.add_argument("--extra", default="")
    ap.add_argument("--dump", default="", help="save exact commands, follow-up token IDs and both engine logs as JSON")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    tok = tokenizer(cfg["tokenizer"])
    ref = run(a, cfg, tok, with_b=False)
    park = run(a, cfg, tok, with_b=True)
    for name, r in (("live (reference)", ref), ("parked (A, B, A)", park)):
        f = r[3].split()
        print(f"{name}: prompt A {r[0]} tokens, follow-up {r[1]} tokens, reused {f[8] if len(f) > 8 else '?'}, "
              f"follow-up in {r[4]:.2f} s; restore lines: {r[5][-1:] or 'none'}")
    same = ref[2] == park[2]
    first = next((k for k in range(min(len(ref[2]), len(park[2]))) if ref[2][k] != park[2][k]), None)
    print("follow-up tokens:", "IDENTICAL" if same else f"DIFFER at {first}", f"({len(ref[2])} / {len(park[2])})")
    print("   ", repr(tok.decode(park[2])[:200]))
    reused = park[6][-1] if park[6] else 0
    print(f"second park of A: {reused} bytes of K/V reused (retained from its restore)")
    if a.cancel_b:
        print("cancelled short admission:", park[7]["interruption"])
    resumed = int(park[3].split()[8])
    kept_history = resumed >= park[0] - 8   # a turn checkpoint may omit the assistant template's final seven tokens
    if a.dump:
        Path(a.dump).write_text(json.dumps({"reference": ref[7], "parked": park[7], "identical": same}, indent=2) + "\n")
    return 0 if same and kept_history and park[5] and reused > 0 and park[7]["snapshot_verified"] else 1


if __name__ == "__main__":
    sys.exit(main())
