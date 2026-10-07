#!/usr/bin/env python3
"""Batch slots, the parts batch_test.py does not reach (#465): a long prompt read WHILE other slots decode (its
chunks interleaved with their windows), and a conversation's next turn continued from the slot that holds it.

  1. solo references (GEN): A, then A's next turn (from the live session), B, C (a long prompt)
  2. the same in the slots: A in slot 0, B in slot 1 next to it, then C in slot 2 while A and B decode - every slot's
     tokens must equal its solo tokens
  3. A's next turn (BGEN into slot 0, which still holds A): read from the slot, its tokens must equal the solo next turn
  4. a long prompt D gives way (BYIELD) at its first chunk boundary, a short E is admitted, D goes on from its slot
     while E decodes: both equal their solo tokens
  6. A in a slot, stopped after 50 tokens and continued on the solo path (GEN of A + its tokens, from the slot): the
     whole equals solo A (what the server does with a request left alone in a slot)
  5. (a measurement) a solo next turn continued from a slot: the drafts accepted
  7. F, A's history as a client sends it back without the reply's thinking, from slot 1's turn checkpoint: equal to
     F solo (which continues from the same checkpoint of the live chain)

Exact comparisons need the same settings as batch_test.py (this script sets STRATA_IQ_MT_MIN=1):
  python tools/batch_interleave_test.py --exe build/strata --config strata-<model>.json \\
      --extra "--pcie-frac 0 --adapt-every 1000000 --no-prefill-borrow"
Without --no-prefill-borrow the slots decoding during C's read see the expert cache without the slots C's prompt
borrowed (those experts run on the CPU, which rounds differently): A's and B's text may then drift from solo.
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batch_test import Engine, tokenizer  # noqa: E402

LONG = ("The history of mathematics is long and full of surprising turns. Early civilizations counted with tally marks, "
        "then with symbols, and later with place-value systems that made arithmetic far easier. ")


def run(eng, out, line, slot=None):
    """Send one GEN/BGEN; returns (tokens of the request line, the BADM flag) - BT/BDONE of other slots are kept."""
    eng.send(line)
    eng.resumed = 0
    got = []
    for l in out:
        if l.startswith("T "):
            got.append(int(l.split()[1]))
        elif l.startswith("RESUME "):
            eng.resumed = int(l.split()[1])
        elif l.startswith(("BT ", "BDONE ")):
            eng.pending.append(l)
        elif l.startswith("ERR"):
            raise SystemExit("engine: " + l)
        elif slot is None and l.startswith("DONE"):
            return got, None
        elif slot is not None and l.startswith("BADM "):
            return got, l.split()[2] == "1"
    raise SystemExit(f"the engine ended during: {line[:40]} (see {eng.log_path})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--long", type=int, default=5000, help="tokens of the long prompt C (several prompt chunks)")
    ap.add_argument("--batch", type=int, default=3, help="use 4 with --extra '--batch-groups 2' for pipeline cache checks")
    ap.add_argument("--dump", default="", help="save exact engine commands and reference/batch token IDs as JSON")
    ap.add_argument("--extra", default="")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    tok = tokenizer(cfg["tokenizer"])

    def chat(q):
        return tok.encode(f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                          parse_special=True)
    body = LONG * (a.long // max(1, len(tok.encode(LONG))) + 1)
    A = chat("Write a Python function that merges overlapping intervals, then explain it.")
    B = chat("Compare TCP and QUIC: handshake, congestion control, multiplexing.")
    C = chat(body + "\nSummarize the text above in three sentences.")
    D = chat(body + "\nWhat are the three most important ideas in the text above?")
    E = chat("Describe the life cycle of a star like the Sun.")
    print(f"prompts: A {len(A)}, B {len(B)}, C {len(C)} tokens", flush=True)
    eng = Engine(a.exe, cfg, a.batch, {**cfg.get("env", {}), "STRATA_IQ_MT_MIN": "1"}, a.extra.split())
    commands = []
    send = eng.send
    def record(line):
        commands.append(line)
        send(line)
    eng.send = record
    eng.pending = []
    out = eng.lines()
    ids = lambda v: ",".join(map(str, v))
    M = a.max_new
    args = list(cfg["args"]) + a.extra.split()
    groups = next((int(args[i + 1]) for i in range(len(args) - 1, -1, -1)
                   if args[i] == "--batch-groups"), 1)
    piped = groups > 1 and len(cfg.get("gpu") or []) > 1

    # 1. solo
    sA, _ = run(eng, out, f"GEN {M} {ids(A)}")
    A2 = A + sA + chat("Now do the same in Rust.")
    sA2, _ = run(eng, out, f"GEN {M} {ids(A2)}")
    # F: A's history as a client sends it back WITHOUT the reply's thinking - it shares A only up to A's last turn
    # boundary (the checkpoint there); solo it continues from that checkpoint of the live chain
    turn_at = max(i for i, t in enumerate(A) if t == A[0])          # A[0] is <|im_start|>
    F = A[:turn_at] + tok.encode("<|im_start|>assistant\nA short answer.<|im_end|>\n", parse_special=True) + \
        chat("Now in Go.")
    sF, _ = run(eng, out, f"GEN {M} {ids(F)}")
    sB, _ = run(eng, out, f"GEN {M} {ids(B)}")
    sC, _ = run(eng, out, f"GEN {M} {ids(C)}")
    C2 = C + sC + chat("Summarize that in one sentence.")
    sC2, _ = run(eng, out, f"GEN {M} {ids(C2)}")
    sD, _ = run(eng, out, f"GEN {M} {ids(D)}")
    sE, _ = run(eng, out, f"GEN {M} {ids(E)}")
    print(f"solo: A {len(sA)}, A2 {len(sA2)}, B {len(sB)}, C {len(sC)}, D {len(sD)}, E {len(sE)} tokens", flush=True)

    # 2. the slots: A, B, then the long C while they decode
    got = {0: [], 1: [], 2: []}
    done = {}
    t0 = time.time()
    for s, P in ((0, A), (1, B), (2, C)):
        first, cont = run(eng, out, f"BGEN {s} {M} {ids(P)}", slot=s)
        got[s] += first
        if not cont:
            done[s] = True
    t_admit = time.time() - t0
    during_c = sum(1 for l in eng.pending if l.startswith("BT "))   # windows the slots ran while C was admitted

    def drain():
        while eng.pending or len(done) < 3:
            l = eng.pending.pop(0) if eng.pending else next(out)
            if l.startswith("BT "):
                _, s, y = l.split()
                got[int(s)].append(int(y))
            elif l.startswith("BDONE "):
                done[int(l.split()[1])] = True
            elif l.startswith("ERR"):
                raise SystemExit("engine: " + l)
    drain()
    print(f"slots: admissions {t_admit:.1f}s; {during_c} slot tokens arrived while the admissions read their prompts",
          flush=True)
    ok = True
    for s, ref, name in ((0, sA, "A"), (1, sB, "B"), (2, sC, "C")):
        same = got[s] == ref
        ok &= same
        d = next((k for k in range(min(len(got[s]), len(ref))) if got[s][k] != ref[k]), None)
        print(f"slot {s} ({name}): {len(got[s])} tokens, solo {len(ref)}: "
              f"{'IDENTICAL' if same else f'DIFFERS at {d}'}", flush=True)

    # 3. A's next turn from slot 0 (the engine's log says "slot 0 gave back ...")
    first, cont = run(eng, out, f"BGEN 0 {M} {ids(A2)}", slot=0)
    reuse2 = eng.resumed
    ok &= reuse2 >= len(A)   # reuse is required, not just output equality after a full re-prefill
    got2, done = first, {}
    if cont:
        while True:
            l = eng.pending.pop(0) if eng.pending else next(out)
            if l.startswith("BT 0 "):
                got2.append(int(l.split()[2]))
            elif l.startswith("BDONE 0 "):
                break
    same = got2 == sA2
    ok &= same
    d = next((k for k in range(min(len(got2), len(sA2))) if got2[k] != sA2[k]), None)
    print(f"A's next turn from its slot: reused {reuse2}, {len(got2)} tokens, solo {len(sA2)}: "
          f"{'IDENTICAL' if same else f'DIFFERS at {d}'}", flush=True)
    # The long conversation in slot 2 must survive three admissions into its group sibling (slot 3).
    # This exercises both pipeline groups and a sparse window containing only the upper-numbered slot.
    if a.batch >= 4:
        for burst in range(3):
            short = E + chat(f"Upper-group interruption {burst}: answer briefly.")
            _, cont = run(eng, out, f"BGEN 3 8 {ids(short)}", slot=3)
            while cont:
                l = eng.pending.pop(0) if eng.pending else next(out)
                if l.startswith("BDONE 3 "):
                    break
                if l.startswith("ERR"):
                    raise SystemExit("engine: " + l)
        long_next, cont = run(eng, out, f"BGEN 2 {M} {ids(C2)}", slot=2)
        reuse_long = eng.resumed
        while cont:
            l = eng.pending.pop(0) if eng.pending else next(out)
            if l.startswith("BT 2 "):
                long_next.append(int(l.split()[2]))
            elif l.startswith("BDONE 2 "):
                break
            elif l.startswith("ERR"):
                raise SystemExit("engine: " + l)
        same_long = long_next == sC2
        ok &= same_long and reuse_long >= len(C)
        print(f"long next turn after three sibling bursts: reused {reuse_long} of {len(C)} history tokens; "
              f"{'IDENTICAL' if same_long else 'DIFFERS'}", flush=True)
    # 4. a long prompt D gives way (BYIELD) to a short E at its first chunk boundary, then goes on from its slot
    got4 = {1: [], 2: []}
    if piped:
        # Upstream read_part bypasses BYIELD for pipelined groups. Still compare D/E output parity;
        # do not re-admit a decoding D into its occupied slot or claim that chunk yielding was tested.
        first, cont2 = run(eng, out, f"BGEN 2 {M} {ids(D)}", slot=2)
        got4[2] += first
        print("SKIP BYIELD: upstream pipelined groups do not support chunk yielding", flush=True)
    else:
        eng.send(f"BGEN 2 {M} {ids(D)}")
        eng.send("BYIELD 2")
        yielded = None
        for l in out:
            if l.startswith("YIELDED "):
                yielded = l
            elif l.startswith(("BT ", "BDONE ")):
                eng.pending.append(l)
            elif l.startswith("ERR"):
                raise SystemExit("engine: " + l)
            elif l.startswith("BADM 2 "):
                break
        print(f"D gave way: {yielded}", flush=True)
        ok &= yielded is not None
    first, cont1 = run(eng, out, f"BGEN 1 {M} {ids(E)}", slot=1)
    got4[1] += first
    if not piped:
        first, cont2 = run(eng, out, f"BGEN 2 {M} {ids(D)}", slot=2)
        got4[2] += first
    done4 = {s for s, c in ((1, cont1), (2, cont2)) if not c}
    while len(done4) < 2:
        l = eng.pending.pop(0) if eng.pending else next(out)
        if l.startswith("BT "):
            _, s, y = l.split()
            if int(s) in got4:
                got4[int(s)].append(int(y))
        elif l.startswith("BDONE "):
            done4.add(int(l.split()[1]))
    for s, ref, name in ((1, sE, "E (short, admitted while D waited)"), (2, sD, "D (gave way, then went on)")):
        same = got4[s] == ref
        ok &= same
        d = next((k for k in range(min(len(got4[s]), len(ref))) if got4[s][k] != ref[k]), None)
        print(f"slot {s} {name}: {len(got4[s])} tokens, solo {len(ref)}: "
              f"{'IDENTICAL' if same else f'DIFFERS at {d}'}", flush=True)
    # 6. back to the solo path (what the server does with a request left alone in a slot): A in slot 1, BSTOP after
    # 50 tokens, then GEN of A + what it produced - the engine continues from the slot with MTP drafts again
    first, cont = run(eng, out, f"BGEN 1 {M} {ids(A)}", slot=1)
    got6, sent = list(first), False
    while cont:
        l = eng.pending.pop(0) if eng.pending else next(out)
        if l.startswith("BT 1 "):
            got6.append(int(l.split()[2]))
            if len(got6) >= 50 and not sent:
                eng.send("BSTOP 1")
                sent = True
        elif l.startswith("BDONE 1 "):
            break
    # Overwrite the main session and run the stopped slot's sibling three times. Pipeline padding used to
    # destroy slot 1 here, making GEN below read A's entire context again. Slot 1 must keep every committed token.
    if len(got6) < M:
        for burst in range(3):
            short = E + chat(f"Short interruption {burst}: answer briefly.")
            _, short_cont = run(eng, out, f"BGEN 0 8 {ids(short)}", slot=0)
            while short_cont:
                l = eng.pending.pop(0) if eng.pending else next(out)
                if l.startswith("BDONE 0 "):
                    break
                if l.startswith("ERR"):
                    raise SystemExit("engine: " + l)
    tail6, _ = run(eng, out, f"GEN {M - len(got6)} {ids(A + got6)}") if len(got6) < M else ([], None)
    reuse6 = eng.resumed
    if len(got6) < M:
        ok &= reuse6 >= len(A) + len(got6) - 1
    same = got6 + tail6 == sA
    ok &= same
    both = got6 + tail6
    d = next((k for k in range(min(len(both), len(sA))) if both[k] != sA[k]), None)
    print(f"A in a slot, then solo again after {len(got6)} tokens: reused {reuse6}, {len(both)} tokens, solo {len(sA)}: "
          f"{'IDENTICAL' if same else f'DIFFERS at {d}'}", flush=True)

    # 5. (a measurement, not a check) a SOLO next turn continued from slot 0: its tokens are exact either way, but the
    # draft layer's own K/V was built for another conversation, so fewer drafts may be accepted than in a solo next
    # turn whose drafter read the conversation (A2 in step 1)
    A3 = A2 + got2 + chat("Now in Go.")
    eng.send(f"GEN {M} {ids(A3)}")
    for l in out:
        if l.startswith(("BT ", "BDONE ")):
            eng.pending.append(l)
        elif l.startswith("DONE"):
            f = l.split()
            print(f"solo next turn from a slot: drafts accepted {f[6]} of {f[7]} ({f[1]} tokens in {f[4]} ms)",
                  flush=True)
            break
    # 7. F from slot 1's turn checkpoint (slot 1 holds A since step 6): the engine log says "(its turn checkpoint)"
    first, cont = run(eng, out, f"BGEN 2 {M} {ids(F)}", slot=2)
    got7 = list(first)
    while cont:
        l = eng.pending.pop(0) if eng.pending else next(out)
        if l.startswith("BT 2 "):
            got7.append(int(l.split()[2]))
        elif l.startswith("BDONE 2 "):
            break
    same = got7 == sF
    ok &= same
    d = next((k for k in range(min(len(got7), len(sF))) if got7[k] != sF[k]), None)
    print(f"F (A's history without the reply's thinking) from slot 1's turn checkpoint: {len(got7)} tokens, solo "
          f"{len(sF)}: {'IDENTICAL' if same else f'DIFFERS at {d}'}", flush=True)
    eng.send("QUIT")
    eng.p.wait(timeout=180)
    if a.dump:
        Path(a.dump).write_text(json.dumps({"commands": commands, "solo": [sA, sA2, sB, sC, sD, sE, sF],
                                          "batch": got, "next_turn": got2, "yielded": got4,
                                          "resumed_solo": both, "checkpoint": got7,
                                          "long_next_turn": long_next if a.batch >= 4 else None,
                                          "solo_long_next_turn": sC2,
                                          "reuse_long_next_turn": reuse_long if a.batch >= 4 else None,
                                          "reuse_next_turn": reuse2, "reuse_solo": reuse6, "ok": ok}, indent=2) + "\n")
    log = Path(eng.log_path).read_text(errors="replace")
    for l in log.splitlines():
        if "drafts accepted" in l:
            print("  log:", l.split("strata serve: ")[-1][:160], flush=True)
    for key in ("gave back", "its turn checkpoint", "takes", "the prompt was read in", "gives way"):
        print(f"engine log '{key}': {log.count(key)} lines", flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
