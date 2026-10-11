#!/usr/bin/env python3
"""4-way concurrent A/B client for --batch-groups 1 (+ --batch-mtp) vs --batch-groups 2.

Runs on the server host against 127.0.0.1:8082 (strata OpenAI server). Stdlib only.
Headline numbers come from the server's final streaming chunk (usage + timings:
prompt_ms / predicted_ms, drafts accepted), not from client clocks; client
perf_counter values are kept as TTFB / wall-clock cross-checks.

Modes (both greedy, thinking disabled, 4 requests per round sent simultaneously):
  decode   --rounds N  N rounds of 4 unique short prompts, --max-tokens (default 256);
           every stream must end finish_reason=length and cached_tokens=0.
  prefill  --rounds N  N rounds of 4 unique ~2000-token prompts (disjoint entry
           ranges), max_tokens 8, cached_tokens=0; per-request real prefill rate
           = prompt_n / prompt_ms from the server.

Usage:
  python3 ab_client.py --mode decode --rounds 5 --label "A: groups 1 + batch-mtp" --out decode-a.json
"""
import argparse
import json
import sys
import threading
import time
import urllib.request

TOPICS = [
    "the history of the Suez Canal", "how desalination plants work",
    "the life cycle of Pacific salmon", "the invention of the printing press",
    "how a four-stroke engine works", "the ecology of coral reefs",
    "the development of the telephone", "how solar panels convert sunlight",
    "the history of the Silk Road", "how vaccines train the immune system",
    "the geology of Iceland", "the story of the transcontinental railroad",
    "how refrigeration changed food supply", "the formation of the Grand Canyon",
    "how lighthouses work", "the domestication of horses",
    "the history of tea trade", "how bridges resist loads",
    "the migration of monarch butterflies", "how paper is made",
    "the discovery of penicillin", "how weather balloons gather data",
    "the rise and fall of the Hanseatic League", "how hydroelectric dams work",
    "the exploration of the Mariana Trench", "how musical instruments produce pitch",
    "the history of olive cultivation", "how airships stayed airborne",
]


def build_payload(prompt, max_tokens):
    return {
        "model": "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "reasoning_effort": "off",
        "chat_template_kwargs": {"enable_thinking": False},
    }


def stream_chat(base, payload, on_first=None, timeout=900):
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    out = {"t_send": time.perf_counter(), "t_first": None, "t_last": None}
    chars = reasoning_chars = 0
    usage = timings = finish = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
                timings = ev.get("timings")
                ch = ev.get("choices") or [{}]
                finish = ch[0].get("finish_reason") or finish
            else:
                d = (ev.get("choices") or [{}])[0].get("delta") or {}
                if d.get("reasoning_content"):
                    reasoning_chars += len(d["reasoning_content"])
                if d.get("content"):
                    chars += len(d["content"])
                    now = time.perf_counter()
                    if out["t_first"] is None:
                        out["t_first"] = now
                        if on_first:
                            on_first()
                    out["t_last"] = now
    out.update(chars=chars, reasoning_chars=reasoning_chars, usage=usage,
               timings=timings, finish=finish)
    return out


def fetch_json(base, path, timeout=30):
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return json.loads(r.read().decode())


def decode_prompts(round_idx, n=4):
    # round 0 is the warm-up; measured rounds follow, all topics unique across the run
    return [f"Write a factual, informative essay of at least 400 words about "
            f"{TOPICS[(round_idx * n + i) % len(TOPICS)]}. "
            f"Answer with the essay text only, no headings, no lists."
            for i in range(n)]


def prefill_prompt(tag, n_lines=58):
    lines = []
    for k in range(n_lines):
        n = tag * 100 + k
        lines.append(f"Entry {n}: the maintenance ledger lists part {n*7 % 997} replaced at "
                     f"cycle {n*13 % 61} with pressure {n*31 % 100} bar and note {n*17 % 997} "
                     f"recorded by shift {n*3 % 8}; the follow-up inspection on day "
                     f"{n*11 % 365} found no fault.")
    return ("Below are archival ledger notes for reference.\n\n" + "\n".join(lines) +
            "\n\nReply with the single word OK and nothing else.")


def run_round(base, mode, round_idx, max_tokens):
    if mode == "decode":
        prompts = decode_prompts(round_idx)
    else:
        prompts = [prefill_prompt(round_idx * 4 + i + 1) for i in range(4)]
    results = [None] * 4
    first_events = [threading.Event() for _ in range(4)]
    barrier = threading.Barrier(4)

    def worker(i):
        payload = build_payload(prompts[i], max_tokens)
        barrier.wait()
        results[i] = stream_chat(base, payload, on_first=first_events[i].set)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    snapshot = None
    deadline = time.time() + 600
    while time.time() < deadline:
        if all(e.is_set() for e in first_events):
            time.sleep(0.5)          # let all four slots be inside their decode windows
            try:
                snapshot = fetch_json(base, "/metrics")
            except Exception as e:   # snapshot is best-effort
                snapshot = {"error": repr(e)}
            break
        if all(r is not None for r in results):
            break
        time.sleep(0.05)
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    for i, r in enumerate(results):
        r["ttfb_s"] = None if r["t_first"] is None else round(r["t_first"] - r["t_send"], 4)
        r["stream_s"] = None if (r["t_first"] is None or r["t_last"] is None) else round(r["t_last"] - r["t_first"], 4)

    summary = {"mode": mode, "round": round_idx, "wall_s": round(wall, 3),
               "prompts": prompts, "requests": []}
    if mode == "decode":
        ok = all(r["finish"] == "length" for r in results)
        cached_ok = all((r["usage"] or {}).get("prompt_tokens_details", {}).get("cached_tokens", 0) == 0 for r in results)
        tokens = [ (r["usage"] or {}).get("completion_tokens", 0) for r in results ]
        srv_tps = [ (r["timings"] or {}).get("predicted_per_second") for r in results ]
        firsts = [r["t_first"] for r in results if r["t_first"] is not None]
        lasts = [r["t_last"] for r in results if r["t_last"] is not None]
        agg = None
        if len(firsts) == 4 and len(lasts) == 4:
            agg = round(sum(tokens) / (max(lasts) - min(firsts)), 2)
        summary.update(finish_all_length=ok, cached_all_zero=cached_ok,
                       tokens=tokens, server_tps_per_request=srv_tps,
                       aggregate_wall_tps=agg,
                       sum_server_tps=round(sum(t for t in srv_tps if t), 2) if all(srv_tps) else None)
    else:
        cached_ok = all((r["usage"] or {}).get("prompt_tokens_details", {}).get("cached_tokens", 0) == 0 for r in results)
        prompt_n = [ (r["timings"] or {}).get("prompt_n") for r in results ]
        prompt_ms = [ (r["timings"] or {}).get("prompt_ms") for r in results ]
        real_tps = [ (r["timings"] or {}).get("prompt_per_second") for r in results ]
        eff_tps = []
        for r in results:
            u = r["usage"] or {}
            t = r["timings"] or {}
            if u.get("prompt_tokens") and t.get("prompt_ms"):
                eff_tps.append(round(u["prompt_tokens"] / (t["prompt_ms"] / 1000), 1))
            else:
                eff_tps.append(None)
        firsts = [r["t_first"] for r in results if r["t_first"] is not None]
        sends = [r["t_send"] for r in results]
        agg = None
        if len(firsts) == 4:
            agg = round(sum(prompt_n) / (max(firsts) - min(sends)), 2)
        summary.update(cached_all_zero=cached_ok, prompt_n=prompt_n, prompt_ms=prompt_ms,
                       real_prefill_tps_per_request=real_tps,
                       effective_prefill_tps_per_request=eff_tps,
                       aggregate_wall_prefill_tps=agg)
    summary["requests"] = [{k: r[k] for k in ("ttfb_s", "stream_s", "chars", "reasoning_chars",
                                              "finish", "usage", "timings")} for r in results]
    if snapshot is not None:
        live = snapshot.get("live")
        summary["live_snapshot"] = live
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8082")
    ap.add_argument("--mode", choices=["decode", "prefill"], required=True)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    max_tokens = args.max_tokens or (256 if args.mode == "decode" else 8)

    print(f"# {args.label} mode={args.mode} rounds={args.rounds} max_tokens={max_tokens}",
          file=sys.stderr, flush=True)
    rounds = []
    for r in range(args.warmup + args.rounds):
        s = run_round(args.base, args.mode, r, max_tokens)
        s["warmup"] = r < args.warmup
        rounds.append(s)
        line = (f"round {r}{' (warmup)' if s['warmup'] else ''}: "
                f"wall={s['wall_s']}s")
        if args.mode == "decode":
            line += (f" tokens={s['tokens']} len_ok={s['finish_all_length']} "
                     f"cached0={s['cached_all_zero']} agg_wall_tps={s['aggregate_wall_tps']} "
                     f"srv_tps={s['server_tps_per_request']}")
        else:
            line += (f" prompt_n={s['prompt_n']} cached0={s['cached_all_zero']} "
                     f"real_tps={s['real_prefill_tps_per_request']} agg={s['aggregate_wall_prefill_tps']}")
        print(line, file=sys.stderr, flush=True)

    measured = [s for s in rounds if not s["warmup"]]

    def med(vals):
        v = sorted(x for x in vals if x is not None)
        return v[len(v) // 2] if v else None

    if args.mode == "decode":
        summary = {
            "median_aggregate_wall_tps": med([s["aggregate_wall_tps"] for s in measured]),
            "median_sum_server_tps": med([s["sum_server_tps"] for s in measured]),
            "per_round_aggregate_wall_tps": [s["aggregate_wall_tps"] for s in measured],
            "all_rounds_valid": all(s["finish_all_length"] and s["cached_all_zero"] for s in measured),
        }
    else:
        summary = {
            "median_real_prefill_tps_per_request": med([t for s in measured
                                                        for t in s["real_prefill_tps_per_request"]]),
            "median_aggregate_wall_prefill_tps": med([s["aggregate_wall_prefill_tps"] for s in measured]),
            "per_round_aggregate_wall_prefill_tps": [s["aggregate_wall_prefill_tps"] for s in measured],
            "all_rounds_cached_zero": all(s["cached_all_zero"] for s in measured),
        }
    out = {"label": args.label, "mode": args.mode, "max_tokens": max_tokens,
           "summary": summary, "rounds": rounds}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
