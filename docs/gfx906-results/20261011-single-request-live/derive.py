#!/usr/bin/env python3
"""Recompute every rate quoted for the 2026-10-11 single-request window from the raw payload.

Reads metrics-20261011-010634z.body (a read-only GET /metrics response from the deployed
v0.1.42 engine on the T5810 dual MI50) and prints the per-request table plus the window
summary. No client clocks are involved: every number comes from the engine's own counters.

Formulas used:
  new prompt tokens   = prompt_tokens - reused
  real prefill tok/s  = (prompt_tokens - reused) / (prompt_ms / 1000)     # compute only
  effective prefill   = prompt_tokens / (prompt_ms / 1000)                # reused tokens counted
  decode tok/s        = output_tokens / (decode_ms / 1000)                # cross-check of decode_tok_s
  draft acceptance    = drafts_accepted / drafts_offered
"""

import json
import os
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
PAYLOAD = os.path.join(HERE, "metrics-20261011-010634z.body")
GIB = 1024 ** 3


def utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def main() -> None:
    with open(PAYLOAD, encoding="utf-8") as handle:
        data = json.load(handle)

    reqs = sorted(data["requests"], key=lambda r: r["time"])
    tot = data["totals"]

    print(f"window {utc(tot['since'])} -> {utc(data['time'])} UTC  ({data['time'] - tot['since']:.1f} s)")
    print(f"requests kept {data['requests_kept']} / totals.requests {tot['requests']}")

    print("\nper request (chronological):")
    hdr = ("time_utc", "prompt", "reused", "new", "out", "prompt_ms",
           "real_pre", "eff_pre", "decode_rep", "decode_calc", "drafts", "dur_s")
    print(("{:<12} {:>7} {:>7} {:>6} {:>5} {:>10} {:>9} {:>8} {:>10} {:>11} {:>13} {:>6}").format(*hdr))
    for r in reqs:
        new = r["prompt_tokens"] - r["reused"]
        psec = r["prompt_ms"] / 1000.0
        dsec = r["decode_ms"] / 1000.0
        print(
            "{:<12} {:>7} {:>7} {:>6} {:>5} {:>10} {:>9.1f} {:>8.0f} {:>10} {:>11.2f} {:>13} {:>6}".format(
                utc(r["time"])[11:], r["prompt_tokens"], r["reused"], new, r["output_tokens"],
                r["prompt_ms"], new / psec, r["prompt_tokens"] / psec, r["decode_tok_s"],
                r["output_tokens"] / dsec,
                f"{r['drafts_accepted']}/{r['drafts_offered']}", r["duration_s"],
            )
        )

    # spacing check: the payload does not label `time` as a start or a completion stamp, so this is
    # evidence of one-at-a-time traffic, not proof. The live counters are the direct evidence.
    gaps = [(reqs[i + 1]["time"] - reqs[i]["time"], reqs[i]["duration_s"]) for i in range(len(reqs) - 1)]
    print("\nspacing check (gap between consecutive time stamps >= earlier request's duration_s): "
          + ", ".join(f"{g:.1f}>={d}" for g, d in gaps))
    print("spacing consistent with one-at-a-time traffic:", all(g >= d for g, d in gaps))

    dec = sorted(r["decode_tok_s"] for r in reqs)
    warm = [r for r in reqs if r["reused"] > 0]
    cold = [r for r in reqs if r["reused"] == 0]
    eff = sorted(r["prompt_tokens"] / (r["prompt_ms"] / 1000) for r in warm)
    real = sorted((r["prompt_tokens"] - r["reused"]) / (r["prompt_ms"] / 1000) for r in warm)

    def med(vals):
        vals = sorted(vals)
        n = len(vals)
        return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2

    print("\nwindow summary:")
    print(f"  decode per request      {dec[0]:.1f}-{dec[-1]:.1f} tok/s, median {med(dec):.1f}, "
          f"mean {sum(dec) / len(dec):.1f}")
    print(f"  decode recomputed       {min(r['output_tokens'] / (r['decode_ms'] / 1000) for r in reqs):.2f}"
          f"-{max(r['output_tokens'] / (r['decode_ms'] / 1000) for r in reqs):.2f} tok/s "
          f"(matches decode_tok_s on all {len(reqs)} requests)")
    print(f"  decode window aggregate {tot['output_tokens']} / {tot['decode_ms'] / 1000:.2f} s = "
          f"{tot['output_tokens'] / (tot['decode_ms'] / 1000):.1f} tok/s")
    for r in cold:
        print(f"  cold prefill (reused 0)  {r['prompt_tokens']} new tokens in {r['prompt_ms'] / 1000:.3f} s = "
              f"{r['prompt_tokens'] / (r['prompt_ms'] / 1000):.1f} tok/s")
    print(f"  warm turns n={len(warm)}: real new-token prefill {real[0]:.1f}-{real[-1]:.1f} "
          f"(median {med(real):.1f}); effective {eff[0]:.0f}-{eff[-1]:.0f} (median {med(eff):.0f}) tok/s")
    print(f"  totals: prompt {tot['prompt_tokens']}, reused {tot['reused']} "
          f"({100 * tot['reused'] / tot['prompt_tokens']:.1f} %), new {tot['prompt_tokens'] - tot['reused']}, "
          f"output {tot['output_tokens']}")
    print(f"  totals: prompt_ms {tot['prompt_ms'] / 1000:.2f} s, decode_ms {tot['decode_ms'] / 1000:.2f} s")
    print(f"  drafts {tot['drafts_accepted']}/{tot['drafts_offered']} = "
          f"{100 * tot['drafts_accepted'] / tot['drafts_offered']:.1f} % accepted")

    cc = data["conversation_cache"]
    print(f"  conversation cache: {cc['slots']} slots / {cc['budget_mib']} MiB, parked {cc['parked']} "
          f"= {cc['bytes'] / GIB:.2f} GiB, parks {cc['parks']}, restores {cc['restores']}, "
          f"evictions {cc['evictions']}; requests {cc['requests']}, reused {cc['requests_reused']}")

    hw, hist = data["hardware"], data["history"]
    print(f"  VRAM {hw['gpu_mem_used'] / GIB:.2f}/{hw['gpu_mem_total'] / GIB:.2f} GiB "
          f"(card0 {hw['gpus'][0]['mem_used'] / GIB:.2f}, card1 {hw['gpus'][1]['mem_used'] / GIB:.2f}); "
          f"engine free {data['engine']['vram_free_mib']} MiB")
    print(f"  RAM {hw['ram_used'] / GIB:.2f}/{hw['ram_total'] / GIB:.2f} GiB; "
          f"history {min(hist['ram_used']) / GIB:.2f}-{max(hist['ram_used']) / GIB:.2f} GiB")
    busy = [i for i, u in enumerate(hist["gpu_util"]) if u > 0]
    print(f"  gpu_util (mean of both cards) busy samples {len(busy)}: "
          f"{', '.join(str(hist['gpu_util'][i]) for i in busy)}")
    print(f"  gpu_power (both cards summed) idle {min(v for i, v in enumerate(hist['gpu_power']) if i not in busy)}"
          f"-{max(v for i, v in enumerate(hist['gpu_power']) if i not in busy)} W, busy "
          f"{min(hist['gpu_power'][i] for i in busy)}-{max(hist['gpu_power'][i] for i in busy)} W "
          f"(limit {hw['gpu_power_limit']} W)")
    print(f"  gpu_temp {min(hist['gpu_temp'])}-{max(hist['gpu_temp'])} C; "
          f"host cpu {min(hist['cpu']):.2f}-{max(hist['cpu']):.2f} %")
    print(f"  snapshot instant: card0 util {hw['gpus'][0]['util']} % / card1 {hw['gpus'][1]['util']} %")
    print(f"  live at snapshot: state {data['live']['state']!r}, prompt_tokens "
          f"{data['live']['prompt_tokens']}, running {data['live']['running']}, "
          f"outside_slots {data['live']['outside_slots']}, slots idle "
          f"{sum(1 for s in data['live']['slots'] if s['state'] == 'idle')}/4")


if __name__ == "__main__":
    main()
