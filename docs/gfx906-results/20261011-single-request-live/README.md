# Single-request live traffic on the deployed v0.1.42 engine — read-only `/metrics` window (2026-10-11 01:01:09–01:06:34 UTC, GSQ-RCO IQ3_S)

This record holds one read-only `GET /metrics` response from the production T5810 service, plus the rates
derived from it. It is the newest workload observation for the fork and the source of the numbers in
*MI50 workload results* in `README.md`. The payload was **not** produced by a benchmark client: it is the
engine's own accounting of the owner's real traffic during one 325.5-second session window.

- Raw payload, byte-for-byte as returned: [`metrics-20261011-010634z.body`](metrics-20261011-010634z.body)
- Every number below is recomputed from that file by [`derive.py`](derive.py)
  (`python3 docs/gfx906-results/20261011-single-request-live/derive.py`)

## What ran

| Item | Value | Source in the payload |
| --- | --- | --- |
| Engine | `0.1.42` (the deployed `build-text-rocm10/strata` = `0550f3f2…`, merge `7f104978` = upstream `61b3fb5d`, installed 2026-10-11 after the batch-groups A/B) | `engine.version`, `engine.engine`, deployment record in `docs/GFX906.md` |
| Model | `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S`, context 262,144, images on | `engine.model`, `engine.context`, `engine.images` |
| KV / experts | `--kv int8 --kv-resident 32768`; arena 47,962 MiB / 24,576 slots (22,675 MiB / 12,288 on the primary card); free VRAM 2,186 MiB | `engine.kv`, `engine.kv_resident`, `engine.arena_mib`, `engine.expert_*`, `engine.vram_free_mib` |
| Speculation | `--spec 4`, `spec_min_p 0.50`, `mtp_max 0`, `lookup 0` | `engine` |
| Batching | 4 slots, `--batch-groups 2`, `slot_cache 1`, `pcie_frac 0.00`, `pool_workers 6`, `vram_elastic 0` | `engine.batch_slots`, `engine.batch_groups`, `engine.slot_cache`, `engine.pcie_frac` |
| Conversation cache | 8,192 MiB / 4 slots, free floor 16,384 MiB | `engine.conversation_cache_*` |
| Host | 2× Radeon Instinct MI50 32 GB (`gfx906`), Xeon E5-1650 v3, 12 threads | `hardware_static` |

## Requests: single, one at a time

The window holds 11 completed requests. Consecutive `time` stamps differ by at least the earlier request's
`duration_s` (38.8≥38.8, 10.1≥10, 14.4≥14.1, 23.0≥20.6, 37.5≥37.2, 43.3≥43.2, 9.7≥9.6, 2.7≥2.6, 4.2≥2.6,
108.3≥13.8 s). The payload does not label `time` as a start or a completion stamp, so this spacing is strong
evidence rather than proof of one-at-a-time traffic. The live counters are the direct evidence: `live.running: 1`,
`live.queued: 0`, `live.waiting: 0`, all four batch slots `idle` with `held_tokens 0`, and
`live.outside_slots: 1` — the in-flight request was served outside the batch slots. A 12th request was
reading its 618-token prompt at the snapshot instant and is not counted in `totals`.

These are therefore **single-request (one-at-a-time) rates**, not batch rates. They are not comparable with
the four-concurrent figures of the 2026-10-10/11 A/B.

## Per-request table (chronological, all values from the payload)

| Time (UTC) | Prompt | Reused | New | Output | `prompt_ms` | Prefill real tok/s | Prefill effective tok/s | Decode tok/s | Drafts accepted/offered | `duration_s` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 01:01:23.755 | 22,197 | 0 | 22,197 | 108 | 36,894.9 | **601.6** | 602 | 57.6 | 74/102 (72.5 %) | 38.8 |
| 01:02:02.596 | 24,460 | 22,071 | 2,389 | 101 | 7,742.7 | 308.5 | 3,159 | 44.9 | 59/103 (57.3 %) | 10.0 |
| 01:02:12.672 | 28,577 | 24,563 | 4,014 | 262 | 9,197.2 | 436.4 | 3,107 | 53.8 | 168/250 (67.2 %) | 14.1 |
| 01:02:27.059 | 29,397 | 28,839 | 558 | 886 | 3,023.4 | 184.6 | 9,723 | 50.5 | 528/786 (67.2 %) | 20.6 |
| 01:02:50.021 | 31,697 | 30,282 | 1,415 | 1,667 | 4,863.4 | 290.9 | 6,517 | 51.5 | 1017/1455 (69.9 %) | 37.2 |
| 01:03:27.550 | 33,803 | 33,363 | 440 | 2,450 | 2,761.8 | 159.3 | 12,239 | **60.6** | 1676/2074 (80.8 %) | 43.2 |
| 01:04:10.850 | 36,292 | 36,255 | 37 | 538 | 506.9 | 73.0 | 71,596 | 59.2 | 367/465 (78.9 %) | 9.6 |
| 01:04:20.508 | 36,869 | 36,832 | 37 | 127 | 506.8 | 73.0 | 72,749 | 60.6 | 87/108 (80.6 %) | 2.6 |
| 01:04:23.255 | 37,026 | 36,995 | 31 | 142 | 453.3 | 68.4 | **81,681** | 66.9 | 105/114 (92.1 %) | 2.6 |
| 01:04:27.414 | 37,265 | 37,170 | 95 | 728 | 1,630.3 | 58.3 | 22,858 | 60.1 | 501/629 (79.7 %) | 13.8 |
| 01:06:15.667 | 618 | 0 | 618 | 49 | 3,939.7 | 156.9 | 157 | **67.1** | 34/37 (91.9 %) | 4.7 |

All 11 requests ended `finish: "stop"` (the model stopped early; none hit a token cap), with
`reasoning_recoveries: 0` and `pcie_share: 0` / `ram_blobs: 0` / `file_blobs: 0` — no expert traffic left
VRAM in any request.

## How each rate was computed

| Rate | Formula | Note |
| --- | --- | --- |
| Prefill — **real** | `(prompt_tokens − reused) / (prompt_ms / 1000)` | compute throughput; reused K/V tokens excluded |
| Prefill — **effective** | `prompt_tokens / (prompt_ms / 1000)` | whole prompt per prompt-phase second; shows how fast a warm turn is accepted, **not** compute throughput |
| Decode | `output_tokens / (decode_ms / 1000)` | cross-checks the payload's own `decode_tok_s`: identical on all 11 requests (44.86–67.08 vs 44.9–67.1), so these `/metrics` per-request records are self-consistent. The HTTP `timings.prompt_ms` collapse recorded in `docs/GFX906.md` does not affect these counters. |
| Draft acceptance | `drafts_accepted / drafts_offered` | MTP `--spec 4` was active on every request |

Counters **not** used as rates: `live.tok_s` / `hardware.tok_s` (counts offered rows and draft tokens, see
the metric-accounting finding in `docs/GFX906.md`) and `live.prefill_tok_s_mean` /
`history.prefill_tok_s_mean` (a rolling value for the request still in its prompt phase at the snapshot:
57.7 → 156.6, no completed request behind it).

## Window summary

| Item | Value |
| --- | ---: |
| Window | 325.5 s (2026-10-11 01:01:09.103 → 01:06:34.583 UTC) |
| Requests / prompt tokens / reused | 11 / 318,201 / 286,370 (90.0 %) |
| New prompt tokens / output tokens | 31,831 / 7,058 |
| Prompt phase / decode phase | 71.52 s / 125.47 s |
| **Decode, per request** | **44.9–67.1 tok/s**, median 59.2, mean 57.5 |
| Decode, window aggregate | 7,058 / 125.47 s = **56.3 tok/s** (single stream at a time) |
| **Prefill — real, cold (no reuse)** | **601.6 tok/s** at 22,197 new tokens (36.895 s); **156.9 tok/s** at 618 new tokens (3.940 s — fixed prompt-phase overhead dominates a small prompt) |
| Prefill — real, warm turns (new tokens only) | 58.3–436.4 tok/s, median 159.3 (9 requests) |
| **Prefill — effective (warm turns)** | 3,107–81,681 tok/s, median 12,239; best turn: 37,026-token prompt, 36,995 reused, prompt phase 0.453 s |
| MTP drafts | 4,616 accepted / 6,123 offered = **75.4 %** |
| Conversation cache | 4 slots / 8,192 MiB: 3 parked (2.30 GiB), 3 parks, 0 restores, 0 evictions; 9 of 11 requests reused |
| VRAM | 61.26 / 63.97 GiB (card0 30.10, card1 31.15); engine free 2,186 MiB; `gpu_mem_used` history holds 61.26 GiB across all 60 samples (0.53 MiB spread) |
| Host RAM | 69.99 / 107.96 GiB at snapshot; history 68.71–70.06 GiB |
| GPU temperature | 35–46 °C across 60 samples; 39–46 °C in the 8 busy samples |
| GPU power (both cards summed) | idle 37–49 W, busy 133–226 W of the 450 W limit |
| Host CPU | 0.17–10.35 % |
| GPU utilization | `gpu_util` is the **mean of both cards**: 49.5–50 % in all 8 busy samples. The snapshot instant reads card0 0 % / card1 100 %, so this metric cannot separate an even split from one idle card. |

## Conditions and limits

- One read-only `/metrics` read of a production session carrying owner traffic. No client-side timing, no
  controlled rounds, no repeated runs, no warm-up/cold separation.
- Single-request traffic only: nothing here measures batch or `--batch-groups` behaviour, and the in-flight
  request ran outside the batch slots.
- Prompts are the owner's real conversations; their content, sampling settings and token counts were not
  controlled or recorded here. `hit_rate: 1` and the reuse counters are taken as reported.
- No accuracy, output-quality, MI60, full-262K-context or long-soak measurement. The window is 5.4 minutes.
- `gpu_pcie_rx_mb`, `disk_read_mb` and `disk_write_mb` are `null` on this host, so PCIe and disk traffic are
  unmeasured; the per-request `pcie_share: 0` / `ram_blobs: 0` / `file_blobs: 0` counters are the only
  expert-placement evidence.
- The engine session started 2026-10-11 01:01:09 UTC, so `totals` covers only this window; earlier sessions
  of the same binary are not included.
