# T5810 IQ3_S first live observation — monitor snapshot, no tests

First live-serving record of the **GSQ-RCO IQ3_S** pack on the dual-MI50 T5810 server, captured from the
built-in status/monitor endpoint at **2026-10-02 08:19:40 UTC**. This is an operational snapshot of real
sequential agentic traffic, **not a controlled benchmark**: no test harness, no fixed prompts, no
token-parity or output-quality evaluation. Provenance of the model switch:
[IQ3_S preparation](../20261002-iq3-s-preparation/README.md) and
[in-place v0.1.34 deployment](../20261002-in-place-v0.1.34/README.md).

`monitor-snapshot.json` in this directory is the verbatim endpoint output. All figures below are derived
from its fields; nothing else was run or captured.

## Captured state

- Engine `0.1.34` (same real-gfx906 binary as the in-place deployment, SHA256 `d8a59558877074f6…`),
  model `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S`, context 262,144, int8 KV, `spec: 4`, `spec_min_p: 0.50`
  (the MTP/speculative settings were retained from the IQ2_XS config at preparation), `pcie_frac: 0.00`.
- Experts: 24,072 slots / 46,918 MiB arena in RAM (`arena_mib: 47962`); primary GPU 12,288 slots /
  22,675 MiB resident (`expert_slots_primary` / `expert_cache_primary_mib`).
- Hardware block: 2× MI50 (32 GiB each), 41–42 °C and 20–22 W per card, idle at capture time; the 60-sample
  history covers only the ~8 idle minutes after the last recorded request (2026-10-02 08:11:14 UTC).

## Derived figures (from the snapshot fields)

| Figure | Value | Source fields |
| --- | ---: | --- |
| Lifetime decode mean | 39.25 tok/s | `totals.output_tokens 158,099` / `totals.decode_ms 4,028,305.6` |
| Recent 12-request decode | 37.0–47.0 tok/s, mean 40.1 | per-request `output_tokens` / `decode_ms` (5,422 tok / 135,220.2 ms) |
| Lifetime prompt/KV reuse | 97.0% | `reused 23,327,626` / `prompt_tokens 24,040,403` |
| Requests / window | 186 over 14,040.8 s (~3.9 h) | `totals.since` (04:25:40 UTC), `time`, `requests_kept` |
| Recent contexts | 167,225–173,532 prompt tokens | `requests[].prompt_tokens` |
| Recent turns added | 40–1,029 new tokens per request | `prompt_tokens − reused` |
| Prompt time (uncached tokens only) | 0.785–4.737 s | `requests[].prompt_ms` |
| Effective new-token prefill | 446.7 tok/s lifetime (712,777 new of 24.04M); 51–217 tok/s per recent request | `(prompt_tokens − reused) / prompt_ms` |
| Expert cache hit | 98.6–99.4% | `requests[].hit_rate` (`hits/lookups`, share of experts served from VRAM — not prompt reuse) |
| Free VRAM | 2,876 MiB at startup (`vram_free_mib`); 2,489 + 403 MiB per card at capture | `engine`, `hardware.gpus` |
| RAM used | 55.66 GB (51.84 GiB) of 115.93 GB | `hardware.ram_used` / `ram_total` |

## Comparison with the deployed v0.1.31 IQ2_XS observation

Presented side by side in the repository README ("Recent MI50 workload results"). Summary: lifetime decode
mean 39.25 vs 42.39 tok/s (−7.4%) at roughly twice the context depth (167K–174K vs 83K–100K prompt tokens)
and a ~42% larger expert arena (47,962 vs 33,812 MiB); lifetime reuse 97.0% vs 96.9%; free VRAM 2,876 vs
11,012 MiB; RAM +~15 GB. The windows differ in engine version, workload and cache state, so the decode delta
cannot be attributed to the quantization without a matched A/B.

## Limitations

- Single snapshot taken while idle; serving CPU/PCIe/disk counters were not captured (`psutil: false`,
  `gpu_pcie_rx/tx`, `disk_*` null).
- `mtp_max: 0` means the draft count is uncapped, not that MTP is off; whether the drafter loaded is not
  visible in the snapshot.
- No token-parity check, no fresh-prefill benchmark, no long-soak stability claim.
- Headroom at capture: GPU 1 held 403 MiB free and total free VRAM sat ~316 MiB above the 2,560 MiB
  conversation-cache floor (`conversation_cache_min_free_mib`) at 167K–174K-token contexts; conversations
  growing toward 262K need headroom re-checked on this quant.
