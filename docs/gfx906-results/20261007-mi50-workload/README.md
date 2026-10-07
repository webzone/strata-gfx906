# T5810 long-context agent workload snapshot — 2026-10-07

One rolling `/metrics` payload from the production `0.0.0.0:8082` Strata service on the dual-MI50 T5810
server. `README.md` in the repository root distils it under *MI50 workload results*.

## Provenance

- Supplied by the owner as a captured monitor payload; the agent did not capture it from the endpoint.
- Authenticity check (read-only, 2026-10-07 03:15:15 UTC, `curl http://127.0.0.1:8082/metrics`): the live
  endpoint reports the same window start `totals.since = 1791340677.6021786`, the same engine block
  (`0.1.40`, `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S`, 24,576 / 12,288 expert slots, 47,962 MiB arena,
  conversation cache 8192 MiB / 4 slots / 16,384 MiB floor, 4 batch slots) and the same
  `hardware_static.gpu_name` string and `mem_total` per card (34,342,961,152 bytes). The counters had
  advanced to 50 requests, 5,563,981 prompt tokens, 5,436,075 reused, 57,990 output tokens and
  49,573 / 38,665 drafts. So the payload is the same window at an earlier point, not a different run.
- Service under test: text engine `build-text-rocm10/strata`, SHA256
  `bc1102ba03c9ee2379699f9fb5c3cf7ce717132e7065efc5ac141606c5996853` (upstream gfx906 path
  `STRATA_HIP_GFX906=ON`, source v0.1.40.1 / `82f46a8` plus engine fix `74583c6`), vision encoder
  `build-vision-rocm10/bin/strata-vision`, SHA256
  `709fc4d2c34abbede8b17cb93a4a9571c931338339d3848691c5b59f156d8452`. Both load ROCm 10 only
  (`10.0.0-gfx906+20260917140126`). Model: GSQ-RCO IQ3_S, 262,144 configured context, `--kv int8`,
  `--kv-resident 32768`, `--spec 4`, `--batch 4 --batch-groups 2`, `--layer-split 24`, `--pcie-frac 0`.
- Window in the payload: 2026-10-07 02:37:57.602 → 03:03:45.966 UTC (1,548.4 s). The service started at
  02:36 UTC, started by the owner, not by the agent.

## Files

| File | Content |
|------|---------|
| `metrics-snapshot.json` | The payload. Values verbatim. Re-indented for readability. The all-zero `history` sample arrays (60 idle samples of GPU util, memory, temperature, power, CPU, RAM, tok/s) are omitted; every other key and value is kept. |
| `evidence-manifest.sha256` | SHA256 of the files in this directory. |

## Figures derived from the payload

| Derived value | How it is computed |
|---------------|--------------------|
| Reuse rate 97.20% | `totals.reused / totals.prompt_tokens` = 3,596,208 / 3,699,741 |
| New prompt tokens 103,533 | `totals.prompt_tokens - totals.reused` |
| Output over the window 25.9 tok/s | `totals.output_tokens / (time - totals.since)` = 40,072 / 1,548.4 s |
| Decode occupancy 47.4% | `totals.decode_ms / 1000 / (time - totals.since)` = 733.5 s / 1,548.4 s |
| Draft acceptance 76.87% | `totals.drafts_accepted / totals.drafts_offered` = 26,529 / 34,513 |
| Sample reuse 99.86%, weighted decode 52.9 tok/s | sums over the 12 records in `requests` (the most recent of 38): 1,563,716 / 1,565,899 and 6,571 / 124.3 s |
| Conversation cache 4.60 GiB | `conversation_cache.bytes` = 4,939,329,976 |
| VRAM 61.35 / 63.97 GiB | `hardware.gpu_mem_used` / `hardware.gpu_mem_total`, GiB = 2^30 bytes |
| Host RAM 72.46 / 107.96 GiB | `hardware.ram_used` / `hardware.ram_total` |

## Limits

- One rolling window of owner agent traffic. Not a controlled benchmark, not a matched A/B, no client-side
  timing, no fixed prompt set.
- A new-token prefill rate is not derivable: `prompt_ms` also covers cache restore and admission work.
- No accuracy, output-quality, full-262K-context, MI60, or long-soak measurement.
- `hardware_static.gpu_name` says "MI50 16GB"; `mem_total` reports 31.98 GiB per card. The name string is a
  monitor label.
- The GPU utilization and tok/s history arrays in the payload are all zero because the capture happened when
  the service was idle. No load curve is available from this payload.
