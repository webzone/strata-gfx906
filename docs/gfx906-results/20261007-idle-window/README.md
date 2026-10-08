# T5810 idle engine window snapshot — 2026-10-07 21:53:56 → 21:57:42 UTC

One `/metrics` payload from the `0.0.0.0:8082` Strata service on the dual-MI50 T5810 server. It covers
**225.9 s of an engine session that served 0 requests**, so it carries resource state and engine
configuration, **no prefill rate and no decode rate**. `README.md` in the repository root distils it
under *MI50 workload results*.

## Provenance

- Supplied by the owner as a captured monitor payload; the agent did not capture it at capture time.
- Window inside the payload: `totals.since = 1791410036.79785` → `time = 1791410262.72242`, i.e.
  2026-10-07 21:53:56.798 → 21:57:42.722 UTC (225.925 s). `totals` and `conversation_cache` counters
  are all zero, so this is the start of an engine session with no traffic yet.
- Currency check (read-only, 2026-10-08 22:38:39 UTC and 22:45:34 UTC,
  `curl http://127.0.0.1:8082/metrics` on the
  machine): the live endpoint reports `totals.since = 1791486926.334384` (2026-10-08 19:15:26 UTC, the
  restart recorded in [GFX906.md](../../GFX906.md#metrics-cors-deployment-on-t5810-2026-10-08-1915-utc)),
  with 113 requests, 5,757,184 prompt tokens, 5,530,572 reused, 62,652 output tokens and 53,758 / 40,208
  drafts offered/accepted. The `engine` block is unchanged (`0.1.40`, same context, KV, expert arena,
  conversation cache and batch-slot values), and at 22:45:34 UTC `hardware_static.gpu_name` reads
  "Radeon Instinct MI50 32GB + Radeon Instinct MI50 32GB" with the driver-string note — the label fix
  post-dates the supplied payload. **The supplied payload is therefore a superseded engine
  session, not the session serving 8082 at the time of the check.**
- Service identity behind the payload is the deployment record, not the payload itself: text engine
  `build-text-rocm10/strata` SHA256 `bc1102ba03c9ee2379699f9fb5c3cf7ce717132e7065efc5ac141606c5996853`,
  vision encoder `build-vision-rocm10/bin/strata-vision` SHA256
  `709fc4d2c34abbede8b17cb93a4a9571c931338339d3848691c5b59f156d8452`, both loading ROCm 10 only
  (`10.0.0-gfx906+20260917140126`). Model GSQ-RCO
  IQ3_S, 262,144 configured context, `--kv int8`, `--kv-resident 32768`, `--spec 4`,
  `--batch 4 --batch-groups 2`, `--layer-split 24`, `--pcie-frac 0`.
- The payload's whole `engine` block and whole `hardware_static` block are byte-identical to the
  `engine` and `hardware_static` blocks in
  [the 2026-10-07 production workload snapshot](../20261007-mi50-workload/metrics-snapshot.json): same
  model string, 262,144 context, `int8` KV / 32,768 resident cells, 24,576 / 12,288 expert slots,
  47,962 / 22,675 MiB expert cache, `spec 4`, `mtp_max 0`, `lookup 0`, `pool_workers 6`,
  `pcie_frac 0.00`, `spec_min_p 0.50`, conversation cache 8,192 MiB / 4 slots / 16,384 MiB floor,
  `batch_slots 4`, `slot_cache 1`, engine `0.1.40`.

## Files

| File | Content |
|------|---------|
| `metrics-snapshot.json` | The payload. Every key and value verbatim, including all six `history` sample arrays (60 samples each). Re-indented for readability. |
| `evidence-manifest.sha256` | SHA256 of the files in this directory. |

## Figures derived from the payload

| Derived value | How it is computed |
|---------------|--------------------|
| Window 225.9 s | `time - totals.since` = 1791410262.72242 − 1791410036.79785 |
| Requests in the window 0 | `totals.requests`, `live.running = 0`, `live.queued = 0`, `live.waiting = 0`, `requests = []`, `requests_kept = 0` |
| VRAM 61.16 / 63.97 GiB (95.6 %) | `hardware.gpu_mem_used` / `gpu_mem_total`, GiB = 2^30 bytes |
| VRAM per card 30.06 / 31.98 GiB (94.0 %) and 31.10 / 31.98 GiB (97.2 %) | `hardware.gpus[0]` and `[1]` |
| Engine-reported free VRAM 2,186 MiB (2.13 GiB) | `engine.vram_free_mib` |
| Expert arena 47,962 MiB (46.84 GiB), primary card 22,675 MiB (22.14 GiB) | `engine.arena_mib`, `engine.expert_cache_primary_mib` |
| Host RAM 67.56 / 107.96 GiB (62.6 %) | `hardware.ram_used` / `ram_total` |
| Host RAM over the window 67.55–67.58 GiB (31 MiB spread) | min/max of `history.ram_used` (60 samples) |
| GPU temperature 44–46 °C, mean 45.4 °C | `history.gpu_temp` (60 samples) |
| GPU package power 41–48 W, mean 42.0 W = 9.3 % of the 450 W limit | `history.gpu_power`, `hardware.gpu_power_limit` |
| GPU utilization 0 % in all 60 samples | `history.gpu_util` |
| Host CPU 0.25–2.08 %, mean 1.51 % | `history.cpu` |
| Conversation cache 0 parked, 0 bytes, 0 parks, 0 restores, 0 evictions | `conversation_cache` |
| tok/s history 0 in all 60 samples | `history.tok_s`, `history.prefill_tok_s_mean` |

## Limits

- **No rate of any kind is derivable from this payload.** `totals.requests = 0`, `totals.prompt_tokens = 0`,
  `totals.output_tokens = 0`, `totals.prompt_ms = 0`, `totals.decode_ms = 0`,
  `drafts_offered = drafts_accepted = 0`, and every `history.tok_s` and `history.prefill_tok_s_mean`
  sample is 0. It is an idle resource snapshot, not a workload result.
- One engine session of 225.9 s at idle. It says nothing about behaviour under load, and it does not
  supersede the measured rates in
  [the 2026-10-07 single-shot vs 4-concurrent probe](../20261007-single-vs-4concurrent/README.md).
- The session is superseded: the live endpoint at 2026-10-08 22:38:39 UTC reports a later session start
  (2026-10-08 19:15:26 UTC) with 113 requests served.
- `hardware_static.gpu_name` says "MI50 16GB"; `mem_total` reports 31.98 GiB per card. The name string is
  a driver label, corrected in a later service start — see
  [the monitor note](../../GFX906.md#monitor-a-32-gb-mi50-shown-as-mi50-16gb-2026-10-08).
- `hardware_static.psutil` is `false`, and `gpu_pcie_rx_mb`, `gpu_pcie_tx_mb`, `disk_read_mb` and
  `disk_write_mb` are `null` in both the snapshot and the history arrays: PCIe and disk counters are not
  available on this host.
- No accuracy, output-quality, full-262K-context, MI60 or long-soak measurement.
