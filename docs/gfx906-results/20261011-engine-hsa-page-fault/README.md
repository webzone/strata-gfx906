# T5810 8082 text engine: one-off HSA page-fault crash and restart — 2026-10-11

All times UTC. Machine: T5810, 2× MI50 (gfx906), deployed pair per the device record: text engine
`build-text-rocm10/strata` (v0.1.42, SHA256 `0550f3f2…`) + fork HIP vision `709fc4d2…`, both ROCm 10.

## What happened

- 2026-10-11 00:53 — owner rebooted after DIMM maintenance (pulled DIMM7, installed DIMM6).
- 00:59:03 — owner started the 8082 service through `./run-iq3-s.sh` (server PID 7055, vision 7107,
  text engine 7184). Config unchanged: `run-iq3-s.json` with `--batch-groups 2`, conversation cache
  8192 MiB / 4 slots / min-free 16384 MiB, `--batch 4`, layer split 24, MTP spec 4.
- ~01:00–02:00 — real traffic: several 96K–137K-token conversations (drafts accepted, KV streaming
  with part of block reads served from RAM, 6 KV checkpoints on the last request).
- 02:00:06 — the text engine died. The engine log ends:

  ```
  strata serve: prompt 137069 tokens = 136927 reused + 142 read in 1963 ms (72.3 tok/s), 2094 generated in 38937 ms (53.8 tok/s), drafts accepted 1385 of 1802, 6 checkpoints
  strata serve: decode expert cache hit rate: 100.0% (1205760 hits / 1205760 lookups)
  strata serve: KV streaming: 96.11% of 32251866 block reads hit VRAM, 5056.5 MiB read from RAM
  Warning: Queue error - HSA_STATUS_ERROR_MEMORY_FAULT
  rocBLAS error during freeing of allocated memory in handle destructor: rocblas_status_internal_error
  ```

  The kernel logged ten `[gfxhub0] no-retry page fault` events in the same second, all on GPU0
  (`0000:07:00.0`), all `Process strata pid 7184`, all `IH client 0x1b (UTCL2)`, faulty client
  `TCP (0x8)` (shader side), `PERMISSION_FAULTS: 0x5`, `RW: 0x1` (write), at host-mapped addresses
  `0x0000622590d04000` … `0x0000622590d1c000` (nine sparse 4 KiB pages over a ~96 KiB span of one
  host region). `MAPPING_ERROR: 0x0`.
- The server survived with `"loaded": false` and an empty `/v1/models`; the dead engine stayed a
  zombie child (PID 7184). The engine had run 61 minutes since load (started 00:58:52, loaded ~01:00).

## Evidence (verbatim)

- `engine-log-tail.txt` — last 60 lines of
  `logs/cachefix-rocm10-20261007/iq3-engine.log` on the machine (append-only across runs; the crash
  lines are the file's final lines, mtime 02:00:06).
- `kernel-page-faults.txt` — `journalctl -b -k` excerpt for the fault second.

## Assessment (bounded)

- First `no-retry page fault` in the available kernel journal (checked across boots back 7 days).
  The same binary served the same heavy long-conversation workload for days (2026-10-09/10) without it.
- No MCE / EDAC / `hardware error` entries in this boot's journal; host RAM clean at crash time.
- A GPU write permission fault to a host VA range means the shader no longer had a valid mapping for
  it — consistent with a host buffer freed/unmapped while an in-flight shader write still referenced
  it, i.e. a rare use-after-unmap race in the engine's host-buffer lifecycle (the post-request
  housekeeping paths are the GPU→host write paths: conversation parking, KV checkpoint saves), or a
  one-off driver/hardware event after the DIMM rearrangement. The evidence does not distinguish these,
  so no code fix and no config change was made.

## Action taken (2026-10-11 01:55–02:15 round)

1. Evidence captured (the two files above) before touching the service.
2. Graceful stop: `SIGTERM` to server 7055; all three processes gone in 2 s, port 8082 free, both
   GPUs drained to 0 % VRAM, zombie reaped.
3. Restart through the unchanged `./run-iq3-s.sh` (setsid, console to
   `logs/serve-console-20261011.log`): server PID 31772, vision PID 31784, engine PID 31817.
   Load to ready ~120 s; `2186 MiB of VRAM free with everything loaded` (matches the 2026-10-11
   A/B record).
4. Acceptance: `/health` → `status ok, loaded true, images true, api_key false, max_context 262144`;
   real greedy request answered `STRATA OK` (`finish_reason: stop`, 4 completion tokens,
   `cached_tokens 0`); `mi50-t5810/scripts/strata-current.sh` → exit 0 (engine `0550f3f2…`, vision
   `709fc4d2…`, only `/opt/rocm-10.0` runtime libs).

Side effect of the restart: the RAM-only conversation cache (parked conversations, KV checkpoints in
RAM) is empty; clients re-prefill.

## If it recurs

- Capture `journalctl -k` around the event plus the engine log, and keep the workload notes.
- Reproduce with instrumentation (`AMD_LOG_LEVEL`, core dumps) before touching code; the audit
  targets are the conversation-cache eviction/parking paths and the KV-checkpoint save path
  (host buffers freed relative to stream order).
- `--conversation-cache-mib 0` is the available mitigation (drops parked-conversation reuse); it was
  deliberately not applied because it changes the 2026-10-11 A/B-winner config for an unconfirmed cause.
