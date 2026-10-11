# T5810 8082 text engine: recurring HSA page-fault crash in the conversation-snapshot copy — 2026-10-11

All times UTC. Machine: T5810, 2× MI50 (gfx906), deployed pair per the device record: text engine
`build-text-rocm10/strata` + fork HIP vision `709fc4d2…`, both ROCm 10.

**Verdict: this is a software bug in the engine's conversation-cache snapshot path (a rare
free/unmap-while-in-flight race), not a hardware fault.** It has now occurred twice with the same
signature (2026-10-08 on the v0.1.40.1 engine, 2026-10-11 on the v0.1.42 engine), both times after
heavy 130K–155K-token traffic with the conversation cache actively parking/evicting.

## The two occurrences (same signature)

| | 2026-10-08 08:27:41 | 2026-10-11 02:00:06 |
|---|---|---|
| Engine binary | `bc1102ba…` (v0.1.40.1) | `0550f3f2…` (v0.1.42) |
| Kernel page faults | 10 × `no-retry`, GPU0 `0000:07:00.0`, gfxhub0, **TCP (shader)** client, write (`RW: 0x1`, status `0x00841051`) | identical shape, status `0x00841051` |
| Host VA region | `0x5e0dd3b36000…` | `0x622590d04000…0x622590d1c000` (9 sparse pages, ~96 KiB) |
| Process | strata pid 1544046 | strata pid 7184 |
| Engine log | `HSA_STATUS_ERROR_MEMORY_FAULT` + **"checkpoint save: conversation snapshot running-state copy: an illegal memory access was encountered"** (engine-log line 4666) | `HSA_STATUS_ERROR_MEMORY_FAULT` (the detail line did not print; fault during the same post-request capture window) |
| Preceding churn | 155,969-token conversation parked moments before; evictions 6→7→8; two small requests parked in between | 137,069-token request finished; capture in flight; earlier lines show `evictions=41`, parked ~8 GiB |

Both runs had served hundreds of long requests (10-08 run: 67 requests, 20 park events, 0.93M parked
tokens; 10-11 run: 143 requests, 126 park events, 8.06M parked tokens). The fault rate is roughly two
crashes in four days of this workload; earlier runs with the same churn (e.g. 10-09: 313 park events)
did not crash — a timing race, not a deterministic fault.

Boot -2 (2026-10-02 → 10-05) has zero page faults.

## What the fault means

A shader-side **write** to a host-mapped region for which the GPU page tables no longer had a valid
mapping. Host RAM going bad does not invalidate GPU mappings (it corrupts data, or raises EDAC/MCE —
none present); VRAM is ECC (`sramecc+:xnack-`). A mapping that disappears is a software lifecycle
event: the buffer was freed/unmapped while a GPU write into it was still in flight. The engine hands
large host buffers to the GPU (fine-grain PCIe is forced), and the conversation-cache capture path
(conversation parking / KV checkpoint saves) is the code that writes GPU→host — both crashes sit
exactly in that path.

**Correction of an earlier claim in this directory's first version:** the "first fault in the kernel
journal, checked back 7 days" statement was wrong — `journalctl -k` shows the current boot only, so
the check had silently covered just the 2026-10-11 boot. Re-running it per boot (`journalctl -k -b -1`)
found the identical 2026-10-08 event. The bug is recurring, not one-off.

## Evidence (verbatim)

- `engine-log-tail.txt` — last 60 lines of the 2026-10-11 run from
  `logs/cachefix-rocm10-20261007/iq3-engine.log` (the crash lines are the file's final lines for that
  run, mtime 02:00:06).
- `kernel-page-faults.txt` — `journalctl -b -k` excerpt for the 2026-10-11 fault second.
- The 2026-10-08 counterpart is in the same append-only engine log (lines 4640–4667: the fault plus
  the "illegal memory access" detail line) and in boot -1's kernel journal (10 faults at 08:27:41,
  strata pid 1544046, VA `0x5e0dd3b36000`).

## Action taken (2026-10-11 01:55–02:15 round)

1. Evidence captured, then graceful stop (SIGTERM, 2 s, port free, both GPUs drained to 0 % VRAM).
2. Restart through the unchanged `./run-iq3-s.sh` (setsid, console to
   `logs/serve-console-20261011.log`): server PID 31772, vision PID 31784, engine PID 31817, ready
   after ~120 s (`2186 MiB of VRAM free with everything loaded`).
3. Acceptance: `/health` ok `loaded:true images:true`; real greedy request answered `STRATA OK`
   (`finish_reason: stop`, `cached_tokens 0`); `strata-current.sh` exit 0.
4. No config or code change: the fix belongs in the snapshot-capture path and needs a reproducing
   workload first. The deployed artifacts and `run-iq3-s.json` are byte-unchanged.

Side effect of the restart: the RAM-only conversation cache (parked conversations, KV checkpoints)
is empty; clients re-prefill.

## Resolution (2026-10-11 03:40–04:10 round)

The faulting buffer was pinned down as host memory: every fault address of each crash lies inside a single
2 MiB span of the process heap (10-08 `0x5e0dd3b36000`–`0x5e0dd3b49000`; 10-11
`0x622590d04000`–`0x622590d1c000`). The deployment runs no `--kv-grow`, so no runtime VMM unmap exists; a
mapping that can vanish under an in-flight write is a plain host allocation that was freed. With
`HSA_ENABLE_SDMA=0` every copy runs as a shader blit (the fault client is TCP): a blit touching a plain host
buffer goes through a device mapping dropped when the pages are released, so a free before the blit executes
faults the GPU on a write, and the sticky error surfaces at whichever call polls next — which is why the
10-08 log named a checkpoint-save copy and the 10-11 run died at a KV-streaming line while neither enqueued
the faulting write.

The fix is in the copy layer only (`src/core/conversation_copy.hpp`, via a private non-blocking stream per
caller): both endpoints are classified with `cudaPointerGetAttributes`; device and device-accessible host
endpoints keep the direct copy, a plain host endpoint stages through a 4 MiB pinned bounce buffer (per
thread), and host-to-host is a CPU `memmove`. No blit can reference pageable memory any more, so the fault
class is dead by construction for all conversation traffic. The exact racing free was not pinpointed (a
cross-thread release or a torn-down turn's buffer); the mechanism is removed regardless. Measured overhead:
one extra CPU copy on plain-host legs, ~0.2–0.3 s per 3.3 GiB park/restore, microseconds for the small
running-state copies.

Verification on the T5810 gfx906 HIP stack (ROCm 10.0, `build-text-rocm10-copyfix`, service untouched):
`conversation_copy_test` 165 checks across 9 endpoint pairs × 9 sizes (heap classifies `device=0
accessible=0`, pinned `0/1`, device `1/1`); `conversation_validation_test` 972 host-only checks;
`conversation_snapshot_test` 3901 checks through the new layer on a real GPU; the engine links clean.

**Deployed 2026-10-11 04:02–04:12 UTC during an idle window** (running 0, queued 0): the v0.1.42 binary
backed up as `build-text-rocm10/strata.v0142-0550f3f2.bak`, the verified build installed to
`build-text-rocm10/strata` (ff24e50b980fa555…), restart via the unchanged `./run-iq3-s.sh`, engine PID
67141. Acceptance: `/health` ok `loaded:true`; real generation "STRATA OK" (`finish_reason: stop`); a
park/restore round-trip through the new copy layer ("conversation cache: parked 104 tokens … parked=3")
with the engine alive; `strata-current.sh` exit 0. `run-iq3-s.json` stayed byte-unchanged. Open: a soak
under the owner's normal 130K+ token traffic (the two crashes spanned four days of that pattern). Upstream
carries the same copy layer, so the change applies there verbatim.

## Original next steps (superseded by the resolution above)

- ~~Reproduce under control: drive several 130K+ token conversations with frequent switches (to force
  parking + evictions) on a maintenance window; the two crashes both followed that pattern.~~
  Not needed for the mechanism fix; a soak under the repaired engine remains open below.
- ~~Audit the capture path for frees relative to stream order…~~ Done as a mechanism fix in the copy layer
  instead of a per-call-site sync: every plain-host blit endpoint is gone.
- Note the fix must go upstream too: the fault reproduces on both v0.1.40.1 (`bc1102ba…`) and
  v0.1.42 (`0550f3f2…`), i.e. it is in inherited upstream code, not a fork-only path. Still true; the fix
  applies verbatim upstream.
- Available mitigation (not applied — it costs a full re-prefill of long contexts on every switch,
  ~5 min at the measured ~400 tok/s for 130K tokens): `--conversation-cache-mib 0`. Now unnecessary; the
  fixed copy layer keeps the cache.
- Open: a soak of the repaired engine under the owner's normal 130K+ token traffic, confirming no
  recurrence over the weeks-long horizon the two crashes spanned.
