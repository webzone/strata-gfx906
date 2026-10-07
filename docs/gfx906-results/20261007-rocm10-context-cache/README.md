# ROCm 10 context-cache fix — independent audit receipt (2026-10-07, final)

Role: read-only audit of the C++ fix `74583c6` and the serve slot/promotion contract, plus
small-auditable-evidence collection. No source changes, no service start/stop by this agent; the
8095 acceptance server was root-owned, the acceptance omp sent its requests, and this audit only
read the captures. Production 8082 remained **stopped** the whole time. Selected large raw captures
(token JSONs, the long HTTP request/SSE pair) are hash-pinned in `evidence-manifest.sha256`;
the image fixture, build log, and mode-0600 test config are retained privately without hashes.
Small stdout logs, raw headers, the vision response, and the root promotion/postflight receipts are
copied under `remote/`.

## Build identity

See `build-identity.json` (verbatim). Summary:

- Merge `07d8a1c` (second parent `82f46a8` = upstream v0.1.40.1); engine fix `74583c6`
  (`src/program/generate.cpp`, `src/core/verify.cpp`, `include/strata/core/verify.hpp`).
- Text candidate `build-text-rocm10-cachefix/strata`, SHA256
  `bc1102ba03c9ee2379699f9fb5c3cf7ce717132e7065efc5ac141606c5996853`; all HIP libraries resolve to
  `/opt/rocm-10.0/core-10.0`. Vision reuses the unchanged ROCm 10 binary
  `build-vision-rocm10/bin/strata-vision` (`709fc4d2…`). llama.cpp pinned `3cf03257`.
- Suites as run: 455 web/Linux server tests — OK with 5 skips; 332 setup tests — OK with 1 skip;
  44 gfx906-host tests — OK with none skipped. Full compiler-warning logs are not committed.

## Root causes and the fix

Three independent causes forced full re-prefill between turns (`owner-report.txt`: 65,846-token
prompt, 301-token tool call, 618-token request cancelled at 611 prompt tokens with 0 outputs, then
the next turn re-prefilled 66,274 tokens from zero — consistent with cause 3 below; the raw log
proves the between-turn loss but not the exact internal path):

1. **Pipeline idle-state clobber.** Pre-fix, each pipeline window launched all `GS` rows of a group;
   idle slots were sent as pad rows whose commit wrote their session state, destroying any cached
   conversation in them.
2. **Unconditionally uncacheable completion.** Every `BDONE` set `sl.cached = false`
   ("a pipelined slot is not reused as a cache"), so a finished turn's K/V was never reused.
3. **Solo main-session overwrite with the RAM cache disabled.** On the solo path the next request
   overwrote the main session between turns; with `conversation_cache_mib = 0` (T5810's state) there
   was no RAM snapshot to restore, so the whole history was re-prefilled. The 65K parking test below
   reproduces this loss explicitly in a controlled solo run.

Fix: `PGroup` freezes `size` + `rows[]` (active slots only) at launch; `Verifier::batch_launch(base,
rows, S, …)` runs and commits exactly those rows; commit state is indexed by the real slot id
(`h_commitb_[rows[first] * CB]`), and each group keeps its fixed, disjoint mapped-host hand-off
region `[pick*GS, pick*GS+GS)` (unused rows inside it are simply never written). Captured graphs are
keyed by the full row layout (`bkey` = hbase + every row id + doorbell variant; `batch_graph_limit_`
LRU-bounds layouts). On any `BDONE` (stop/cancel/length), text slots now keep
`sl.cached = prompt_cache > 0 && !sl.img` — both the solo-batch path (`generate.cpp:7580`) and the
pipeline path (`:7679`). `BT`/`BDONE` lines carry real slot ids. Four slots / two groups unchanged.
Cause 3 is covered by the existing opt-in RAM conversation cache (park/restore), which the
acceptance enables.

## Host conversation cache: enablement and eviction boundary

- Engine flags: `--prompt-cache N` (must be > 0), `--conversation-cache-mib 8192`,
  `--conversation-cache-slots 4`, `--conversation-cache-min-free-mib 16384` (16 GiB RAM floor).
  With `--conversation-cache-mib > 0`, serve mode fatally requires `--prompt-cache > 0` and
  `--conversation-cache-slots > 0` (`generate.cpp:1860`).
- Budget behavior: parking estimates the snapshot; if it exceeds the budget, `make_room` evicts
  oldest-first; physical-RAM admission is re-checked against the floor before and after capture.
  Slots are a bounded first tier (evicted LRU by the frontend when all slots hold different
  conversations; the outgoing branch is then parked to the 8 GiB RAM tier). **Eviction is the
  explicit boundary — no slot or conversation has an unlimited permanent cache.**

## park_current timing on a cancelled short prefill

`park_current` has exactly one call site: request start (`generate.cpp:8295`), before the outgoing
branch is overwritten (checkpoint rewind / incoming restore / slot-source copy), guarded by
`(!from_live || incoming || slot_source >= 0)` and `live_ok`. A request cancelled mid-prefill sets
`cancelled` (`:8983`), refills lent slots, and does **not** swap `live` (`:9818`: "a prompt stopped
halfway leaves the session somewhere between two chunks") — nothing is parked at cancel time by
design, and the invalid partial state is never cached. Salvage is the checkpoints taken while
reading (turn/message boundaries, `prompt_cache_every`); they stay valid and prefix-match the next
request (`:8239`). Observed sequence in the parking test below: completed A was parked before B's
admission; the cancelled B produced no cached snapshot; A was restored from its previously saved
snapshot.

## Frontend slot contract (serve/server.py)

`pick_slot` (`:1294`): among free slots, the longest held proper prefix of the new prompt → else an
empty slot → else the holding slot used longest ago. `YIELDS_MAX = 2` (`:1013`) bounds how often one
request's prompt read gives way. `slot_held[b]` is updated at `BDONE`
(`prompt + out[gen0:-1]`, exactly the engine's fed tokens) and on yield, cleared at admission. The
engine independently re-validates any slot source (`!sl.active && sl.cached && sl.cvec match` and
token-exact `starts_with`), so a stale frontend hint cannot corrupt state; a failed
`copy_from_slot` falls back to a full read from zero.

## Validated scope

Model split: the CLI checks below ran **IQ2_XS** (acceptance model); the 8095 HTTP/vision/API
candidate ran **IQ3_S** (root-owned server; the acceptance omp sent the requests, this audit only
read the captures).

- **Pipeline parity (IQ2_XS, CLI) — PASS.** `tools/batch_interleave_test.py --batch 4 --long 32768`
  (actual context 31,908) `--max-new 200 --extra "--batch-groups 2 --trim-stage-weights
  --kv-resident 32768 --pcie-frac 0 --adapt-every 1000000 --no-prefill-borrow"`: exit 0; A/B/C and
  D/E and checkpoint F output tokens identical to the solo run. Long-lived slot 2 survived three
  slot-3 bursts and reused 31,961 tokens with a next-turn output identical to solo; slot 1 stopped
  at 51 tokens, rode out three slot-0 bursts, then a GEN reused 75 + 1 fresh token with a 200-token
  output identical to solo.
- **BYIELD — SKIP, honestly.** Upstream pipeline `read_part` does not support BYIELD. The first run
  that required BYIELD failed and is retained verbatim
  (`remote/iq2-pipeline-unsupported-byield.stdout.log`); the re-run skips that check. It is **not**
  claimed as passing.
- **65K parking (IQ2_XS, native background tool) — PASS** (raw `remote/iq2-parking.*`). Sequence:
  completed A (65,902-token prompt) was parked **before** B's admission; B was the 618-token short
  request cancelled at 611 prompt tokens with 0 outputs and was **not** cached; the next request
  restored A's saved snapshot — 65,986 tokens (live) in 484.5 ms — and the 66,011-token follow-up
  reused 65,986 with 25 fresh tokens in 2.46 s (vs 1.96 s live reference), output 80/80 tokens
  identical. `SNAPSHOT_VERIFY draft=ce89238121a2ad38 cells=65988 mode=2 source=ram resident=32848`;
  the re-park reused 1,007,179,776 bytes of retained K/V. Limitation: the native tool does not print
  a numeric exit code — script return-0 is inferred from its final dump and ok-conditions; the raw
  stdout/engine lines quoted here are direct observation.
- **HTTP burst (IQ3_S, 8095 candidate) — PASS** (raw `remote/http-bursts.stdout.log`,
  `remote/http-server.stdout.log`). Initial 36,048-token full prefill occurred once; later prompts
  resumed with one fresh token each (36069=36068+1, 36074=36073+1, 36087=36086+1, 36093=36092+1,
  36110=36109+1) — no 36K re-prefill after bursts. Summary `ok:true`, 87.292 s, completion 1,024,
  742 chunks. As with the parking run, the native tool prints no numeric exit code — return-0 is
  inferred from `ok:true` and the final dump; the raw captures are direct observation.
- **Vision smoke (IQ3_S, 8095 candidate) — PASS**: test-client SSH Python subprocess rc 0 (the
  engine itself stayed running until root's shutdown), HTTP 200, `Access-Control-Allow-Origin: *`,
  exact headline "MEN WALK ON MOON", prompt 332 / completion 7
  (`remote/vision-smoke.response.json`, `remote/vision-smoke.http.txt`).
- **API/CORS (IQ3_S, 8095 candidate) — PASS** (`remote/api-validation/`): OPTIONS preflights from an
  arbitrary HTTPS origin and from `Origin: null` both return **HTTP 204 No Content** with
  `Access-Control-Allow-Origin: *`; `/health` reports `images: true`; `/v1/status` serving 4.
  An explicit origin list restricts, `[]` disables, and changes need a restart.

## Promotion (root, after the acceptance completed)

`remote/promotion.json` (root's receipt) and `remote/postflight.json` (resource/port proof) record:

- The 8095 server exited gracefully with rc 0 after SIGINT; no 8082/8095 listeners; both GPUs 0%
  use and 0% VRAM allocated; RAM 104 GiB available; root 22 GiB free; `/data` 107 GiB; Docker empty.
- The exact validated candidate `bc1102ba…` was atomically promoted into
  `build-text-rocm10/strata`; the vision binary `709fc4d2…` is unchanged.
- Both IQ3 and IQ2 configs (mode 0600) now select the formal ROCm 10 engine with
  `cors_origins: ["*"]`, 8192 MiB cache / 4 slots / 16384 MiB floor, `--kv-resident 32768`
  and trim-stage weights on IQ2, `LD_LIBRARY_PATH` on the ROCm 10 core lib and `ROCM_PATH` on the
  ROCm 10 core. Only `exe`, `log`, env, CORS, and args changed; old env values and old arg prefixes
  are preserved exactly (IQ3 appended only the three cache flag pairs; IQ2 the same plus
  kv-resident/trim). Model, sampling, GPUs, split-24, host `0.0.0.0:8082`, and the owner's no-API-key
  preference are unchanged. IQ3 keeps parallel 4 + vision; IQ2 stays serial 1 without vision.
- Config backups: remote `promotion-backups/`. **8082 was NOT started** — startup remains the
  owner's manual step through the existing launcher with its port/lock guard.

## Status

No audit findings remain open. Remaining step: root commits and pushes these docs.
