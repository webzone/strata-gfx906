# gfx906 prefill optimization — active implementation contract

This work continues on `gfx906-prefill-opt`, initially `e7b1d0d`, not the owner deployment branch.
The user's objective is to implement the agreed gfx906 prefill plan end to end. Historical 2026-10-03
33K IQ2_XS results remain historical; they are not evidence for larger inputs, IQ3_S or later source.

## Required work and completion audit

- [x] Implement a reproducible, serial, pipe-only validation runner with private immutable evidence,
  complete raw stdin/stdout/stderr, exact tokenizer-measured inputs, version/model identity checks,
  deployment integrity checks, owner launch lock, busy-card/listener/RAM/disk admission and bounded
  own-process shutdown/deferred-VRAM cleanup. No service control or deployment overwrite.
- [x] Validate control 2048 versus MTP-only 3072/4096, experimental QSA OFF, on pinned IQ2_XS then IQ3_S:
  code, Chinese long-form and multi-turn inputs at 64K/128K, then 192K; repeat key comparisons at least
  three times. Require zero reuse for cold samples. Record generated IDs, real prefill/TTFT/decode,
  acceptance/rejection, expert streaming/refill, cache/buffer pressure and memory peaks. Disclose any
  output differences, failures or incomplete coverage rather than relabelling them as passes.
- [x] Inspect real expert group distributions and quantify format-specific MMQ/FP16 costs. Implement
  and independently numerically test gfx906 expert-matrix improvements justified by measurements.
  Do not force unsupported formats through MMQ, mutate the pinned dependency, change precision
  blindly or report resident microbenchmark timings as model throughput.
- [x] Re-measure stage balance using the improved MTP path. Test nearby layer splits only where the
  measured critical path justifies them, with each device owning its caches and mapped aliases,
  producer synchronization, no P2P/tensor parallelism and no competing GPU workload.
- [x] Diagnose the reproducible QSA-on/4096 token-42 divergence with real operator/layer/logit evidence,
  independent numeric reference and isolated controls. Resolve an implementation error if found;
  otherwise establish and document the numeric mechanism/limits, retaining strict tests and opt-in.
  Never loosen an oracle threshold to conceal a discrepancy.
- [x] Run relevant CPU/GPU/regression/model gates on final application source, preserve raw evidence,
  document measured scope/limits in `docs/GFX906.md`, commit related changes and push the dedicated
  branch without force. Audit all rows against authoritative artifacts before goal completion.

## Safety and scope

MI60, full 262K/long-soak quality, tensor parallelism, system tuning and RAM conversation parking are
not silently claimed by this work. Deployment is a separate owner-confirmed operation, not authorized
by the optimization objective. Fresh resource checks precede builds/tests; an occupied owner lock or
active GPU/API workload halts admission rather than stopping it. Keep at least 4 GiB disk reserve,
use existing weights/native-from-GGUF paths and this repository's `.venv`, and preserve original
binary/config/launcher hashes. `STRATA_EXPERIMENTAL_GFX906` and new computation paths remain opt-in.

## Initial admission snapshot (historical)

2026-10-03 04:07 UTC: local branch was clean at `e7b1d0d`; remote owner checkout remained `c883c4f`.
The isolated application was `49285ae` with SHA256
`05b646dd91bef728a9b31068d1eebec35001c9a050d3bbc68ac58b06687f78ec`.
Both MI50s had 0% utilization, 10866688 B VRAM use and no KFD PIDs; no 8082 listener/owner lock was
observed. Available RAM was 107772 MiB. These are preflight observations, not permanent availability.
Full private preflight is `logs/gfx906-prefill-goal-20261003T040756Z/resources-initial.txt` locally.
At that initial snapshot, all completion rows above were unverified. The final requirement audit below
supersedes that pending status without relabelling any historical source or benchmark.

## Continued implementation journal

- Runner `dc0a1b4` was committed/pushed after 169 setup / 64 gfx906 Python tests (including seven
  new runner tests). Strict standalone admission-policy compilation also passed.
- Its bounded **4096-token IQ2_XS pilot**, compiled application still **49285ae**, completed two
  isolated arms, 2048 control and 4096+MTP, each code/Chinese/multi-turn: six zero-reuse requests,
  clean child exit 0, all generated IDs equal, complete raw protocol and one-second sampled peaks.
  Deployment hashes matched before/after. This is one repetition and a short pilot, NOT the long
  matrix. 2048 control measured 8528.1/8323.6/8432.5 ms; 4096+MTP measured 9646.6/9499.7/9548.1 ms,
  so the larger chunk was slower here; pipeline fill/drain makes short-input extrapolation unsafe.
  Private evidence is `logs/gfx906-prefill-goal-20261003T040756Z/pilot-4096` on T5810. The runner's
  original whitespace-separated IDs were accepted/count-verified by the actual engine; subsequent
  source uses the frontend's canonical comma syntax and adds a frontend-contract regression.
- Continued source adds binary/model immutability checks, complete cold-profile/MTP-path gates,
  per-layer row distributions and phase costs, opt-in grouped gfx906 MMQ J selection, and a fresh-
  fixture GPU gate with full control equality plus sampled CPU double oracle. Current CPU checks:
  66 gfx906 Python tests, 44 admission-policy checks with `-Wall -Wextra -Werror`; no GPU results yet
  at that point for these expert changes.
- Isolated **cbdc94c** HIP application and all requested probes linked successfully. The first private
  build driver's post-link hash audit failed (`digest(str)`), retained as exit 1; corrected explicit
  recheck/build and dependency/deployment audit returned 0. Candidate SHA256:
  `167b38cc5c57b0984f8653da8e8161a5e9bc00a7a40cafb7bff0d706f8559c8b`.
- Both MI50s passed fresh 29952 wave64 checks, clock 24.9995/24.9991 MHz. Both passed 24 MMQ shapes ×
  five dispatches, full bit-identical tile/control outputs and the unchanged sampled CPU double gate.
  Resident operator gains depend on format/rows; they are NOT model speed. Public small raw logs and
  scope: `gfx906-results/20261003-mmq-tiles-cbdc94c/README.md`. Complete fixtures stay private.
- The current-source 64K pilot is running serially on T5810: IQ2_XS and IQ3_S, code/Chinese/multi-turn,
  2048 control and MTP-only 3072/4096, QSA/tile forcing OFF, per-layer profiling ON, one repetition.
  Driver PID was **371361**, evidence `logs/gfx906-prefill-goal-20261003T040756Z/long64k-pilot/`;
  authoritative completion is `long64k-pilot.exit` plus per-arm exits/profiles/comparison/integrity,
  NOT the remembered PID. Do not compete with it or rebuild its binary while it runs.
  Longer/three-repeat validation, actual row-informed tile adoption, split work and QSA diagnosis
  remain unfinished. Source snapshot is remote `src-cbdc94c`, build `build-cbdc94c`; owner checkout,
  old 49285ae candidate and deployed engine/configs remain distinct and unchanged.
- That **cbdc94c 64K pilot is now complete**, exit 0, all six arms/18 requests and own children exited 0.
  The comparison artifact is authoritative: IQ2_XS 3072/4096 **both differ** from 2048 on Chinese and
  multi-turn IDs (code agrees). IQ3_S 3072 differs on Chinese; IQ3_S 4096 matches all three. This is
  one repetition, not a universal consistency gate. Mean prefill: IQ2_XS 97.684/89.837/86.710 s and
  IQ3_S 103.178/92.780/88.138 s for 2048/3072/4096 respectively. Do not equate these gains with passing
  consistency, and compare same-chunk MTP-OFF controls to separate chunk math from MTP effects.
- Real layer routing is heavily skewed (several max rows reach a full chunk); per-product 32-expert
  maxima differ from those aggregated layer maxima. Continued CPU-tested source now adds an exact
  `auto` tile opt-in leaving Q2_0/tiny/very large or unsupported products unchanged; HIP/model gates
  are pending. It also adds pre-attention compact real-input captures, committed-logit snapshots,
  replay against the unchanged double oracle, and common-prefix/input-isolation analysis. Strict
  mock-runtime serialization tests cover both HIP-admitted and non-HIP build paths, not GPU execution.
  Current checks: 71 gfx906 Python tests with repository .venv/bin on PATH (no skips), 55 CPU policy
  checks; earlier missing-PATH run's five CMake skips are retained and not relabelled as passes.
  None of the long/repetition/adoption/split/QSA completion rows is yet satisfied.
- Source **9211185** compiled/audited successfully; application SHA256
  `5b52e618e427bff1582fb2011a80ec924ea1bc245d3f5f475f1061178ca01f61`. Both MI50s passed 40 grouped
  matrix fixtures × six dispatches including `auto`, full bit-identical outputs, 48 graph replays/card,
  and unchanged sampled CPU double gates. Fresh wave clocks: 25.0000/24.9997 MHz, 29952 checks/card.
  Small raw receipt: `gfx906-results/20261003-mmq-auto-9211185/README.md`. This is not model adoption.
- The saved original 33475-token IQ2_XS source-review input was tokenizer-reverified and generated
  through both QSA paths with zero reuse/clean exits on 9211185, reproducing first output-ID divergence
  **42**. Immutable real first-chunk inputs/outputs exist for all twelve QSA layers in each arm.
  A private absolute-script import failure was retained, then corrected with isolated-source PYTHONPATH.
  Subsequent analysis correctly found different downstream compact-page geometry; the analyzer now
  preserves changed-input observations as non-isolated rather than refusing them. Original failure
  evidence remains. No logits were written because the original hook covered one-shot generation,
  not the pipe service's separate verify loop; continued source fixes the actual service hook and
  owner-device logits read, with a static entry-point regression (not hardware proof yet).
- Real replay loader rejected all 24 captures **before numeric execution**: production page_size is
  **4**, compact pages 757–812, while the historical synthetic fixture used page_size 256 and an
  unjustified 512-page loader limit. Continued source replaces that limit with bounded geometry and
  the existing 64-MiB byte bound, and adds four-cell/1024-page synthetic cases. The numeric thresholds
  remain exactly 0.003 + 0.0005*abs(reference), relative L2 <=2e-4; recovery execution is pending.
- Offline `tools.gfx906_prefill_compare` verifies full requested fixture/model/arm/repetition coverage,
  actual raw submitted IDs and complete raw T/DONE results, deployment/binary hashes, and separates
  canonical-2048 from same-chunk MTP-OFF controls. Missing paired controls yield null, never equality.
  It audited all 18 completed cbdc94c pilot cases against original protocol. Latest CPU regressions:
  **169 setup / 76 gfx906 Python tests**, no skips, diff check clean. Final-source HIP, broader model
  repetitions/contexts, accepted expert optimization, nearby splits and QSA diagnosis remain pending.
- Source **78d2fbf6c3a90af5ef5b625d36f1ee8db03fd65c** built/audited successfully, SHA256
  `5c9466bffd2c093a250d5343cabdd0cf442133337200b04719ed4c608e5ed753`. Its new diagnostic/replay
  suite exited **0**: both original 33475-token arms generated 64 IDs with zero reuse, clean exits,
  all 24 sampled real QSA fixtures passed old/new/dispatcher/graph/actual-output double-reference gates
  without threshold changes, and both 64-vector committed-logit sequences match actual raw T IDs.
  First QSA layer3 has identical inputs and output max-abs 4.2915e-6 / relative-L2 4.6021e-7; subsequent
  layers show changed operands and amplification. At common-prefix index42/position33516, ID561 versus
  ID1690 flips from gap +0.24339485 to -0.00288963; full-logit max-abs 0.51663 / relative-L2 0.04716.
  This is bounded numeric mechanism evidence, not all-row correctness or global quality; fixed-window/
  teacher-prefix isolation and broader checks remain pending. Receipt: `gfx906-results/20261003-qsa-real-78d2fbf/`.
- The fully raw-audited cbdc94c 64K pilot now has a bounded public receipt and exact output/identity/
  memory/stage evidence: `gfx906-results/20261003-long64k-pilot-cbdc94c/`. The source version is unchanged.
- New **78d2fbf** model work is admitted and running serially under the owner lease. Before launch,
  43,227,627,520 B disk free admitted an **8-GiB combined evidence allowance +4-GiB reserve**; no weights
  or expert-pack preparation. Private driver `model-validation-78d2fbf.sh` first runs focused64k:
  2048 MTP-OFF, 2048 MTP-ON, 2048 MTP+auto, plus 3072/4096 MTP-OFF, two pinned models/three kinds,
  one repetition with bounded product traces (30 requests). Only after complete clean protocol/integrity
  audit does it run required **64K/128K/192K ×three kinds ×three arms ×three repeats ×two models =162
  cold requests**, without product tracing. Requested source is read from the archived build receipt,
  not guessed; each phase has its own fresh evidence and failure/exit files. Recorded driver PID508609
  is not completion evidence. First six IQ2_XS control/MTP requests completed; comparisons, auto arms,
  IQ3_S and full matrix are not yet verified. Do not compete with it or rebuild its executing binary.
- Future runner source now estimates repeated raw CSV submissions, startup/memory evidence,
  layer profiles and up to 32 MMQ trace records/layer/chunk rather than a fixed eight-MiB allowance.
  **77 gfx906 CPU Python tests** passed without skips. Recomputed conservative budgets for the active
  archived-78d2fbf driver are 1,622,032,384 B focused +2,053,840,896 B full matrix =3,675,873,280 B,
  below its separately admitted 8,589,934,592 B evidence allowance. The active application/runner
  archive was not modified; periodic 4-GiB disk-floor monitoring and new-directory checks remain active.
- **78d2fbf focused64k is complete**, driver/audit0, ten clean engines/30 cold requests with full raw
  protocol/integrity audit. Both IQ2_XS and IQ3_S 2048 MTP and MTP+auto match all three control ID
  sequences (one repetition). Auto pilot mean-vs-MTP reductions are 0.751%/0.937%, instrumented and
  not adoption evidence. **MTP-OFF** 3072/4096 already reproduce IQ2_XS Chinese/chat differences;
  IQ3_S3072-OFF differs on Chinese,4096-OFF matches this repetition. Thus larger chunk alone suffices
  for these observed differences; do not blame MTP or assert universal equivalence. Receipt:
  `gfx906-results/20261003-focused64k-78d2fbf/`. Full private audit retains all request-level layers/products.
- The full 162-request matrix started after the focused audit. Latest observed IQ2_XS r0 completed
  nine control +nine3072 requests through **196608** tokens;4096 is in progress, not acceptance.
- New offline profile summaries keep separate automatic/manual identity and actual per-product
  maximum histograms/minima/sums, rather than only the largest layer/product maximum. **78 gfx906 /
  169 setup CPU tests** passed, no skips. The active immutable archived runner/application was unchanged.
- Serialized post-matrix queue PID541915 waits for the previous complete driver/audit exit0, without
  holding the owner lease. Fresh guarded admissions then schedule 54 repeated64K auto/control/MTP
  requests,72 immediate-neighbor split requests (23/25,24/24,25/23 justified by ~5.95s/request measured
  next-stage wait/drain), fixed-window original-source QSA controls (`--mtp-max-t 1 --suffix-draft 0`,
  actual INFO/DONE/decode-window/logit gates), and final-source policy/wave/attention/native-alias/MMQ
  probes plus requested-missing-device failures. No parallel GPU work. Extra model/diagnostic/probe
  evidence requires20GiB allowance +4GiB reserve; operator suite rechecks16GiB allowance +reserve.
  These are queued work, not completed gates; no deployment/default changes or active binary rebuilds.
- Offline final audit was tightened: reject global failure evidence even if every case file exists,
  validate exact published model revision/two shard hashes/sizes/stored fingerprints, require the exact
  requested model/arm/repetition set (not only its count), verify DONE prompt count against original
  submitted IDs, complete cold request/stage row counts and two-card one-second memory coverage.
  Split comparisons now include all matching 24/24 repetitions with the same chunk/MTP/tile; absent
  matched split controls are null, never a pass. **83 gfx906 CPU tests** passed, no skips; strengthened
  audit of all18 original cbdc94c 64K cases passed using retained full raw evidence locally. Identity
  validation checks stored full-hash attestations, not a new live weight rehash. Active archives remain
  immutable; final post-run audit must use this stronger offline tool before completing the goal.
- At **2026-10-04 01:48 UTC**, the existing long-matrix log held **144/162** completed request records;
  the isolated 78d2fbf engine was loading the IQ3_S r2 2048 control arm. No current-phase failure files
  were observed. This snapshot is progress, not completed acceptance, and does not establish later
  availability or output equality. Post-matrix suites had not started at that observation.
- Continued offline audit cross-checks saved process argv/config/model paths against the arm controls,
  raw greedy GEN limit/seed, real INFO/READY against saved startup, and rebuilds request/stage/draft/
  layer summaries from raw stderr. It requires the actual last-stage MTP batch path when requested,
  complete requested 48-layer coverage, bounded generated counts and positive TTFT/wall. Raw ordered
  two-card memory samples must independently reproduce every declared peak/minimum/count. Stored
  metadata remains an archive-lineage check, not fresh live environment/weight attestation or model
  parity. **89 gfx906 /169 setup CPU tests** passed, no skips; full retained cbdc94c 64K pilot re-audit
  passed all six arms/18 cases, and diff check passed. Active compiled/runner archives were unchanged.

## Final requirement audit — 2026-10-04

Completed-source receipt: [`gfx906-results/20261004-final-prefill-78d2fbf/README.md`](gfx906-results/20261004-final-prefill-78d2fbf/README.md).
Application/GPU-source trees remain identical to compiled **78d2fbf**; offline auditing is **ff9972d**.
All model suites and final serial drivers exited 0. A later code change would require new matching gates.

| Requirement | Authoritative evidence and resolution |
|---|---|
| Serial immutable runner/safety | 318 complete cold requests, raw GEN/T/DONE, recorded argv/config/startup, every requested fixture/model/arm/repetition, raw profile reconstruction and raw two-card memory extrema passed the strengthened ff9972d audit. Owner lease/resource admission, own exits/deferred drain, model/binary/deployment identities were retained. No service operation. |
| Long-context MTP/model validation | All162 requested IQ2_XS/IQ3_S 64K/128K/192K ×code/Chinese/chat ×three policies ×three repeats passed complete protocol/integrity audit. IDs, real prefill/TTFT/decode, speculation, streaming/refill/cache/timers and sampled peaks are preserved. Larger chunks have disclosed ID differences; no universal chunk3072/4096 recommendation. |
| Measured expert improvement | Actual product distributions and format-specific MMQ/FP16 costs justify guarded autoJ32/J64. Final-source40 fixtures ×six dispatches/card and48 graph outputs/card retain full control equality and sampled CPU double gates. Repeated54-request64K model gate retains all control IDs; auto adds0.4603%/1.0396% mean reduction over MTP, not the much larger resident operator gains. Unsupported formats/Q2_0/dense/MTP/FP16 fallbacks and pinned kernels/precision are unchanged. |
| Measured pipeline/split | All72 requested64K neighboring-split trials are fully raw-audited. Critical-path/wait/residency/pressure measurements are recorded. IQ2_XS neighbors do not beat24/24; IQ3_S neighbors change IDs despite a faster25/23 result. Keep24/24 rather than adopting an unverified numerical path. Ownership/aliases/producer ordering remain; no P2P/tensor parallelism. |
| QSA diagnosis | Actual fixedT=1 controls were verified from INFO, DONE and64 one-row decode windows. Raw chosen IDs match128 committed-logit vectors; index42 flip and initial same-input layer3 rounding drift remain. All24 sampled real-input old/new/dispatcher/graph/actual-output independent double gates passed unchanged. Speculative-window variation is eliminated for this diagnostic; unsampled rows/global quality are not claimed. QSA stays OFF for accepted candidates. |
| Final gates/documentation/integrity | CPU89 gfx906/169 setup, policy55; both-card attention/native-alias/MMQ/graph and29952 wave checks/card passed on unchanged78d2fbf. Requested-missing probes returned expected1, not skips. All six binary and six deployment hashes unchanged; cards drained/no KFD/API/Docker workload at completion. Public receipt retains bounded IDs/counters/timers/hashes/raw logs; full fixtures remain private. Related changes are committed/pushed only on the dedicated origin branch. |

The bounded accepted result is **2048 /24-24 /MTP batch+auto /QSA OFF** for the repeated64K
fixtures, with2.6815%/3.2539% mean cold-prefill reduction over control. This does not replace the
required long-context matrix with a smaller test: that entire matrix was completed and its negative
consistency results are retained. Rejected QSA/larger-chunk/neighboring-split variants are not passes.
Deployment/default changes, MI60, full262K/long-soak/global quality, every-row/full-model CPU parity
and complete split RAM-conversation parking remain outside the agreed implementation scope.
