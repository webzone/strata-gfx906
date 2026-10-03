# gfx906 prefill optimization — active implementation contract

This work continues on `gfx906-prefill-opt`, initially `e7b1d0d`, not the owner deployment branch.
The user's objective is to implement the agreed gfx906 prefill plan end to end. Historical 2026-10-03
33K IQ2_XS results remain historical; they are not evidence for larger inputs, IQ3_S or later source.

## Required work and completion audit

- [ ] Implement a reproducible, serial, pipe-only validation runner with private immutable evidence,
  complete raw stdin/stdout/stderr, exact tokenizer-measured inputs, version/model identity checks,
  deployment integrity checks, owner launch lock, busy-card/listener/RAM/disk admission and bounded
  own-process shutdown/deferred-VRAM cleanup. No service control or deployment overwrite.
- [ ] Validate control 2048 versus MTP-only 3072/4096, experimental QSA OFF, on pinned IQ2_XS then IQ3_S:
  code, Chinese long-form and multi-turn inputs at 64K/128K, then 192K; repeat key comparisons at least
  three times. Require zero reuse for cold samples. Record generated IDs, real prefill/TTFT/decode,
  acceptance/rejection, expert streaming/refill, cache/buffer pressure and memory peaks. Disclose any
  output differences, failures or incomplete coverage rather than relabelling them as passes.
- [ ] Inspect real expert group distributions and quantify format-specific MMQ/FP16 costs. Implement
  and independently numerically test gfx906 expert-matrix improvements justified by measurements.
  Do not force unsupported formats through MMQ, mutate the pinned dependency, change precision
  blindly or report resident microbenchmark timings as model throughput.
- [ ] Re-measure stage balance using the improved MTP path. Test nearby layer splits only where the
  measured critical path justifies them, with each device owning its caches and mapped aliases,
  producer synchronization, no P2P/tensor parallelism and no competing GPU workload.
- [ ] Diagnose the reproducible QSA-on/4096 token-42 divergence with real operator/layer/logit evidence,
  independent numeric reference and isolated controls. Resolve an implementation error if found;
  otherwise establish and document the numeric mechanism/limits, retaining strict tests and opt-in.
  Never loosen an oracle threshold to conceal a discrepancy.
- [ ] Run relevant CPU/GPU/regression/model gates on final application source, preserve raw evidence,
  document measured scope/limits in `docs/GFX906.md`, commit related changes and push the dedicated
  branch without force. Audit all rows against authoritative artifacts before goal completion.

## Safety and scope

MI60, full 262K/long-soak quality, tensor parallelism, system tuning and RAM conversation parking are
not silently claimed by this work. Deployment is a separate owner-confirmed operation, not authorized
by the optimization objective. Fresh resource checks precede builds/tests; an occupied owner lock or
active GPU/API workload halts admission rather than stopping it. Keep at least 4 GiB disk reserve,
use existing weights/native-from-GGUF paths and this repository's `.venv`, and preserve original
binary/config/launcher hashes. `STRATA_EXPERIMENTAL_GFX906` and new computation paths remain opt-in.

## Current authoritative state

2026-10-03 04:07 UTC: local branch was clean at `e7b1d0d`; remote owner checkout remained `c883c4f`.
The isolated application was `49285ae` with SHA256
`05b646dd91bef728a9b31068d1eebec35001c9a050d3bbc68ac58b06687f78ec`.
Both MI50s had 0% utilization, 10866688 B VRAM use and no KFD PIDs; no 8082 listener/owner lock was
observed. Available RAM was 107772 MiB. These are preflight observations, not permanent availability.
Full private preflight is `logs/gfx906-prefill-goal-20261003T040756Z/resources-initial.txt` locally.
All completion rows above remain unverified for the continued goal.

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
  for these expert changes. Compile/numeric/model/performance gates and QSA investigation remain.
