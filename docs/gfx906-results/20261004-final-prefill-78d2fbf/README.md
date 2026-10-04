# Final gfx906 cold-prefill implementation audit — 2026-10-04

## Identity, scope and evidence

Hardware: T5810, two AMD MI50 32-GiB cards, actual gfx906/wave64, Xeon E5-1650 v3,
110555 MiB OS-visible RAM, system ROCm 7.2.4. Application source is
**`78d2fbf6c3a90af5ef5b625d36f1ee8db03fd65c`**, v0.1.37; executable SHA256:
**`5c9466bffd2c093a250d5343cabdd0cf442133337200b04719ed4c608e5ed753`**.
Later commits changed offline auditing/documentation, not application/GPU-source trees.
The final archive audit used `ff9972d`, not the older archived runner's weaker audit.

Both models are pinned `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` IQ2_XS and original
IQ3_S, revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09`, with published shard sizes/full
SHA256 attestations checked. llama.cpp/ggml remains pinned to
`3cf03257f219afbe7334045ff7c6a06ac68c627d`; no vendor-kernel or precision changes.
The historical deployed v0.1.34 binary, two configs and three launchers were unchanged.
This is not a benchmark against that deployment and did not start an API/service.

`validation.json` contains all **318 cold model requests**: focused pilot 30, long matrix 162,
auto repeats 54 and neighboring splits 72. It preserves complete generated IDs, fixture hashes,
real DONE/prefill/TTFT/decode/speculation/cache counters, request/device timing and memory evidence,
comparisons against every available control repetition, actual routing/product distributions,
fixed-window QSA analysis and final probe identities/exits/hashes. Schema2 aggregates the12342
per-callback MTP records per request/device/batched flag (count, token sum, position range, wall sum);
full individual callbacks remain in the private raw audit. No cases/IDs/comparisons/stages/counters/
memory/product histograms/raw hashes were removed. Full raw inputs, stdout/stderr,
profiles, process/config/startup metadata, one-second memory samples and numerical fixtures remain
private. The archive auditor checked the entire requested geometry, complete raw GEN/T/DONE,
zero reuse, clean own exits, recorded control/launch identity, raw profile reconstruction and raw
memory extrema, plus binary/deployment integrity. These are stored evidence checks, not a fresh
live model-weight/environment attestation.

Private remote root: `logs/gfx906-prefill-goal-20261003T040756Z/`. Full synthetic GPU fixtures
remain there; no weights, binaries or full compiler-warning logs are committed. Local audited copy:
`logs/gfx906-prefill-goal-20261003T040756Z/final-evidence-20261004T072300Z/`.
All serial drivers returned 0. Requested-missing-device probes deliberately returned 1 and were
verified as expected failures, not silent passes/skips.

## Bounded accepted computation result

For the **submitted 65536-token fixtures**, retain chunk **2048**, split **24/24**, experimental
QSA **OFF**. Explicit opt-ins `STRATA_GFX906_MTP_BATCH=1` and `STRATA_GFX906_MMQ_J=auto` provide
a measured improvement while retaining all generated IDs across code, Chinese and multi-turn
fixtures, three repetitions, both models. Each arm has nine requests/model, greedy top-k1,
seed42, up to 64 generated IDs, no prompt checkpoint cache or short-read. Layer timing was enabled;
product tracing/additional diagnostic synchronization was OFF for these repeats. No defaults or
deployment were changed. This recommendation does **not** extrapolate to 128K/192K, 262K, MI60,
all inputs, full answers or global quality.

| Model | 2048 control prefill | MTP only | MTP + auto | Reduction vs control | Auto alone vs MTP |
|---|---:|---:|---:|---:|---:|
| IQ2_XS | 97.5268 s | 95.3505 s | **94.9116 s** | **2.6815%** | **0.4603%** |
| IQ3_S | 103.0874 s | 100.7807 s | **99.7331 s** | **3.2539%** | **1.0396%** |

These are mean real cold-prefill times, not resident operator speedups. TTFT/decode and every
repetition are retained separately in the JSON. Small automatic-tile gains must not be inflated
into the much larger historical warm MMQ microbenchmark gains.

Auto uses each admitted 32-expert product's **maximum rows**, not a layer maximum: J32 for49–96,
J64 for97–192, original dispatcher otherwise. Unsupported geometry/formats, Q2_0 down, tiny/large
products and dense/single-matrix MTP retain the original path. In the focused three-request trace,
IQ2_XS used33503 forced products (IQ2_S/IQ2_XXS); IQ3_S used74160 (IQ2_S/IQ3_XXS/IQ3_S/IQ4_XS
and IQ4_NL down). Exact auto/manual labels, minima/maxima and histograms are retained. IQ1_M FP16
fallback remains unchanged; its measured phase costs are disclosed rather than forcing it through
MMQ. Format-cost sums overlap devices/requests and are not whole-request wall time; dequant is mixed work.

IQ2_XS repeats had zero cold expert streaming/DMA/refill. IQ3_S repeats recorded320544 streamed/DMA
experts and5481 refilled slots across nine requests in each policy, with the same counts for MTP/auto.
Observed MTP+auto arm-wide RSS/VRAM maxima were about34.382/22.680+24.917 GiB for IQ2_XS and
48.258/29.408+31.479 GiB for IQ3_S. These include load/all requests and one-second sampling, not
instantaneous or per-context peaks; IQ3_S's second card has little headroom.

## Required long-context matrix: complete, larger chunks not token-equivalent

Two models ×three contexts ×three kinds ×three policies ×three repetitions = **162 complete cold
requests**, all raw-audited. Means below cover nine requests/context/policy. Both models' three
2048-control repetitions retained identical IDs. Larger-chunk MTP policies did **not** match all
canonical outputs; successful generation and faster timing do not erase those differences.

| Model/policy | 65536 prefill | 131072 prefill | 196608 prefill | Canonical IDs across all contexts |
|---|---:|---:|---:|:---:|
| IQ2_XS 2048 control |97.6272 s|197.0431 s|303.2344 s|yes|
| IQ2_XS 3072 MTP |89.8535 s|181.6429 s|280.4553 s|**no**|
| IQ2_XS 4096 MTP |86.7435 s|175.1432 s|270.5871 s|**no**|
| IQ3_S 2048 control |103.1112 s|207.9577 s|319.8900 s|yes|
| IQ3_S 3072 MTP |92.4790 s|186.5825 s|287.7349 s|**no**|
| IQ3_S 4096 MTP |88.1040 s|178.0667 s|274.5158 s|**no**|

Exact mismatched context/kind/repetition/first-ID coordinates are retained, not sampled away.
The separate64K paired MTP-OFF pilot proves larger chunk alone suffices for specific observed64K
differences; it does not isolate every128K/192K cause. Missing same-chunk-OFF controls at longer
contexts remain null, not equality. Long-context speculation/refill/streaming/cache/device timers
and sampled memory are retained. Highest observed IQ3_S arm-wide RSS was48.513 GiB; second-card
VRAM reached31.613 GiB. No universal larger-chunk recommendation or full-context quality claim.

## Measured pipeline decision: keep24/24

Immediate neighbors were justified by measured GPU0 next-stage wait/drain, not arbitrary split
estimates. **72 cold requests** tested control and MTP+auto splits23/25,24/24,25/23, three kinds ×
three repetitions ×two models. Each device owns its layers/caches; mapped aliases/producer ordering
remain, no P2P or tensor parallelism. All original controls and every matching24/24 repetition
participate in the comparisons.

| Model, MTP+auto |23/25 prefill|24/24 prefill|25/23 prefill|Neighbor outputs |
|---|---:|---:|---:|---|
| IQ2_XS |99.8104 s|**94.8992 s**|95.1461 s|both neighbors match; neither improves wall time|
| IQ3_S |107.9890 s|99.7210 s|**95.3274 s**|both neighbors differ from canonical/matched24/24|

IQ2_XS GPU0 wait/drain fell16.182→5.929→2.866 s/request, but25/23 had slightly worse total time:
minimizing one overlapping interval is not minimizing the critical path. IQ3_S waits were
24.396/10.874/3.074 s/request; streaming counts changed498816/320544/185472 across nine requests,
refills5526/5481/5454. Different residency/numerical paths accompany partition changes; the cause
of neighboring-split ID differences is not isolated here. The faster IQ3_S25/23 result is **not
adopted as token-consistent**. Keep24/24; do not select a split from overlapping timer sums alone.

## QSA diagnosis with genuinely fixed verifier windows

Original tokenizer-verified33475-token IQ2_XS source review, chunk4096/split24, cache0/short-read0,
MTP batching OFF, MMQ forcing OFF. Both paths ran `--mtp-max-t 1 --suffix-draft 0`. Raw INFO reported
`mtp_max=1 lookup=0`; DONE reported zero offered/accepted drafts; actual decode logs confirmed
**64 windows, avg T1.00,1.00 tokens/window**. Both produced64 IDs, zero reuse, child exit0; every
committed-logit chosen ID matches raw stdout T IDs. Diagnostics synchronize and are **not timings**.

The common-prefix flip remains **index42/position33516**, control ID561 versus online ID1690:
control pair gap+0.2433948517, online−0.0028896332; pair-logit changes−0.085357666/+0.160926819.
Full-vocabulary max-abs0.5166330338, relative-L2 0.0471610414. Control's distinct top-two margin is
0.0092411041 (runner-up ID8618), not the flip-pair gap.

All **24 real fixtures**, three sampled query rows per QSA layer in each arm, passed old/new/
dispatcher/graph/actual-output against unchanged independent f16-dequantization/double-softmax
reference: element gate `0.003+0.0005*abs(reference)`, relative-L2≤2e−4. First QSA layer3 has all ten
inputs byte-identical; old/new output max-abs4.29153442e−6 and relative-L2 4.60211759e−7. Later
queries/KV/selections differ, so later layer comparisons are not same-input operator isolation.
This establishes bounded initial rounding-order drift followed by amplification and a real logit
ordering reversal without speculative-window variation; it does not prove every unsampled row,
whole-model CPU parity or global quality. No implementation error was established within these
numeric checks, and no threshold was relaxed. Experimental QSA stays OFF for accepted candidates.

## Final-source regressions and cleanup

- CPU policy: **55 checks**. Latest source Python gates: **89 gfx906 /169 setup**, no skips.
  The first completion rerun lacked `.venv/bin` on PATH and skipped five CMake tests; that raw run
  remains private. The corrected final rerun used the project `.venv/bin` PATH and passed all89.
- Each physical MI50: **29952 independent logical-wave32-on-wave64 checks**, including2D/3D blocks,
  signed INT8 dot4/overflow and shuffles, zero errors. Fresh clocks24.9978/24.9991 MHz.
- Each card: attention old/new/dispatcher/graph versus unchanged independent CPU double oracle,
  including production four-cell-page admission and historical synthetic boundaries/masks.
- Native mapped embedding:768 exact signed values plus graph replay/card, no P2P.
- Each card: **40 MMQ fixtures ×six dispatches**, full bit-identical tile/control outputs and
  **48 graph output files**, sampled pinned CPU dequantizer/double oracle≤0.02; observed worst
  sampled relative-L2 0.0040455994. Oracle samples≤16 elements/nonempty expert, not an independently
  implemented GGUF decoder or every-output double oracle. This final run did not benchmark MMQ.
- Requested-missing attention/native card exits1 matched expected1. No missing-card skip is a pass.
- All six candidate probe/application hashes and six protected deployment hashes unchanged.
- Final post-driver completion07:22:30 UTC. At07:23 UTC: both cards0%,10866688 B VRAM/card,
  no KFD PID, Docker workload or8082 listener,107691 MiB available RAM, about30 GiB disk free.
  This is a completion snapshot, not permanent availability. No service restored or restarted.
  Fresh08:36 UTC locked integrity audit again found all eight phase/driver exits0, matching all six
  candidate/six deployment hashes and both fully hashed shard fingerprints, cards idle/no KFD/Docker/
  API listener,107689 MiB available RAM and31469350912 B disk free; owner checkout stillc883c4f.

MI60, full262K/long-soak/global quality, all-row QSA/full-model CPU parity and split RAM-conversation
parking remain outside this implementation audit. Rejected experimental variants remain opt-in,
not approved defaults. Deployment requires separate owner confirmation.
