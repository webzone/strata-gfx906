# Paired 64K MTP / per-group MMQ-auto pilot — 78d2fbf

## Scope and evidence

Two MI50s, real gfx906 HIP Release, ROCm 7.2.4, portable host code, pinned ggml, original pinned
GSQ-RCO IQ2_XS then IQ3_S. Application source `78d2fbf6c3a90af5ef5b625d36f1ee8db03fd65c`,
SHA256 `5c9466bffd2c093a250d5343cabdd0cf442133337200b04719ed4c608e5ed753`; not the owner-deployed
v0.1.34 binary. Ten arms/30 requests, code/Chinese/multi-turn at exactly65536 submitted tokens,
one repetition, greedy64, split24, cache0/short-read0, QSA OFF. Per-layer profiling and grouped-product
trace ON, so these are instrumented pilot timings, **not repeated performance/adoption acceptance**.

Runner and complete raw-stdin/T/DONE/profile/fixture/binary/deployment audit exited0; all ten children
exited0, zero reuse, clean deferred-VRAM drain, deployment hashes unchanged. Full raw evidence is
private `logs/gfx906-prefill-goal-20261003T040756Z/focused64k-78d2fbf/`; full audited report is adjacent
privately as `focused64k-78d2fbf-audit.json`. Bounded `validation.json` retains actual complete generated
IDs, request metrics, memory samples' observed peaks, stage intervals/counters, aggregate product
choices/covered rows and raw-file hashes. It deliberately omits the huge per-layer/product trees;
full raw traces and exact original submitted/rendered inputs were not discarded.

## Results (mean across three different inputs; not three repetitions)

| Model | 2048 OFF | 2048 MTP | 2048 MTP+auto | 3072 OFF | 4096 OFF |
|---|---:|---:|---:|---:|---:|
| IQ2_XS prefill s |97.629|95.589|94.871|92.168|88.919|
| IQ3_S prefill s |103.244|100.825|99.881|94.719|90.388|

Both models' 2048 MTP and MTP+auto match **all three control output-ID sequences**, one repetition only.
Auto versus same-chunk MTP pilot mean reduction is **0.751% /0.937%**, much smaller than selected
resident-operator gains. Repeated untraced runs are queued before adoption. Prefill is not TTFT:
IQ2_XS MTP/auto mean TTFT95.665/94.946s; IQ3_S100.893/99.957s.

IQ2_XS **3072 OFF and4096 OFF also differ** from2048 on Chinese/chat (first IDs53/16), matching the
historical larger-chunk MTP pilot's divergence coordinates. IQ3_S3072 OFF differs on Chinese;
4096 OFF matches all three in this repetition. Thus larger chunk alone is sufficient to cause these
observed canonical-control differences; MTP is not required. This is same-source paired evidence,
not a claim that every later output or context has the same cause. Required broader/repeated
MTP3072/4096 validation is still separate and unfinished.

## Real expert and pipeline observations

For IQ2_XS auto across the three cold requests, 33,503 grouped gate/up products selected an override:
IQ2_S J32=5,563/J64=20,376; IQ2_XXS J32=1,313/J64=6,251. Maximums observed in those aggregate groups
are96/192 respectively. Other products retain the original dispatcher; Q2_0 down is never forced.
Per-product maxima are distinct from whole-layer maxima. New offline summarization retains actual
maximum histograms/minima/sums and the automatic marker; no new GPU work or active-archive mutation.

Aggregate actual format-attributed gate/up MMQ intervals, IQ2_XS MTP versus auto: type22
78.742→76.899s and type16 21.148→20.597s. FP16 gate/up type29 is12.567→12.561s; down type42 has
MMQ72.124→71.787s and FP16 8.443→8.438s. These are summed intervals across both device timelines and
three requests, **not additive whole-request wall time**. Dequantization intervals contain mixed
work and must not be assigned exclusively to the gate/up format.

GPU0 next-stage wait/drain is18.583s with MTP and17.851s with auto, about6.19/5.95s/request. Its timeline
includes these waits; GPU1's timeline has different fill/drain bounds. This justifies measuring immediate
23/25,24/24,25/23 neighbors, not selecting one from timeline sums. Three-repeat actual-model auto and
neighboring-split suites are queued **after** the full long matrix, with fresh lock/resource/space
admission and no overlapping GPU work. No deployment/default is changed automatically.

One-second RSS/RAM/VRAM monitoring gives observed maxima/minima, not instantaneous peak proof.
No downloads, expert-pack duplication, API/service/package/driver/ROCm/network/tuning operations.
MI60, full262K, long-soak/global quality and performance beyond the measured inputs remain unvalidated.
