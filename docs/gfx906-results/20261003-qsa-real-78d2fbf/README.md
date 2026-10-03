# Real-input QSA / committed-logit diagnosis — 78d2fbf

## Verified scope

Isolated real-gfx906 HIP application/probes, source and SHA256 in `candidate.json`, on two MI50s,
ROCm 7.2.4, pinned ggml. The original IQ2_XS 33,475-token source-review request was re-tokenized and
matched its archived IDs. QSA-OFF and QSA-ON used the same 4096 chunk, split24, MTP batch OFF,
MMQ forcing OFF, zero prefix reuse and greedy top-k1. Both generated 64 IDs and exited 0.
Synchronizing captures/logit copies mean **these runs are not performance acceptance**.

All twelve QSA layers captured queries 0/2048/4095 of the first 4096-row chunk: raw queries,
selected compact INT8 KV pages/half scales, selections/steps/table and actual outputs. Actual pages
contain **four cells**, compact pools 757–812 pages, not the historical page256 synthetic layout.
Each of **24 real fixtures** replayed old/new/dispatcher/graph and actual output against the
independent f16-dequantization/double-softmax oracle. All passed unchanged absolute bound
`0.003 + 0.0005*abs(reference)` and relative-L2 `<=2e-4`. The replay summary records no failures.
This checks three real queries/layer, **not every original row or whole-model CPU parity**.

Both arms saved all 64 committed verifier-logit vectors (248,320 floats each), with index/position/
chosen ID. A separate audit matched every logged chosen ID to actual raw stdout T IDs and stored
results. Full vectors/inputs remain private; bounded analysis and first-layer raw replay logs are adjacent.

## Observed first divergence and amplification

The original first output-ID difference reproduced at **index42**, position33516, with all 42 prior
IDs equal. Control chose ID561; online chose ID1690. Control logits: ID561=20.232673645,
ID1690=19.989278793 (gap **0.243394852**). Online logits: ID561=20.147315979,
ID1690=20.150205612 (gap **-0.002889633**). Thus changes -0.085357666/+0.160926819 reverse
that pair. Full-vocabulary max-absolute difference is 0.516633034, relative-L2 0.047161041.
The control's separate top-two margin is 0.009241104; it must not be confused with the flip-pair gap.

First QSA layer3 is a genuinely isolated operator comparison: **all ten captured inputs identical**.
Its output difference has max-absolute **4.29153442e-6**, relative-L2 **4.60211759e-7**; both paths,
actual output and graph are close to the unchanged independent oracle. The online tile-by-tile FP32
softmax/value accumulation and old partition/merge use different rounding orders, not identical math
execution order. Later captures have changed queries, KV/scales/selections and sometimes compact-page
counts: these are upstream-amplified observations, **not isolated same-input operator comparisons**.
For example, layer7 query relative-L2 is 0.0078833; layer47 query/output are 0.31536/0.13153.

Evidence is consistent with small initial rounding differences amplified through subsequent recurrent/
quantized/sparse-selection computation and ending in a greedy logit-order flip. It does not establish
that every unsampled row is correct, isolate all intermediate quantization/selection boundaries, or
eliminate differing speculative-window effects. Additional fixed-window/teacher-prefix isolation and
broader quality checks remain required. **Keep QSA OFF for accepted candidates**; operator tolerances
are not model ID equality. No threshold/precision was changed to hide this divergence.

## Retained recovery and safety evidence

Earlier 9211185 failures were retained: private absolute-script import, analyzer refusal of legitimate
downstream page-count differences, missing pipe logits from a one-shot-only hook, and loader's arbitrary
512-page refusal before numeric execution. 78d2fbf fixes the service hook/owner-device read and bounded
four-cell replay admission. New build and complete diagnostic/replay suite exited 0; original deployed
six files and candidate hashes remained unchanged, own processes drained under the owner lease.
No API/service/package/driver/ROCm/model-preparation operations occurred. MI60, full262K and global/
long-soak quality are not established.
