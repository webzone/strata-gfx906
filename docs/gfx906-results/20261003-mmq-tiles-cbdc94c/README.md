# MI50 grouped MMQ tile validation (not model prefill performance)

Compiled application source: `cbdc94c2d06103d1f70e3c1e2d7c741ad3cba9d1`, Strata v0.1.37.
Candidate SHA256: `167b38cc5c57b0984f8653da8e8161a5e9bc00a7a40cafb7bff0d706f8559c8b`.
Hardware: T5810, two actual MI50 32-GB gfx906 wave64 cards; system ROCm 7.2.4, portable host code,
opted-in HIP/native experts/MMQ. Dependency: pinned `3cf03257f219afbe7334045ff7c6a06ac68c627d`.
The owner deployment and sampled dependency-source hashes remained unchanged.

## Verified scope

- Isolated configure/build succeeded. The first driver failed *after* all requested targets linked,
  because its post-build audit passed a string to the `Path`-expecting `digest` helper. That exit-1
  driver/log is retained; corrected resource-admitted recheck/build and integrity audit both exited 0.
- CPU policy: 44 checks. Python: 169 setup / 66 gfx906 checks, no skips.
- Both cards: 29952 wave64 ballot/shuffle/signed-dot4 checks, including 2D/3D blocks, zero errors.
  Fresh wall-clock calibrations were 24.9995 / 24.9991 MHz.
- Both cards: 24 grouped matrix fixtures × five requested dispatches (unchanged control, J16/32/48/64),
  all exited 0. Gate/up uses 1280×2560 and down 2560×640, 32 experts, illustrative mean rows 40/80/160,
  empty/single/skewed groups, permuted source/destination rows, signed f16 scales and exact zero rows.
- All compared output values were finite and **bit-identical to the unchanged dispatcher**. The separate
  pinned CPU dequantizer/double products cover **up to 16 sampled elements per nonempty expert**, not
  every output; the unchanged relative-L2 tolerance is 0.02 to account for intended Q8_1 activations.
  This is an independent CPU execution/reference path, not an independently implemented GGUF decoder.

## Resident operator observations

The raw result logs report median HIP-event microseconds over seven retained samples after warmup.
They are neither model tok/s nor cold model prefill/TTFT. Illustrative fixtures show substantial
format/row-dependent differences: J32 reduces several mean-40 gate/up medians by approximately
27–37%, and J64 reduces several mean-80 medians by approximately 27–42%. Q2_0 down shows little
benefit; several larger groups have no useful difference. **Do not force one tile globally or infer
whole-model speed**. Actual expert-routing distributions and cache-pressure/model comparisons are
required before adopting a selection rule. Unsupported tiles retain GGML dispatch.

## Evidence and limits

Small raw numerical/timing logs are retained here. Complete pre-launch weights, activations, bounds,
permutations and all five outputs for every fixture remain private under:
`/home/chris/dev/strata-gfx906/logs/gfx906-prefill-goal-20261003T040756Z/mmq-device[01]-fixtures/`.
The owner launch lease and resource admission preceded the serial two-card suite; no model/API/service
was started by this suite. It passed with final idle-card/deployment-integrity checks.

The broader optimization goal is **not complete**: the IQ2_XS/IQ3_S 64K pilot has started, but its outputs,
longer 128K/192K and three-repetition comparisons, real-model tile adoption, split rebalancing and QSA
real-layer/logit diagnosis remain unverified. MI60 and long-soak/full-262K quality remain unvalidated.
