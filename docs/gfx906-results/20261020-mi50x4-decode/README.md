# Exact gfx906 decode speed-ups: raw evidence (4x MI50, 2026-10-20)

The summary and the setup are in `docs/GFX906.md`, section "Exact decode speed-ups for the STRATA_HIP_GFX906 build
(2026-10-20)".

| File | What it is |
|---|---|
| `ab-results.jsonl` | One JSON record per engine variant, rounds `a1` to `a6`: the variant (engine binary, environment), every request's `decode_tok_s`, output tokens, prompt tokens and time, drafts accepted / offered, and the SHA-1 prefix of the output text. |
| `ab.py` | The harness that produced it: a fixed greedy workload, no random prefix, one engine restart per variant. |
| `microbench-idle-gpu.txt` | Kernel timings on an idle MI50 (the server stopped): the BF16 GEMV grid-size probe (`bf16b2.hip`), the BF16 router variants, the Q8_0 wave GEMV variants (`q8bench.hip`) and `fused_gr_bench` (hyper-connection read). |
| `q8bench.hip`, `bf16b2.hip`, `bf16b3.hip` | The stand-alone micro-benchmarks (`hipcc -O3 --offload-arch=gfx906`). `bf16b3.hip` (padding blocks) was only run while the server was busy on the same card, so its numbers are not recorded here; the engine A/B (`a4_e8_nopad`) is the evidence for the padding. |

Engine variants in `ab-results.jsonl` (all built from `d892f1bb` plus the changes named):

| Name | Changes |
|---|---|
| stock | the deployed 0.2.0 image's engine (`d892f1bb`) |
| e1 / e2 | `sh_stream_on()` off on gfx906, multi-block argmax, BF16 GEMV prefetch (e2 also had a wave64 router top-10 that measured 0% and was dropped) |
| e3 | e2 + Q8_0 wave GEMV `R = 2, U = 2` for 2+ columns |
| e5 / e6 | e3 + hyper-connection kernel variants (e6 also expert `row_dot` prefetch); both slower, dropped |
| e8 | e3 + BF16 grid padding + the gfx906 default table (`STRATA_GR_UP8=0 STRATA_GR_DOWN_BW4=0` turn off the dropped hyper-connection variants it still carried) |
| e10 | the committed set: e8 without the hyper-connection variants, plus the batched INT8 V loads in decode attention |
