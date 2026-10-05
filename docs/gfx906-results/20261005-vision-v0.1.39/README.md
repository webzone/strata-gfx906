# Experimental gfx906 HIP vision — 2026-10-05 UTC

Baseline: fork `c50a20628575403fdc53342d4f81babdff4cb20f`, upstream Strata
**v0.1.39 / `6f32ec0`**, pinned llama.cpp/ggml
`3cf03257f219afbe7334045ff7c6a06ac68c627d`. T5810: two MI50 32 GiB cards,
real `gfx906:sramecc+:xnack-`, ROCm 7.2.4, Release portable AVX2.
The owner stopped the existing service manually before GPU work.

## What passed

- The optional HIP encoder compiled and ran on each MI50 separately. GPU logs identify
  `CLIP using ROCm0 backend` (visible ordinal zero after selecting either physical card).
- Five complete embeddings at a 300-token cap passed on each card, including square,
  wide and tall patch grids and a real newspaper image. GPU0 used three repeats;
  GPU1 used one. Every retained SVE was checked for dimensions and finite values;
  the numerical comparison uses the last repeat, and timings use the median.
- Six embeddings at a 1,024-token cap passed on GPU0, including an actual 32 x 32
  output grid / 1,024 tokens from a 2,048 x 2,048 BMP. This run used one repeat.
- Predeclared thresholds remained **relative L2 <= 0.005** and **minimum token cosine
  >= 0.999**. The maximum observed relative L2 was 0.002182; minimum token cosine
  was 0.999720. These comparisons use the HIP binary's CPU mode with identical
  losslessly expanded FP32 weights and ordinary attention.
- The independent Python BF16 bit-shift expansion produced exactly the same CPU and
  HIP embeddings as the in-memory loader, bit for bit on all five 300-cap inputs.
  The diagnostic F32 GGUF is retained remotely; runtime reads the original BF16 GGUF.
- Real GSQ-RCO **IQ2_XS**, two GPUs with layer split 24, 4K context, int8 KV and the
  existing MTP runtime generated `MEN WALK ON MOON<|im_end|>`. GPU, matching FP32 CPU
  and original upstream BF16 CPU embeddings produced the same seven token IDs.
  Each prompt contained 332 tokens (300 image tokens), with zero prompt reuse.
  Both the encoder and text-engine processes exited with code zero.
- CMake rejected missing opt-in and a wrong architecture. Runtime rejected a missing
  GPU and forced Flash Attention before `READY`. A final reconfigure/build was a no-op.

## Encoder measurements

GPU0 at a 300-token cap, median of three ENC calls, six CPU threads, ordinary attention:

| Input | Output grid | Matching CPU ms | HIP ms | Encoder ratio |
| --- | --- | ---: | ---: | ---: |
| square.bmp | 7 x 7 | 639 | 38 | 16.8x |
| wide.bmp | 24 x 12 | 5,376 | 197 | 27.3x |
| tall.bmp | 12 x 24 | 5,379 | 198 | 27.2x |
| large.bmp | 17 x 17 | 5,788 | 216 | 26.8x |
| photo.jpeg | 20 x 15 | 5,720 | 202 | 28.3x |

At the higher cap, the 1,024-token limit fixture took CPU **38,408 ms** and HIP
**1,204 ms**, one call each. Post-warm-up total device VRAM was 2,204,991,488 bytes
(2.05 GiB) at cap 300, and 3,308,556,288 bytes (3.08 GiB) at cap 1,024. The other
card remained at its roughly 10 MiB idle baseline during each encoder run.
This is encoder timing, not model tok/s, end-to-end API latency, a comparison to
CPU Flash Attention, or a controlled text throughput benchmark. The three seven-token
generations are a correctness smoke, too short for a performance claim.

## Precision boundary and failed attempts

The direct pinned HIP backend with the original BF16 loading failed all five initial
comparisons: relative L2 0.0217–0.0841, minimum token cosine as low as 0.8846.
Its raw results remain in `compare-300-gpu0/`; they are **not accepted speed results**.
Global BF16/F32 GEMM overrides and a mixed per-matrix precision attempt also failed;
their raw probes and node captures remain under the remote evidence root.

The accepted build losslessly expands source BF16 weights to FP32 and avoids repeated
BF16 activation rounding. A vision-only unary overlay emulates the CPU GELU's FP16
input/output rounding and +/-10 branches, using the GPU formula rather than the CPU
lookup table. Build-directory overlays leave the pinned checkout byte-identical:
1,763 relevant source files matched its fixed source archive after building.

**This is not upstream BF16 embedding parity.** Even the FP32 CPU reference differs
from the upstream BF16 CPU arithmetic by 0.62–3.47% relative L2 on these fixtures;
accepted HIP differs by 0.62–3.52%. Newspaper minimum-token cosine against that old
CPU arithmetic was 0.9539 for HIP. `legacy-arithmetic-comparison.json` records both
comparisons. The matching headline is one sampled model result, not logits/layer
parity, a broad vision quality evaluation, or proof of identical future responses.

## Identity, reproduction and retained data

- Original mmproj: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` revision
  `ed59f92082b1e93c0e96d60a8b11aab089b52f09`,
  `mmproj-Qwen3.8-Flash-Next-BF16.gguf`, 907,543,008 bytes,
  SHA256 `b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0`.
  `mmproj-source.json` is the source metadata; comparison summaries contain the actual hash.
- Newspaper: pinned llama.cpp `tools/mtmd/test-1.jpeg`, retained here verbatim as
  `reference-newspaper.jpeg`, 124,071 bytes, SHA256
  `2dff664c0c8aaea18aff8cbe7e868845b775e90cdd7a0bac98df709b131deaa3`.
  BMP generation is deterministic in `tools/gfx906_vision_check.py`; dimensions and
  hashes are in the summaries. All raw BMPs and all repeated SVE outputs remain remote.
- Text model: existing IQ2_XS shards/pack and MTP runtime, using the pinned identities
  in `tools/gfx906_model.py`. `model-source-rechecked.json` records a fresh full
  SHA256 check of both original GGUF shards after the smoke. No new main-model pack
  or duplicate `experts.bin` was created. Text engine SHA256:
  `33fd7f7865d19ce1d942ad388d1f9ae5f1c3b7a3579df5fe904c22fda8d6d005`.
- Encoder hash, exact copied source hashes and paths: `receipt.json`.
  Actual CMake cache switches and rejection checks: `build-record.json`.
  Compiler warning logs and binaries remain remote and are not committed.
- Raw root: `/data/strata-gfx906/vision-gfx906-20261005` on T5810.
  Diagnostic scripts and failed harness attempts are preserved there. The first model
  attempt omitted required MTP arguments; the next omitted API-message normalization;
  a successful smoke initially lost its exit-code handle during cleanup. The final
  recorded run corrected the harness and captured both zero exit codes.

Reproduce the encoder comparison after building as in `docs/GFX906.md`, with idle cards:

```bash
.venv/bin/python tools/gfx906_vision_check.py \
  --exe build-vision-hip/bin/strata-vision \
  --mmproj /path/to/mmproj-Qwen3.8-Flash-Next-BF16.gguf \
  --model /path/to/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf \
  --photo docs/gfx906-results/20261005-vision-v0.1.39/reference-newspaper.jpeg \
  --output /path/to/new-evidence-dir --device 0 --max-tokens 300 --repeat 3
```

`generation-smoke.py` and the two lossless-expansion diagnostic scripts retain the
T5810-specific reproduction paths. They are experiments, not server launchers; inspect
resources and paths before running them. The service regression log uses mock engines,
so its large printed tok/s figures are not model measurements.

Regression results: setup **258 tests, one existing skip**; Linux gfx906 **44, no skips**;
service/runconfig **148, no skips**. Local macOS gfx906 testing skipped five CMake checks
because CMake was unavailable; the Linux run covers them. No full model logits parity,
MI60, simultaneous dual-encoder, long-context images, production HTTP or long-soak
acceptance. The installer defaults remain opt-in. No 8082 service was started, config
changed, package installed, or driver/network/tuning service modified. Postflight GPUs
returned to roughly 10 MiB each; RAM was released and disks retained their safety floor.
