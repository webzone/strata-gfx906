# Strata for AMD Instinct MI50 / MI60

**`strata-gfx906` is an AMD-focused Strata fork for gfx906 accelerators.** The original Strata was built primarily for NVIDIA GPUs and CUDA; this fork adds a native AMD HIP/ROCm path tuned for the wave64 architecture in AMD Instinct MI50 and MI60 cards. It targets the real `gfx906` architecture and does not spoof another GPU generation.

The gfx906 backend is experimental and must be enabled explicitly. The validation documented in this repository has been performed on MI50 hardware. MI60 is an intended same-architecture target, but has not been independently validated here.

## Recent MI50 workload results

The following figures summarize recent runs of the current GSQ-RCO IQ2_XS quantized workload on MI50. They describe different stages and cache conditions; they are not directly comparable to one another or to measurements from the original NVIDIA-oriented project.

| Stage | Observed result | Workload / interpretation |
| --- | ---: | --- |
| Decode | **~15.1 tokens/s** (13.2–17.5) | Stable observed range for the current quantized model under CPU/GPU hybrid execution. |
| Uncached prefill, compute throughput | **~490 tokens/s** | Long-text input with approximately 32K new tokens and no KV-cache reuse; this is the raw prefill-compute baseline. |
| End-to-end prefill after a KV-cache hit | **11,000–27,800 effective tokens/s** | Prompt handling with very high KV-cache reuse. The effective rate includes reused prompt tokens. |
| First-token latency after a cache hit | **~1.6–4.4 s** | Approximately 48K-token prompt with substantial KV reuse; this is not a fresh, uncached 48K prefill. |

These are operational observations, not a controlled single-card-versus-dual-card benchmark; complete raw timing/configuration records for this summary are not archived here. In particular, cache-assisted effective throughput must not be read as uncached prefill compute speed. The separate short-prompt smoke measurements and their conditions are documented in the [gfx906 development guide](docs/GFX906.md).

## What this fork supports

- AMD Instinct **MI50 / MI60 (`gfx906`)** with Linux, ROCm, HIP, and hipBLAS. MI50 is the hardware validated so far; consult the guide before assuming MI60 compatibility in a particular system.
- The **GSQ-RCO IQ2_XS** Qwen3.8-Flash-Next model path used by the current validation. The model's expert tensors use several quantization types; the filename does not describe every tensor.
- Single-GPU inference and experimental multi-GPU **contiguous-layer / pipeline splitting**. This is not tensor parallelism.
- CPU/GPU hybrid expert execution and speculative decoding with the supported MTP setup.

This is not a general claim that every feature or model in upstream Strata is available on gfx906. For example, HIP image input is not available in the current validated path, and the HTTP server processes requests serially.

## Build and validation

Use an existing ROCm 7 / HIP / hipBLAS installation. gfx906 support is opt-in; do not set `HSA_OVERRIDE_GFX_VERSION` or use a wheel built for a different GPU architecture. The tested build and regression commands, model identity, storage requirements, and smoke-test reproduction steps are in **[docs/GFX906.md](docs/GFX906.md)**.

The archived MI50 regression run completed **40 tests, with 1 skipped and 0 failed**, plus **18/18** Python entrypoint/download-safety tests. The skip and tests excluded by the preset are described in the [evidence notes](docs/gfx906-results/20261001/README.md); these software checks do not establish production readiness or long-run stability.

## Project background

This repository is based on upstream Strata 0.1.30. It carries a gfx906-specific HIP implementation and experimental MI50/MI60 support work; it is not the upstream project's general NVIDIA/CUDA release. Upstream source: [Niko1221/Strata](https://github.com/Niko1221/Strata). For architecture details, compatibility limitations, and reproducible evidence, start with the [gfx906 development guide](docs/GFX906.md).

## License

Strata is open source under the [MIT License](LICENSE). Third-party components and model files may carry separate licenses; see their respective notices and source pages.
