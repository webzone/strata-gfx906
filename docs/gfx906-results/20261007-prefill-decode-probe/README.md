# T5810 cold prefill and decode probe — 2026-10-07

Three one-shot requests against the live `0.0.0.0:8082` Strata service on the dual-MI50 T5810 server.
They measure the two rates the project reports: **real (cold) prefill** and **decode**.

## Provenance

- Run by this agent, 2026-10-07 03:26:40 → 03:29:24 UTC (`probe-results.json` mtime `03:29:24`).
- Service state before each probe was read from `/metrics`: `live.running = 0`, `live.queued = 0`, all four
  slots idle. The service was not restarted and no setting was changed.
- Build under test: text engine `build-text-rocm10/strata`, SHA256
  `bc1102ba03c9ee2379699f9fb5c3cf7ce717132e7065efc5ac141606c5996853` (upstream gfx906 path
  `STRATA_HIP_GFX906=ON`, source v0.1.40.1 / `82f46a8` plus engine fix `74583c6`); vision encoder
  `build-vision-rocm10/bin/strata-vision`, SHA256
  `709fc4d2c34abbede8b17cb93a4a9571c931338339d3848691c5b59f156d8452`. Both on ROCm
  `10.0.0-gfx906+20260917140126`. Model GSQ-RCO IQ3_S, 262,144 context, `--kv int8`,
  `--kv-resident 32768`, `--spec 4`, `--batch 4 --batch-groups 2`, `--layer-split 24`, `--pcie-frac 0`.

## Method

- Prompt: generated filler lines, each with a unique index and salt, so no K/V reuse was possible across
  probes or turns. One user message, then `Question: what is 6 times 7? Answer with the number only.`
- Request: `max_tokens 64`, `temperature 0.0`, `stream: false`, `POST /v1/chat/completions`.
- Numbers below are the engine's own `timings` values from each response, not client estimates.
  `prompt_per_second` is the real prefill rate because `cached_tokens = 0` for all three probes.
  `predicted_per_second` is the decode rate with MTP `--spec 4` active.
- All three answers were correct (`42`).
- Cost: 137.8 s of GPU prefill plus 2.2 s of decode on an otherwise idle service.

## Results

| Prompt tokens | Reused | `prompt_ms` | **Real prefill tok/s** | **Decode tok/s** | Drafts accepted / offered | Client wall s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6,642 | 0 | 13,647.5 | **486.7** | **59.7** | 30 / 36 | 14.36 |
| 26,765 | 0 | 41,420.0 | **646.2** | **57.7** | 30 / 34 | 42.21 |
| 53,846 | 0 | 80,342.1 | **670.2** | **55.5** | 30 / 36 | 81.23 |

Real prefill rises with prompt size: 486.7 tok/s at 6.6K tokens, 670.2 tok/s at 53.8K. Decode falls
slightly as the context grows: 59.7 → 55.5 tok/s.

## The effective prefill rate is a different quantity

The production window in `../20261007-mi50-workload/` gives the reuse-inclusive figure:
`prompt_tokens / prompt_ms` = 3,699,741 / 248.2 s = **14,904 tok/s** across the window, and
**29,129–246,878 tok/s** per request. Those prompts hit the K/V cache (for example 133,594 of 134,007
tokens reused), so the figure counts tokens that were never computed. It describes how fast a warm turn is
accepted. It is not compute throughput and must not be quoted as a prefill rate.

## Limits

- Synthetic filler text, one run per size, and only 41 completion tokens per run: the decode sample is
  short, and the draft acceptance in these probes (30 of 34–36) is not the production acceptance rate
  (76.87% in the production window).
- Not a matched A/B: no comparison against another engine build, another ROCm version, or another split.
- No accuracy, perplexity, full-262K-context, MI60, or long-soak measurement.
- The service was idle during the probe. Concurrent owner traffic was not running, so these rates are
  single-stream rates, not concurrent throughput.

## Files

| File | Content |
|------|---------|
| `probe-results.json` | Raw per-probe records (verbatim engine usage/timings fields plus client wall time). |
| `evidence-manifest.sha256` | SHA256 of the files in this directory. |
