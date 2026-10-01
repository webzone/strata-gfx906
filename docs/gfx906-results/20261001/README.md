# T5810 / MI50 Audit Evidence — 2026-10-01

See the [gfx906 development guide](../../GFX906.md). Model weights and binaries are not committed;
model and ggml revisions are pinned in that guide.

- `ctest-final.log`: 41 tests, 40 passed / 1 hipBLASLt skipped / 0 failed; two tests were excluded by the preset.
- `python-tests-final.log`: 18 CPU-only installer-entrypoint and download-safety regressions.
- `hip-layer-handoff.log`: active 25 MHz clock calibration, both handoff directions, and same-device handoff;
  137,980,416 exact checks.
- `smoke-{dual,single,dual-threshold1,reverse}.json`: real-model configuration, prompt/generated IDs,
  text, exit status, and engine counters. All four modes matched across four prompts (153 generated tokens
  per mode); `min-p=1` still offered 22 drafts and is not MTP-off.
- `cli-no-mtp-v2.json`: native-pack single-GPU CLI without loading MTP, using the supported
  `--spec 2 --prefill 128` verifier path; arithmetic tokens matched.
- `reference.*`: pinned CPU-only llama.cpp `llama-completion` with the same GGUF and 23-token greedy
  prompt template; returned `12`, exit code 0. This is not a full-logit or per-layer oracle.
- `model-plan-ready.json` and `model-sha256-verified.log`: model identity, byte counts, LFS hashes,
  and download completion.
- `environment-final.json`: 03:31 UTC GPU, service, disk, RAM, listener, and GPU UMC snapshot.
- `summary.json`: recomputed throughput and token-consistency summary from the captured test results.

Initial test failures are retained: an oracle used an invalid `gemm` mode, idle-sleep clock calibration
was invalid for profiling, and the native-pack CLI rejected `--spec 0`. Corrected reruns are recorded;
the initial failures and hipBLASLt skip are not counted as passes. Full build/package/engine logs remain
on the host under `/home/chris/dev/strata-gfx906/logs`; complete compiler-warning logs are not committed.

## Recheck

```bash
python3 tools/gfx906_summary.py docs/gfx906-results/20261001
```

The validator uses explicit exceptions rather than assertions, so `python3 -O` does not bypass checks.
The first `smoke-dual.json` derived `decode_tok_s` field used `generated-1`; **do not use that derived
field**. The summary recomputes speed from raw `engine.generated/decode_ms`, and the smoke tool was
corrected afterward. The 128-token observation is about 17.8 tok/s (`128 / 7195.3 ms`).

Smoke timings cover their measured decoder windows, not model loading or cache fill. These are a few
smoke tests, not an isolated performance A/B. Some comparison runs overlapped CPU-reference
compilation; do not turn the observations into a general multi-GPU/speculative-decoding speedup claim.
Raw prompt/response transcripts are preserved verbatim as test data.
