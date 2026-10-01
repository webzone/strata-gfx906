# MI50 v0.1.31 Deployment Evidence — 2026-10-01

Built/deployed source: `6188acba7d48056312675a0c97940f5015bb5343`, incorporating upstream v0.1.31 /
`9259cad4cfa3543cd3b8decab5962672b968c649`. The owner confirmed the RAM ECC issue resolved and requested
JSON environment configuration and T5810 deployment. Full private config, old-engine/config backups and
compiler/install/control logs stay on the host, not in this archive. No model weights or binaries are committed.

## Results and scope

- Project-local Python gfx906 regressions: **28 passed**, no final skips/failures. An initial invocation
  without `.venv/bin` on PATH skipped four CMake gates; the archived rerun includes all four.
- Full HIP engine build with preset `hip-gfx906-multigpu`, two build workers, real gfx906 and native experts.
- Selected CTest preset: **45 selected, 44 passed, one hipBLASLt skip, zero failures**. The preset excludes
  `ple_parity` and `platform_memory_test`; do not call this every possible project test.
- Wave64 probe: 29,952 exact checks; active clock calibration on each card approximately 25 MHz.
- Dual mapped-host handoff: 137,980,416 exact checks, no enabled peer access. Five independent CPU
  dequantization/double-accumulation operator oracles passed; neither is a full-model CPU parity oracle.
- Full SHA256 verification of both pinned IQ2_XS GGUF shards; 1,583 pinned ggml/llama source files matched.
- Real generation at 4K context: one card, two cards and reversed card order, four prompts / **153 identical
  generated IDs per mode**. Engine exit was zero in all modes.
- Actual API startup with configured **262,144-token capacity**, int8 KV, split 24, prefill 2,048 and MTP
  window 32,768. `/v1/models` responded and two short HTTP prompts returned `12` and `MI50 ready.`.
  This is not a 262K prompt, long-soak validation or a production rollout.

The actual engine process received `GPU_MAX_HW_QUEUES=8`, `HSA_ENABLE_SDMA=0`,
`HSA_FORCE_FINE_GRAIN_PCIE=1` and `HIP_VISIBLE_DEVICES=0,1`. `HSA_OVERRIDE_GFX_VERSION` was absent.
The manual host config/launcher are `run-iq2-xs.json` and `run-iq2-xs.sh`; the config is private mode 0600.
API listener: **127.0.0.1:8095 only**, without public exposure; no autoboot/autorestart was configured.
Other workloads, Docker containers, drivers, networking, tuning and thermal services were not changed.
The host EDAC CE/UE counters stayed 0/0 and disk free stayed above 4 GiB. Existing model/native/MTP files
were reused, with no new model preparation or duplicate main-model expert file.

## Artifacts

- `build-state.json`, `deploy-state.json`, `deployment-receipt.json`: commands/status, model hashes,
  health guards, executable identity and whitelisted runtime environment. PIDs are dated snapshots.
- `ctest.log`, `ctest.xml`, `ctest-full.log`: raw CTest console, JUnit and `Testing/Temporary/LastTest.log`.
  JUnit truncates some passed-test stdout at 1,024 bytes; use the full LastTest log for complete output.
- `hip_layer_handoff.log`, `hip_wave64_probe.log`, `hip_device1.log`: complete stdout extracted from
  the original full LastTest log, not new test executions after the API was started.
- `model-sha256.log`: full GGUF LFS hash checks; no download was performed.
- `smoke-{single,dual,reverse}.json`: unchanged real-model prompt/generated IDs, text and engine counters.
- `model-smoke.py`: exact bounded replay helper. It reads the private host config, caps context/window
  at 4K, prefill at 128 and disables prompt-cache reuse for those tests. It does not create an API listener.
- `api-*.request.json`, `api-*.response.json`, `api-models.json`: real HTTP inputs and raw response bodies.
- `gfx906-python.log`: local regression rerun. `manifest.json` hashes the artifacts.

Private host logs/backups: `/home/chris/dev/strata-gfx906/logs/deploy-v0.1.31-20261001T202157Z/`.
No further GPU tests were run after the API retained both GPUs.

## Performance boundary

The 128-token counting prompt's real decoder windows were 2,933.6 ms (one card), 2,246.3 ms (two cards)
and 2,317.4 ms (reversed), about 43.6 / 57.0 / 55.2 generated tokens/s. This easy sequence accepted
96/96 offered MTP drafts; the dual runs had 100% expert-cache hits. These are small warm-cache observations,
not representative chat benchmarks, an isolated environment A/B, or proof that any of the three requested
variables improved performance. Do not extrapolate dot4 timings, these rates or historical v0.1.30 results
to a general speedup. Full logits, long context/soak, optional GR v3, other models and MI60 remain unvalidated.
