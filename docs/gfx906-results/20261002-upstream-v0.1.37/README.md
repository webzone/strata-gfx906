# Upstream v0.1.37 source integration — 2026-10-02

Upstream: `Niko1221/Strata@db4f91a` (source version **v0.1.37**, tags v0.1.35 / v0.1.36 / v0.1.37).
Previous upstream base: v0.1.34 / `1678de333d0e0711bc414ad992b640e1a37dd814`.
Fork before merge: `8db53c2` (branch `gfx906`). Merge commit: `cdf45bb`; follow-up test fix: `b038ded`.
40 upstream commits were absorbed with a reviewed merge commit; published fork history was not rebased.
llama.cpp/ggml remains pinned to `3cf03257f219afbe7334045ff7c6a06ac68c627d`.

This record covers **source integration, a real gfx906 HIP compilation on the T5810, and host-only tests**.
It is **not** GPU test-suite coverage, model parity, a speed measurement, or a deployment. The owner's
IQ3_S service was running on both cards throughout; it was not contacted, stopped, restarted or
replaced, and the live checkout `/home/chris/dev/strata-gfx906` and its `build-hip/strata` were not modified.

## Preflight

`preflight.log` — 2026-10-02T21:24:42Z. The owner-started IQ3_S server/engine pair (PID 44208/44218)
was listening on `0.0.0.0:8082`, both MI50s were model-resident (VRAM 92% / 98%, GPU use 0%), 56 GiB RAM
available, 58 GiB free on `/`, 111 GiB free on `/data`, load average 4.49. Nothing was stopped to free
resources; all build and test work ran in a separate worktree on `/data`.

## What was run

All C++ work ran in the detached worktree `/data/strata-gfx906/wt-v0137` at `b038ded`, with
`third_party/llama.cpp` symlinked to the existing pinned source. No packages, wheels or weights were
installed or downloaded. ROCm 7.2.4 in `/opt/rocm`, system compiler, project `.venv` reused read-only.

| Raw log | Result | Scope |
|---|---|---|
| `hip-build-summary.txt` | build complete, exit 0 | `cmake --preset hip-gfx906-multigpu` + `cmake --build --preset hip-gfx906-multigpu`: Release, `CMAKE_HIP_ARCHITECTURES=gfx906`, `STRATA_EXPERIMENTAL_GFX906=ON`, `STRATA_PREFILL_MMQ=ON`, `STRATA_PORTABLE=ON`, CUDA off. 231 targets, 0 `error:`/`FAILED:` lines, 1,413 compiler warnings (upstream `-Wunused-value` on `hipError_t`, e.g. `src/core/device.cu`; `STRATA_WERROR=OFF`). The full 1.31 MB build log stays on the host and is not committed. |
| `rebuild-verify.log` | exit 0, `ninja: no work to do` | Confirms the first build reached completion after the source checkout. |
| `device-list.log` | 2 devices | `build-hip/strata-device --list-devices` on the merged binary: both cards report `arch gfx906, 32.0 GiB, wave64`. Device enumeration only — no allocation, no kernel launch. |
| `ctest-host-only.log` | 3 passed / 0 failed | GPU-independent CTest targets from the gfx906 HIP build: `file_expert_source_test`, `expert_profile_save_test` (new upstream target), `ple_reader_selftest`. |
| `py-merged.log` | 280 + 40 passed, 0 failed | On the T5810 with the merged source: all `tools/test_*.py` (280) and the fork's `tests/test_gfx906*.py` (40). |

The Python suite prints rates such as `61.2 tok/s` from **fake tuning engines**; those are not model tok/s.

## What was NOT run, and why

`preflight.log` shows both MI50 cards model-resident. Running the GPU suites (`hip_*`, `*_parity`,
`hip_wave64_probe`, `hip_layer_handoff`, `hip_device1`, the five `hip_gfx906_*_oracle` CPU-oracle targets
that still link the HIP runtime, and the prefill/MMQ checks) would allocate against a busy card and could
disturb the live service, so they were **skipped, not passed**. No model generation, API request, MTP,
long-context, dual-GPU handoff, clock or stability measurement was performed on the merged binary.
No deployment, launcher, config, driver, ROCm or tuning change was made.

The merged upstream changes that need that window are the v0.1.36/v0.1.37 prompt-expert and sampler work.
Upstream keeps them CUDA-only: `moe_fused.cu` / `moe_fused_iq.cu` build only in the CUDA `strata_mmq`
branch (`STRATA_PREFILL_FUSED` defaults off for HIP, `prefill.cpp` stubs `fused::*` to `false`), and the
sm_90+ thread-block cluster kernels in `qsa_select.cu` / `sampler.cu` are inside `#if !defined(__HIPCC__)`
with `qsa_block_topk_cluster` / `sample_greedy_cluster` returning `false` under HIP. The HIP build proves
they compile out; it does not prove MI50 behaviour of the surrounding rewritten `prefill.cpp`,
`sampler.cu`, `expert_cache.cpp`, `expert_source.cpp` and `mtp.cpp`.

## Conflict resolutions (fork behaviour kept, upstream features taken)

- `setup.py`: upstream's #446 runtime-only-ROCm warning, plus the fork's rule that gfx906 needs a system
  ROCm 7 and never RDNA TheRock wheels. The fork's `select_amd_gpus()` / engine-rebuild start path now also
  carries upstream's `hip_runtime_beside_exe` (#468 #461), `split_budget` (#498), the atomic `write_config`
  (#459) and the new `--vram-reserve-mib` keep value (#493).
- `tools/mtp_fetch.py`: upstream's `HF_ENDPOINT` mirror support (#495) while the fork still ignores
  `STRATA_MTP_REVISION` and never falls back to mutable `main` (pinned revision
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`, SHA256 checks still applied through a mirror).
- `README.md`: MI50 evidence and fork install text kept; only the source-version lines moved to
  v0.1.37 / `db4f91a`. Historical v0.1.31 / v0.1.34 benchmark and deployment labels are unchanged.
- `tests/test_gfx906_upstream.py` now pins source version 0.1.37 and `MIN_ENGINE (0, 1, 37)`.

## Fork test fix (`b038ded`)

`test_gfx906_setup.StorageAndRanges.test_complete_model_recheck_needs_only_safety_floor` was already
failing before this merge on both macOS and the T5810: commit `6abb5a9` moved `tools/gfx906_model.py`
from a module-level `FILES` tuple to `MODEL_FILES[args.model]`, so the test patched a name `main()` no
longer reads and asserted the real 68 GB IQ2_XS plan against a 5 GiB fake disk. It now patches
`MODEL_FILES` with a two-shard fake pack and asserts `remaining_bytes == 0`. The fork suite is
40/40 on the T5810 (5 skips are macOS-only fixtures, so 0 skips on Linux).

## Reproduce

```bash
# on the T5810, isolated from the live checkout
cd ~/dev/strata-gfx906 && git fetch origin
git worktree add --detach /data/strata-gfx906/wt-v0137 b038ded
ln -s ~/mi50-t5810/strata-wave64-20260930/llama.cpp-3cf03257f219afbe7334045ff7c6a06ac68c627d \
      /data/strata-gfx906/wt-v0137/third_party/llama.cpp
cd /data/strata-gfx906/wt-v0137
cmake --preset hip-gfx906-multigpu
cmake --build --preset hip-gfx906-multigpu
./build-hip/strata-device --list-devices
ctest --preset hip-gfx906-multigpu -R "file_expert_source_test|expert_profile_save_test|ple_reader_selftest"
.venv/bin/python -m unittest discover -s tools -p 'test_*.py'
.venv/bin/python -m unittest discover -s tests -p 'test_gfx906*.py'
```

The full GPU suites (`ctest --preset hip-gfx906-multigpu`) belong in an idle-GPU window with the owner's
agreement; they were not run here.

`manifest.json` records the hashes, sizes and commands for the committed raw logs. No weights,
executables, secrets or production configs are included.
