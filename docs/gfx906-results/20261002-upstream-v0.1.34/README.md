# Upstream v0.1.34 source integration — 2026-10-02

Upstream: `Niko1221/Strata@1678de333d0e0711bc414ad992b640e1a37dd814` (source version v0.1.34).
Previous upstream base: v0.1.31 / `9259cad4cfa3543cd3b8decab5962672b968c649`.
Fork before merge: `be23587e559060c9d6a704e6b34490f5be9fce7b`, branch `gfx906`.
128 upstream commits were incorporated with a merge commit, not a rebase.
llama.cpp/ggml remains pinned to `3cf03257f219afbe7334045ff7c6a06ac68c627d`.

This is **CPU/software source integration**, not HIP compilation, GPU/model parity, a speed measurement or deployment.
Local host: macOS arm64, Python 3.14.7, CMake 4.4.3, existing project `.venv` and pinned llama.cpp source.
Local GPU/RAM/disk/container/listener state was inspected before testing; no workload was stopped.
The T5810 authority record reports an existing inference workload; this merge did not contact or modify that host.
No dependencies or weights were downloaded. All prior raw evidence is unchanged.

## Final results

| Raw log | Passed | Skipped | Failed | Scope |
|---|---:|---:|---:|---|
| `gfx906-final.log` | 40 | 0 | 0 | Entry points, saved opt-in, system-ROCm-only gate, architecture lists, CPU vision coexistence, HIP masks, Windows RDNA mocks, MTP pin/offline verify, PID safety. |
| `setup-final.log` | 132 | 0 | 0 | Upstream setup suites including AMD/Windows, choices, risk, golden configs, pins, RoPE, Unsloth and vocabulary. |
| `tools-final.log` | 73 | 0 | 0 | Calibration, shard identity, synthetic GGUF packing, MTP HTTP/ranges/hashes/safety, MCP fake processes and HTTP. |
| `serve.log` | 137 | 9 | 0 | Mock/synthetic tokenizer, server, lifecycle, monitor, MCP and structured-output checks. |
| `cpu-ctest.log` | 9 | 0 | 0 | Selected GPU-independent artifact/speculation and conversation cache/memory targets. |

Python: **382 passed / 9 skipped / 0 failed** (391 tests discovered). Selected CTest: **9 passed / 0 failed**.
The nine Python skips are five real-pack tokenizer fixtures, two Windows-only job-object tests and two optional
`jsonschema` cases (package not installed). Tests may print rates from fake engines; these are not model tok/s.

C++ configuration explicitly disabled CUDA/HIP/native experts. Only the listed nine portable targets were built/run,
not the full engine or AVX2 pool. The architecture gates stop before HIP compiler/SDK discovery.
Windows tests mock detection/prebuilt archives and do not run a Windows engine. CPU vision tests mock its builder;
there was no image-encoder build or image inference. Conversation-cache tests are synthetic, not real-model parity.

## First attempts and intermediate output

Captured logs are retained verbatim, including failures and partial output. They are not counted as final passes.

- `gfx906-first.log`: original 28 checks passed before adding v0.1.34-specific regressions.
- `setup-first.log`: one Windows-archive fixture hardcoded `strata.exe` while the test host expected `strata`;
  46 golden subcases normalized the substring `strata` inside log filenames as an executable. Corrected only
  those cross-platform fixtures. The upstream golden JSON remains unchanged; all its successful profiles now
  compare exactly, rather than recording a new baseline.
- `tools-first.log`: three fake install lifecycle checks failed because the no-`/proc` fallback counted exited,
  unreaped processes as live; another validation check placed a new data folder under macOS `/var`, correctly
  rejected as a system folder. The fallback now uses `ps` state and start time, rejecting zombies and PID reuse.
  Final tests use project-local `.venv/test-tmp`, not `/var`; the system-folder safety rule was not relaxed.
- `serve-first.log`: partial output after the initial combined run hit its 300-second tool deadline. It has no
  terminal unittest summary and is not a pass. The complete standalone rerun is `serve.log`. A later process
  inspection found no surviving fake install/server processes from the interrupted run.
- `gfx906.log`, `setup.log`, `tools.log`: successful intermediate runs (39/132/73 tests). A final Windows RDNA
  saved-start regression was then added, including absolute DLL paths and the runtime's actual GPU ordinal;
  the final logs above include all 40 gfx906 checks and rerun setup/tools after that fix.
- `cpu-configure.log`, `cpu-build.log`: small raw configure/build output for the selected targets; not a HIP build.

`manifest.json` records byte counts, SHA256 values, commands and the final suite results. Raw logs use the existing
`.gitattributes` binary-text/whitespace exceptions. No production configs, secrets, weights or executables are included.

## Integration policy differences

The default-OFF gfx906 gate, wave64/logical-wave32 intrinsics, signed dot4, clock conversion, per-device mapped-host
handoff and producer synchronization are retained. RDNA4 matrix-core kernels stay gfx12-specific; gfx906 is Linux-only
with a real system ROCm SDK, never RDNA wheels or an architecture override. The existing resident-from-GGUF expert
path and acceptance model pins remain; no duplicate main-model `experts.bin` was created.

Upstream's 31 pinned MTP SHA256 values and offline `verify` are included, with the fork's non-destructive policy:
conflicting inventories, oversized tensors and corrupt existing/downloaded bytes are retained and cause an explicit
failure, rather than deletion/refetch or a mutable-source fallback. The adapted MTP tests assert byte preservation,
source/range validation, final hash enforcement and a valid pinned partial resume. They use fake checkpoint bytes,
not a new hash check of the T5810 tensors. Inspect the source and use a separate output directory after a conflict.

The optional Linux CPU encoder remains unvalidated on gfx906. The new generic MCP installer has no gfx906 opt-in
parameter, so MI50 installation still uses the fork's documented CLI. PID fallback changes affect only non-Windows
systems without Linux `/proc`; the Windows/Linux identity checks remain intact.

## Reproduction

Run from the repository root with the existing pinned source under `third_party/llama.cpp` and packages from
`requirements-gfx906.txt`. This test environment additionally has `cmake==4.4.3` in `.venv`; `jsonschema` is absent.
No installer main flow, model loading or external inference service is needed.

```bash
export PATH="$PWD/.venv/bin:$PATH"
mkdir -p .venv/test-tmp
export TMPDIR="$PWD/.venv/test-tmp" TMP="$PWD/.venv/test-tmp" TEMP="$PWD/.venv/test-tmp"

.venv/bin/python -m unittest discover -s tests -p 'test_gfx906*.py' -v
.venv/bin/python -m unittest discover -s tools -p 'test_setup_*.py' -v
.venv/bin/python -m unittest tools.test_calibrate tools.test_shards tools.test_iq_pack tools.test_mtp_fetch tools.test_strata_mcp -v
.venv/bin/python -m unittest discover -s serve -p 'test_*.py' -v

.venv/bin/cmake -S . -B build-source-sync-cpu \
  -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF \
  -DSTRATA_NATIVE_EXPERTS=OFF -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_BUILD_CONVERSATION_TESTS=ON
.venv/bin/cmake --build build-source-sync-cpu --parallel 2 --target \
  gguf_reader_test gguf_split_test suffix_drafter_test controller_test \
  draft_policy_test conv_cache_test coupled_draft_test conversation_cache_test conversation_memory_test
.venv/bin/ctest --test-dir build-source-sync-cpu \
  -R '^(gguf_reader_test|gguf_split_test|suffix_drafter_test|controller_test|draft_policy_test|conv_cache_test|coupled_draft_test|conversation_cache_test|conversation_memory_test)$' \
  --output-on-failure
```

Do not replace this with full discovery of `tools/test_*.py`: the separate conversation-cache scripts there require
real engines/fixtures and are not part of this CPU-only validation.

## Outstanding hardware validation

A separate idle-GPU window must compile the real gfx906 engine and rerun the wave64 probes, new fast router and
operator oracles, active per-card clock calibration, both-device mapped-host handoff and pinned IQ2_XS generation.
Do not stop the existing service to obtain resources. Full CPU logits/layer parity, optional numeric switches,
new expert/PLE formats, CPU vision, long-context/long-soak stability and MI60 require additional validation.
Nothing here establishes a post-merge speedup or authorizes replacing the running v0.1.31 service.
