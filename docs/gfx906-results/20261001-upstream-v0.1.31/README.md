# Upstream v0.1.31 source integration — 2026-10-01

Upstream: `Niko1221/Strata@9259cad4cfa3543cd3b8decab5962672b968c649` (v0.1.31).
Fork before merge: `b8fe485a937ee373f615ba2dd023c25ca0c43f25`, on `gfx906`.
Original port base: `30ec18ec7094550fcc594fd948220d511d80464e` (v0.1.30).
llama.cpp/ggml source pin remains `3cf03257f219afbe7334045ff7c6a06ac68c627d`.

This evidence covers **CPU/software regression of the merged source**, not GPU acceptance,
model parity, model speed or production deployment. Execution host: macOS arm64, Python 3.14.7,
CMake 4.4.3. Python packages and CMake were installed only in the ignored project `.venv`.
Pinned llama.cpp source was fetched for the synthetic GGUF packing tests; no weights were downloaded.

## Results and raw output

| Raw log | Passed | Skipped | Failed | Scope |
|---|---:|---:|---:|---|
| `gfx906.log` | 28 | 0 | 0 | Original 18 entrypoint/download tests plus 10 new upstream-sync regressions. |
| `upstream-python.log` | 61 | 0 | 0 | Setup AMD/low-RAM/pins/RoPE, shard and calibration logic. |
| `pack.log` | 22 | 0 | 0 | Synthetic GGUF split/native-format/conversion identity checks. |
| `serve.log` | 93 | 7 | 0 | Mock/synthetic server, MCP and tokenizer checks. |
| `cpu-ctest.log` | 7 | 0 | 0 | Selected portable GPU-independent C++ tests. |

Python totals: **204 passed, 7 skipped, 0 failed** (211 discovered).
Selected CTest totals: **7 passed, 0 skipped, 0 failed**. These are not a full engine/HIP test suite.

Five server-suite skips require a real pack tokenizer fixture; two are Windows-only job-object tests.
The server tests start temporary local mock listeners and fake MCP processes, not an inference service.
Their printed token rates are mock timing and **must not be reported as model tok/s**.

Architecture-gate tests execute only the beginning of `cmake/hip_backend.cmake`, stopping before
`enable_language(HIP)`: default gfx906 rejection, explicit gfx906 plus feature suffix/mixed architecture
acceptance, retained upstream RDNA support and continued rejection of gfx908. They do not compile GPU code.
HIP builder/start/update tests mock the SDK, compiler and child process; they verify ordered device lists,
real compile targets, saved opt-in and that missing system ROCm never looks up/installs RDNA wheels.

## First attempts (retained, not counted as passes)

- `gfx906-first.log`: one new test fixture did not mock `source_version()` after replacing `ROOT` with a
  temporary directory, causing a missing `CMakeLists.txt` error. Four gate tests were skipped because
  CMake was not on PATH. The fixture was corrected; pinned CMake was installed in `.venv` and added to PATH.
- `gfx906-cmake-first.log`: the default-OFF gate correctly rejected gfx906, but a test assertion expected
  an unwrapped diagnostic. CMake wraps it across lines. The assertion now normalizes diagnostic whitespace.

Both logs are preserved verbatim. The final `gfx906.log` ran all 28 checks without skips or failures.
Raw logs are protected by `.gitattributes`; `manifest.json` records their SHA256 values. Full compiler,
package-install and unrelated machine diagnostic logs are intentionally not committed.

## Reproduction

Run from the repository root. Obtain the pinned llama.cpp source under `third_party/llama.cpp`
for the GGUF packing tests. The test environment used `requirements-gfx906.txt` and `cmake==4.4.3`.
Do not install dependencies into system Python.

```bash
export PATH="$PWD/.venv/bin:$PATH"
.venv/bin/python -m unittest discover -s tests -p 'test_gfx906*.py' -v
.venv/bin/python -m unittest tools.test_setup_amd tools.test_setup_lowram tools.test_setup_pins tools.test_setup_rope tools.test_shards tools.test_calibrate -v
.venv/bin/python -m unittest tools.test_iq_pack -v
.venv/bin/python -m unittest discover -s serve -p 'test_*.py' -v

.venv/bin/cmake -S . -B build-source-sync-cpu \
  -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF \
  -DSTRATA_NATIVE_EXPERTS=OFF -DSTRATA_BUILD_TESTS=ON
.venv/bin/cmake --build build-source-sync-cpu --parallel 2 --target \
  gguf_reader_test gguf_split_test suffix_drafter_test controller_test \
  draft_policy_test conv_cache_test coupled_draft_test
.venv/bin/ctest --test-dir build-source-sync-cpu \
  -R '^(gguf_reader_test|gguf_split_test|suffix_drafter_test|controller_test|draft_policy_test|conv_cache_test|coupled_draft_test)$' \
  --output-on-failure
```

Only those seven C++ targets were built/run. On arm64, this selection deliberately avoids the x86 AVX2/AVX-512
expert targets, full engine and exhaustive BF16 bit-pattern test. No missing real-model or HIP fixture is
presented as a pass.

## Hardware boundary

The T5810 authority record still requires holding model stress/stability work pending host RAM ECC/MCE
investigation. A read-only resource check was made, but no remote build or model run was performed for
this merge. No driver/ROCm/network/tuning/thermal changes, container restart or remote API listener were made.

The byte-permutation/SWAR, IQ/Q2_0 kernels, extended width-8 wave64 probe, HIP LDS limits, per-card active
clock calibration, operator oracles, two-card handoff and real IQ2_XS generation must be rerun after the
hardware issue is addressed. MI60 requires separate hardware validation. The original v0.1.30 GPU results
in `../20261001` remain unchanged; they do not validate v0.1.31 or establish a post-merge speedup.
