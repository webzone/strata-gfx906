# v0.2.13.1 gfx906 build check on T5810 — 2026-11-12 05:35–06:20 UTC

Source: fork `gfx906` at `ca5a7ec9` (merge of upstream tag `v0.2.13.1` / `628f7495`). Machine checkout
fast-forwarded `43fe263f` → `ca5a7ec9`; tracked files clean. Nothing was started, stopped or restarted:
the service (server 2866485, vision 2866494, text 2866533) held both cards the whole window
(0 % util, 94 % / 97 % VRAM).

Build directories are new; the deployed paths were untouched by the build.

| Item | Result |
|---|---|
| Text configure (`build-text-rocm10-v02131`) | exit 0 — `Strata: HIP (gfx906 compat) enabled, arch gfx906`, Clang 23.0.0 from `/opt/rocm-10.0/core-10.0` |
| Text build | exit 0, 587 targets, 0 errors, 3843 `warning:` lines, 13 m 30 s at `-j 2` |
| Text artifact | `strata` 42,090,608 B, SHA256 `74bfc9bffe99bc60…`, `strata --version` = `strata 0.1.59.1`, 20 libraries from `/opt/rocm-10.0/core-10.0`, 0 from `/opt/rocm-7.2.4`, `.hip_fatbin` with gfx906 code |
| Vision configure (`build-vision-rocm10-v02131`) | exit 0, `STRATA_VISION_HIP=ON`, `STRATA_VISION_HIP_ARCHS=gfx906`, `LLAMA_DIR=…/third_party/llama.cpp` |
| Vision build | exit 0, 275 targets, 0 errors, 0 warnings; program `bin/strata-vision-hip`, 48,830,632 B, SHA256 `0e72e7601ca79ea7…`, 20 ROCm 10 libraries, 0 from 7.2.4 |
| `tools/check_vendored.py` | ok — 127 files (94 upstream, 2 edited, 9 added, 10 copy, 12 patched), pin `3cf03257f` |

The ggml source moved this range: the engine now builds from the vendored `third_party/strata-ggml`
instead of `-DSTRATA_GGML_DIR=…/third_party/llama.cpp`, so configure prints `ggml commit: ca5a7ec9`
(the enclosing repository's commit) rather than `3cf03257f`. `tools/check_vendored.py` is the pin proof
for this tree: every listed upstream hash matches llama.cpp at `3cf03257f219afbe7334045ff7c6a06ac68c627d`.

Host ctest, GPUs visible as in previous rounds (`ctest -j 2`, 291.88 s): **268/277 passed (97 %)**,
17 skipped, `isa_guard_check` passed (2.45 s). The 9 failures are the same known set as the v0.2.12
round, and the three `conversation_snapshot_test` failures of that round now pass:

| Test | Cause |
|---|---|
| `mmvq_il_parity`, `mmvq_il_parity_89table`, `mmvq_il_parity_pf` | NVIDIA interleave tables / `native_q8_1_interleave is CUDA only` |
| `pf_wmma_parity` | WMMA prompt path not launched on gfx906 |
| `verify_parity` | `argmax_rows: not built for this backend` |
| `native_dense_parity` | f16 dequant bit differences on the dense lane (12 cases) |
| `ple_parity` | missing optional fixture `../../Q2_0/…-00002-of-00002.gguf` |
| `platform_memory_test` | `ulimit -l` = 8192 KiB blocks `mlock` |
| `conversation_copy_streams_test` | first copy stream + event took 167,936 KiB of device memory while both cards are full |

A first ctest pass was run with `HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES=-1` and gave 149/277: 128
tests print `no ROCm-capable device is detected`. That run is kept only as a record of why the device
must stay visible; it is not the result.

Full build logs (`build-text.log` 3.1 MB, `build-vision.log`, `ctest-gpus.log`) stay on the machine under
`~/dev/strata-gfx906/logs/v02131-build-20261112T053522Z/`.
