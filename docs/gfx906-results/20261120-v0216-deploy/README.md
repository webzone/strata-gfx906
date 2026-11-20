# T5810 v0.2.16 install without restart — 2026-11-20

Source: fork commit `3961026f0` (merge of upstream tag `v0.2.16` / `64524fb9f`), engine version `0.1.62`.
Machine: T5810, dual MI50, ROCm 10 (`/opt/rocm-10.0/core-10.0`), checkout `/home/chris/dev/strata-gfx906`.
Owner instruction: install, no restart, no tests. Nothing was started, stopped or restarted, and no model
request or CTest run was made.

| Item | Value |
|---|---|
| Text engine installed | `build-text-rocm10/strata`, SHA256 `a9c9ac4a7347060393dc5a08fac33a976a268f73cc06e4797d398b4d389dea3b`, 26,960,536 B, `strata 0.1.62` |
| Text build | `build-text-rocm10-v0216` from `engine/`, `-DSTRATA_HIP_GFX906=ON -DSTRATA_ENABLE_HIP=OFF -DCMAKE_HIP_ARCHITECTURES=gfx906 -DSTRATA_PORTABLE=ON -DSTRATA_PREFILL_MMQ=OFF -DSTRATA_BUILD_TESTS=ON -DSTRATA_BUILD_CONVERSATION_TESTS=ON`, `-j 2`; 35,919 log lines, 0 errors, 3,837 warnings |
| Picture reader | compile check only: `build-vision-rocm10-v0216/bin/strata-vision-hip`, SHA256 `51f830a42d7cf6f9b5d589142912a129f13a28cb5b0e4a7db5c4c4f4b74c1b05`, 49,578,864 B, 0 errors, 0 warnings; the deployed reader `943a54bd…` stayed in place |
| Backup kept | `build-text-rocm10/strata.v0215-b424d55b.bak`, SHA256 `b424d55bea4a8b791c4d2594e4d31907bf1893fe29773542fc4c4877e9187568` |
| Running trio | PIDs `2863631` (server), `2863677` (text engine), `2863645` (reader) unchanged before and after; `/proc/2863677/exe` shows `(deleted)` |
| Service state | `/health` `status: ok, loaded: true`; `/metrics` engine still `0.1.61` |
| Run config | `run-iq3-s.json` md5 `e996f0f9e1e60b2dade05f188ee8c4e3`, unchanged in this round (last owner edit 2026-11-17 19:17 UTC) |
| Layout move | engine → `engine/`, serving layer → `terminal/`, picture reader → `terminal/tools/vision`; `tools/gfx906_launch.sh` now runs `terminal/serve/server.py` with `terminal/.venv/bin/python`, and `terminal/.venv` links to the existing checkout `.venv` |
| Machine scripts | `~/mi50-t5810/scripts/verify-v0216.sh` (post-restart), `rollback-v0216.sh` (`--dry-run` verified), `strata-current.sh` expectation moved to `a9c9ac4a…` |

Files: `deploy-v0216.sh` (the one-command script as run), `deploy-steps.log` (its step output, checkout
rename noise removed), `build-identity.json`, `configure-text.log`, `configure-vision.log`,
`vendored.txt` (pin check), `checkout.txt`. Full build logs stay on T5810 under
`logs/v0216-build-20261120T025858Z/`.

Not tested: model parity, generation output, tok/s, CTest, the gfx906 picture reader at runtime, and the
8082 restart path. The compile of the engine and of the reader is the only new evidence.
