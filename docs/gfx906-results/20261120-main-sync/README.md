# T5810 sync of upstream `main` `58ee32277` — 2026-11-20 04:07 UTC

One script ran on T5810: `tools/deploy-main-sync.sh` (`--dry-run` first, then the real run, exit 0). A
second run with `--tip` repeated every check after the record commit was pushed. No build, no stop, no
restart, no model request, no CTest run. The fork merge is `b3aaee7e4`
(upstream `main` `58ee32277`, four commits after tag `v0.2.16` / `64524fb9f`).

| Check | Result | File |
|---|---|---|
| Checkout | `3c28629c9` → `b3aaee7e4` (the merge), then `e2eb4c4191c9f7f08142d439d0b1cc6e39b31233` (the commit that carries this record), branch `gfx906`, upstream `58ee32277` and merge `b3aaee7e4` both ancestors | `checkout.txt` |
| Round scope | 11 paths from the merge (+192 / −94): `README.md`, `SUPPORTERS.md`, 7 translations, `tools/docker/build.sh`, `docs/GFX906.md`; 23 paths since `3c28629c9` once this directory is in the tree | `sync-scope.txt`, `sync-paths.txt` |
| Engine scope | 0 paths under `engine/`, `terminal/`, `serve/`, `tools/vision`, `tools/hip` → no rebuild needed | `sync-paths.txt` |
| Workflow files | `git ls-files .github/workflows` prints nothing | `workflows.txt` (empty) |
| Vendored pin | `check_vendored: ok`, pin `3cf03257f`; strata-ggml 62 files (2 edited), strata-gguf-py 116, strata-llama 918, stb 2 | `vendored.txt` |
| Deployed text engine | SHA256 `a9c9ac4a7347060393dc5a08fac33a976a268f73cc06e4797d398b4d389dea3b`, 26,960,536 B, `strata --version` → `strata 0.1.62` | `sync-run.log` |
| Deployed picture reader | SHA256 `943a54bd7a683446a63904ae37f39f2f40add0f79a10b8a33ef9e60582f02d7f`, 49,529,624 B | `sync-run.log` |
| Run config | `run-iq3-s.json` md5 `e996f0f9e1e60b2dade05f188ee8c4e3`, parses | `sync-run.log` |
| Server compile | `terminal/serve/server.py` compiles with the checkout venv (Python 3.14.4); `terminal/.venv -> ../.venv` | `sync-run.log` |
| Processes | PIDs `2863631`, `2863645`, `2863677` unchanged before and after the sync | `pids.txt` (empty diff) |
| Listener | `0.0.0.0:8082` still held by PID `2863631` | `sync-run.log` |
| Health | `status: ok`, `loaded: true`, model `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S` | `health.json` |
| Running engine | `/metrics` still reports `"engine": "0.1.61"` — the installed `0.1.62` file takes effect on restart | `engine-version.txt` |
| `strata-current.sh` | exit 1: PID `2863677` still holds the deleted v0.2.15 inode | `strata-current.txt` |

Machine state before the sync: 8082 listening; `MemAvailable` 6,183 MiB; 42 GiB free disk; VRAM used
31,948,894,208 B on GPU 0 and 33,610,711,040 B on GPU 1 by the live engine. No container was running.

## What this round does not prove

No build, no model request and no CTest run. So there is no new throughput, parity or gfx906 runtime
evidence here. The `0.1.62` binary installed in the v0.2.16 round (`docs/gfx906-results/20261120-v0216-deploy/`)
remains compile-verified only. The restart path was not exercised.

## Machine records

`/home/chris/dev/strata-gfx906/logs/main-sync-20261120T040711Z/` and
`logs/main-sync-20261120T040925Z/`, plus `logs/main-sync-run.log` and `logs/main-sync-run-tip.log`
(`sync-run.log` and `sync-run-tip.log` here are those two files). The script lives at
`/home/chris/dev/strata-gfx906/tools/deploy-main-sync.sh` (untracked, like the earlier rounds' deploy
scripts); `deploy-main-sync.sh` in this directory is that same file.

## Owner's next step

Restart 8082 with `./run-iq3-s.sh`, then `bash ~/mi50-t5810/scripts/verify-v0216.sh` (the post-restart
check for the `0.1.62` engine, still valid because this round changed no binary).
