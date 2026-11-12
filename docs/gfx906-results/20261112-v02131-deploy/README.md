# v0.2.13.1 text engine deployed on T5810, no restart — 2026-11-12 06:23 UTC

Owner instruction: push, and deploy to t5810, but do not restart. Nothing was started, stopped or
restarted. `tools/deploy-v02131.sh` (dry run first) refuses unless the source and deployed hashes both
match, then `install -m 0755`.

- Text: `build-text-rocm10/strata` is now **`74bfc9bffe99bc60…`** (engine `0.1.59.1`). The replaced
  v0.2.12 binary is backed up as `build-text-rocm10/strata.v0212-4e0cfecb.bak` (`4e0cfecb…`, hash
  re-checked after the copy).
- Vision: **not replaced.** `git diff 146df3c4 628f7495 -- tools/vision` is empty and the llama.cpp pin
  is unchanged, so the deployed `build-vision-rocm10/bin/strata-vision` (`14350915…`) is already current
  for this source. The v0.2.13.1 rebuild exists only as a compile check
  (`build-vision-rocm10-v02131/bin/strata-vision-hip`, `0e72e760…`, same 48,830,632 B; the hash differs
  because the build directory is embedded in the binary).
- Running trio unchanged: server 2866485, vision 2866494, text 2866533, up since 2026-11-08 01:28 UTC.
  `/proc/2866533/exe` shows `(deleted)`: the process still executes the v0.2.12 inode. `/metrics` still
  reports engine `0.1.58`.
- `strata-current.sh` now expects `74bfc9bffe99bc60…` for text and keeps `14350915…` for vision; until
  the restart it exits 1 with 跑的不是配置所指文件 … (deleted) — deployed file ≠ running process, by
  design. Previous copy: `strata-current.sh.bak-20261112T062306Z`.
- Post-install greedy request answered `STRATA OK` (19 prompt tokens, 4 completion tokens,
  `cached_tokens: 0`) on the still-running v0.2.12 engine. Both cards still 94 % / 97 % VRAM.
- `run-iq3-s.json` byte-unchanged, md5 `41bbcf3a737668e65f52f8b69a73f941`. `./run-iq3-s.sh` picks the
  new engine up with no config edit.
- New scripts: `~/mi50-t5810/scripts/verify-v02131.sh` (post-restart: both hashes, engine `0.1.59.1`,
  ROCm runtime of both processes, one `STRATA OK` generation, cache counters, `strata-current.sh`) and
  `rollback-v02131.sh` (`--dry-run` verified; restores the `4e0cfecb…` text binary and the previous
  script expectations; `--stop-current` is the only restart path; vision is untouched).
- Disk: `/` 44 GiB free before and after (81 %). New build directories add 889 MB.

**Risk:** the untested `0.1.59.1` binary now sits at the engine path, so an engine death before the
restart makes the server fatal-recover into it. `rollback-v02131.sh` and the `.bak` stay. The restart
drops the RAM conversation cache by design — clients re-prefill once.
