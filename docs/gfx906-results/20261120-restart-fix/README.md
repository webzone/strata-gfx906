# T5810 restart failure and fix — 2026-11-20 08:13–08:27 UTC

## Symptom

The owner restarted 8082 (`./run-iq3-s.sh`). The server exited before it bound the port:

```
serve.server_parts.engines.EngineDied: the engine exited before it was ready
(see /home/chris/dev/strata-gfx906/logs/cachefix-rocm10-20261007/iq3-engine.log)
the engine process ended: exit code 1
```

The engine's own last line was:

```
strata generate: read_expert_profile: cannot open /home/chris/dev/strata-gfx906/data/expert-profile.bin
```

## Root cause

Upstream's 0.2.16 layout move (`4058aa782`, "Layout move for 0.2.16") moved the tracked file
`data/expert-profile.bin` to `terminal/data/expert-profile.bin`. The v0.2.16 merge therefore deleted the
old path from the checkout. The saved run config `run-iq3-s.json` still names
`/home/chris/dev/strata-gfx906/data/expert-profile.bin`, so the engine could not read its expert profile.

The old trio kept working only because its process held the deleted inode. The restart lost that inode,
so the same config path became fatal. Nothing else in the config moved: every other path in
`run-iq3-s.json` still exists (`exe`, `tokenizer`, `cwd`, `vision.exe`, `vision.mmproj`, `vision.model`,
`--pack`, `--native`, `--ple-gguf`, `--mtp`).

## Fix

The fork keeps its own untracked copy at the path the config names, byte-identical to upstream's new file:

```
8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf  data/expert-profile.bin
8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf  terminal/data/expert-profile.bin
196,632 B each
```

The owner's config stays byte-unchanged (md5 `e996f0f9e1e60b2dade05f188ee8c4e3`, last owner edit
2026-11-17 19:17 UTC), so the deploy guards in `tools/deploy-main-sync.sh` stay valid. An untracked file
survives later layout moves, which a tracked path does not.

One script does the work and verifies its own result: `tools/restart-8082.sh` (`--dry-run` first, then the
real run; `--adopt` re-verifies a service that is already starting). It checks the port and process state,
the two deployed binary hashes, the config md5 and every config path, restores the profile copy, starts
`./run-iq3-s.sh` detached, waits for `loaded: true`, then runs `verify-v0216.sh` and `strata-current.sh`.

## Second fault found by the restart

`strata-current.sh` reported `违反规则: 运行产物缺构建身份记录`. The v0.2.15 record used flat keys
(`text_sha256`, `vision_sha256`) and the v0.2.16 record used `build_dir` / `flags` / `llama_pin`, while the
checker reads top-level `upstream`, `merge`, `engine_change`, `llama_cpp` and per-side `path`, `sha256`,
`bytes`. The running pair had no record in that shape, so the picture reader printed `NORECORD`. This round
wrote `build-identity.json` (this directory) in the shape the checker reads. After it, `strata-current.sh`
prints `结论: 符合现役规则(上游文本 + fork HIP 视觉,均 ROCm 10)` and exits 0.

## Verification after the restart (`verify-v0216.sh`, exit 0)

- Text binary `a9c9ac4a7347060393dc5a08fac33a976a268f73cc06e4797d398b4d389dea3b` and picture reader `14350915…` match the expected artifacts.
- Running engine `0.1.62` — the first time this binary has run on the card.
- One real greedy generation answered `STRATA OK` (19 prompt / 4 completion tokens, `cached_tokens: 0`).
- PIDs: server `3950911`, picture reader `3950924`, text engine `3950947`; `0.0.0.0:8082` listening.
- Live traffic after the restart, one engine line: prompt 154,598 tokens = 154,396 reused + 202 read in
  2,200 ms (**91.8 tok/s real**, **70,272 effective tok/s**), 710 generated in 13,702 ms
  (**51.8 tok/s decode**), drafts accepted 441 of 628, decode expert cache hit rate 100.0 %, KV streaming
  92.15 % of 2,775,978 block reads hit VRAM.

## What this does not prove

No CTest run and no benchmark. The rates above are one live-traffic line, not a protocol run, so they are
not comparable with the v0.1.42 workload records in `docs/GFX906.md`. A same-protocol workload run on
`0.1.62` is still missing.

## Files

| File | Content |
|---|---|
| `failed-start-1.log` | the failed start: the server traceback and the engine's last lines |
| `launch.log` | the successful start: expert load and readiness |
| `verify.txt` | `verify-v0216.sh` output, exit 0 |
| `strata-current.txt` | `strata-current.sh` output after the identity record, exit 0 |
| `build-identity.json` | the build-identity record for the running pair, in the shape `strata-current.sh` reads |
| `service-state.txt` | listener, PIDs, health, engine version, both binary hashes, config md5, both profile copies |
| `restart-8082.sh` | the script that fixed and verified this round |

Machine logs: `logs/restart-20261120T081520Z.log` (failed), `logs/restart-20261120T081859Z/` (started),
`logs/restart-20261120T082641Z/` (adopt run: identity, verify, strata-current).
