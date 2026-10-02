# T5810 v0.1.34 preparation — no tests, no startup

The owner requested deployment for `0.0.0.0:8082`, then explicitly instructed that no tests or startup be performed.
This receipt records **source/configuration preparation and a successful HIP build**, not runtime acceptance.
No new engine/server process, inference request, CTest, GPU probe or image test was run.

## Prepared artifacts

- Source: `0e12f868f11497a224172294bffbf7658d935534`, Strata v0.1.34, upstream `1678de3`.
- Isolated deployment worktree: `/data/strata-gfx906/releases/v0.1.34-0e12f86`.
- Binary: `build-hip/strata`, 17,592,272 bytes, SHA256
  `d8a59558877074f605116e396f305d0d1d1cf8a7c2f2512918b1c4fbb9f440cb`.
- Private config: `/data/strata-gfx906/releases/v0.1.34-0e12f86/run-iq2-xs.json`, mode 0600.
  It sets host `0.0.0.0`, port `8082` and a generated API key. The key/config is not committed.
- Manual entry point: `/home/chris/dev/strata-gfx906/run-iq2-xs-v0.1.34.sh`, a versioned symlink to
  the new release's foreground launcher. The old `run-iq2-xs.sh` is unchanged.
- Existing project `.venv` and llama.cpp source at pin `3cf03257f219afbe7334045ff7c6a06ac68c627d` are linked
  into the release. No packages or weights were downloaded.

The original checkout stayed at `be23587e559060c9d6a704e6b34490f5be9fce7b`; only its remote tracking refs were
fetched and the isolated worktree/launcher alias were added. Its old binary, config and launcher were not replaced.
Existing IQ2_XS model shards, pack, MTP runtime, engine arguments, GPU order `[0, 1]`, sampling, layer split,
262K/int8/spec settings and three ROCm environment entries were preserved. No main-model `experts.bin` was generated.

## Build evidence

System ROCm 7.2.4 / CMake 4.2.3 / Python 3.14.4 on T5810. Release, real gfx906 architecture, explicit experimental
opt-in, portable AVX2 host, native experts and MMQ. Configuration succeeded (exit 0); target `strata` built with
2 jobs (exit 0). All three test registration switches were OFF.

Exact commands:

```bash
cd /data/strata-gfx906/releases/v0.1.34-0e12f86
cmake --preset hip-gfx906 \
  -DSTRATA_BUILD_TESTS=OFF -DSTRATA_HIP_MULTIGPU_TESTS=OFF \
  -DSTRATA_BUILD_CONVERSATION_TESTS=OFF -DSTRATA_NATIVE_EXPERTS=ON
cmake --build --preset hip-gfx906 --target strata
```

The compile environment used system `/opt/rocm`, cleared `HSA_OVERRIDE_GFX_VERSION` and placed temporary files
in the release's `.build-tmp` on `/data`. Complete raw configure/build logs remain private at
`/data/strata-gfx906/releases/v0.1.34-0e12f86/logs/configure.log` and `build.log`; full compiler-warning logs are
not committed. `deployment-receipt.json` is a small, secret-free receipt of exits/settings/hash/static observations.
Compilation and file/config inspection do not establish GPU numeric correctness, model parity or speed.

## Process-state boundary

Preflight found the legacy server/engine PIDs 2931/2936 running on 8082. The initial finalization guard later failed
because both had exited. A read-only follow-up found no Strata process or 8082 listener, idle cards (~10 MiB VRAM each),
and about 105 GiB available RAM. The exit reason was not established; no stop/start/restart command was issued by
the preparer, and the service was not restored. No visible OOM/GPU-reset messages appeared in the bounded kernel
query; this is not proof of the exit cause. EDAC CE/UE was 0/0 in the snapshots, not long-soak validation.
Root free stayed about 5.7 GiB, `/data` about 111 GiB. No Docker, network, driver/SDK, tuning or thermal service changed.

## Manual startup

On T5810, when the owner is ready and port 8082/GPU resources are available:

```bash
cd /home/chris/dev/strata-gfx906
./run-iq2-xs-v0.1.34.sh
```

Use the API key stored in the new private config. There is no autoboot/autorestart.
The new executable was never run during this preparation, so model generation, numeric/operator probes,
clock calibration, inter-GPU handoff, API readiness, long-context behavior and stability remain unverified for it.
