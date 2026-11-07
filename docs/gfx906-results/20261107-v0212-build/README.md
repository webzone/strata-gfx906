# v0.2.12 / 0.1.58 gfx906 build + deploy + parameter optimization (2026-11-07)

- Upstream tag `v0.2.12` / `146df3c4`, merge `519f148e`.
- Text `build-text-rocm10-v0212/strata` SHA256 `4e0cfecb5c038c4d…` (42,068,896 B), version `0.1.58`.
- Vision `build-vision-rocm10-v0212/bin/strata-vision-hip` SHA256 `1435091542d4b53b…` (48,830,632 B).
- Host ctest: 266/278 passed (96%), `isa_guard_check` passed (2.45 s).
- Deployed with `install -m 0755` over `build-text-rocm10/strata` and `build-vision-rocm10/bin/strata-vision`.
- Backups preserved: `build-text-rocm10/strata.v0211-6f461815.bak` / `build-vision-rocm10/bin/strata-vision.v0211-ab6c9ef0.bak`.
- No restart. Running processes (server 1988446, text 1988499, vision 1988460) kept v0.2.11-3 / `0.1.57` inodes.
- Startup parameters in `run-iq3-s.json` audited against T5810 hardware:
  - `--conversation-cache-min-free-mib`: `16384` -> `4096`
  - `--conversation-cache-mib`: `16384` -> `6144`
  - Fixed 100% admission skip failure (`need 948 MiB plus 16384 MiB floor, 11198 MiB available`) where zero conversations could be parked.
  - Takes effect upon next owner restart of 8082 (`./run-iq3-s.sh`).
- Full machine logs: `~/dev/strata-gfx906/logs/v0212-build-20261107T225206Z/` and `logs/v0212-deploy-20261107T231456Z/`.
