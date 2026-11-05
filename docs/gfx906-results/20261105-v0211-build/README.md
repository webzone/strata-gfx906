# v0.2.11-3 / 0.1.57 gfx906 build + deploy (2026-11-05)

- Upstream tip `9f7d977d` (v0.2.11 + 3), merge `9512c550`, ISA-guard fix `c1e19467`.
- Text `build-text-rocm10-v0211/strata` SHA256 `6f4618150e833f1c…` (41,760,312 B), version `0.1.57`.
- Vision `build-vision-rocm10-v0211/bin/strata-vision-hip` SHA256 `ab6c9ef0076e68a7…` (48,830,632 B).
- Deployed with `install -m 0755` over `build-text-rocm10/` and `build-vision-rocm10/bin/strata-vision`.
- No restart. Running processes kept v0.2.4.2+ / `0.1.49.2` inodes. `run-iq3-s.json` unchanged.
- Full machine logs: `~/dev/strata-gfx906/logs/v0211-build-20261105T202801Z/` and
  `logs/v0211-deploy-20261105T203936Z/`.
