# T5810 v0.1.34 in-place deployment — no tests, no startup

The owner confirmed manually stopping the legacy service and requested overwriting the original deployment
instead of keeping the separately staged release. The no-test/no-start instructions remain in force.

## Installed state

- Checkout: `/home/chris/dev/strata-gfx906`, branch `gfx906`, fast-forwarded to `74e6fd0`.
- Binary: `/home/chris/dev/strata-gfx906/build-hip/strata`, replacing the old engine in place.
  Compiled from `0e12f868f11497a224172294bffbf7658d935534`; the checkout differs only in documentation.
  17,592,272 bytes; SHA256 `d8a59558877074f605116e396f305d0d1d1cf8a7c2f2512918b1c4fbb9f440cb`.
- Config: `/home/chris/dev/strata-gfx906/run-iq2-xs.json`, mode 0600, host `0.0.0.0`, port `8082`.
  The private API key generated during preparation was retained. No key/config is committed.
- Launcher: `/home/chris/dev/strata-gfx906/run-iq2-xs.sh`, replaced in place, foreground only.

Existing IQ2_XS shards/pack, MTP runtime, all model arguments, GPU order `[0, 1]`, sampling,
262K/int8/spec/split-24 settings, the three ROCm environment values, project `.venv` and pinned llama.cpp source
were preserved. No packages, weights or duplicate main-model `experts.bin` were created.
The original `build-hip` directory was successfully reconfigured with all three test switches OFF.
The already built real gfx906 HIP Release binary was installed atomically without another build; its hash matches.
Static ELF dependency inspection found no dependency on the temporary release directory.

The temporary `/data/strata-gfx906/releases/v0.1.34-0e12f86` Git worktree and
`run-iq2-xs-v0.1.34.sh` alias were removed after installing and preserving the raw build evidence.
No separate runnable release remains. Original configure/build logs and receipt were copied verbatim to
`/home/chris/dev/strata-gfx906/logs/deploy-v0.1.34-in-place-20261002T023704Z/preparation/`.
The parent log directory contains the private in-place configure/log/receipt files outside `preparation/`.

## Evidence boundary

`deployment-receipt.json` records the installation/configuration/file checks and owner-confirmed stop.
**No CTest, GPU numerical probe, inference/API/image request or server startup/restart was performed.**
At finalization there was no Strata process or 8082 listener. Root free stayed about 5.7 GiB.
No autoboot/autorestart, Docker, network, driver/SDK or tuning/thermal service changes were made.
Compilation and static installation checks are not GPU/model/API readiness, parity, speed or stability acceptance.
The [initial staging receipt](../20261002-prepared-v0.1.34/README.md) is historical, not the current layout;
its raw JSON remains unchanged, including the then-unknown exit cause subsequently clarified by the owner.

## Manual startup on T5810

When the owner is ready and port/GPU resources are available:

```bash
cd /home/chris/dev/strata-gfx906
./run-iq2-xs.sh
```

Clients must use the API key in the installed private config. The deployment remains stopped until manual startup.
