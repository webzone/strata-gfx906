# IQ3_S upstream text engine + HIP GPU vision restoration

T5810, two MI50 32 GiB / gfx906, ROCm 7.2.4, Xeon E5-1650 v3.
Accepted **2026-10-05 05:43:18 UTC**, service left running on **192.168.2.8:8082**.
The owner requested the original faster v0.1.39 upstream text binary, explicitly retained GPU vision
and four concurrency, and manually stopped the preceding service. Fresh preflight observed both cards
at roughly 10 MiB/0% use, about 105 GiB available RAM, no Docker workload and no 8082 listener.

## What changed

Only the saved IQ3_S config's `exe` and `log` changed. The text engine is restored to its original
`/data/strata-gfx906/eval-v0139/build-upstream/strata`, SHA256
`1fc9cde6f2a67b3a830186b022d8fd320049bd56ce275b0576a39964f726040f`.
This is the **existing upstream HIP backend mode** of the synchronized v0.1.39 checkout, built with
`STRATA_HIP_GFX906=ON`, real architecture gfx906 and `STRATA_ENABLE_HIP=OFF`;
[the retained build-mode record](text-build-mode.json) also records prefill MMQ OFF and portable AVX2.
The artifact was reused byte-for-byte, not rebuilt. Its build cache points at the shared checkout and
does not independently establish the original compile-time commit; the recorded binary hash identifies
the exact original owner-deployed artifact. The frontend source at startup was `c465b79` (runtime code
unchanged since `2a96127`). The fork's source/backend and `build-hip/strata` were not replaced.

The independent HIP vision binary is unchanged, SHA256
`18f00324223fb1f511a04c216cb4b786287d96166ffa04f9520cde006b1801fb`.
It reads the same pinned original BF16 mmproj, expands weights exactly to FP32 in memory, uses
physical GPU0, six CPU preparation threads and a 1,024 image-token cap. Its FP32/original-BF16
precision limitations remain in [the encoder validation](../20261005-vision-v0.1.39/README.md).
Existing GGUF, pack, MTP, 262K/int8/KV-resident-32768, split-24, spec-4, four slots, sampling and env
are retained. The IQ2_XS config still selects the fork binary and is unchanged.

The existing `./run-iq3-s.sh` lock/port guard started this configuration. It remains `0.0.0.0:8082`,
without an API key per owner preference. PIDs at acceptance: server **27648**, vision **27660**,
upstream text engine **27678**. No other workload was stopped; no packages, networking, drivers/ROCm,
thermal/tuning or automatic-startup settings changed.

## Runtime acceptance

| Check | Observed result | Evidence |
| --- | --- | --- |
| Actual GPU vision | Encoder `--gpu`, HIP mask `0`, HIP/gfx906 log, 1,024-token warm-up and GPU allocation | [Process/config snapshot](postflight.json), [engine log](server-engine.log.acceptance-snapshot) |
| Actual upstream engine | Original binary path/hash, text mask `0,1`, `--batch 4 --vision`, version 0.1.39 | [Receipt](deployment.json), process snapshot |
| Text | `17 + 25` → `42`, HTTP 200 | [Response](api-checks/text.response.json) |
| Real newspaper photo | `MEN WALK ON MOON`, HTTP 200, 2.87 s | [Response](api-checks/image.response.json), [API summary](api-checks/summary.json) |
| Four simultaneous texts | Four HTTP 200/nonempty responses; four admitted requests and four occupied slots observed together | [Live samples](api-checks/parallel-live.json), API summary |
| Fresh 1,024-token image during four texts | All five HTTP 200/nonempty; image 17.26 s including its wait | [Mixed summary](mixed-check/summary.json), [live samples](mixed-check/live.json) |
| Three sequential text generations | 512 tokens each; decode 49.7 / 52.7 / 55.2 tok/s; weighted 52.41 | [Decode summary](decode-check/summary.json), exact requests/responses |
| LAN health from development Mac | `ok`, loaded, images true, no API key | [Raw response](lan-health.json) |

The decode cases use short fresh prompts, greedy sampling, thinking disabled and MTP enabled, with GPU
vision loaded and four slots configured. The weighted rate is total real generated tokens divided by
total engine-reported decode seconds. It is **not matched A/B**, long-context pi-agent performance or
four-client throughput, and does not guarantee every future turn reaches 50 tok/s. The mixed case
checks liveness; its synthetic image is not an image-quality benchmark. Full 262K, broad vision quality,
model parity, MI60 and long-soak validation remain outstanding. No additional unit/CTest suite was
run during this binary/config-only restoration; the unchanged frontend/encoder regression evidence
is retained with the preceding deployment.

Raw startup/stdout snapshots are retained verbatim. The request scripts are archived beside this file.
All original requests, including large base64 image payloads, remain in
`/home/chris/dev/strata-gfx906/logs/restore-upstream-vision-20261005T053548Z/` alongside mode-0600
before/staged config backups. Public image fixtures/hash identities are the same as the preceding
[IQ3_S image validation](../20261005-iq3-s-vision-deployment/README.md). The scripts protect their
output directories; reproducing these checks needs a new directory and an owner-authorized window.
The live logs continue receiving traffic; archived acceptance snapshots are not rewritten.
