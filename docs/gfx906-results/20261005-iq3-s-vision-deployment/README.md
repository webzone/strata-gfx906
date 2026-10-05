# T5810 IQ3_S HIP vision deployment — 2026-10-05

The owner manually stopped the prior 8082 workload, authorized IQ3_S vision deployment, and
explicitly requested retaining four concurrent requests. Acceptance snapshot: **03:51:18 UTC**.
The newly started service was left running; no historical container was restored.

## Identity and configuration

| Item | Deployed value |
| --- | --- |
| Application source | `2a96127b76ef9fdf957f274cf62f51e56e5023d6`, upstream v0.1.39 / `6f32ec0` |
| Hardware | Two MI50 32 GiB / gfx906, Xeon E5-1650 v3, ROCm 7.2.4 |
| HTTP | `0.0.0.0:8082`; LAN `192.168.2.12`; no API key per owner preference |
| Model ID | `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S` |
| Vision | Real HIP GPU0, original BF16 mmproj expanded losslessly to FP32 in memory, six threads, cap 1,024 |
| Text engine | Real gfx906 fork, split 24 across GPU0/GPU1, `--batch 4 --vision` |
| Context/KV | 262,144 configured, int8, 32,768 resident cells |
| Speculation | Existing MTP runtime, spec-4, min-p 0.5; existing sampling/env retained |
| PIDs at snapshot | Server 1015929, vision 1015939, text engine 1015961 |
| Manual entry point | `/home/chris/dev/strata-gfx906/run-iq3-s.sh` |

The original checkout/config were updated in place. Existing GGUF, native pack and MTP runtime were
reused. Fresh full hashes of both IQ3_S shards passed before allocation. The saved IQ2_XS configuration
is byte-identical, but its launcher shares the newly installed v0.1.39 fork text binary. All model,
mmproj and binary hashes are in [deployment.json](deployment.json). Its `source_revision` records the
earlier preparation revision; `deployed_app_revision` records the application revision actually started.
The text binary was copied byte-for-byte from the validated v0.1.39 fork build, and the vision binary
from the preceding isolated HIP validation. No new system packages were installed.

## Checks and evidence

| Check | Result | Raw evidence |
| --- | --- | --- |
| HTTP health/models/status/metrics | Images enabled, loaded, actual batch slots 4 | [API summary](api-checks/summary.json) and corresponding response files |
| Text | `17 + 25` → `42`, HTTP 200 | [Response](api-checks/text.response.json) |
| Real photo | `MEN WALK ON MOON`, HTTP 200 | [Response](api-checks/image.response.json) |
| Four simultaneous texts | Four HTTP 200/nonempty responses; maximum live requests and occupied slots both 4 | [Live samples](api-checks/parallel-live.json), API summary |
| Four texts plus fresh image | All five HTTP 200/nonempty responses; image input 1,024 tokens | [Mixed summary](mixed-check/summary.json), [live samples](mixed-check/live.json), engine log |
| Local service/runconfig regression | 150 passed | [Raw unit log](service-tests.log) |
| Linux coordination regression | Two passed | [Raw unit log](batched-vision-linux-tests.log) |
| LAN access from development Mac | Health `ok`, images true | [LAN health response](lan-health.json) |
| Resource/process/config checks | No HSA override; GPU masks `0` for vision and `0,1` for text; listener owned by new server | [Postflight](postflight.json) |

The raw startup/runtime snapshots are [engine log](server-engine.log.acceptance-snapshot) and
[server stdout](server.stdout.log.acceptance-snapshot). The live private log files continue receiving
traffic; the archived snapshots are not rewritten. Unit logs include deliberately injected failures
and mock timing rates; these are **not deployed model failures or performance measurements**.

The request scripts are [API check](check-iq3-vision-api.py) and
[mixed check](check-iq3-vision-mixed.py). Exact original requests, including base64 payloads, are retained
under `/home/chris/dev/strata-gfx906/logs/deploy-iq3-s-vision-20261005/{api-checks,mixed-check}/`.
The real image is the already archived [reference newspaper](../20261005-vision-v0.1.39/reference-newspaper.jpeg),
SHA256 `2dff664c0c8aaea18aff8cbe7e868845b775e90cdd7a0bac98df709b131deaa3`.
The fresh synthetic BMP is retained at
`/data/strata-gfx906/vision-gfx906-20261005/accepted-1024-limit-gpu0/limit.bmp`, SHA256
`ebb1d9168f07b51d8e91c9b6353d638b19ff5fb1d5ebd24eacc2b3d57f841d63`.
The scripts refuse an existing output directory, protecting original evidence; they encode the exact
request bodies and fixture locations used. Re-running requires a new evidence directory and an
owner-authorized workload window.

The GPU coordination fix holds new admissions while existing requests drain before image encoding;
it does not reduce the engine's four slots. Upstream `/props.total_slots` and `/slots` are compatibility
stubs hardcoded to one. Use `/v1/status.concurrency.serving`, `/metrics.engine.batch_slots` and actual
live slot samples when assessing concurrency.

## Scope and recovery

This is bounded HTTP/liveness acceptance with short prompts and one real-photo transcription. The
1,024-token synthetic image checks the configured cap, not visual quality. 262K is a retained setting,
not full-context validation. Broad vision quality, model/logit parity, fairness, long-soak stability,
and MI60 validation remain outstanding. These request durations are not a controlled throughput
benchmark. The FP32 CPU/GPU and original BF16 arithmetic limits remain documented in the
[encoder validation](../20261005-vision-v0.1.39/README.md).

The private deployment directory retains mode-0600 config/launcher backups and the previous shared
text binary under `backup/`. Its live `deployment.json` and full requests are authoritative for this
deployment. The launcher still holds its model lock and refuses an occupied 8082; switching models
requires the owner to stop the current service first. No automatic startup/restart, driver/ROCm,
network or thermal-service changes were made.
