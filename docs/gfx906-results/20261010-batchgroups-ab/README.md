# batch-groups A/B on the T5810 dual MI50 — `--batch-groups 1 --batch-mtp` vs `--batch-groups 2`, four concurrent (2026-10-10/11 UTC)

Requested comparison: `--batch-groups 1` with MTP at 4 concurrent slots vs `--batch-groups 2` at 4
concurrent slots. Layer-split slot drafters (`--batch-mtp` with a layer split) exist only from upstream
v0.1.42, so both arms ran the **same freshly built v0.1.42 engine** — the rebuild is bit-identical to the
2026-10-10 18:32–18:51 build-check artifact recorded in `docs/GFX906.md`
(SHA256 `0550f3f224351a7927fece988ae596cd0f5bbaa8d6e39a59dfd85c1e991226b2`, merge `7f104978` =
upstream `61b3fb5d`).

## Setup (identical for both arms)

- T5810, 2× AMD Radeon Instinct MI50 32 GB (`gfx906:sramecc+:xnack-`), Xeon E5-1650 v3,
  ROCm `10.0.0-gfx906+20260917140126` (both binaries load `/opt/rocm-10.0/core-10.0` only).
- Model: GSQ-RCO IQ3_S (HF revision `ed59f92…`), native pack + PLE shard, `--layer-split 24` over GPU0/1,
  `--batch 4`, `--kv int8 --kv-resident 32768`, `--spec 4 --mtp …/mtp/rt --mtp-window 32768`,
  `--pcie-frac 0 --adapt-every 0`, `--prompt-cache 6`, conversation cache 8192 MiB / 4 slots,
  `--vram-reserve-mib 1024`, `--trim-stage-weights`, `--vision` (fork encoder `709fc4d2…`, unchanged).
  The engine logged **100 % of experts resident in VRAM on both cards** in both arms, so no CPU expert
  path is involved in either arm.
- Arm A: `--batch-groups 1 --batch-mtp`. The engine logged `--batch-mtp: 4 slot drafters on CUDA1
  (48 MiB of private K/V state and buffers each, sharing the solo drafter's weights)`.
- Arm B: `--batch-groups 2` (the pipelined default; batch-window MTP off by construction, solo MTP on).
- Server: the T5810 checkout's `serve/server.py` on `0.0.0.0:8082`, no API key; one engine process per arm,
  GPUs verified idle before each arm.
- Client: `ab_client.py` (stdlib only, on the server host, `127.0.0.1:8082`). Headline numbers come from
  the server's final streaming chunk (`timings`: `prompt_ms` / `predicted_ms`, `usage.cached_tokens`),
  not from client clocks. All four requests of a round are sent simultaneously over threads.
- Greedy (`temperature 0`), thinking disabled (`reasoning_effort: "off"` +
  `chat_template_kwargs {"enable_thinking": false}`; verified 0 `reasoning_content` chars).

## Decode test — 4 concurrent, 1 warm-up + 5 measured rounds

Each round: 4 unique short prompts (24 distinct essay topics, no reuse), `max_tokens 256`; every stream
ended `finish_reason=length` with `cached_tokens=0` (round validity checked and recorded per round).

| Arm | Aggregate decode tok/s (median of 5 rounds) | Rounds (wall) | Per-request server tok/s (median) | TTFB median / max |
| --- | ---: | --- | ---: | ---: |
| A: `--batch-groups 1 --batch-mtp` | **48.3** | 48.3 / 54.7 / 45.5 / 56.9 / 46.1 | 14.8 | 2.7 s / 4.9 s |
| B: `--batch-groups 2` | **87.9** | 76.9 / 87.7 / 88.1 / 88.0 / 87.9 | 24.2 | 2.6 s / 3.6 s |

Arm A's window log (serial path prints per-run summaries): ~6.07–6.37 rows/window at 92–98 ms/window,
**MTP proposals accepted 62.6–67.3 %** (the drafts ran; the BDONE protocol does not report them
per-request). Arm B (pipelined) prints no window summaries. Even Arm B's worst round (76.9) is far above
Arm A's best (56.9). With all experts resident in VRAM this matches the upstream note that `--batch-mtp`
pays only when experts do not fit in VRAM, and the fork's own 2× R9700 all-resident measurement
(80.2 → 62.7 tok/s against the pipelined default).

## Prefill test — 4 concurrent, 1 warm-up + 3 measured rounds

Each round: 4 unique ~3,100-token prompts built from disjoint entry ranges (never repeated), `max_tokens 8`,
`cached_tokens=0` verified per request. Per-request real rate = the engine's `prompt N tokens = 0 reused +
N read in M ms` lines for the test reads (the slot hand-off re-reads, `… reused + 1 read in ~0 ms`, and
unrelated live traffic are excluded; live traffic that arrived between Arm A rounds is visible in the
extracts and touched no measured round).

| Arm | Real prefill per read (16 reads) | Aggregate wall tok/s (3 rounds) |
| --- | --- | ---: |
| A: `--batch-groups 1 --batch-mtp` | 412.7–419.6 tok/s | 405.9 / 409.1 / 411.0 |
| B: `--batch-groups 2` | 405.4–412.5 tok/s | 400.0 / 402.3 / 404.8 |

Prefill is a near tie (Arm A ~1.5–2 % faster); the decode result decides. The well-known pipelined-prefill
advantage (a long prompt arriving **beside decoding streams**) is a different scenario and was not tested.

## Outcome

**`--batch-groups 2` wins the 4-concurrent decode comparison by ~82 % aggregate (87.9 vs 48.3 tok/s) and
ties prefill.** The production `run-iq3-s.json` already carried `--batch-groups 2`, so the JSON settings
are unchanged; the v0.1.42 engine was installed to the deployed path `build-text-rocm10/strata`
(`bc1102ba…` kept as `build-text-rocm10/strata-v0.1.40.1`). The service was left **stopped** for the
owner to start manually.

## Files

- `ab_client.py` — the test client (exact prompts, checks, and metric sources are in the code).
- `decode-{a42,b42}.trimmed.json` / `prefill-{a42,b42}.trimmed.json` — per-round raw results
  (prompts and large `/metrics` snapshots trimmed; untrimmed copies remain on the server host in
  `logs/ab-batchgroups-20261010/`).
- `config-a-v42.json` / `config-b-v42.json` — the two engine configs (differ only in
  `--batch-groups`/`--batch-mtp` and the log path).
- `engine-log-extracts.txt` — startup verification, window summaries, draft acceptance, and prompt-read
  lines for both arms.
- `build-identity.json` — source identity and switches of the deployed v0.1.42 artifact.

## Limits

Not measured: sampled decoding, prompt lengths beyond ~3.1k tokens, prompts arriving beside decoders,
more than 4 concurrent requests, MI60, long soak, accuracy/parity beyond the recorded `finish_reason`
and `cached_tokens` checks. One engine per arm, one round order (A then B); Arm A's round-to-round
spread (45.5–56.9) is larger than Arm B's (87.7–88.1) for reasons not further investigated.
