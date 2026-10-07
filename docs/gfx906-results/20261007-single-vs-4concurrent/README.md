# T5810 single-shot vs 4-concurrent probe — 2026-10-07

Two bounded phases against the live `0.0.0.0:8082` Strata service on the dual-MI50 T5810 server:
**one request alone**, then **four requests released together**, same prompt shape and same
`max_tokens`. The question is what four concurrent streams actually deliver on this build.

## Provenance

- Run by this agent, 2026-10-07 19:46:57 → 19:50:44 UTC, from the T5810 host against
  `http://127.0.0.1:8082` (no network variable in the client wall time).
- Client: `tools/gfx906_concurrency_probe.py` (this repository). It sends the requests, snapshots
  `/metrics` every second while they are in flight, and captures the engine log lines written during
  the phase. It starts nothing, stops nothing and changes no setting.
- Service state was read from `/metrics` before each phase; `--wait-idle 60` was used so a phase starts
  only when `live.running = 0` and `live.queued = 0`.
- Build under test: text engine `build-text-rocm10/strata`, SHA256
  `bc1102ba03c9ee2379699f9fb5c3cf7ce717132e7065efc5ac141606c5996853` (verified against the running
  process `/proc/<pid>/exe`; upstream gfx906 path `STRATA_HIP_GFX906=ON`, source v0.1.40.1 / `82f46a8`
  plus engine fix `74583c6`), vision encoder `build-vision-rocm10/bin/strata-vision`, both on ROCm
  `10.0.0-gfx906+20260917140126`. Model GSQ-RCO IQ3_S, 262,144 configured context, `--kv int8`,
  `--kv-resident 32768`, `--spec 4`, `--batch 4 --batch-groups 2`, `--layer-split 24`, `--pcie-frac 0`,
  `--conversation-cache-mib 8192 --conversation-cache-slots 4`.
- Raw copies also remain on the server at
  `/home/chris/dev/strata-gfx906/logs/concurrency-probe-20261007T194232Z/`.

## Method

- Prompt: `items` filler lines, each carrying a unique index and salt, so no K/V could be reused inside
  or between phases; then a long-form task, so the answer runs to `max_tokens` instead of stopping early
  (`finish_reason: length` in every record below).
- Request: `max_tokens 256`, `temperature 0.0`, `stream: false`,
  `chat_template_kwargs: {"enable_thinking": false}`, `POST /v1/chat/completions`.
- Phase A (single): `--streams 1`, 3,976 prompt tokens. An earlier attempt with 9,877 prompt tokens is
  kept as `probe-single-9877.json`.
- Phase B (4-concurrent): `--streams 4`, `--stagger-ms 0` (all four released together), 3,976 prompt
  tokens each, salt base 601 so nothing was reused from phase A.
- Rates below come from the engine's own log lines (`prompt N tokens = X reused + Y read in Z ms`,
  `M generated in W ms`) and from the engine's per-request `timings`. Client-side wall time is labelled
  as such. See "Which timing field to trust" below.

## Results

| Phase | Streams | Prompt tokens | Reused | Output tokens | Prefill (real) | Decode per stream | Aggregate decode | Client wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A single | 1 | 3,976 | 0 | 256 | **459.5 tok/s** (8,654 ms) | **55.2 tok/s** | 55.2 tok/s | 17.15 s |
| A single (earlier) | 1 | 9,877 | 0 | 256 | **568.3 tok/s** (17,380 ms) | 25.7 tok/s (co-tenant joined) | — | 27.49 s |
| B 4-concurrent | 4 | 4 × 3,976 = 15,904 | 0 | 4 × 256 = 1,024 | **477.3–481.7 tok/s per stream**, ≈**480 tok/s aggregate** | **7.5 / 9.9 / 14.7 / 28.3 tok/s** | **29.9 tok/s** | 42.6 s |

Phase B detail (engine log, one run):

- Prefill did **not** parallelize: the four admissions read 3,976 tokens each in 8,254 / 8,269 / 8,273 /
  8,330 ms, one after another, so 15,904 tokens took ≈33 s — the same ≈480 tok/s a single stream gets.
- The batch window reported `batch (pipelined, 2 groups of 2): 513 group-steps, 1019 rows in 34244 ms
  = 29.8 rows/s (admissions included)`. 1,019 rows produced 1,024 accepted tokens: **≈1.0 accepted token
  per row**.
- The single-stream run reported `batch (pipelined, 2 groups of 2): 100 group-steps, 100 rows in 4528 ms
  = 22.1 rows/s` and `203 generated in 3675 ms (55.2 tok/s)`, with MTP `drafts accepted 121 of 182`:
  **≈2.0 accepted tokens per row**.
- So in this run the batch path raised rows/s by ~35% (22.1 → 29.8) while accepted tokens per row fell
  from ≈2.0 to ≈1.0. Net accepted decode: **55.2 tok/s single-stream vs 29.9 tok/s for four streams**.
- The engine's live `/metrics` `tok_s` counter read **109–114** while all four slots were decoding. That
  counter counts offered rows/draft tokens, not accepted tokens; the accepted-token rate over the same
  window is 29.9 tok/s. Do not compare it with a single-stream `predicted_per_second`.
- `/metrics` confirmed four busy slots and `live.running = 4` for the whole phase, with no co-tenant
  request (`contaminated_by_other_traffic: false`).

## Which timing field to trust

With batch slots active, the per-request HTTP `timings.prompt_ms` / `prompt_per_second` is **not** the
whole-prompt prefill figure. In the 9,877-token single-shot it reported `prompt_ms: 0.9` and
`prompt_per_second: 10,974,444`, while the engine log for the same request says
`prompt 9877 tokens = 0 reused + 9877 read in 17380 ms (568.3 tok/s)`. The per-request
`predicted_per_second` also spans queue and admission time, so a concurrent stream's value is a lower
bound on steady-state decode, not a decode-only rate. This probe therefore reports the engine log lines
for prefill and decode, and labels every client-derived number.

## Service conditions during the probe

- The service is a production service and was **not** empty. Before the probe it held four parked owner
  conversations (8 GiB conversation cache, 4 slots) and was serving 74K–78K-token agent prompts.
- Phase A (single) started at `live.running = 0`, then shared the engine with one co-tenant for the first
  13 one-second samples (`live.running = 2`), i.e. during its prefill; the last four samples, which cover
  its 3.7 s decode segment, read `live.running = 1`. The 459.5 tok/s prefill rate therefore carries
  co-tenant load, while the 55.2 tok/s decode rate does not. The 9,877-token single-shot, which also had
  a clean prefill window, reached 568.3 tok/s.
- Phase B ran with only the four probe streams (`live.running = 4` throughout).
- Conversation-cache eviction counters moved from 42 to 43 during phase B; KV streaming hit VRAM 79–81%
  with 18–22 MiB read from RAM. This was real memory pressure, not an empty machine.
- Cost: phase B held the engine for 42.7 s of wall time (≈33 s of prefill work, 34.2 s batch window with
  admissions included). Phase A held it for 17.2 s. No restart, no config change, no model change.

## Limits

- One run per phase, synthetic filler text, `max_tokens 256`, `temperature 0`, thinking disabled. The
  decode sample is 256 tokens per stream.
- Not a matched A/B: no other engine build, ROCm version, `--batch` value or `--batch-groups` value was
  tested. It does not isolate whether the batch path disables MTP speculation by design or loses it to
  admission churn; the draft counters for the batch admissions read `0 of 0` and need a source-level check.
- No accuracy, output-quality, full-262K-context, MI60, or long-soak measurement.
- The archived 4-way figure for v0.1.39 (`~80.4 tok/s`, `docs/GFX906.md`) is a live-counter figure under
  warm-cache agent traffic; it is not the same quantity as the 29.9 tok/s accepted-token rate here.

## Files

| File | Content |
|------|---------|
| `probe-single-3976.json` | Phase A: one request, 3,976 prompt tokens. Verbatim usage/timings, `/metrics` before-during-after, engine log delta. |
| `probe-single-9877.json` | Earlier single-shot at 9,877 prompt tokens; kept for the prefill rate and the `prompt_ms` anomaly. |
| `probe-concurrent4-3976.json` | Phase B: four simultaneous requests, 3,976 prompt tokens each. Same fields. |
| `evidence-manifest.sha256` | SHA256 of the files in this directory. |
