# MI50 cold-prefill prototypes — 2026-10-03

## Identity and boundaries

The engine and HIP probes were compiled from **`49285ae` on `gfx906-prefill-opt`**, Strata **v0.1.37**,
using system ROCm 7.2.4, real `gfx906`, portable host code, native experts/MMQ and two build jobs.
Later Python summarizer/documentation changes do not relabel that executable's source revision.
Hardware was T5810's **two MI50 32-GB cards**, `gfx906:sramecc+:xnack-`, physical wave64.
The pinned llama.cpp source was `3cf03257f219afbe7334045ff7c6a06ac68c627d`.
Both acceptance IQ2_XS GGUF shards passed **full LFS SHA256** verification at HF revision
`ed59f92082b1e93c0e96d60a8b11aab089b52f09`; identities and the candidate binary hash are in
[validation.json](validation.json).

All model runs used one pipe-only process at a time under the owner launch lock, GPUs `[0,1]`,
24/24 layers, INT8 KV, 262144 context capacity, MTP window 32768/spec 4, native resident-from-GGUF
experts, fixed profile, `--adapt-every 0`, greedy temperature 0/top-k 1/seed 42, and
`STRATA_PREFILL_TIMING=1`. `--short-read 0` exercised batching on the padded short prompts.
Checkpoint cache was **0**, except the explicitly separate restore probes (cache 2).
There was no HTTP listener, deployment overwrite, package/weight download, duplicate `experts.bin`,
P2P enablement, GPU impersonation, or driver/network/tuning/thermal-service change.
The suite paused when the owner service held the lock and resumed after the owner stopped it.
The original deployed binary, both configs and three launch scripts remained byte-identical.

## Synthetic checks — not model parity or model speed

- 169 setup and **57 gfx906 Python tests**, no skips; strict CPU policy build and **29 checks** passed.
- Both cards passed wave64 half-wave/2D/3D/shuffle/signed-dot4/overflow probes: **29952 checks each**,
  zero errors. Active wall-clock calibration: **24.9988 / 24.9984 MHz**.
- Both cards passed the independent CPU f16-dequantizer/double-softmax QSA oracle, decode control,
  new online-softmax kernel, actual dispatcher and graph replay. Fixtures cover capacities
  1/63/64/65/257/2051, masks, nonresident pages, empty/tile-boundary cases and large logits.
- Native mapped embedding: **768 exact signed values plus graph replay per card**. Separate aliases
  were resolved on each consuming device; their virtual addresses need not differ.
- Requested missing attention/card and single-card alias runs returned **1**, not a pass or skip.
- The optional resident 2048-query/cap-2051 operator benchmark used a **sampled eight-query double
  oracle**, plus complete control/new/dispatcher comparisons. Medians: card 0 **57.361 → 53.004 ms**;
  card 1 **57.424 → 53.259 ms**. These are warm operator milliseconds, **not model tok/s**.

The small `.log` files are verbatim probe output. No compiler-warning dump or binary fixture is committed.

## Real cold generation comparison

The source-review request had **33475 tokenizer-measured tokens**, zero reuse, and generated **64 IDs**.
Each arm also generated from three identical padded arithmetic/literal/Python prompts. All 16 arms had
identical submitted IDs for these four requests and exited 0. Exact inputs, output sequences and full
stderr are preserved privately. **Successful generation is not automatically token consistency.**

| Chunk | Experimental QSA | Split MTP batch | Prefill seconds | Four-request ID consistency |
|---:|:---:|:---:|---:|:---:|
| 2048 | off | off | **52.058** (two-run mean) | yes |
| 3072 | off | off | 49.082 | yes |
| 4096 | off | off | **48.816** (two-run mean) | yes |
| 2048 | on | off | 51.831 | yes |
| 2048 | off | on | 49.873 | yes |
| 2048 | on | on | 49.666 | yes |
| 4096 | on | off | 48.516 | **no** |
| 4096 | off | on | **46.619** (two-run mean) | **yes, all 89 IDs** |
| 4096 | on | on | 46.277 (two-run mean) | **no, reproduced** |

The measured token-consistent candidate is **chunk 4096, MTP batch ON, experimental QSA OFF**:
**10.45% less prefill time** than the candidate's 2048 control, and **4.50% less** than its 4096 control.
Mean measured pipe TTFT was **46.665 s**. This compares controls of the **same v0.1.37 candidate**,
not the older deployed v0.1.34 engine. Only the stated controls/candidates were repeated twice;
this is one artificial source workload, not a statistically broad production claim.

**Important limitation:** enabling experimental QSA at chunk 4096 changed the source-review output
starting at **zero-based generated ID index 42**. Attention-only and both-on isolation reproduced the
same alternative paragraph; MTP-only and both 4096 controls matched all reference IDs.
The operator's numerical tolerance pass does **not** prove whole-model token parity. The exact reason
and quality impact are not established by these tests. Keep QSA opt-in and do not recommend this
4096 QSA combination as token-consistent. No threshold was relaxed to hide the difference.

At 2048, old MTP prefill callback wall time was about **2.53 s**, versus **0.312 s** with KV-only batching
on device 1 (17 cold callbacks). That is not an 88% whole-model improvement: measured whole prefill
improvement was about 4.2%. The first request exceeds the MTP window; the early-skip/partial boundary
was exercised, but this does not cover every window size/boundary.

## Bottleneck evidence and additional guards

Cold-only profiles exclude the three short requests. On the 2048 control, gate/up plus down GEMMs
accounted for **30.4% / 35.9%** of each device's event timeline; QSA attention **10.4% / 11.0%**.
CPU grouping work was only **28.7 / 33.7 ms**, whereas grouping synchronization waited for already
queued GPU work for **42.77 / 44.55 s**. These are different quantities, not additive bottlenecks.
GPU0's next-stage wait/drain was **6.49 s**. GPU grouping and split rebalancing remain deferred;
removing CPU counting alone is not justified by these measurements.

MMQ handled **374/408** layer-chunks on GPU0 and **391/408** on GPU1. FP16 fallback layer IDs were
**8, 13 / 37**, zero-based. The model's mixed formats were not forced through MMQ. These cold runs
had **zero expert streaming and zero refill**; results do not predict the tighter IQ3_S cache budget.
GPU timelines include waits/idle; stage wall times overlap and must not be summed across devices.

With cache 2, both control and both-on/2048 restored **16384 tokens** for a 33494-token followup.
After a different short request, cold re-execution reused 0; all **three followup IDs** matched restored
execution. This is a bounded checkpoint probe, **not** full-state/logits parity or split conversation parking.
With both switches requested and `STRATA_MTP_BATCH=0`, all new batching callbacks fell back, and all
four original output sequences matched control. Acceptance and rejection occurred in actual generation;
full speculative acceptance distributions, full 262K context, long-soak quality and MI60 remain unvalidated.

## Reproduction and retained evidence

The review body is generated without a download:

```python
code = '\n'.join(
    f'export function rule{i}(x) {{ return x === {i} ? x + {i+1} : x - {i}; }}'
    for i in range(1024)
)
prompt = 'Review this source and describe its behavior precisely in a paragraph.\n' + code
# Render through packs/iq2_xs/tokenizer/chat_template.jinja with enable_thinking=False;
# Tokenizer.from_gguf(the pinned IQ2_XS first shard), parse_special=True.
```

`validation.json` contains input-ID hashes, complete generated ID sequences, per-arm effective switches,
real engine/TTFT metrics, cold-only stage/draft profiles, consistency failures and scope limits.
The `*-cold-profile.raw.log` files are exact line excerpts; full logs were not silently truncated/replaced.
Private complete evidence is under
`/home/chris/dev/strata-gfx906/logs/gfx906-prefill-opt-bbc39eb/` (its directory name is historical;
`source-ref.txt` identifies **49285ae**). GPU fixture arrays are in `attn-device[01]-fixtures/` and
`native-embed-fixture/`; model `*.request.json`, `result.json` and `engine.log` are in `model-ARM/`.
The driver/harness are retained there. An initial harness failed to retain the process handle across
v0.1.37 `unload()`; that unverified-exit control is excluded. Subsequent runs retained the handle and
waited boundedly for deferred driver VRAM release, never treating a busy card as available.

The original service remains owner-managed. No API service was restarted after these isolated tests.
