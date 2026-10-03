# IQ2_XS / IQ3_S 64K cold pilot — cbdc94c

This is **one repetition**, not acceptance of the required 64K/128K/192K three-repeat matrix.
It tests isolated application `cbdc94c2d06103d1f70e3c1e2d7c741ad3cba9d1`, SHA256
`167b38cc5c57b0984f8653da8e8161a5e9bc00a7a40cafb7bff0d706f8559c8b`, on T5810's two MI50s,
ROCm 7.2.4, real gfx906, Strata v0.1.37 and pinned ggml. It is not an A/B against deployed v0.1.34.

Both original pinned GGUF identities passed full-hash/fingerprint validation. Code, Chinese
maintenance records and four-turn chat fixtures each contained exactly **65,536 tokenizer-measured
input IDs**, identically tokenized for both models. Every sample had reused=0 and prompt_read=65536.
The six own engines and overall driver exited 0; six protected deployment files remained unchanged.
No API/service/deployment/package/weight operations occurred.

## Actual generation observations

Mean across three different input kinds, **not three repetitions of one input**:

| Model | Chunk / MTP batch | Prefill seconds | TTFT seconds | All three ID sequences equal 2048 control? |
|---|---|---:|---:|---|
| IQ2_XS | 2048 / OFF | 97.684 | 97.758 | reference |
| IQ2_XS | 3072 / ON | 89.837 | 89.911 | **no** |
| IQ2_XS | 4096 / ON | 86.710 | 86.785 | **no** |
| IQ3_S | 2048 / OFF | 103.178 | 103.247 | reference |
| IQ3_S | 3072 / ON | 92.780 | 92.851 | **no** |
| IQ3_S | 4096 / ON | 88.138 | 88.210 | yes, this single repetition only |

QSA and forced MMQ tiles were OFF in every arm. For IQ2_XS, both larger chunks first differed
from control at zero-based output ID **53** on Chinese and **16** on chat; code matched. IQ3_S
3072 differed at **34** on Chinese, with code/chat matching. Each code/Chinese result generated
64 IDs; chat stopped after 34 IQ2_XS or 46 IQ3_S IDs. These are real complete bounded generations.
Different IDs are not relabelled as passing parity or judged global quality regressions from this
small corpus. There are no same-chunk MTP-OFF controls in this pilot, so chunk versus MTP causes
are not separated yet; absent comparisons in the evidence are **null**, never passes.

## Pressure and balance

IQ2_XS cold expert streaming was zero. IQ3_S GPU1 streamed 35,616 expert entries/request at
2048, about 28,590 at 3072, and 24,416 at 4096. These are invocation-local counters, not counts of
unique missing experts. One-second observed GPU1 VRAM maxima reached approximately 33.8 GB for
IQ3_S; observed engine RSS reached about 52.1 GB. Sampled maxima are not instantaneous peak proof.
Full memory records/startup/cache-buffer diagnostics remain private.

At 4096+MTP, GPU0 next-stage wait/drain was 6.49–7.35 seconds for IQ2_XS and 9.08–9.72 for IQ3_S.
This supports measuring nearby splits, not choosing a new split by inference. Device timelines
include waits and overlap; do not add them to obtain request wall time or attribute CPU group waits
to actual counting work. Per-product expert maxima must not be inferred from layer-wide maxima.

## Evidence

`validation.json` retains exact generated IDs/text/engine counters, fixture hashes, model/application
identities, stage/draft summaries, sampled memory peaks and raw transcript hashes. The offline audit
verified full requested fixture/model/arm/repetition coverage, original raw GEN IDs and T/DONE
outputs, clean exits and deployment/binary immutability. Complete raw inputs/stdin/stdout/stderr,
per-layer profiles and memory samples stay private in
`logs/gfx906-prefill-goal-20261003T040756Z/long64k-pilot/`.

MI60, full 262K, long-soak/global quality, RAM conversation parking and deployment adoption are not
validated. Longer/repeated comparisons, expert-policy model gates, split tuning and QSA diagnosis
remain required work.
