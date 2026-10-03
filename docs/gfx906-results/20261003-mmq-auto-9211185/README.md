# Dual-MI50 MMQ auto operator gate — 9211185

## Identity and scope

Application/probes: `9211185ca51b587379dae4090543b58079e8deae`, isolated real-gfx906 HIP Release,
ROCm 7.2.4, pinned ggml `3cf03257f219afbe7334045ff7c6a06ac68c627d`, portable host code.
Application SHA256 is `5b52e618e427bff1582fb2011a80ec924ea1bc245d3f5f475f1061178ca01f61`.
Build and admission/integrity audit exited 0. Neither the owner checkout nor deployed engine/configs/
launchers were replaced. This receipt does not relabel older cbdc94c or 49285ae evidence.

Both physical MI50 cards passed 29,952 wave64 checks with zero ballot/other errors. Fresh clock
calibrations were 25.0000/24.9997 MHz. Requested absent hardware remains a failure, not a skip.

## MMQ gate

Each card ran **40 grouped matrix fixtures × six requested dispatches**: original control, J16,
J32, J48, J64 and `auto`. Gate/up formats: IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS;
down formats: Q2_0, IQ4_NL. Mean rows/expert were 5/20/40/80/160, with 32 experts, empty/single/
skewed groups, signed half scales and source/destination permutations. Every output was finite
and **bit-identical to the unchanged dispatcher**. Each card additionally saved **48 graph-replay
outputs**, all bit-identical to control. Private fixture inventories confirmed 40 directories,
240 ordinary outputs and 48 graph outputs/card. All four commands and suite driver exited 0.

The separate CPU execution uses the pinned format decoder with double product accumulation,
checking at most 16 elements per nonempty expert, not every output. Its unchanged relative-L2
bound is 0.02 for intended Q8_1 activation quantization. It is not an independently implemented
GGUF decoder. The full GPU tile/control comparisons cover all outputs.

The opt-in `auto` policy resolves per 32-expert product: measured formats use J32 at max rows
49–96 and J64 at 97–192, otherwise original dispatch. Q2_0 declines automatic tuning; dense/MTP
single matrices and unvalidated format/geometry pairs remain original dispatch. Aggregated
layer maxima must not be substituted for product maxima.

Seven retained HIP-event samples after warmup are resident **operator** observations, not cold
model prefill, TTFT, decode, tok/s or universal tile recommendations. Model token/performance/
cache-pressure adoption, long repetitions, split tuning and QSA diagnosis remain unfinished.
MI60/full 262K/long-soak quality remain unvalidated.

## Evidence and safety

Small raw wave/MMQ logs and candidate identity are adjacent. Full fresh prelaunch inputs,
all outputs/graphs, lease-protected resource preflight, sampled dependency hashes, unchanged
six-file deployment audit and post-probe idle drain remain private under
`logs/gfx906-prefill-goal-20261003T040756Z/`. A 16-GiB probe allowance plus 4-GiB reserve was
admitted; no weights/download/preparation/API/service/system-package operations occurred.
