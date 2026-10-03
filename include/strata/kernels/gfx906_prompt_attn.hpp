// Experimental MI50/MI60 INT8-KV prefill attention; no matrix instructions or split-K scratch.
#pragma once

#include "strata/kernels/qsa_decode_attn.hpp"

namespace strata::kernels {

// False means nothing was launched: keep the existing prompt/decode fallback.
// Requires an experimental HIP build, actual gfx906, STRATA_GFX906_PREFILL_ATTN=1,
// complete INT8 pools, and the real 24/2/256 geometry. Decode never calls this entry.
// FP32 math, but online softmax changes the accumulation order: not bitwise parity.
bool gfx906_prompt_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids,
                              const int32_t* steps, int64_t cap, const QsaShapes& s,
                              float* attn, int64_t queries, void* stream);

} // namespace strata::kernels
