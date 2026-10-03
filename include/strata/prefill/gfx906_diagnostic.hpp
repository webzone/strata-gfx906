// Opt-in, synchronizing real-input QSA diagnostics. Not a performance path.
#pragma once
#include "strata/kernels/qsa_decode_attn.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace strata::prefill::gfx906 {
struct AttentionCapture {
    std::string directory;
    std::vector<int64_t> rows;
};
// STRATA_GFX906_QSA_CAPTURE=1, private existing CAPTURE_DIR and exact CAPTURE_POS.
// Inputs are saved before attention launches; finish saves the actual selected-path outputs.
// Only real gfx906, complete INT8 KV and 24/2/256 geometry are admitted. No default GPU work.
bool capture_attention(const float* q, const kernels::QsaAttnPools& pools, const int32_t* ids,
                       const int32_t* steps, int64_t cap, const kernels::QsaShapes& shapes,
                       int64_t queries, int layer, int64_t pos0, void* stream,
                       AttentionCapture& capture, std::string& error);
bool finish_attention(const AttentionCapture& capture, const float* attn, void* stream, std::string& error);
// One immutable vector per committed prediction, before speculative state commit. Existing private
// directory; a repeated GEN deliberately fails rather than replacing another request's evidence.
bool capture_logits(int64_t index, int64_t position, int32_t token, const std::vector<float>& logits,
                    std::string& error);
} // namespace strata::prefill::gfx906
