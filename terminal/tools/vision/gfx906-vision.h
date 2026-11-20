// Used only by the opt-in HIP vision build's mtmd source overlay.
// BF16 -> FP32 is exact: no new GGUF, quantization or rounded source weights.
#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include <istream>
#include <stdexcept>
#include <vector>

static ggml_tensor * strata_vision_tensor(ggml_context * ctx, const ggml_tensor * src) {
    return src->type == GGML_TYPE_BF16
        ? ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, src->ne)
        : ggml_dup_tensor(ctx, src);
}

static void strata_vision_read_bf16(std::istream & input, ggml_tensor * dst) {
    const auto n = ggml_nelements(dst);
    std::vector<ggml_bf16_t> source(n);
    std::vector<float> expanded(n);
    input.read(reinterpret_cast<char *>(source.data()), n * sizeof(ggml_bf16_t));
    if (!input) throw std::runtime_error("truncated BF16 vision tensor");
    ggml_bf16_to_fp32_row(source.data(), expanded.data(), n);
    ggml_backend_tensor_set(dst, expanded.data(), 0, ggml_nbytes(dst));
}
