// gfx906-only prefill experiment. Retains two logical wave32 groups per physical wave64.
#include "strata/kernels/gfx906_prompt_attn.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/prefill/gfx906_policy.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cstdio>
#include <cstdlib>

#if defined(STRATA_USE_HIP) && defined(STRATA_EXPERIMENTAL_GFX906)
#include <array>
#include <mutex>
#endif

namespace strata::kernels {
#if defined(STRATA_USE_HIP) && defined(STRATA_EXPERIMENTAL_GFX906)
namespace {
constexpr int HD = 256, G = 12, TILE = 64, THREADS = 256, WARPS = THREADS / 32;

__device__ __forceinline__ float sum32(float x) {
#pragma unroll
    for (int d = 16; d; d >>= 1) x += __shfl_xor_sync(0xffffffffu, x, d);
    return x;
}
__device__ __forceinline__ float max32(float x) {
#pragma unroll
    for (int d = 16; d; d >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, d));
    return x;
}

__global__ void __launch_bounds__(THREADS) online_int8(const float* __restrict__ queries, QsaAttnPools p,
                                                       const int32_t* __restrict__ selections,
                                                       const int32_t* __restrict__ steps, int cap,
                                                       int page_size, float* __restrict__ output) {
#if defined(__gfx906__)
    const int qi = blockIdx.y, kvh = blockIdx.x;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const float* q = queries + (size_t) qi * 24 * HD + kvh * G * HD;
    const int32_t* ids = selections + (size_t) qi * cap;
    const int width = max(0, min(cap, steps[(size_t) qi * kStepCount + kStepWidth]));
    __shared__ __align__(16) float sq[G][HD];
    __shared__ float prob[G][TILE];
    __shared__ long long rows[TILE];
    __shared__ float running_m[G], running_l[G], alpha[G], beta[G];
    for (int i = t; i < G * HD; i += THREADS) sq[i / HD][i % HD] = q[i];
    if (t < G) { running_m[t] = -FLT_MAX; running_l[t] = 0.0f; }
    float acc[G];
#pragma unroll
    for (int h = 0; h < G; ++h) acc[h] = 0.0f;
    __syncthreads();

    for (int start = 0; start < width; start += TILE) {
        const int n = min(TILE, width - start);
        if (t < TILE) {
            long long row = -1;
            if (t < n) {
                const int cell = ids[start + t];
                if (cell >= 0) {
                    const int page = p.page_table[cell / page_size];
                    // Nonresident pages are masked exactly as in the old attention kernel.
                    if (page >= 0) row = ((long long) page * 2 + kvh) * page_size + cell % page_size;
                }
            }
            rows[t] = row;
        }
        __syncthreads();
        for (int c = warp; c < TILE; c += WARPS) {
            if (c >= n || rows[c] < 0) {
                if (lane < G) prob[lane][c] = -FLT_MAX;
                continue;
            }
            const long long row = rows[c];
            const int d = lane * 8;
            const float scale = __half2float(__ushort_as_half(p.k_scale[row * (HD / KV_Q8_GROUP) + d / KV_Q8_GROUP]));
            const uint2 packed = *reinterpret_cast<const uint2*>(p.k_q + row * HD + d);
            const int8_t* codes = reinterpret_cast<const int8_t*>(&packed);
            float k[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) k[j] = (float) codes[j] * scale;
#pragma unroll
            for (int h = 0; h < G; ++h) {
                const float4 a = *reinterpret_cast<const float4*>(&sq[h][d]);
                const float4 b = *reinterpret_cast<const float4*>(&sq[h][d + 4]);
                float score = k[0] * a.x + k[1] * a.y + k[2] * a.z + k[3] * a.w +
                              k[4] * b.x + k[5] * b.y + k[6] * b.z + k[7] * b.w;
                score = sum32(score);
                if (lane == 0) prob[h][c] = score * (1.0f / 16.0f);
            }
        }
        __syncthreads();
        for (int h = warp; h < G; h += WARPS) {
            const float a = prob[h][lane], b = prob[h][lane + 32];
            const float m = max32(fmaxf(a, b));
            const float ea = lane < n && rows[lane] >= 0 ? __expf(a - m) : 0.0f;
            const float eb = lane + 32 < n && rows[lane + 32] >= 0 ? __expf(b - m) : 0.0f;
            prob[h][lane] = ea;
            prob[h][lane + 32] = eb;
            const float l = sum32(ea + eb);
            if (lane == 0) {
                if (l > 0.0f) {
                    const float nm = fmaxf(running_m[h], m);
                    alpha[h] = running_l[h] > 0.0f ? __expf(running_m[h] - nm) : 0.0f;
                    beta[h] = __expf(m - nm);
                    running_l[h] = fmaf(running_l[h], alpha[h], l * beta[h]);
                    running_m[h] = nm;
                } else {
                    // Includes fully masked tiles after a valid tile; never evaluate -inf - -inf.
                    alpha[h] = 1.0f;
                    beta[h] = 0.0f;
                }
            }
        }
        __syncthreads();
#pragma unroll
        for (int h = 0; h < G; ++h) acc[h] *= alpha[h];
        for (int c = 0; c < n; ++c) {
            if (rows[c] < 0) continue;
            const long long row = rows[c];
            const float scale = __half2float(__ushort_as_half(p.v_scale[row * (HD / KV_Q8_GROUP) + t / KV_Q8_GROUP]));
            const float v = (float) p.v_q[row * HD + t] * scale;
#pragma unroll
            for (int h = 0; h < G; ++h) acc[h] = fmaf(prob[h][c] * beta[h], v, acc[h]);
        }
        __syncthreads(); // All readers finish before the next tile overwrites rows/prob/coefficients.
    }
#pragma unroll
    for (int h = 0; h < G; ++h)
        output[((size_t) qi * 24 + kvh * G + h) * HD + t] = running_l[h] > 0.0f ? acc[h] / running_l[h] : 0.0f;
#endif
}

bool current_is_gfx906() {
    // Shared between the stage threads, unlike an unsynchronized static device array.
    static std::mutex lock;
    static std::array<int, 64> cached{};
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= (int) cached.size()) return false;
    std::lock_guard<std::mutex> guard(lock);
    if (cached[device] == 0) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return false;
        cached[device] = strata::prefill::gfx906::architecture(prop.gcnArchName) && prop.warpSize == 64 ? 1 : -1;
    }
    return cached[device] == 1;
}
} // namespace
#endif

bool gfx906_prompt_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids,
                              const int32_t* steps, int64_t cap, const QsaShapes& s,
                              float* attn, int64_t queries, void* stream) {
#if defined(STRATA_USE_HIP) && defined(STRATA_EXPERIMENTAL_GFX906)
    if (!strata::prefill::gfx906::opt_in(std::getenv("STRATA_GFX906_PREFILL_ATTN")) ||
        !strata::prefill::gfx906::attention_geometry(queries, cap, s.n_head, s.n_head_kv, s.head_dim, s.page_size) ||
        !q || !ids || !steps || !attn || !pools.page_table || !pools.k_q || !pools.v_q ||
        !pools.k_scale || !pools.v_scale || pools.k_q4 || pools.v_q4 || !current_is_gfx906()) return false;
    online_int8<<<dim3(2, (unsigned) queries), THREADS, 0, (cudaStream_t) stream>>>(
        q, pools, ids, steps, (int) cap, (int) s.page_size, attn);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        // A launched path must not silently fall back after an execution failure.
        std::fprintf(stderr, "gfx906 prompt attention: %s\n", cudaGetErrorString(error));
        std::abort();
    }
    return true;
#else
    (void) q; (void) pools; (void) ids; (void) steps; (void) cap;
    (void) s; (void) attn; (void) queries; (void) stream;
    return false;
#endif
}
} // namespace strata::kernels
