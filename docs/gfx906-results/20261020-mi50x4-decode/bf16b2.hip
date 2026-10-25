// Why does a 2560 -> 512 bf16 GEMV take 45 us on gfx906?  Isolate the parts.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { std::printf("%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e)); std::exit(1); } } while (0)
__device__ __forceinline__ float bf(uint16_t v) { return __uint_as_float((uint32_t) v << 16); }
__device__ __forceinline__ float ws32(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
    return v;
}
// MODE 0: engine (loop + warp tree + block partials); 1: no block reduction (thread 0's own acc);
// 2: no shuffles at all; 3: loads only (sum without fma chain dependency)
template <int B, int MODE>
__global__ void k(const float* __restrict__ x, const uint16_t* __restrict__ w, float* __restrict__ y, int n_in) {
    const int t = threadIdx.x;
    const uint32_t* w2 = reinterpret_cast<const uint32_t*>(w + (size_t) blockIdx.x * n_in);
    const float2* x2 = reinterpret_cast<const float2*>(x);
    __shared__ float partials[32];
    if (t < 32) partials[t] = 0.0f;
    __syncthreads();
    float acc = 0.0f;
    for (int p = t; p < n_in / 2; p += B) {
        const uint32_t wv = w2[p];
        const float2 xv = x2[p];
        acc = __fmaf_rn(bf((uint16_t) wv), xv.x, acc);
        acc = __fmaf_rn(bf((uint16_t) (wv >> 16)), xv.y, acc);
    }
    if (MODE == 0) {
        acc = ws32(acc);
        if ((t & 31) == 0) partials[t / 32] = acc;
        __syncthreads();
        if (t < 32) acc = ws32(partials[t]);
    } else if (MODE == 1) {
        acc = ws32(acc);
    }
    if (t == 0) y[blockIdx.x] = acc;
}
// one wave (64 lanes) per row, 4 rows per block, pairs lane + 64 i (a different order: speed only)
__global__ void k_wave(const float* __restrict__ x, const uint16_t* __restrict__ w, float* __restrict__ y, int n_in, int n_out) {
    const int lane = threadIdx.x & 63, row = blockIdx.x * 4 + (threadIdx.x >> 6);
    if (row >= n_out) return;
    const uint4* w4 = reinterpret_cast<const uint4*>(w + (size_t) row * n_in);
    const float4* x4 = reinterpret_cast<const float4*>(x);
    float acc = 0.0f;
    for (int c = lane; c < n_in / 8; c += 64) {
        const uint4 wv = w4[c];
        const float4 a = x4[2 * c], b = x4[2 * c + 1];
        acc = fmaf(__uint_as_float(wv.x << 16), a.x, acc); acc = fmaf(__uint_as_float(wv.x & 0xffff0000u), a.y, acc);
        acc = fmaf(__uint_as_float(wv.y << 16), a.z, acc); acc = fmaf(__uint_as_float(wv.y & 0xffff0000u), a.w, acc);
        acc = fmaf(__uint_as_float(wv.z << 16), b.x, acc); acc = fmaf(__uint_as_float(wv.z & 0xffff0000u), b.y, acc);
        acc = fmaf(__uint_as_float(wv.w << 16), b.z, acc); acc = fmaf(__uint_as_float(wv.w & 0xffff0000u), b.w, acc);
    }
#pragma unroll
    for (int o = 32; o > 0; o >>= 1) acc += __shfl_xor(acc, o, 64);
    if (lane == 0) y[row] = acc;
}
template <int B>
__global__ void k_ld(const float* __restrict__ x, const uint16_t* __restrict__ w, float* __restrict__ y, int n_in, int ld, int perm, int n_out) {
    const int t = threadIdx.x;
    const int row = (int) (((long long) blockIdx.x * perm) % n_out);
    const uint32_t* w2 = reinterpret_cast<const uint32_t*>(w + (size_t) row * ld);
    const float2* x2 = reinterpret_cast<const float2*>(x);
    __shared__ float partials[32];
    if (t < 32) partials[t] = 0.0f;
    __syncthreads();
    float acc = 0.0f;
    for (int p = t; p < n_in / 2; p += B) {
        const uint32_t wv = w2[p];
        const float2 xv = x2[p];
        acc = __fmaf_rn(bf((uint16_t) wv), xv.x, acc);
        acc = __fmaf_rn(bf((uint16_t) (wv >> 16)), xv.y, acc);
    }
    acc = ws32(acc);
    if ((t & 31) == 0) partials[t / 32] = acc;
    __syncthreads();
    if (t < 32) acc = ws32(partials[t]);
    if (t == 0) y[row] = acc;
}
__global__ void k_empty(float* y) { if (threadIdx.x == 0 && blockIdx.x == 0) y[0] = 1.f; }
template<typename L> static float tm(L f) {
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    for (int i = 0; i < 5; ++i) f();
    CK(hipEventRecord(a)); for (int i = 0; i < 300; ++i) f(); CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
    float ms; CK(hipEventElapsedTime(&ms, a, b)); return ms * 1000.f / 300;
}
int main() {
    const int n_in = 2560;
    std::mt19937 rng(7);
    std::vector<uint16_t> hw((size_t) n_in * 4096);
    for (auto& v : hw) { float f = (float) ((int) (rng() % 2001) - 1000) / 8000.f; uint32_t u; std::memcpy(&u, &f, 4); v = (uint16_t) (u >> 16); }
    std::vector<float> hx(n_in); for (auto& v : hx) v = (float) ((int) (rng() % 2001) - 1000) / 1000.f;
    uint16_t* w; float* x; float* y;
    CK(hipMalloc(&w, hw.size() * 2)); CK(hipMemcpy(w, hw.data(), hw.size() * 2, hipMemcpyHostToDevice));
    CK(hipMalloc(&x, n_in * 4)); CK(hipMemcpy(x, hx.data(), n_in * 4, hipMemcpyHostToDevice));
    CK(hipMalloc(&y, 4096 * 4));
    for (int n_out : {128, 256, 384, 512, 640, 768, 1024}) {
        std::printf("n_out %4d: ld2560 %6.1f  ld2688 %6.1f  ld2624 %6.1f  ld2816 %6.1f  perm7 %6.1f  n_in2048 %6.1f n_in3072 %6.1f\n", n_out,
            tm([&] { hipLaunchKernelGGL((k_ld<256>), dim3(n_out), dim3(256), 0, 0, x, w, y, n_in, 2560, 1, n_out); }),
            tm([&] { hipLaunchKernelGGL((k_ld<256>), dim3(n_out), dim3(256), 0, 0, x, w, y, n_in, 2688, 1, n_out); }),
            tm([&] { hipLaunchKernelGGL((k_ld<256>), dim3(n_out), dim3(256), 0, 0, x, w, y, n_in, 2624, 1, n_out); }),
            tm([&] { hipLaunchKernelGGL((k_ld<256>), dim3(n_out), dim3(256), 0, 0, x, w, y, n_in, 2816, 1, n_out); }),
            tm([&] { hipLaunchKernelGGL((k_ld<256>), dim3(n_out), dim3(256), 0, 0, x, w, y, n_in, 2560, 7, n_out); }),
            tm([&] { hipLaunchKernelGGL((k_ld<256>), dim3(n_out), dim3(256), 0, 0, x, w, y, 2048, 2048, 1, n_out); }),
            tm([&] { hipLaunchKernelGGL((k_ld<256>), dim3(n_out), dim3(256), 0, 0, x, w, y, 3072, 3072, 1, n_out); }));
    }
    for (int n_out : {64, 512, 2048, 4096}) {
        std::printf("n_out %4d: engine %6.1f  noblockred %6.1f  noshfl %6.1f  B64 %6.1f  B128 %6.1f  wave-row %6.1f  empty(n_out blocks) %6.1f us\n", n_out,
            tm([&] { hipLaunchKernelGGL((k<256, 0>), dim3(n_out), dim3(256), 0, 0, x, w, y, n_in); }),
            tm([&] { hipLaunchKernelGGL((k<256, 1>), dim3(n_out), dim3(256), 0, 0, x, w, y, n_in); }),
            tm([&] { hipLaunchKernelGGL((k<256, 2>), dim3(n_out), dim3(256), 0, 0, x, w, y, n_in); }),
            tm([&] { hipLaunchKernelGGL((k<64, 0>), dim3(n_out), dim3(64), 0, 0, x, w, y, n_in); }),
            tm([&] { hipLaunchKernelGGL((k<128, 0>), dim3(n_out), dim3(128), 0, 0, x, w, y, n_in); }),
            tm([&] { hipLaunchKernelGGL(k_wave, dim3((n_out + 3) / 4), dim3(256), 0, 0, x, w, y, n_in, n_out); }),
            tm([&] { hipLaunchKernelGGL(k_empty, dim3(n_out), dim3(256), 0, 0, y); }));
    }
    return 0;
}
