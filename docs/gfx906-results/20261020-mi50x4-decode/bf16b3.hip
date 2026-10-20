#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { std::printf("%d %s\n", __LINE__, hipGetErrorString(e)); std::exit(1); } } while (0)
__device__ __forceinline__ float bf(uint16_t v) { return __uint_as_float((uint32_t) v << 16); }
__device__ __forceinline__ float ws32(float v) { for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32); return v; }
template <int B>
__global__ void k(const float* __restrict__ x, const uint16_t* __restrict__ w, float* __restrict__ y, int n_in, int n_out) {
    if ((int) blockIdx.x >= n_out) return;
    const int t = threadIdx.x;
    const uint32_t* w2 = reinterpret_cast<const uint32_t*>(w + (size_t) blockIdx.x * n_in);
    const float2* x2 = reinterpret_cast<const float2*>(x);
    __shared__ float partials[32];
    if (t < 32) partials[t] = 0.0f;
    __syncthreads();
    float acc = 0.0f;
    for (int p = t; p < n_in / 2; p += B) {
        const uint32_t wv = w2[p]; const float2 xv = x2[p];
        acc = __fmaf_rn(bf((uint16_t) wv), xv.x, acc); acc = __fmaf_rn(bf((uint16_t) (wv >> 16)), xv.y, acc);
    }
    acc = ws32(acc);
    if ((t & 31) == 0) partials[t / 32] = acc;
    __syncthreads();
    if (t < 32) acc = ws32(partials[t]);
    if (t == 0) y[blockIdx.x] = acc;
}
template<typename L> static float tm(L f) {
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    for (int i = 0; i < 5; ++i) f();
    CK(hipEventRecord(a)); for (int i = 0; i < 300; ++i) f(); CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
    float ms; CK(hipEventElapsedTime(&ms, a, b)); return ms * 1000.f / 300;
}
int main() {
    const int n_in = 2560;
    std::vector<uint16_t> hw((size_t) n_in * 2048, 0x3c00); std::vector<float> hx(n_in, 0.5f);
    uint16_t* w; float* x; float* y;
    CK(hipMalloc(&w, hw.size() * 2)); CK(hipMemcpy(w, hw.data(), hw.size() * 2, hipMemcpyHostToDevice));
    CK(hipMalloc(&x, n_in * 4)); CK(hipMemcpy(x, hx.data(), n_in * 4, hipMemcpyHostToDevice)); CK(hipMalloc(&y, 4096 * 4));
    for (int n_out : {448, 480, 496, 504, 508, 511, 512, 513, 516, 520, 528, 544, 576, 1024}) {
        std::printf("n_out %4d:", n_out);
        for (int extra : {0, 1, 4, 8, 64}) std::printf("  +%d %6.1f", extra, tm([&] { hipLaunchKernelGGL((k<256>), dim3(n_out + extra), dim3(256), 0, 0, x, w, y, n_in, n_out); }));
        std::printf("\n");
    }
}
