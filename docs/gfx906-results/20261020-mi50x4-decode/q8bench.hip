// Q8_0 x Q8_1 decode GEMV micro-benchmark for gfx906: the engine's native_mmvq_wave_kernel (copied) against
// load-path variants that keep each lane's blocks, dp4a order and the 64-lane butterfly, so outputs must be bitwise equal.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <string>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { std::printf("%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e)); std::exit(1); } } while (0)

struct Q80Block { half d; int8_t qs[32]; };
struct Q81Block { half2 ds; int8_t qs[32]; };
static_assert(sizeof(Q80Block) == 34, "");
static_assert(sizeof(Q81Block) == 36, "");

__device__ __forceinline__ int load_int_b2(const void* ptr, int i32) {
    const auto* x = static_cast<const uint16_t*>(ptr);
    int value = x[2 * i32] << 0;
    value |= x[2 * i32 + 1] << 16;
    return value;
}
__device__ __forceinline__ int dp4a(int a, int b, int c) { return __builtin_amdgcn_sdot4(a, b, c, false); }

// ---- V0: the engine kernel (R rows per wave, 4 waves per block)
template<int NCOLS, int R>
__launch_bounds__(256) __global__ void k_v0(const Q80Block* __restrict__ w, const Q81Block* __restrict__ x,
                                           float* __restrict__ y, int n_in, int n_out) {
    const int lane = int(threadIdx.x) & 63;
    const int row0 = (int(blockIdx.x) * 4 + (int(threadIdx.x) >> 6)) * R;
    if (row0 >= n_out) return;
    const int bpr = n_in / 32;
    const int x_stride = n_in / 32;
    const int kqs = 2 * (lane % 4);
    float tmp[NCOLS][R] = {};
    for (int kbx = lane / 4; kbx < bpr; kbx += 16) {
#pragma unroll
        for (int i = 0; i < R; ++i) {
            if (row0 + i < n_out) {
                const Q80Block* b = w + std::size_t(row0 + i) * bpr + kbx;
                int v0 = load_int_b2(b->qs, kqs), v1 = load_int_b2(b->qs, kqs + 1);
                float d0 = b->d;
#pragma unroll
                for (int j = 0; j < NCOLS; ++j) {
                    const Q81Block* xb = x + std::size_t(j) * x_stride + kbx;
                    int s = dp4a(v0, reinterpret_cast<const int*>(xb->qs)[kqs], 0);
                    s = dp4a(v1, reinterpret_cast<const int*>(xb->qs)[kqs + 1], s);
                    tmp[j][i] += d0 * __low2float(xb->ds) * float(s);
                }
            }
        }
    }
#pragma unroll
    for (int j = 0; j < NCOLS; ++j)
#pragma unroll
        for (int i = 0; i < R; ++i) {
            float v = tmp[j][i];
#pragma unroll
            for (int off = 32; off > 0; off >>= 1) v += __shfl_xor(v, off, 64);
            if (lane == 0 && row0 + i < n_out) y[std::size_t(j) * n_out + row0 + i] = v;
        }
}

// ---- V1: same lane mapping, the 8 weight bytes as one unaligned 8-byte load (gfx9 unaligned-access mode)
__device__ __forceinline__ int2 load8_u(const void* p) {
    int2 r;
    __builtin_memcpy(&r, p, 8);
    return r;
}
template<int NCOLS, int R, int UNR>
__launch_bounds__(256) __global__ void k_v1(const Q80Block* __restrict__ w, const Q81Block* __restrict__ x,
                                           float* __restrict__ y, int n_in, int n_out) {
    const int lane = int(threadIdx.x) & 63;
    const int row0 = (int(blockIdx.x) * 4 + (int(threadIdx.x) >> 6)) * R;
    if (row0 >= n_out) return;
    const int bpr = n_in / 32;
    const int x_stride = n_in / 32;
    const int kqs = 2 * (lane % 4);
    float tmp[NCOLS][R] = {};
    int kbx = lane / 4;
    // UNR blocks per lane in flight: all loads first, then the math in the same kbx order
    for (; kbx + 16 * (UNR - 1) < bpr; kbx += 16 * UNR) {
        int2 v[UNR][R]; float d0[UNR][R];
#pragma unroll
        for (int u = 0; u < UNR; ++u)
#pragma unroll
            for (int i = 0; i < R; ++i) {
                const int row = min(row0 + i, n_out - 1);
                const Q80Block* b = w + std::size_t(row) * bpr + kbx + 16 * u;
                v[u][i] = load8_u(b->qs + 4 * kqs);
                d0[u][i] = __half2float(b->d);
            }
#pragma unroll
        for (int u = 0; u < UNR; ++u)
#pragma unroll
            for (int i = 0; i < R; ++i)
#pragma unroll
                for (int j = 0; j < NCOLS; ++j) {
                    const Q81Block* xb = x + std::size_t(j) * x_stride + kbx + 16 * u;
                    int s = dp4a(v[u][i].x, reinterpret_cast<const int*>(xb->qs)[kqs], 0);
                    s = dp4a(v[u][i].y, reinterpret_cast<const int*>(xb->qs)[kqs + 1], s);
                    tmp[j][i] += d0[u][i] * __low2float(xb->ds) * float(s);
                }
    }
    for (; kbx < bpr; kbx += 16) {
#pragma unroll
        for (int i = 0; i < R; ++i) {
            const int row = min(row0 + i, n_out - 1);
            const Q80Block* b = w + std::size_t(row) * bpr + kbx;
            const int2 v = load8_u(b->qs + 4 * kqs);
            const float d0 = b->d;
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
                const Q81Block* xb = x + std::size_t(j) * x_stride + kbx;
                int s = dp4a(v.x, reinterpret_cast<const int*>(xb->qs)[kqs], 0);
                s = dp4a(v.y, reinterpret_cast<const int*>(xb->qs)[kqs + 1], s);
                tmp[j][i] += d0 * __low2float(xb->ds) * float(s);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < NCOLS; ++j)
#pragma unroll
        for (int i = 0; i < R; ++i) {
            float v = tmp[j][i];
#pragma unroll
            for (int off = 32; off > 0; off >>= 1) v += __shfl_xor(v, off, 64);
            if (lane == 0 && row0 + i < n_out) y[std::size_t(j) * n_out + row0 + i] = v;
        }
}

// ---- V2: as V0 (two 16-bit loads per int) but UNR blocks of loads issued before the math
template<int NCOLS, int R, int UNR>
__launch_bounds__(256) __global__ void k_v2(const Q80Block* __restrict__ w, const Q81Block* __restrict__ x,
                                           float* __restrict__ y, int n_in, int n_out) {
    const int lane = int(threadIdx.x) & 63;
    const int row0 = (int(blockIdx.x) * 4 + (int(threadIdx.x) >> 6)) * R;
    if (row0 >= n_out) return;
    const int bpr = n_in / 32;
    const int x_stride = n_in / 32;
    const int kqs = 2 * (lane % 4);
    float tmp[NCOLS][R] = {};
    int kbx = lane / 4;
    for (; kbx + 16 * (UNR - 1) < bpr; kbx += 16 * UNR) {
        int v0[UNR][R], v1[UNR][R]; float d0[UNR][R];
#pragma unroll
        for (int u = 0; u < UNR; ++u)
#pragma unroll
            for (int i = 0; i < R; ++i) {
                const int row = min(row0 + i, n_out - 1);
                const Q80Block* b = w + std::size_t(row) * bpr + kbx + 16 * u;
                v0[u][i] = load_int_b2(b->qs, kqs); v1[u][i] = load_int_b2(b->qs, kqs + 1);
                d0[u][i] = __half2float(b->d);
            }
#pragma unroll
        for (int u = 0; u < UNR; ++u)
#pragma unroll
            for (int i = 0; i < R; ++i)
#pragma unroll
                for (int j = 0; j < NCOLS; ++j) {
                    const Q81Block* xb = x + std::size_t(j) * x_stride + kbx + 16 * u;
                    int s = dp4a(v0[u][i], reinterpret_cast<const int*>(xb->qs)[kqs], 0);
                    s = dp4a(v1[u][i], reinterpret_cast<const int*>(xb->qs)[kqs + 1], s);
                    tmp[j][i] += d0[u][i] * __low2float(xb->ds) * float(s);
                }
    }
    for (; kbx < bpr; kbx += 16) {
#pragma unroll
        for (int i = 0; i < R; ++i) {
            const int row = min(row0 + i, n_out - 1);
            const Q80Block* b = w + std::size_t(row) * bpr + kbx;
            const int v0 = load_int_b2(b->qs, kqs), v1 = load_int_b2(b->qs, kqs + 1);
            const float d0 = b->d;
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
                const Q81Block* xb = x + std::size_t(j) * x_stride + kbx;
                int s = dp4a(v0, reinterpret_cast<const int*>(xb->qs)[kqs], 0);
                s = dp4a(v1, reinterpret_cast<const int*>(xb->qs)[kqs + 1], s);
                tmp[j][i] += d0 * __low2float(xb->ds) * float(s);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < NCOLS; ++j)
#pragma unroll
        for (int i = 0; i < R; ++i) {
            float v = tmp[j][i];
#pragma unroll
            for (int off = 32; off > 0; off >>= 1) v += __shfl_xor(v, off, 64);
            if (lane == 0 && row0 + i < n_out) y[std::size_t(j) * n_out + row0 + i] = v;
        }
}

template<typename L>
static float time_it(L launch, int reps) {
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    for (int i = 0; i < 5; ++i) launch();
    CK(hipEventRecord(a));
    for (int i = 0; i < reps; ++i) launch();
    CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
    float ms; CK(hipEventElapsedTime(&ms, a, b));
    return ms * 1000.f / reps;
}

int main(int argc, char** argv) {
    const int dev = argc > 1 ? std::atoi(argv[1]) : 0;
    CK(hipSetDevice(dev));
    struct Shape { int n_out, n_in; };
    std::vector<Shape> shapes = {{2560, 4096}, {2560, 6144}, {2560, 8192}, {10240, 2560}, {6144, 2560}, {12288, 2560},
                                 {640, 2560}, {512, 2560}, {106299, 2560}};
    std::mt19937 rng(1);
    // a big pool, rotated so every timed call reads its weights from HBM, not from L2
    const std::size_t pool_bytes = std::size_t(512) << 20;
    Q80Block* pool; CK(hipMalloc(&pool, pool_bytes));
    {
        std::vector<uint8_t> h(pool_bytes);
        for (std::size_t i = 0; i < pool_bytes; i += 34) {
            for (int k = 0; k < 34 && i + k < pool_bytes; ++k) h[i + k] = uint8_t(rng());
            if (i + 1 < pool_bytes) { half d = __float2half(float(int(rng() % 2000) - 1000) / 65536.f); std::memcpy(&h[i], &d, 2); }
        }
        CK(hipMemcpy(pool, h.data(), pool_bytes, hipMemcpyHostToDevice));
    }
    Q81Block* x; float* y0; float* y1;
    CK(hipMalloc(&x, sizeof(Q81Block) * 8 * 8192 / 32));
    {
        std::vector<Q81Block> hx(8 * 8192 / 32);
        for (auto& b : hx) { b.ds = __floats2half2_rn(float(rng() % 1000) / 4096.f, 0.f); for (auto& q : b.qs) q = int8_t(rng()); }
        CK(hipMemcpy(x, hx.data(), sizeof(Q81Block) * hx.size(), hipMemcpyHostToDevice));
    }
    CK(hipMalloc(&y0, sizeof(float) * 8 * 106299)); CK(hipMalloc(&y1, sizeof(float) * 8 * 106299));
    std::printf("%-14s %-5s %9s %9s  %s\n", "shape", "ncols", "variant", "us", "GB/s / bitwise");
    for (auto s : shapes) {
        const std::size_t bytes = std::size_t(s.n_out) * (s.n_in / 32) * 34;
        const int copies = int(std::max<std::size_t>(1, pool_bytes / bytes - 1));
        int rot = 0;
        auto wptr = [&]() { const Q80Block* p = pool + (std::size_t(rot) % copies) * (bytes / 34); ++rot; return p; };
        for (int nc : {1, 2, 4}) {
            std::vector<float> ref(std::size_t(nc) * s.n_out), out(ref.size());
            auto run = [&](const char* name, auto kern, int R, bool is_ref) {
                const unsigned blocks = unsigned((s.n_out + 4 * R - 1) / (4 * R));
                float* yy = is_ref ? y0 : y1;
                hipLaunchKernelGGL(kern, dim3(blocks), dim3(256), 0, 0, (const Q80Block*) pool, x, yy, s.n_in, s.n_out);
                CK(hipDeviceSynchronize());
                CK(hipMemcpy(is_ref ? ref.data() : out.data(), yy, sizeof(float) * ref.size(), hipMemcpyDeviceToHost));
                const float us = time_it([&] { hipLaunchKernelGGL(kern, dim3(blocks), dim3(256), 0, 0, wptr(), x, yy, s.n_in, s.n_out); }, 200);
                bool same = is_ref || std::memcmp(ref.data(), out.data(), sizeof(float) * ref.size()) == 0;
                std::printf("%6dx%-6d  %-5d %9s %9.1f  %6.0f %s\n", s.n_out, s.n_in, nc, name, us, bytes / us / 1e3, same ? "" : "MISMATCH");
            };
#define RUNS(NC) \
            run("v0 R1", k_v0<NC, 1>, 1, true); \
            run("v0 R2", k_v0<NC, 2>, 2, false); \
            run("v1 R1U1", k_v1<NC, 1, 1>, 1, false); \
            run("v1 R1U2", k_v1<NC, 1, 2>, 1, false); \
            run("v1 R1U4", k_v1<NC, 1, 4>, 1, false); \
            run("v1 R2U2", k_v1<NC, 2, 2>, 2, false); \
            run("v2 R1U2", k_v2<NC, 1, 2>, 1, false); \
            run("v2 R1U4", k_v2<NC, 1, 4>, 1, false); \
            run("v2 R2U2", k_v2<NC, 2, 2>, 2, false);
            if (nc == 1) { RUNS(1) } else if (nc == 2) { RUNS(2) } else { RUNS(4) }
        }
    }
    return 0;
}
