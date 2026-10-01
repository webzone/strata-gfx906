// MI50 validation of Strata's logical 32-lane warps on physical wave64.
// Independent CPU references; deliberately different predicates in both halves.
// Build with upstream or patched include/ to obtain the negative control/candidate.
#include <hip/hip_runtime.h>
#include "strata/hip_compat/intrinsics.hpp"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

static void check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}
#define HIP(x) check((x), #x)

struct Result {
    unsigned ballot;
    int dot, overflow, broadcast, down8, down8_1, down8_2, up32, xor16;
    float sum;
    double xor_double;
    unsigned long long broadcast64;
};
__host__ __device__ static unsigned mix(unsigned x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; return x ^ (x >> 16);
}
__global__ static void probe(Result* out) {
    const unsigned linear = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    const unsigned n = blockDim.x * blockDim.y * blockDim.z;
    const unsigned id = blockIdx.x * n + linear, lane = linear & 31u, warp = linear / 32;
    constexpr unsigned mask = 0xffffffffu;
    Result r{};
    r.ballot = __ballot_sync(mask, lane % 3 == warp % 3);
    r.dot = __dp4a((int)mix(id * 2 + 1), (int)mix(id * 2 + 2), (int)mix(id + 99));
    r.overflow = __dp4a(0x7f7f7f7f, 0x7f7f7f7f, std::numeric_limits<int>::max());
    r.broadcast = __shfl_sync(mask, (int)id, 0);
    r.down8 = __shfl_down_sync(mask, (int)id, 4, 8);
    // Upstream 0.1.31's Q2_0 chunk reduction uses width-8 offsets in the order 4, 1, 2.
    // Exercise each offset in both logical half-warps, including 2D/3D blocks.
    r.down8_1 = __shfl_down_sync(mask, (int)id, 1, 8);
    r.down8_2 = __shfl_down_sync(mask, (int)id, 2, 8);
    r.up32 = __shfl_up_sync(mask, (int)id, 1);
    r.xor16 = __shfl_xor_sync(mask, (int)id, 8, 16);
    float sum = (float)id;
    for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(mask, sum, o);
    r.sum = sum;
    r.xor_double = __shfl_xor_sync(mask, (double)id + 0.25, 1);
    r.broadcast64 = __shfl_sync(mask, 0x1234567800000000ull + id, 0);
    out[id] = r;
}
__global__ static void timestamp(unsigned long long* out, int i) { out[i] = wall_clock64(); }
__global__ static void dot_bench(unsigned* out, int iterations) {
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    int x = (int)mix(id + 1), a = (int)mix(id + 2), b = (int)mix(id + 3);
    for (int i = 0; i < iterations; ++i) {
        x = __dp4a(a ^ x, b, x);
        a = a * 1664525u + 1013904223u;
    }
    out[id] = (unsigned)x;
}
static int dot_ref(unsigned a, unsigned b, unsigned c) {
    unsigned sum = c;
    for (int i = 0; i < 4; ++i) {
        int av = (a >> (8*i)) & 255, bv = (b >> (8*i)) & 255;
        av = av < 128 ? av : av - 256; bv = bv < 128 ? bv : bv - 256;
        sum += (unsigned)(av * bv);
    }
    return (int)sum;
}
int main(int argc, char** argv) {
    try {
        const bool negative = argc > 1 && !std::strcmp(argv[1], "--expect-ballot-fail");
        hipDeviceProp_t p{}; HIP(hipGetDeviceProperties(&p, 0));
        std::printf("device=%s arch=%s physical_wave=%d\n", p.name, p.gcnArchName, p.warpSize);
        if (p.warpSize != 64 || std::strncmp(p.gcnArchName, "gfx906", 6)) return 2;
        const dim3 shapes[] = {dim3(32),dim3(64),dim3(96),dim3(128),dim3(256),dim3(16,4,2),dim3(8,4,4)};
        int ballot_errors = 0, other_errors = 0, checks = 0;
        for (auto shape : shapes) {
            unsigned threads = shape.x * shape.y * shape.z, count = threads * 3;
            Result* d = nullptr; HIP(hipMalloc(&d, count * sizeof(Result)));
            probe<<<3, shape>>>(d); HIP(hipGetLastError()); HIP(hipDeviceSynchronize());
            std::vector<Result> got(count); HIP(hipMemcpy(got.data(), d, got.size()*sizeof(Result), hipMemcpyDeviceToHost));
            int be = 0, oe = 0;
            for (unsigned id = 0; id < count; ++id) {
                unsigned linear = id % threads, lane = linear & 31u, warp = linear / 32, base = id - lane;
                unsigned ballot = 0; for (unsigned j=0;j<32;++j) if (j%3==warp%3) ballot |= 1u<<j;
                const auto& r=got[id];
                be += r.ballot != ballot;
                oe += r.dot != dot_ref(mix(id*2+1),mix(id*2+2),mix(id+99));
                oe += r.overflow != dot_ref(0x7f7f7f7fu,0x7f7f7f7fu,0x7fffffffu);
                oe += r.broadcast != (int)base;
                oe += r.down8 != (int)(id + (lane%8 < 4 ? 4 : 0));
                oe += r.down8_1 != (int)(id + (lane%8 < 7 ? 1 : 0));
                oe += r.down8_2 != (int)(id + (lane%8 < 6 ? 2 : 0));
                oe += r.up32 != (int)(lane ? id-1 : id);
                oe += r.xor16 != (int)(base + (lane ^ 8));
                oe += r.sum != (float)(32*base+496);
                oe += r.xor_double != (double)(base+(lane^1))+.25;
                oe += r.broadcast64 != 0x1234567800000000ull+base;
                checks += 12;
            }
            std::printf("shape=%u,%u,%u threads=%u ballot_errors=%d other_errors=%d\n",shape.x,shape.y,shape.z,threads,be,oe);
            ballot_errors += be; other_errors += oe; HIP(hipFree(d));
        }
        std::printf("checks=%d ballot_errors=%d other_errors=%d\n",checks,ballot_errors,other_errors);
        if (negative) return ballot_errors > 0 && other_errors == 0 ? 0 : 1;
        if (ballot_errors || other_errors) return 1;
        unsigned long long* dt=nullptr; HIP(hipMalloc(&dt,2*sizeof(*dt)));
        timestamp<<<1,1>>>(dt,0); HIP(hipDeviceSynchronize());
        auto start=std::chrono::steady_clock::now(); std::this_thread::sleep_for(std::chrono::milliseconds(100));
        timestamp<<<1,1>>>(dt,1); HIP(hipDeviceSynchronize());
        double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        unsigned long long ts[2]; HIP(hipMemcpy(ts,dt,sizeof(ts),hipMemcpyDeviceToHost)); HIP(hipFree(dt));
        std::printf("wall_clock_mhz=%.4f host_elapsed_ms=%.3f\n",(ts[1]-ts[0])/elapsed/1e6,elapsed*1e3);
        constexpr int blocks=256, threads=256, iterations=4096;
        unsigned* d=nullptr; HIP(hipMalloc(&d,blocks*threads*sizeof(unsigned)));
        dot_bench<<<blocks,threads>>>(d,32); HIP(hipDeviceSynchronize());
        hipEvent_t a,b; HIP(hipEventCreate(&a));HIP(hipEventCreate(&b));
        std::vector<float> times;
        for (int rep=0;rep<7;++rep) {
            HIP(hipEventRecord(a));dot_bench<<<blocks,threads>>>(d,iterations);HIP(hipEventRecord(b));HIP(hipEventSynchronize(b));
            float ms=0;HIP(hipEventElapsedTime(&ms,a,b));times.push_back(ms);
            std::printf("dot4_bench rep=%d ms=%.4f Gdot4_per_s=%.4f\n",rep,ms,(double)blocks*threads*iterations/ms/1e6);
        }
        // Independently check the timed kernel too; no timing of wrong arithmetic is accepted.
        std::vector<unsigned> got(blocks*threads);HIP(hipMemcpy(got.data(),d,got.size()*sizeof(unsigned),hipMemcpyDeviceToHost));
        for(unsigned id=0;id<got.size();id+=257) {
            unsigned x=mix(id+1),aa=mix(id+2),bb=mix(id+3);
            for(int i=0;i<iterations;++i){x=(unsigned)dot_ref(aa^x,bb,x);aa=aa*1664525u+1013904223u;}
            if(x!=got[id])throw std::runtime_error("dot bench CPU reference mismatch");
        }
        HIP(hipEventDestroy(a));HIP(hipEventDestroy(b));HIP(hipFree(d));
        std::puts("PASS: independent wave64 half-warp parity and timed dot4 CPU reference");return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"ERROR: %s\n",e.what());return 2;}
}
