// Real two-device regression for the layer split's transport, NOT model parity.
// Calls the SAME copy_from_mapped kernels and MappedHandoff owner as Verifier.
// No P2P enable, weights, API, driver changes or root privileges are needed.
#include "strata/core/device.hpp"
#include "strata/core/mapped_handoff.hpp"
#include "strata/core/on_device.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/dp4a.hpp"
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <future>
#include <stdexcept>
#include <string>
#include <vector>

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
#define GPU(call) ck((call), #call)
static void require(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }

// Powers-of-two arithmetic makes an exact, independent CPU oracle possible.
__global__ static void affine(float* x, size_t n, float scale, float bias) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] * scale + bias;
}
static void transform(float* x, size_t n, float s, float b, cudaStream_t stream) {
    affine<<<(unsigned) ((n + 255) / 256), 256, 0, stream>>>(x, n, s, b);
    GPU(cudaGetLastError());
}
struct Stage {
    int device;
    cudaStream_t stream = nullptr;
    float* data = nullptr;
    Stage(int dev, size_t n) : device(dev) {
        strata::core::OnDevice on(device);
        GPU(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        GPU(cudaMalloc(reinterpret_cast<void**>(&data), n * sizeof(float)));
    }
    ~Stage() {
        strata::core::OnDevice on(device);
        cudaStreamSynchronize(stream);
        cudaFree(data);
        cudaStreamDestroy(stream);
    }
};
struct Graph {
    int device;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    explicit Graph(int d) : device(d) {}
    ~Graph() {
        strata::core::OnDevice on(device);
        if (exec) cudaGraphExecDestroy(exec);
        if (graph) cudaGraphDestroy(graph);
    }
    void finish(cudaStream_t stream) {
        GPU(cudaStreamEndCapture(stream, &graph));
        GPU(cudaGraphInstantiate(&exec, graph, 0));
    }
};

__global__ static void read_wall_clock(unsigned long long* ticks) {
    const unsigned long long begin = wall_clock64();
    // Measure while the device is ACTIVE, as the profiler uses it. Idle
    // power states between two kernels can gate the gfx906 clock counter.
    while (wall_clock64() - begin < 5000000ull) strata_spin_pause();
    ticks[0] = begin;
    ticks[1] = wall_clock64();
}
static void calibrate_clock(const strata::core::DeviceInfo& info) {
    strata::core::OnDevice on(info.ordinal);
    unsigned long long* device = nullptr;
    GPU(cudaMalloc(reinterpret_cast<void**>(&device), 2 * sizeof(*device)));
    read_wall_clock<<<1, 1>>>(device); // Warm module/context before timing.
    GPU(cudaDeviceSynchronize());
    auto begin = std::chrono::steady_clock::now();
    read_wall_clock<<<1, 1>>>(device);
    GPU(cudaDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();
    unsigned long long ticks[2];
    GPU(cudaMemcpy(ticks, device, sizeof(ticks), cudaMemcpyDeviceToHost));
    GPU(cudaFree(device));
    double mhz = (double) (ticks[1] - ticks[0]) / std::chrono::duration<double>(end - begin).count() / 1e6;
    std::printf("device%d active wall_clock64 approximately %.4f MHz\n", info.ordinal, mhz);
    if (info.arch == "gfx906") require(mhz > 24.0 && mhz < 26.0, "gfx906 40 ns timer conversion is not valid on this device");
}

constexpr size_t H = 4, N = 2560, HB = H * N + N + H, GUARD = 32;
constexpr float CANARY = -12345.0f;
static float input(size_t i, int ring) { return ((int) (i % 1024) - 512) * 0.125f + ring * 0.25f; }

static size_t verify_path(int producer, int consumer, int T) {
    GPU(cudaSetDevice(producer));
    const size_t n = (size_t) T * HB;
    strata::core::MappedHandoff hand;
    std::string err;
    if (!hand.init((n + 2 * GUARD) * sizeof(float), {producer, consumer}, err)) throw std::runtime_error(err);
    int current = -1;
    GPU(cudaGetDevice(&current));
    require(current == producer, "mapped handoff did not restore the allocating device");
    require(hand.device_data(999) == nullptr, "unmapped device returned an alias");
    std::fill(hand.host_data(), hand.host_data() + n + 2 * GUARD, CANARY);
    Stage p(producer, n), c(consumer, n);
    Graph pg(producer), cg(consumer);
    {
        strata::core::OnDevice on(producer);
        GPU(cudaStreamBeginCapture(p.stream, cudaStreamCaptureModeThreadLocal));
        transform(p.data, n, 2.0f, 1.0f, p.stream);
        for (int t = 0; t < T; ++t) {
            const size_t at = (size_t) t * HB;
            float* dst = hand.device_data(producer) + GUARD + at;
            strata::kernels::copy_from_mapped(dst, p.data + at, H * N, p.stream);
            strata::kernels::copy_from_mapped(dst + H * N, p.data + at + H * N, N, p.stream);
            strata::kernels::copy_from_mapped(dst + H * N + N, p.data + at + H * N + N, H, p.stream);
        }
        pg.finish(p.stream);
    }
    {
        strata::core::OnDevice on(consumer);
        GPU(cudaStreamBeginCapture(c.stream, cudaStreamCaptureModeThreadLocal));
        strata::kernels::copy_from_mapped(c.data, hand.device_data(consumer) + GUARD, n, c.stream);
        transform(c.data, n, 0.5f, -2.0f, c.stream);
        cg.finish(c.stream);
    }
    std::vector<float> x(n), y(n);
    size_t checks = 0;
    for (int ring = 0; ring < 32; ++ring) {
        for (size_t i = 0; i < n; ++i) x[i] = input(i, ring);
        {
            strata::core::OnDevice on(producer);
            GPU(cudaMemcpyAsync(p.data, x.data(), n * sizeof(float), cudaMemcpyHostToDevice, p.stream));
            GPU(cudaGraphLaunch(pg.exec, p.stream));
            GPU(cudaStreamSynchronize(p.stream)); // Verifier synchronizes BEFORE next_->run.
        }
        for (size_t i = 0; i < n; ++i) {
            require(hand.host_data()[GUARD + i] == x[i] * 2.0f + 1.0f, "producer/host parity");
            ++checks;
        }
        {
            strata::core::OnDevice on(consumer);
            GPU(cudaGraphLaunch(cg.exec, c.stream));
            GPU(cudaMemcpyAsync(y.data(), c.data, n * sizeof(float), cudaMemcpyDeviceToHost, c.stream));
            GPU(cudaStreamSynchronize(c.stream));
        }
        for (size_t i = 0; i < n; ++i) { require(y[i] == x[i] - 1.5f, "consumer/CPU parity"); ++checks; }
        for (size_t i = 0; i < GUARD; ++i) {
            require(hand.host_data()[i] == CANARY && hand.host_data()[GUARD + n + i] == CANARY, "handoff guard overwritten");
            checks += 2;
        }
    }
    std::printf("verify transport GPU%d->GPU%d T=%d: %zu exact checks, 32 graph replays OK\n", producer, consumer, T, checks);
    return checks;
}

static size_t prefill_path(int producer, int consumer, int T) {
    GPU(cudaSetDevice(producer));
    const size_t n = (size_t) T * H * N;
    float* host[2] = {nullptr, nullptr};
    for (auto& h : host) GPU(cudaHostAlloc(reinterpret_cast<void**>(&h), n * sizeof(float), cudaHostAllocPortable));
    Stage p(producer, n), c(consumer, n);
    std::vector<float> x(n), y(n);
    std::future<size_t> pending;
    size_t checks = 0;
    for (int ring = 0; ring < 8; ++ring) {
        for (size_t i = 0; i < n; ++i) x[i] = input(i, ring);
        float* h = host[ring & 1];
        {
            strata::core::OnDevice on(producer);
            GPU(cudaMemcpyAsync(p.data, x.data(), n * sizeof(float), cudaMemcpyHostToDevice, p.stream));
            transform(p.data, n, 2.0f, 1.0f, p.stream);
            GPU(cudaMemcpyAsync(h, p.data, n * sizeof(float), cudaMemcpyDeviceToHost, p.stream));
            GPU(cudaStreamSynchronize(p.stream));
        }
        // Exactly Prefill's two-buffer ordering: previous consumer completes
        // before starting this one; producer may have prepared the other buffer.
        if (pending.valid()) checks += pending.get();
        pending = std::async(std::launch::async, [&, h, ring] {
            strata::core::OnDevice on(consumer);
            GPU(cudaMemcpyAsync(c.data, h, n * sizeof(float), cudaMemcpyHostToDevice, c.stream));
            transform(c.data, n, 0.5f, -2.0f, c.stream);
            GPU(cudaMemcpyAsync(y.data(), c.data, n * sizeof(float), cudaMemcpyDeviceToHost, c.stream));
            GPU(cudaStreamSynchronize(c.stream));
            for (size_t i = 0; i < n; ++i) require(y[i] == input(i, ring) - 1.5f, "prefill/CPU parity");
            return n;
        });
    }
    if (pending.valid()) checks += pending.get();
    for (auto h : host) GPU(cudaFreeHost(h));
    std::printf("prefill transport GPU%d->GPU%d T=%d: %zu exact checks, two-buffer pipeline OK\n", producer, consumer, T, checks);
    return checks;
}

int main() {
    try {
        int count = 0;
        GPU(cudaGetDeviceCount(&count));
        require(count >= 2, "hip_layer_handoff REQUIRES two visible GPUs; this is not a skipped pass");
        for (int d : {0, 1}) {
            const auto info = strata::core::device_info(d); // Enforces compiled arch and experimental wave64 guard.
            std::printf("device%d %s %s wave%d\n", d, info.name.c_str(), info.arch.c_str(), info.warp_size);
            calibrate_clock(info);
            strata::core::MappedHandoff bad;
            std::string err;
            require(!bad.init(0, {d}, err) && !bad.host_data(), "zero-size handoff accepted");
            require(!bad.init(16, {count + 1}, err) && !bad.host_data() && !err.empty(), "bad ordinal accepted");
            cudaGetLastError();
            int current = -1;
            GPU(cudaGetDevice(&current));
            require(current == d, "failed mapping did not restore device");
        }
        int peer = 0;
        GPU(hipDeviceCanAccessPeer(&peer, 0, 1));
        std::printf("P2P capability=%d; transport uses pinned HOST memory, no peer access is enabled\n", peer);
        size_t checks = 0;
        for (auto pair : {std::pair{0, 1}, std::pair{1, 0}}) {
            for (int T : {1, 2, 4, 8}) checks += verify_path(pair.first, pair.second, T);
            for (int T : {32, 128, 512}) checks += prefill_path(pair.first, pair.second, T);
        }
        // The same-device split used by the engine's A/B must still work.
        checks += verify_path(0, 0, 4);
        std::printf("hip_layer_handoff OK: %zu exact float/guard checks; no model-level claim\n", checks);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "hip_layer_handoff: %s\n", e.what());
        return 1;
    }
}
