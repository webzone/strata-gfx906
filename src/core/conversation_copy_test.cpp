// The conversation copies (conversation_copy.hpp) move bytes between every endpoint kind: device allocations,
// pinned staging, mapped aliases and plain heap. A plain host endpoint is staged through pinned bounce scratch
// (no blit ever touches pages a free could take away mid-copy), the device-accessible endpoints keep the direct
// copy, and host-to-host is a plain memcpy. Every path must return the same bytes, on every size that crosses
// a bounce-chunk boundary.  Runs on one device; skips with 77 when the machine has none.
#include "conversation_copy.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

using strata::core::conversation_detail::copy_nonblocking;
using strata::core::conversation_detail::endpoint_kind;
using strata::core::conversation_detail::kCopyBounceBytes;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
void cuda_check(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

enum class Kind { Device, Pinned, Heap };

struct Buffer {
    Kind kind;
    std::unique_ptr<uint8_t[]> heap;   // plain host memory
    uint8_t* p = nullptr;              // device or pinned address
    size_t bytes = 0;

    Buffer(Kind k, size_t n) : kind(k), bytes(n) {
        if (k == Kind::Device) cuda_check(cudaMalloc((void**) &p, n));
        else if (k == Kind::Pinned) cuda_check(cudaHostAlloc((void**) &p, n, cudaHostAllocDefault));
        else heap.reset(new uint8_t[n]);
        fill(0xaa);
    }
    ~Buffer() {
        if (kind == Kind::Device && p) cudaFree(p);
        if (kind == Kind::Pinned && p) cudaFreeHost(p);
    }
    uint8_t* data() { return kind == Kind::Heap ? heap.get() : p; }
    void fill(uint8_t v) {
        if (kind == Kind::Device) cuda_check(cudaMemset(p, v, bytes));
        else std::memset(data(), v, bytes);
    }
    void fill_pattern(size_t n) {
        if (kind == Kind::Device) {
            std::vector<uint8_t> host(n);
            for (size_t i = 0; i < n; ++i) host[i] = (uint8_t) (i * 131 + 7);
            cuda_check(cudaMemcpy(p, host.data(), n, cudaMemcpyHostToDevice));
        } else {
            for (size_t i = 0; i < n; ++i) data()[i] = (uint8_t) (i * 131 + 7);
        }
    }
    std::vector<uint8_t> read_back(size_t n) {
        std::vector<uint8_t> out(n);
        if (kind == Kind::Device) cuda_check(cudaMemcpy(out.data(), p, n, cudaMemcpyDeviceToHost));
        else std::memcpy(out.data(), data(), n);
        return out;
    }
};

const char* name(Kind k) { return k == Kind::Device ? "device" : k == Kind::Pinned ? "pinned" : "heap"; }
}   // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    cuda_check(cudaSetDevice(0));

    // what this runtime reports for each kind - the bounce policy leans on plain heap being not device-accessible
    bool device_flag = false, accessible = false;
    {
        Buffer probe(Kind::Device, 256);
        cuda_check(endpoint_kind(probe.p, device_flag, accessible));
        std::fprintf(stderr, "strata conversation copy: device buffer: device=%d accessible=%d\n", device_flag, accessible);
        check(device_flag && accessible, "a device allocation classifies as accessible device memory");
    }
    {
        Buffer probe(Kind::Pinned, 256);
        cuda_check(endpoint_kind(probe.p, device_flag, accessible));
        std::fprintf(stderr, "strata conversation copy: pinned buffer: device=%d accessible=%d\n", device_flag, accessible);
        check(!device_flag && accessible, "a pinned buffer classifies as device-accessible host memory");
    }
    {
        Buffer probe(Kind::Heap, 256);
        cuda_check(endpoint_kind(probe.data(), device_flag, accessible));
        std::fprintf(stderr, "strata conversation copy: heap buffer: device=%d accessible=%d\n", device_flag, accessible);
        check(!device_flag && !accessible, "plain heap classifies as plain host memory (the bounce path)");
    }

    const size_t sizes[] = {1, 3, 4095, 4096, 4097, kCopyBounceBytes - 1, kCopyBounceBytes, kCopyBounceBytes + 1,
                            2 * kCopyBounceBytes + 7};
    const size_t cap = 2 * kCopyBounceBytes + 16;   // headroom for the +1/+3 interior offsets at the largest size
    const Kind kinds[] = {Kind::Device, Kind::Pinned, Kind::Heap};
    for (Kind sk : kinds) {
        for (Kind dk : kinds) {
            Buffer s(sk, cap), d(dk, cap);
            s.fill_pattern(cap);   // the pattern must cover src+3 .. src+3+n for every size
            for (size_t n : sizes) {
                d.fill(0xaa);
                cuda_check(cudaDeviceSynchronize());   // the fill must not overtake the private-stream copy below
                cuda_check(copy_nonblocking(d.data() + 1, s.data() + 3, n));   // interior pointers on both ends
                const auto got = d.read_back(n + 1);
                for (size_t i = 0; i < n; ++i)
                    if (got[i + 1] != (uint8_t) ((3 + i) * 131 + 7)) {
                        std::fprintf(stderr, "FAIL: %s->%s at %zu bytes: byte %zu is %02x, want %02x\n", name(sk),
                                     name(dk), n, i, got[i + 1], (uint8_t) ((3 + i) * 131 + 7));
                        std::exit(1);
                    }
                check(got[0] == 0xaa, "the copy stayed inside the interior destination");
                ++checks;
            }
            std::fprintf(stderr, "strata conversation copy: %s -> %s ok\n", name(sk), name(dk));
        }
    }
    std::fprintf(stderr, "strata conversation copy: %d checks passed\n", checks);
    return 0;
}
