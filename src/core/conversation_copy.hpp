#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>

namespace strata::core::conversation_detail {

/// Conversation snapshot / checkpoint copies go through a private non-blocking stream of the current device
/// instead of the legacy (null) stream.  A plain cudaMemcpy is ordered against every blocking stream of the
/// device, and while another thread captures a graph on one (a drafter or a stage worker), HIP refuses it with
/// "operation would make the legacy stream depend on a capturing blocking stream".  The caller's contract is
/// unchanged: the device was synchronized before the copy (the copy itself is complete on return).
///
/// The copies run as shader blits, not SDMA (HSA_ENABLE_SDMA=0), so the blit touches a plain host endpoint
/// through a device mapping that dies with the pages.  A host endpoint freed before its blit runs - a free on
/// another thread, or a torn-down turn whose buffer the sticky fault outlives - faults the GPU, and the sticky
/// error surfaces many calls later, naming whichever copy polled next (2026-10-08/10-11 T5810: WRITE page
/// faults inside one 2 MiB span of heap, surfacing at a checkpoint save and at KV streaming).  Plain host
/// endpoints therefore stage through pinned bounce scratch the driver maps for the process lifetime; no blit
/// ever touches pages a free could take away.  Pinned or registered host buffers (the pool aliases, the pack
/// arena) and device allocations keep the direct copy.  An endpoint the runtime cannot classify keeps the old
/// cudaMemcpyDefault behaviour.

inline constexpr size_t kCopyBounceBytes = 4 << 20;   // pinned staging per thread, sized for one segment visit

/// A device allocation is `device`; `accessible` also covers every host buffer the device can address for the
/// allocation's lifetime (pinned staging, the registered pack arena, the mapped pool aliases) - a blit may
/// touch those directly because their mapping cannot be taken away under it.
inline cudaError_t endpoint_kind(const void* p, bool& device, bool& accessible) {
    cudaPointerAttributes a{};
    const cudaError_t e = cudaPointerGetAttributes(&a, const_cast<void*>(p));
    if (e != cudaSuccess) {
        (void) cudaGetLastError();
        return e;
    }
    device = a.type == cudaMemoryTypeDevice;
    accessible = device || a.devicePointer != nullptr;
    return cudaSuccess;
}

inline cudaError_t copy_nonblocking(void* dst, const void* src, size_t bytes) {
    if (!bytes) return cudaSuccess;
    int dev = 0;
    cudaError_t e = cudaGetDevice(&dev);
    if (e != cudaSuccess) return e;
    thread_local std::vector<cudaStream_t> streams;
    if (dev < 0) dev = 0;
    if ((size_t) dev >= streams.size()) streams.resize((size_t) dev + 1, nullptr);   // never destroyed: process lifetime
    if (!streams[(size_t) dev]) {
        e = cudaStreamCreateWithFlags(&streams[(size_t) dev], cudaStreamNonBlocking);
        if (e != cudaSuccess) { streams[(size_t) dev] = nullptr; return e; }
    }
    cudaStream_t s = streams[(size_t) dev];

    bool src_device = false, src_accessible = false, dst_device = false, dst_accessible = false;
    if (endpoint_kind(src, src_device, src_accessible) != cudaSuccess ||
        endpoint_kind(dst, dst_device, dst_accessible) != cudaSuccess) {
        // An endpoint the runtime refuses to classify: the behaviour before the bounce existed.
        e = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, s);
        return e != cudaSuccess ? e : cudaStreamSynchronize(s);
    }
    if (!src_device && !dst_device) {
        // Host to host, through an alias or not: the CPU moves the bytes, the GPU nothing.
        std::memmove(dst, src, bytes);
        return cudaSuccess;
    }
    if (src_device && dst_device) {
        e = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, s);
        return e != cudaSuccess ? e : cudaStreamSynchronize(s);
    }
    if (src_device ? dst_accessible : src_accessible) {
        // The device blit only touches the endpoint whose mapping is stable for the allocation's lifetime.
        e = cudaMemcpyAsync(dst, src, bytes, src_device ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice, s);
        return e != cudaSuccess ? e : cudaStreamSynchronize(s);
    }
    // One endpoint is plain (not device-accessible) host memory: stage the plain leg through pinned bounce
    // scratch so no blit ever writes or reads pages a free could take away while it runs.
    struct Bounce { uint8_t* p; };
    thread_local Bounce* bounce = nullptr;   // never freed: process lifetime, like the streams
    if (bounce == nullptr) {
        void* p = nullptr;
        const cudaError_t e = cudaHostAlloc(&p, kCopyBounceBytes, cudaHostAllocDefault);
        if (e != cudaSuccess) return e;
        bounce = new Bounce{static_cast<uint8_t*>(p)};
    }
    const bool to_host = src_device;   // device to plain host; the other plain case is plain host to device
    for (size_t off = 0; off < bytes; off += kCopyBounceBytes) {
        const size_t n = std::min(kCopyBounceBytes, bytes - off);
        if (to_host) {
            e = cudaMemcpyAsync(bounce->p, static_cast<const uint8_t*>(src) + off, n, cudaMemcpyDeviceToHost, s);
            if (e == cudaSuccess) e = cudaStreamSynchronize(s);
            if (e == cudaSuccess) std::memcpy(static_cast<uint8_t*>(dst) + off, bounce->p, n);
        } else {
            std::memcpy(bounce->p, static_cast<const uint8_t*>(src) + off, n);
            e = cudaMemcpyAsync(static_cast<uint8_t*>(dst) + off, bounce->p, n, cudaMemcpyHostToDevice, s);
            if (e == cudaSuccess) e = cudaStreamSynchronize(s);
        }
        if (e != cudaSuccess) return e;
    }
    return cudaSuccess;
}

} // namespace strata::core::conversation_detail
