// Portable pinned-host bridge for layer-split verify windows.
// Each GPU gets its OWN mapped alias: neither HIP nor CUDA promises that a
// host pointer (or another device's alias) is a valid kernel pointer on it.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {

class MappedHandoff {
public:
    MappedHandoff() = default;
    ~MappedHandoff() { reset(); }
    MappedHandoff(const MappedHandoff&) = delete;
    MappedHandoff& operator=(const MappedHandoff&) = delete;

    // No streams may still be using an old allocation when init/reset is called.
    bool init(size_t bytes, const std::vector<int>& devices, std::string& err) {
        reset();
        if (bytes == 0 || bytes % sizeof(float) != 0 || devices.empty()) {
            err = "layer handoff: invalid size or empty device list";
            return false;
        }
        cudaError_t e = cudaGetDevice(&owner_);
        if (e != cudaSuccess) {
            err = std::string("layer handoff: current device: ") + cudaGetErrorString(e);
            return false;
        }
        e = cudaHostAlloc(reinterpret_cast<void**>(&host_), bytes,
                          cudaHostAllocMapped | cudaHostAllocPortable);
        if (e != cudaSuccess) {
            err = std::string("layer handoff: pinned allocation: ") + cudaGetErrorString(e);
            return false;
        }
        for (int device : devices) {
            // Do not silently map on the wrong GPU if a device switch failed.
            void* alias = nullptr;
            e = cudaSetDevice(device);
            if (e == cudaSuccess) e = cudaHostGetDevicePointer(&alias, host_, 0);
            if (e != cudaSuccess || alias == nullptr) {
                err = "layer handoff: mapping on device " + std::to_string(device) + ": " +
                      (e == cudaSuccess ? "null device pointer" : cudaGetErrorString(e));
                cudaSetDevice(owner_);
                reset();
                return false;
            }
            aliases_.emplace_back(device, static_cast<float*>(alias));
        }
        e = cudaSetDevice(owner_);
        if (e != cudaSuccess) {
            err = std::string("layer handoff: restore current device: ") + cudaGetErrorString(e);
            reset();
            return false;
        }
        bytes_ = bytes;
        return true;
    }

    float* host_data() const { return host_; }
    float* device_data(int device) const {
        for (const auto& entry : aliases_)
            if (entry.first == device) return entry.second;
        return nullptr;
    }
    size_t bytes() const { return bytes_; }

    void reset() noexcept {
        if (host_) {
            int saved = owner_;
            cudaGetDevice(&saved);
            cudaSetDevice(owner_);
            cudaFreeHost(host_);
            cudaSetDevice(saved);
        }
        host_ = nullptr;
        aliases_.clear();
        bytes_ = 0;
    }

private:
    float* host_ = nullptr;
    std::vector<std::pair<int, float*>> aliases_;
    size_t bytes_ = 0;
    int owner_ = 0;
};

} // namespace strata::core
