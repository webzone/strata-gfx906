#pragma once
#include "strata_hip.h"

// The four CUDA half bit-casts.  hip/hip_fp16.h only carries them on its `__clang__ && __HIP__` branch; a host-only
// translation unit (a test .cpp built by the host compiler in this lane) gets hip_fp16_gcc.h instead, which has no
// names of this kind, so `__half_as_ushort` is "not declared in this scope".  Same bits: __half_raw is the unsigned
// short the value lives in, which is exactly what __half_as_ushort returns.
#if !(defined(__clang__) && defined(__HIP__))
inline short __half_as_short(__half x) { return static_cast<__half_raw>(x).x; }
inline unsigned short __half_as_ushort(__half x) { return static_cast<__half_raw>(x).x; }
inline __half __short_as_half(short x) { __half_raw r; r.x = x; return r; }
inline __half __ushort_as_half(unsigned short x) { __half_raw r; r.x = x; return r; }
#endif
