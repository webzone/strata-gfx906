// CPU-only admission policy for the experimental gfx906 prefill paths.
#pragma once

#include <cstdint>
#include <cstring>
#include <initializer_list>

namespace strata::prefill::gfx906 {

// Deliberately exact: an unset switch, "0", or a misspelling must preserve the control path.
inline bool opt_in(const char* value) { return value && std::strcmp(value, "1") == 0; }
inline bool architecture(const char* arch) {
    return arch && std::strncmp(arch, "gfx906", 6) == 0 && (arch[6] == '\0' || arch[6] == ':');
}
inline bool attention_geometry(int64_t queries, int64_t cap, int64_t heads, int64_t kv_heads,
                               int64_t dim, int64_t page_size) {
    return queries > 0 && queries <= 65535 && cap > 0 && cap <= 32768 &&
           heads == 24 && kv_heads == 2 && dim == 256 && page_size > 0;
}
// Tile experiments affect grouped experts only, not single-matrix drafter/dense calls.
// Only pinned-model dimensions are admitted until other geometries are numerically validated.
inline int mmq_tile(const char* value, const char* arch) {
    if (!architecture(arch) || !value) return 0;
    if (std::strcmp(value, "auto") == 0) return -1; // measured per-product policy, never a global default
    for (const auto& entry : {"16", "32", "48", "64"})
        if (std::strcmp(value, entry) == 0) return (entry[0] - '0') * 10 + entry[1] - '0';
    return 0;
}
inline bool mmq_expert_geometry(int experts, int64_t rows, int64_t cols) {
    return experts > 1 && ((rows == 1280 && cols == 2560) || (rows == 2560 && cols == 640));
}
// Conservative measured bands: leave tiny and very large/skewed groups on the vendor selector.
// Q2_0 and unmeasured formats explicitly decline auto tuning at the caller.
inline int mmq_auto_tile(int64_t max_rows, bool measured_format) {
    if (!measured_format) return 0;
    if (max_rows > 48 && max_rows <= 96) return 32;
    if (max_rows > 96 && max_rows <= 192) return 64;
    return 0;
}
inline bool split_draft(bool split, const char* value, const char* arch) {
    return split && opt_in(value) && architecture(arch);
}

} // namespace strata::prefill::gfx906
