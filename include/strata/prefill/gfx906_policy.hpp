// CPU-only admission policy for the experimental gfx906 prefill paths.
#pragma once

#include <cstdint>
#include <cstring>

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
inline bool split_draft(bool split, const char* value, const char* arch) {
    return split && opt_in(value) && architecture(arch);
}

} // namespace strata::prefill::gfx906
