#include "strata/prefill/gfx906_policy.hpp"

#include <cstdio>

int main() {
    using namespace strata::prefill::gfx906;
    int checks = 0;
#define CHECK(x) do { ++checks; if (!(x)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); return 1; } } while (0)
    CHECK(!opt_in(nullptr)); CHECK(!opt_in("")); CHECK(!opt_in("0")); CHECK(!opt_in("true"));
    CHECK(!opt_in("01")); CHECK(!opt_in("1garbage")); CHECK(opt_in("1"));
    CHECK(!architecture(nullptr)); CHECK(!architecture("")); CHECK(!architecture("gfx90a"));
    CHECK(!architecture("gfx9060")); CHECK(!architecture("gfx1100")); CHECK(!architecture("GFX906"));
    CHECK(architecture("gfx906")); CHECK(architecture("gfx906:sramecc+:xnack-"));
    CHECK(attention_geometry(1, 1, 24, 2, 256, 256));
    CHECK(attention_geometry(4096, 2051, 24, 2, 256, 256));
    CHECK(!attention_geometry(0, 2051, 24, 2, 256, 256));
    CHECK(!attention_geometry(65536, 2051, 24, 2, 256, 256));
    CHECK(!attention_geometry(32, 0, 24, 2, 256, 256));
    CHECK(!attention_geometry(32, 32769, 24, 2, 256, 256));
    CHECK(!attention_geometry(32, 2051, 12, 1, 256, 256));
    CHECK(!attention_geometry(32, 2051, 24, 2, 128, 256));
    CHECK(!attention_geometry(32, 2051, 24, 2, 256, 0));
    CHECK(!split_draft(false, "1", "gfx906")); CHECK(!split_draft(true, nullptr, "gfx906"));
    CHECK(!split_draft(true, "0", "gfx906")); CHECK(!split_draft(true, "1", "gfx1201"));
    CHECK(split_draft(true, "1", "gfx906"));
    std::printf("gfx906 prefill policy: %d checks passed (CPU-only; not GPU or model parity)\n", checks);
    return 0;
}
