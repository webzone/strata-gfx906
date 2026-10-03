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
    CHECK(mmq_tile(nullptr, "gfx906") == 0); CHECK(mmq_tile("0", "gfx906") == 0);
    CHECK(mmq_tile("32", "gfx1100") == 0); CHECK(mmq_tile("032", "gfx906") == 0);
    CHECK(mmq_tile("32junk", "gfx906") == 0); CHECK(mmq_tile("128", "gfx906") == 0);
    CHECK(mmq_tile("16", "gfx906") == 16); CHECK(mmq_tile("32", "gfx906") == 32);
    CHECK(mmq_tile("48", "gfx906:xnack-:sramecc+") == 48); CHECK(mmq_tile("64", "gfx906") == 64);
    CHECK(mmq_expert_geometry(32, 1280, 2560)); CHECK(mmq_expert_geometry(2, 2560, 640));
    CHECK(!mmq_expert_geometry(1, 1280, 2560)); CHECK(!mmq_expert_geometry(0, 2560, 640));
    CHECK(!mmq_expert_geometry(32, 128, 256));
    CHECK(mmq_tile("auto", "gfx906") == -1); CHECK(mmq_tile("auto", "gfx90a") == 0);
    CHECK(mmq_tile("AUTO", "gfx906") == 0);
    CHECK(mmq_auto_tile(0, true) == 0); CHECK(mmq_auto_tile(48, true) == 0);
    CHECK(mmq_auto_tile(49, true) == 32); CHECK(mmq_auto_tile(96, true) == 32);
    CHECK(mmq_auto_tile(97, true) == 64); CHECK(mmq_auto_tile(192, true) == 64);
    CHECK(mmq_auto_tile(193, true) == 0); CHECK(mmq_auto_tile(70, false) == 0);
    std::printf("gfx906 prefill policy: %d checks passed (CPU-only; not GPU or model parity)\n", checks);
    return 0;
}
