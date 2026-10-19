#include "framework.hpp"

#include "cassini/data/spk.hpp"

#include <cmath>
#include <vector>

using namespace cassini;
using namespace cassini::data;

// --- Round-trip: constant position (degree 0) ----------------------------
TEST(spk_roundtrip_constant_position) {
    // One record: MID=0, RADIUS=1, X=[1], Y=[2], Z=[3] => position (1,2,3).
    const int rsize = 5;
    std::vector<double> rec = {0.0, 1.0, 1.0, 2.0, 3.0};
    const auto bytes = make_type2_spk(199, 10, 1, -10.0, 10.0, 0.0, 2.0, rsize, rec);

    SpkFile spk;
    CHECK(spk.load_from_memory(bytes));
    CHECK(spk.segments().size() == 1);

    Vec3<double> r, v;
    CHECK(spk.state(199, 10, 0.5, r, v));
    CHECK_NEAR(r.x, 1.0, 1.0e-12);
    CHECK_NEAR(r.y, 2.0, 1.0e-12);
    CHECK_NEAR(r.z, 3.0, 1.0e-12);
    CHECK_NEAR(v.x, 0.0, 1.0e-12);
    CHECK_NEAR(v.y, 0.0, 1.0e-12);
    CHECK_NEAR(v.z, 0.0, 1.0e-12);
}

// --- Round-trip: linear motion (degree 1) -> velocity via derivative ------
TEST(spk_roundtrip_linear_motion_deg1) {
    const int rsize = 8;  // degree 1: MID, RADIUS, X[2], Y[2], Z[2]
    std::vector<double> rec = {
        0.0, 1.0,       // MID, RADIUS
        1.0, 2.0,       // X: 1 + 2s
        0.0, 0.0,       // Y: 0
        0.0, 0.0,       // Z: 0
    };
    const auto bytes = make_type2_spk(199, 10, 1, -10.0, 10.0, 0.0, 2.0, rsize, rec);

    SpkFile spk;
    CHECK(spk.load_from_memory(bytes));

    Vec3<double> r, v;
    CHECK(spk.state(199, 10, 0.5, r, v));
    CHECK_NEAR(r.x, 2.0, 1.0e-12);  // 1 + 2*0.5
    CHECK_NEAR(v.x, 2.0, 1.0e-12);  // dX/dt = 2
    CHECK_NEAR(v.y, 0.0, 1.0e-12);
}

// --- Reader rejects garbage ----------------------------------------------
TEST(spk_rejects_garbage) {
    std::vector<std::uint8_t> junk(2048, 0);
    SpkFile spk;
    CHECK(!spk.load_from_memory(junk));
}

// --- Endianness: big-endian file read correctly ---------------------------
TEST(spk_big_endian_read) {
    const int rsize = 5;
    std::vector<double> rec = {0.0, 1.0, 1.0, 2.0, 3.0};
    const auto bytes = make_type2_spk(199, 10, 1, -10.0, 10.0, 0.0, 2.0, rsize, rec,
                                      Endian::Big);

    SpkFile spk;
    CHECK(spk.load_from_memory(bytes));
    Vec3<double> r, v;
    CHECK(spk.state(199, 10, 0.5, r, v));
    CHECK_NEAR(r.x, 1.0, 1.0e-12);
    CHECK_NEAR(r.y, 2.0, 1.0e-12);
    CHECK_NEAR(r.z, 3.0, 1.0e-12);
}

CAS_TEST_MAIN()
