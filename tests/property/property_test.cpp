// Cassini — property-based tests for integrator invariants. Each property is
// checked over many randomized trials (deterministic seeds for reproducibility).
#include "framework.hpp"

#include "cassini/core/constants.hpp"
#include "cassini/core/rng.hpp"
#include "cassini/physics/diagnostics.hpp"
#include "cassini/physics/forces.hpp"
#include "cassini/physics/rk45.hpp"
#include "cassini/physics/system.hpp"
#include "cassini/physics/wisdom_holman.hpp"

#include <cmath>

using namespace cassini;
using namespace cassini::physics;

namespace {
System random_two_body(std::uint64_t seed) {
    Pcg32 rng(seed);
    const double m0 = 1.0;
    const double m1 = rng.uniform(1.0e-6, 1.0e-4);
    const double a = rng.uniform(0.5, 5.0);
    const double mu = constants::G * (m0 + m1);
    const double v = std::sqrt(mu / a);
    // COM at origin, zero total momentum (WH works in the COM frame).
    System sys;
    sys.add(m0, Vec3<double>(-m1 / (m0 + m1) * a, 0, 0), Vec3<double>(0, -m1 / (m0 + m1) * v, 0));
    sys.add(m1, Vec3<double>(m0 / (m0 + m1) * a, 0, 0), Vec3<double>(0, m0 / (m0 + m1) * v, 0));
    return sys;
}
}  // namespace

// --- WH is time-reversible (symmetric map) --------------------------------
TEST(wh_time_reversible) {
    for (int trial = 0; trial < 50; ++trial) {
        System sys = random_two_body(static_cast<std::uint64_t>(1000 + trial));
        System orig = sys;
        WisdomHolman wh;
        const double dt = 5.0;
        wh.step(sys, dt);
        wh.step(sys, -dt);
        for (std::size_t i = 0; i < sys.size(); ++i) {
            CHECK_NEAR(sys.pos[i].x, orig.pos[i].x, 1.0e-12);
            CHECK_NEAR(sys.pos[i].y, orig.pos[i].y, 1.0e-12);
        }
    }
}

// --- RK45 energy is bounded over random trials ----------------------------
TEST(rk45_energy_bounded) {
    for (int trial = 0; trial < 20; ++trial) {
        System sys = random_two_body(static_cast<std::uint64_t>(2000 + trial));
        NewtonianForce force;
        RK45 rk(force, 1.0e-12);
        const double e0 = total_energy(sys);
        const double period = 2.0 * 3.141592653589793 * std::sqrt(1.0 / constants::G);
        for (int i = 0; i < 100; ++i) rk.step(sys, period / 100.0);
        const double drift = std::fabs(total_energy(sys) / e0 - 1.0);
        CHECK(drift < 1.0e-8);
    }
}

// --- WH preserves energy to a bounded, non-secular error ------------------
TEST(wh_energy_no_secular_drift_property) {
    for (int trial = 0; trial < 10; ++trial) {
        System sys = random_two_body(static_cast<std::uint64_t>(3000 + trial));
        WisdomHolman wh;
        const double e0 = total_energy(sys);
        const double period = 2.0 * 3.141592653589793 * std::sqrt(1.0 / constants::G);
        for (int i = 0; i < 200; ++i) wh.step(sys, period / 50.0);
        const double drift = std::fabs(total_energy(sys) / e0 - 1.0);
        CHECK(drift < 1.0e-6);
    }
}

CAS_TEST_MAIN()
