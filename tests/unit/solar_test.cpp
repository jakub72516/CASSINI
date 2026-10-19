#include "framework.hpp"

#include "cassini/core/constants.hpp"
#include "cassini/math/kepler.hpp"
#include "cassini/physics/diagnostics.hpp"
#include "cassini/physics/solar_system.hpp"
#include "cassini/physics/wisdom_holman.hpp"

#include <cmath>

using namespace cassini;
using namespace cassini::physics;

namespace {
System sun_earth() {
    System sys;
    sys.add(1.0, Vec3<double>(0, 0, 0), Vec3<double>(0, 0, 0));
    const double a = 1.0;
    const double v = std::sqrt(constants::G / a);
    sys.add(3.00349e-6, Vec3<double>(a, 0, 0), Vec3<double>(0, v, 0));
    // Shift to COM frame so total momentum is zero (WH assumes COM frame).
    const Vec3<double> p = 3.00349e-6 * Vec3<double>(0, v, 0);
    sys.vel[0] = p * (-1.0 / 1.0);
    return sys;
}
}  // namespace

// --- Wisdom-Holman single planet = exact Kepler drift ---------------------
TEST(wh_single_planet_matches_kepler) {
    System sys = sun_earth();
    WisdomHolman wh;

    const Vec3<double> dr0 = sys.pos[1] - sys.pos[0];
    const Vec3<double> dv0 = sys.vel[1] - sys.vel[0];
    const double mu = constants::G * (1.0 + 3.00349e-6);

    const math::KeplerSolution ref = math::kepler_propagate(dr0, dv0, 1.0, mu);
    wh.step(sys, 1.0);

    const Vec3<double> dr1 = sys.pos[1] - sys.pos[0];
    const Vec3<double> dv1 = sys.vel[1] - sys.vel[0];
    CHECK_NEAR(dr1.x, ref.r.x, 1.0e-12);
    CHECK_NEAR(dr1.y, ref.r.y, 1.0e-12);
    CHECK_NEAR(dv1.x, ref.v.x, 1.0e-12);
    CHECK_NEAR(dv1.y, ref.v.y, 1.0e-12);
}

// --- Wisdom-Holman long-term energy conservation (no secular drift) -------
TEST(wh_energy_no_secular_drift) {
    System sys = build_solar_system();
    WisdomHolman wh;
    const double e0 = total_energy(sys);

    // 400 years in ~16-day steps.
    const double dt = 16.0;
    const int steps = static_cast<int>(365.25 * 400 / 16);
    double max_drift = 0.0;
    for (int i = 0; i < steps; ++i) {
        wh.step(sys, dt);
        const double drift = std::fabs(total_energy(sys) / e0 - 1.0);
        if (drift > max_drift) max_drift = drift;
    }
    // Symplectic integrator: energy error is bounded, not secular. A 2nd-order
    // map with 16-day steps should stay far below 1e-6 relative.
    CHECK(max_drift < 1.0e-6);
}

// --- Solar system construction --------------------------------------------
TEST(solar_system_zero_momentum) {
    System sys = build_solar_system();
    CHECK(sys.size() == 9);
    Vec3<double> p{};
    for (std::size_t i = 0; i < sys.size(); ++i) p += sys.mass[i] * sys.vel[i];
    CHECK(norm(p) < 1.0e-15);
}

TEST(solar_system_earth_at_one_au) {
    System sys = build_solar_system();
    // Earth (index 3) should be ~1 AU from the Sun.
    const double r = norm(sys.pos[3] - sys.pos[0]);
    CHECK_NEAR(r, 1.0, 0.02);
}

CAS_TEST_MAIN()
