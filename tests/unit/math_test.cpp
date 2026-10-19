#include "framework.hpp"

#include "cassini/core/constants.hpp"
#include "cassini/core/vec3.hpp"
#include "cassini/math/kepler.hpp"
#include "cassini/math/stumpff.hpp"

#include <cmath>

using cassini::Vec3;
using namespace cassini::math;

// --- Stumpff functions ----------------------------------------------------
TEST(stumpff_c0_definition) {
    // c0(z) = cos(sqrt(z)) for z > 0.
    for (double z = 0.2; z < 5.0; z += 0.3) {
        const Stumpff s = stumpff(z);
        CHECK_NEAR(s.c0, std::cos(std::sqrt(z)), 1.0e-12);
    }
}

TEST(stumpff_hyperbolic_branch) {
    // c0(z) = cosh(sqrt(-z)) for z < 0.
    const Stumpff s = stumpff(-4.0);
    CHECK_NEAR(s.c0, std::cosh(2.0), 1.0e-12);
    // c1(z) = sinh(x)/x with x = 2.
    CHECK_NEAR(s.c1, std::sinh(2.0) / 2.0, 1.0e-12);
}

// --- Universal-variable Kepler solver -------------------------------------
TEST(kepler_circular_orbit_returns) {
    const double mu = cassini::constants::G;  // AU^3/day^2
    const double a = 1.0;                     // 1 AU
    const Vec3<double> r0(a, 0.0, 0.0);
    const Vec3<double> v0(0.0, std::sqrt(mu / a), 0.0);
    const double period = 2.0 * 3.141592653589793 * std::sqrt(a * a * a / mu);

    const KeplerSolution sol = kepler_propagate(r0, v0, period, mu);
    CHECK(sol.converged);
    CHECK_NEAR(sol.r.x, a, 1.0e-9);
    CHECK_NEAR(sol.r.y, 0.0, 1.0e-9);
    CHECK_NEAR(sol.v.y, std::sqrt(mu / a), 1.0e-9);
}

TEST(kepler_elliptic_half_period) {
    const double mu = cassini::constants::G;
    const double a = 2.0;
    const double e = 0.5;
    const double rp = a * (1.0 - e);                    // perihelion
    const double vp = std::sqrt(mu * (1.0 + e) / (a * (1.0 - e)));
    const Vec3<double> r0(rp, 0.0, 0.0);
    const Vec3<double> v0(0.0, vp, 0.0);
    const double period = 2.0 * 3.141592653589793 * std::sqrt(a * a * a / mu);

    const KeplerSolution sol = kepler_propagate(r0, v0, period / 2.0, mu);
    CHECK(sol.converged);
    // At half period the body is at aphelion: x = -a(1+e).
    CHECK_NEAR(sol.r.x, -a * (1.0 + e), 1.0e-8);
    CHECK_NEAR(sol.r.y, 0.0, 1.0e-8);
}

TEST(kepler_hyperbolic_escape) {
    const double mu = cassini::constants::G;
    // Speed > escape speed => hyperbolic (alpha < 0), starting at perihelion.
    const Vec3<double> r0(1.0, 0.0, 0.0);
    const Vec3<double> v0(0.0, std::sqrt(3.0 * mu), 0.0);
    const KeplerSolution sol = kepler_propagate(r0, v0, 1.0, mu);
    CHECK(sol.converged);
    CHECK(cassini::norm(sol.r) > 1.0);  // moving away from perihelion
}

CAS_TEST_MAIN()
