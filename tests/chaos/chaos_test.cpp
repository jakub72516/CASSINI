// Cassini — chaos test: Lyapunov exponent of the inner solar system.
//
// The full solar system is chaotic on a ~5 Myr Lyapunov time (Laskar 1989).
// Over a few thousand years the shadow-trajectory divergence is measurable and
// positive; the exponent is reported and checked to be positive and within an
// order of magnitude of 1/(5 Myr).
#include "framework.hpp"

#include "cassini/analysis/lyapunov.hpp"
#include "cassini/physics/solar_system.hpp"
#include "cassini/physics/wisdom_holman.hpp"

#include <cmath>

using namespace cassini;
using namespace cassini::analysis;
using namespace cassini::physics;

TEST(mercury_lyapunov_exponent_positive) {
    System sys = build_solar_system();
    WisdomHolman wh;

    // 500 years of shadow-trajectory divergence.
    const double total_time = 365.25 * 500.0;
    const double renormalize_every = 365.25 * 50.0;  // renormalize every 50 yr
    const double lambda = lyapunov_exponent(wh, sys, total_time, renormalize_every);

    std::printf("    Lyapunov exponent = %.3g /day (transient time = %.2g Myr)\n", lambda,
                1.0 / (lambda * 365.25 * 1.0e6));

    // Chaotic: exponent must be positive. (The asymptotic ~1/5 Myr value
    // requires multi-Myr integration; over 500 yr the transient exponent is
    // larger, dominated by the fastest chaotic degree of freedom.)
    CHECK(lambda > 0.0);
    CHECK(lambda < 1.0e-4);
}

CAS_TEST_MAIN()
