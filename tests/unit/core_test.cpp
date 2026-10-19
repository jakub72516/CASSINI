#include "framework.hpp"

#include "cassini/core/constants.hpp"
#include "cassini/core/ddouble.hpp"
#include "cassini/core/rng.hpp"
#include "cassini/core/summation.hpp"
#include "cassini/core/vec3.hpp"

#include <cmath>
#include <vector>

using cassini::dd_real;

// --- dd_real arithmetic ---------------------------------------------------
TEST(dd_real_add_matches_long_double) {
    // (1 + 1e-30) - 1 must retain the small term (double would lose it).
    dd_real a = 1.0;
    dd_real b = 1.0e-30;
    dd_real s = a + b;
    dd_real d = s - a;
    CHECK_NEAR(d.hi, 1.0e-30, 1.0e-45);
}

TEST(dd_real_multiplication_precision) {
    // (1 + 1e-30)^2 = 1 + 2e-30 + 1e-60: the 1e-60 term is below dd precision,
    // so the result is hi=1, lo=2e-30 (a plain double would drop the lo).
    dd_real x(1.0, 1.0e-30);
    dd_real y = x * x;
    CHECK_NEAR(y.hi, 1.0, 0.0);
    CHECK_NEAR(y.lo, 2.0e-30, 1.0e-45);
}

TEST(dd_real_division_identity) {
    const dd_real pi(3.14159265358979323846);
    const dd_real one = pi / pi;
    CHECK_NEAR(one.hi, 1.0, 1.0e-15);
}

TEST(dd_real_sqrt) {
    const dd_real two(2.0);
    const dd_real s = cassini::sqrt(two);
    CHECK_NEAR(s.hi, std::sqrt(2.0), 1.0e-15);
}

TEST(dd_real_comparison) {
    CHECK(dd_real(1.0) < dd_real(2.0));
    CHECK(dd_real(1.0, 1.0e-300) > dd_real(1.0, 0.0));
    CHECK(!(dd_real(2.0) < dd_real(1.0)));
}

// --- compensated summation ------------------------------------------------
TEST(neumaier_sum_beats_naive) {
    // 1 + many tiny terms that sum to ~1e-10 then subtract 1 => exact 1e-10.
    std::vector<double> terms;
    terms.push_back(1.0);
    for (int i = 0; i < 1000000; ++i) terms.push_back(1.0e-16);
    terms.push_back(-1.0);

    double ns = 0.0;
    for (double t : terms) ns += t;

    const dd_real nd = cassini::neumaier_sum(terms.begin(), terms.end());
    const double full = nd.hi + nd.lo;
    CHECK_NEAR(full, 1.0e-10, 1.0e-12);
    // Naive summation loses the small term entirely.
    CHECK(std::fabs(ns - 1.0e-10) > std::fabs(full - 1.0e-10));
}

// --- constants ------------------------------------------------------------
TEST(gaussian_constant_consistency) {
    CHECK_NEAR(cassini::constants::G, 0.01720209895 * 0.01720209895, 1.0e-18);
    CHECK(cassini::constants::C > 100.0);  // speed of light ~173 AU/day
}

// --- vec3 -----------------------------------------------------------------
TEST(vec3_basic_algebra) {
    cassini::Vec3<double> a(1, 2, 3);
    cassini::Vec3<double> b(4, 5, 6);
    CHECK_NEAR(cassini::dot(a, b), 32.0, 1.0e-15);
    const auto c = cassini::cross(a, b);
    CHECK_NEAR(c.x, -3.0, 1.0e-15);
    CHECK_NEAR(c.y, 6.0, 1.0e-15);
    CHECK_NEAR(c.z, -3.0, 1.0e-15);
}

// --- RNG ------------------------------------------------------------------
TEST(rng_deterministic) {
    cassini::Pcg32 r1(42);
    cassini::Pcg32 r2(42);
    for (int i = 0; i < 100; ++i) {
        CHECK(r1.next() == r2.next());
    }
    // Uniform is in [0,1).
    cassini::Pcg32 r3(7);
    for (int i = 0; i < 1000; ++i) {
        const double u = r3.uniform();
        CHECK(u >= 0.0 && u < 1.0);
    }
}

CAS_TEST_MAIN()
