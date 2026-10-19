#include "framework.hpp"

#include "cassini/core/constants.hpp"
#include "cassini/physics/diagnostics.hpp"
#include "cassini/physics/forces.hpp"
#include "cassini/physics/ias15.hpp"
#include "cassini/physics/system.hpp"

#include <cmath>

using namespace cassini;
using namespace cassini::physics;

namespace {
// Sun + Earth, zero total momentum, relative circular orbit of radius a.
System two_body(double a = 1.0) {
    const double M = 1.0, m = 3.00349e-6;
    const double mu = constants::G * (M + m);
    const double v = std::sqrt(mu / a);
    System sys;
    sys.add(M, Vec3<double>(-m / (M + m) * a, 0, 0), Vec3<double>(0, -m / (M + m) * v, 0));
    sys.add(m, Vec3<double>(M / (M + m) * a, 0, 0), Vec3<double>(0, M / (M + m) * v, 0));
    return sys;
}

double relative_error_after_orbit(int steps) {
    const double a = 1.0;
    const double mu = constants::G * (1.0 + 3.00349e-6);
    const double period = 2.0 * 3.141592653589793 * std::sqrt(a * a * a / mu);
    System sys = two_body(a);
    NewtonianForce force;
    IAS15 ias15(force);
    const Vec3<double> dr0 = sys.pos[1] - sys.pos[0];
    for (int i = 0; i < steps; ++i) ias15.step(sys, period / steps);
    const Vec3<double> dr1 = sys.pos[1] - sys.pos[0];
    return norm(dr1 - dr0);
}
}  // namespace

// --- IAS15 energy conservation -------------------------------------------
TEST(ias15_energy_conserved) {
    System sys = two_body();
    NewtonianForce force;
    IAS15 ias15(force);
    const double e0 = total_energy(sys);
    const double period = 2.0 * 3.141592653589793 * std::sqrt(1.0 / constants::G);
    for (int i = 0; i < 200; ++i) ias15.step(sys, period / 200.0);
    CHECK_NEAR(total_energy(sys) / e0 - 1.0, 0.0, 1.0e-13);
}

// --- IAS15 reaches near machine precision with few steps ------------------
TEST(ias15_machine_precision_accuracy) {
    // A 15th-order integrator resolves one orbit to ~1e-14 with only 20 steps,
    // whereas lower-order methods (RK45 at same cost) would be ~1e-6.
    const double e = relative_error_after_orbit(20);
    CHECK(e < 1.0e-10);
}

CAS_TEST_MAIN()
