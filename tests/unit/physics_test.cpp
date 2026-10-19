#include "framework.hpp"

#include "cassini/core/constants.hpp"
#include "cassini/core/ddouble.hpp"
#include "cassini/physics/diagnostics.hpp"
#include "cassini/physics/forces.hpp"
#include "cassini/physics/rk45.hpp"
#include "cassini/physics/system.hpp"

#include <cmath>

using namespace cassini;
using namespace cassini::physics;

namespace {
System circular_two_body(double a = 1.0, double mu = constants::G) {
    System sys;
    sys.add(1.0, Vec3<double>(0, 0, 0), Vec3<double>(0, 0, 0));   // central
    const double v = std::sqrt(mu / a);
    sys.add(1.0e-6, Vec3<double>(a, 0, 0), Vec3<double>(0, v, 0));
    return sys;
}

System mercury_system() {
    const double a = 0.38709893;   // AU
    const double e = 0.20563069;
    const double mu = constants::G;  // AU^3/day^2
    const double rp = a * (1.0 - e);
    const double vp = std::sqrt(mu * (1.0 + e) / (a * (1.0 - e)));
    System sys;
    sys.add(1.0, Vec3<double>(0, 0, 0), Vec3<double>(0, 0, 0));  // Sun
    sys.add(1.6601e-7, Vec3<double>(rp, 0, 0), Vec3<double>(0, vp, 0));  // Mercury
    return sys;
}
}  // namespace

// --- RK45 energy conservation --------------------------------------------
TEST(rk45_energy_conserved_circular) {
    System sys = circular_two_body();
    NewtonianForce force;
    RK45 rk(force, 1.0e-13);

    const double e0 = total_energy(sys);
    const double period = 2.0 * 3.141592653589793 * std::sqrt(1.0 / constants::G);
    const int steps = 2000;
    const double dt = period / steps;

    for (int i = 0; i < steps; ++i) rk.step(sys, dt);

    const double e1 = total_energy(sys);
    CHECK_NEAR(e1 / e0 - 1.0, 0.0, 1.0e-10);
}

// --- RK45 position accuracy ----------------------------------------------
TEST(rk45_returns_to_start_after_one_orbit) {
    // Zero-total-momentum two-body setup (COM at origin, at rest), so the
    // relative orbit is exactly periodic.
    const double M = 1.0;
    const double m = 1.0e-6;
    const double a = 1.0;
    const double mu = constants::G * (M + m);
    const double v = std::sqrt(mu / a);

    System sys;
    sys.add(M, Vec3<double>(-m / (M + m) * a, 0, 0), Vec3<double>(0, -m / (M + m) * v, 0));
    sys.add(m, Vec3<double>(M / (M + m) * a, 0, 0), Vec3<double>(0, M / (M + m) * v, 0));

    NewtonianForce force;
    RK45 rk(force, 1.0e-13);

    const Vec3<double> dr0 = sys.pos[1] - sys.pos[0];
    const double period = 2.0 * 3.141592653589793 * std::sqrt(a * a * a / mu);
    for (int i = 0; i < 1000; ++i) rk.step(sys, period / 1000.0);

    const Vec3<double> dr1 = sys.pos[1] - sys.pos[0];
    CHECK_NEAR(dr1.x, dr0.x, 1.0e-8);
    CHECK_NEAR(dr1.y, dr0.y, 1.0e-8);
}

// --- Mercury 1PN perihelion advance ---------------------------------------
TEST(mercury_1pn_perihelion_advance) {
    const double a = 0.38709893;
    const double e = 0.20563069;
    const double mu = constants::G;
    const double c = constants::C;

    // Analytic 1PN advance per orbit: 6*pi*mu / (c^2 * a * (1 - e^2))  [rad].
    const double expected = 6.0 * 3.141592653589793 * mu / (c * c * a * (1.0 - e * e));

    ForceOptions opts;
    opts.pn_order = 1;
    NewtonianForce force(opts);
    RK45 rk(force, 1.0e-14);

    System sys = mercury_system();
    const double period = 2.0 * 3.141592653589793 * std::sqrt(a * a * a / mu);

    // Initial Lenz-Runge vector (points to perihelion, along +x).
    const Vec3<double> e0 = lenz_runge(sys, 1);
    const double steps = 40000;
    const double dt = period / steps;
    for (int i = 0; i < static_cast<int>(steps); ++i) rk.step(sys, dt);

    const Vec3<double> e1 = lenz_runge(sys, 1);
    // In-plane rotation angle of the Lenz-Runge vector = apsidal advance.
    const double dphi = std::atan2(cross(e0, e1).z, dot(e0, e1));
    CHECK_NEAR(dphi, expected, 0.01 * expected);
}

// --- Angular momentum & Lenz-Runge invariants -----------------------------
TEST(angular_momentum_conserved) {
    System sys = circular_two_body();
    NewtonianForce force;
    RK45 rk(force, 1.0e-13);

    const Vec3<double> L0 = angular_momentum(sys);
    const double period = 2.0 * 3.141592653589793 * std::sqrt(1.0 / constants::G);
    for (int i = 0; i < 2000; ++i) rk.step(sys, period / 2000.0);
    const Vec3<double> L1 = angular_momentum(sys);

    CHECK_NEAR(L1.x, L0.x, 1.0e-11);
    CHECK_NEAR(L1.z, L0.z, 1.0e-11);
}

CAS_TEST_MAIN()
