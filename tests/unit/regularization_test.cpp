#include "framework.hpp"

#include "cassini/core/constants.hpp"
#include "cassini/math/kepler.hpp"
#include "cassini/physics/regularization.hpp"
#include "cassini/physics/step_controller.hpp"
#include "cassini/physics/tides.hpp"
#include "cassini/physics/yarkovsky.hpp"

#include <cmath>

using namespace cassini;
using namespace cassini::physics;

// --- Levi-Civita transform round-trip ------------------------------------
TEST(levi_civita_roundtrip) {
    const double x = 1.0, y = -2.5, vx = 0.3, vy = -0.7;
    const LCState s = lc_forward(x, y, vx, vy);
    double x2, y2, vx2, vy2;
    lc_inverse(s, x2, y2, vx2, vy2);
    CHECK_NEAR(x2, x, 1.0e-12);
    CHECK_NEAR(y2, y, 1.0e-12);
    CHECK_NEAR(vx2, vx, 1.0e-12);
    CHECK_NEAR(vy2, vy, 1.0e-12);
}

// --- Regularized Kepler integration matches exact solver -----------------
TEST(levi_civita_regularized_orbit) {
    const double mu = constants::G;
    const double a = 1.0, e = 0.9;
    const double rp = a * (1.0 - e);
    const double vp = std::sqrt(mu * (1.0 + e) / (a * (1.0 - e)));
    // Planar orbit in xy; perihelion along +x.
    const Vec3<double> r0(rp, 0.0, 0.0);
    const Vec3<double> v0(0.0, vp, 0.0);
    const double E = 0.5 * dot(v0, v0) - mu / rp;  // negative (bound)

    LCState s = lc_forward(r0.x, r0.y, v0.x, v0.y);
    const double period = 2.0 * 3.141592653589793 * std::sqrt(a * a * a / mu);
    const double ds = 0.05;
    double t = 0.0;
    while (t < period) {
        double dt;
        s = lc_regularized_step(s, E, ds, dt);
        t += dt;
    }

    double x, y, vx, vy;
    lc_inverse(s, x, y, vx, vy);
    const math::KeplerSolution ref = math::kepler_propagate(r0, v0, t, mu);
    CHECK(std::fabs(x - ref.r.x) < 1.0e-3);
    CHECK(std::fabs(y - ref.r.y) < 1.0e-3);
}

// --- 3D KS position map --------------------------------------------------
TEST(ks_position_map) {
    // u = (1, 0, 0, 0) -> r = (1, 0, 0), |u|^2 = 1.
    const Vec3<double> r = ks_position(1.0, 0.0, 0.0, 0.0);
    CHECK_NEAR(r.x, 1.0, 1.0e-15);
    CHECK_NEAR(r.y, 0.0, 1.0e-15);
    CHECK_NEAR(r.z, 0.0, 1.0e-15);
}

// --- Step controller -----------------------------------------------------
TEST(step_controller_dynamical_time) {
    // Earth-like orbit: tau = sqrt(a^3/mu) ~ 58 days.
    System sys;
    sys.add(1.0, Vec3<double>(0, 0, 0), Vec3<double>(0, 0, 0));
    sys.add(3.0e-6, Vec3<double>(1.0, 0, 0), Vec3<double>(0, std::sqrt(constants::G), 0));
    StepController sc;
    const double tau = sc.dynamical_time(sys);
    CHECK_NEAR(tau, std::sqrt(1.0 / constants::G), 0.01 * std::sqrt(1.0 / constants::G));
    CHECK(sc.suggest_step(sys) < tau);
}

// --- Tidal dissipation is small and correct sign -------------------------
TEST(tidal_acceleration_dissipates) {
    // Sun (primary, fast spin) / Earth (secondary) at 1 AU.
    const Vec3<double> r(1.0, 0, 0);
    const Vec3<double> v(0.0, std::sqrt(constants::G), 0.0);
    const double R_sun_au = constants::meters_to_au(6.957e8);
    const double a_t = norm(tidal_acceleration(r, v, 1.0, 3.0e-6, R_sun_au, 0.03, 100.0, 0.5, constants::G));
    CHECK(a_t > 0.0);
    CHECK(a_t < 1.0e-8);  // tidal perturbation is tiny
}

// --- Yarkovsky acceleration is small and along velocity ------------------
TEST(yarkovsky_acceleration_small) {
    const Vec3<double> r(2.0, 0, 0);
    const Vec3<double> v(0.0, std::sqrt(constants::G / 2.0), 0.0);
    const double R = constants::meters_to_au(500.0);  // 500 m asteroid
    // rho = 2500 kg/m^3 in natural units (Msun/AU^3).
    const double rho = 2500.0 * constants::AU_m * constants::AU_m * constants::AU_m /
                       constants::MSUN_kg;
    const Vec3<double> a =
        yarkovsky_acceleration(r, v, R, 0.1, rho, 1.0, 0.1);
    CHECK(norm(a) < 1.0e-12);
    // Aligned with velocity direction (prograde spin).
    CHECK(dot(a, v) > 0.0);
}

CAS_TEST_MAIN()
