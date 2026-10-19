// Cassini — JPL regression harness: compares the simulator's propagated state
// against DE440/Horizons over the next century. Only runs when a DE440 .bsp
// kernel is available (CASSINI_DE440_PATH or ./data/de440.bsp); otherwise all
// cases are skipped.
#include "framework.hpp"

#include "cassini/core/constants.hpp"
#include "cassini/data/de440.hpp"
#include "cassini/physics/forces.hpp"
#include "cassini/physics/rk45.hpp"
#include "cassini/physics/solar_system.hpp"
#include "cassini/physics/system.hpp"
#include "cassini/physics/wisdom_holman.hpp"

#include <cmath>
#include <cstdlib>
#include <string>

using namespace cassini;
using namespace cassini::data;
using namespace cassini::physics;

namespace {
std::string de440_path() {
    if (const char* p = std::getenv("CASSINI_DE440_PATH")) return p;
    return "data/de440.bsp";
}
}  // namespace

TEST(jpl_inner_planet_position_error_100yr) {
    De440 de440;
    if (!de440.load(de440_path())) {
        std::printf("    SKIP: de440.bsp not found (set CASSINI_DE440_PATH)\n");
        return;
    }

    // Seed the simulator from DE440 at J2000 (ET = 0) for the 8 planets.
    System sys;
    const int bodies[8] = {static_cast<int>(Body::Mercury),  static_cast<int>(Body::Venus),
                           static_cast<int>(Body::Earth),    static_cast<int>(Body::Mars),
                           static_cast<int>(Body::Jupiter),  static_cast<int>(Body::Saturn),
                           static_cast<int>(Body::Uranus),   static_cast<int>(Body::Neptune)};
    const double masses[8] = {1.66012e-7, 2.44784e-6, 3.00349e-6, 3.22715e-7,
                              9.54592e-4, 2.85857e-4, 4.36625e-5, 5.15139e-5};

    // Sun at origin (barycentric correction is negligible for this test).
    sys.add(1.0, Vec3<double>(0, 0, 0), Vec3<double>(0, 0, 0));
    for (int i = 0; i < 8; ++i) {
        Vec3<double> r, v;
        if (!de440.heliocentric_state(bodies[i], 0.0, r, v)) {
            std::printf("    SKIP: missing DE440 segment for body %d\n", bodies[i]);
            return;
        }
        sys.add(masses[i], r, v);
    }

    WisdomHolman wh;
    const double dt = 16.0;      // 16-day symplectic step
    const double years = 100.0;
    const int steps = static_cast<int>(365.25 * years / dt);

    for (int i = 0; i < steps; ++i) wh.step(sys, dt);

    // Compare final heliocentric positions for inner planets (index 1..4).
    const double et_final = 365.25 * years * constants::DAY_s;
    double max_err_km = 0.0;
    for (int i = 0; i < 4; ++i) {
        Vec3<double> r_ref, v_ref;
        de440.heliocentric_state(bodies[i], et_final, r_ref, v_ref);
        const Vec3<double> dr = sys.pos[i + 1] - sys.pos[0] - r_ref;
        const double err_km = norm(dr) * data::KM_PER_AU;
        if (err_km > max_err_km) max_err_km = err_km;
    }
    std::printf("    max inner-planet position error after 100 yr: %.6g km\n", max_err_km);
    CHECK(max_err_km < 1.0);  // < 1 km
}

CAS_TEST_MAIN()
