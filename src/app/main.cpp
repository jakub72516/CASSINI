// Cassini CLI — minimal entry point for milestone 1-2.
// Runs a two-body RK45 demo and reports energy/angular-momentum drift.
#include <cstdio>

#include "cassini/core/constants.hpp"
#include "cassini/physics/diagnostics.hpp"
#include "cassini/physics/forces.hpp"
#include "cassini/physics/rk45.hpp"
#include "cassini/physics/system.hpp"

int main() {
    using namespace cassini;
    using namespace cassini::physics;

    System sys;
    sys.add(1.0, Vec3<double>(0, 0, 0), Vec3<double>(0, 0, 0));
    const double a = 1.0;
    const double v = std::sqrt(constants::G / a);
    sys.add(1.0e-6, Vec3<double>(a, 0, 0), Vec3<double>(0, v, 0));

    NewtonianForce force;
    RK45 rk(force, 1.0e-13);

    const double period = 2.0 * 3.141592653589793 * std::sqrt(a * a * a / constants::G);
    const double e0 = total_energy(sys);

    std::printf("two-body demo: %zu bodies, period %.6f days\n", sys.size(), period);
    const int orbits = 100;
    for (int i = 0; i < orbits; ++i) {
        for (int k = 0; k < 2000; ++k) rk.step(sys, period / 2000.0);
        const double drift = std::fabs(total_energy(sys) / e0 - 1.0);
        std::printf("  orbit %3d: |dE/E| = %.3e\n", i + 1, drift);
    }
    return 0;
}
