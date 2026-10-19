// Cassini — brute-force gravity throughput benchmark (scaling baseline).
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include "cassini/core/rng.hpp"
#include "cassini/physics/forces.hpp"

using namespace cassini;
using namespace cassini::physics;

int main() {
    Pcg32 rng(12345);
    NewtonianForce force;

    for (std::size_t n : {100u, 1000u, 5000u}) {
        std::vector<Vec3<double>> pos(n);
        std::vector<Vec3<double>> vel(n);
        std::vector<double> mass(n);
        std::vector<Vec3<double>> acc(n);
        for (std::size_t i = 0; i < n; ++i) {
            pos[i] = {rng.uniform(-50, 50), rng.uniform(-50, 50), rng.uniform(-50, 50)};
            vel[i] = {0, 0, 0};
            mass[i] = (i == 0) ? 1.0 : rng.uniform(1.0e-9, 1.0e-6);
        }

        const auto t0 = std::chrono::steady_clock::now();
        force.accelerations(pos, vel, mass, acc);
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("N=%5zu  brute-force force eval: %8.3f ms\n", n, ms);
    }
    return 0;
}
