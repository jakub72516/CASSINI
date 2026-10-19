#include "framework.hpp"

#include "cassini/core/rng.hpp"
#include "cassini/gravity/cuda_force.hpp"
#include "cassini/physics/forces.hpp"

#include <cmath>
#include <vector>

using namespace cassini;
using namespace cassini::physics;
using namespace cassini::gravity;

TEST(cuda_force_matches_brute_force) {
    const int n = 4096;
    Pcg32 r1(42), r2(7);
    std::vector<Vec3<double>> pos(n), vel(n, Vec3<double>{});
    std::vector<double> mass(n);
    for (int i = 0; i < n; ++i) {
        pos[i] = {r1.uniform(-50, 50), r1.uniform(-50, 50), r1.uniform(-50, 50)};
    }
    mass[0] = 1.0;
    for (int i = 1; i < n; ++i) mass[i] = r2.uniform(1.0e-4, 1.0e-2);

    std::vector<Vec3<double>> a_bf(n), a_cuda(n);
    NewtonianForce bf;
    bf.accelerations(pos, vel, mass, a_bf);
    CudaForce cuda;
    cuda.accelerations(pos, vel, mass, a_cuda);

    double max_rel = 0.0;
    for (int i = 0; i < n; ++i) {
        const double mag = norm(a_bf[i]);
        if (mag < 1e-12) continue;
        const double rel = norm(a_cuda[i] - a_bf[i]) / mag;
        if (rel > max_rel) max_rel = rel;
    }
    CHECK(max_rel < 1.0e-10);
}

CAS_TEST_MAIN()
