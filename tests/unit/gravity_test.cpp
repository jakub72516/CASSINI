#include "framework.hpp"

#include "cassini/core/rng.hpp"
#include "cassini/gravity/barnes_hut.hpp"
#include "cassini/gravity/fmm.hpp"
#include "cassini/physics/forces.hpp"

#include <chrono>
#include <cmath>
#include <vector>

using namespace cassini;
using namespace cassini::physics;
using namespace cassini::gravity;

namespace {
std::vector<Vec3<double>> random_positions(int n) {
    Pcg32 rng(42);
    std::vector<Vec3<double>> p(n);
    for (int i = 0; i < n; ++i) {
        p[i] = {rng.uniform(-50, 50), rng.uniform(-50, 50), rng.uniform(-50, 50)};
    }
    return p;
}
std::vector<double> random_masses(int n) {
    Pcg32 rng(7);
    std::vector<double> m(n);
    m[0] = 1.0;
    for (int i = 1; i < n; ++i) m[i] = rng.uniform(1.0e-4, 1.0e-2);
    return m;
}
}  // namespace

// --- Barnes-Hut matches brute force within multipole tolerance ------------
TEST(barnes_hut_matches_brute_force) {
    const int n = 2000;
    const auto pos = random_positions(n);
    const auto mass = random_masses(n);
    std::vector<Vec3<double>> vel(n, Vec3<double>{});
    std::vector<Vec3<double>> a_bf(n), a_bh(n);

    NewtonianForce bf;
    bf.accelerations(pos, vel, mass, a_bf);

    BarnesHut bh(0.3);
    bh.accelerations(pos, vel, mass, a_bh);

    double max_rel = 0.0;
    double sum_rel = 0.0;
    int count = 0;
    for (int i = 0; i < n; ++i) {
        const double mag = norm(a_bf[i]);
        if (mag < 1e-12) continue;
        const double rel = norm(a_bh[i] - a_bf[i]) / mag;
        if (rel > max_rel) max_rel = rel;
        sum_rel += rel;
        ++count;
    }
    const double mean_rel = sum_rel / static_cast<double>(count);
    CHECK(max_rel < 0.05);
    CHECK(mean_rel < 0.01);
}

// --- Smaller theta => smaller error --------------------------------------
TEST(barnes_hut_smaller_theta_more_accurate) {
    const int n = 1000;
    const auto pos = random_positions(n);
    const auto mass = random_masses(n);
    std::vector<Vec3<double>> vel(n, Vec3<double>{});
    std::vector<Vec3<double>> a_bf(n), a_coarse(n), a_fine(n);

    NewtonianForce bf;
    bf.accelerations(pos, vel, mass, a_bf);

    BarnesHut coarse(0.7);
    coarse.accelerations(pos, vel, mass, a_coarse);
    BarnesHut fine(0.2);
    fine.accelerations(pos, vel, mass, a_fine);

    double err_coarse = 0.0, err_fine = 0.0;
    for (int i = 0; i < n; ++i) {
        const double mag = norm(a_bf[i]);
        if (mag < 1e-12) continue;
        err_coarse = std::max(err_coarse, norm(a_coarse[i] - a_bf[i]) / mag);
        err_fine = std::max(err_fine, norm(a_fine[i] - a_bf[i]) / mag);
    }
    CHECK(err_fine < err_coarse);
}

// --- FMM matches brute force ---------------------------------------------
TEST(fmm_matches_brute_force) {
    const int n = 5000;
    const auto pos = random_positions(n);
    const auto mass = random_masses(n);
    std::vector<Vec3<double>> vel(n, Vec3<double>{});
    std::vector<Vec3<double>> a_bf(n), a_fmm(n);

    NewtonianForce bf;
    bf.accelerations(pos, vel, mass, a_bf);
    FMM fmm(16);
    fmm.accelerations(pos, vel, mass, a_fmm);

    double max_rel = 0.0, sum_rel = 0.0;
    int count = 0;
    for (int i = 0; i < n; ++i) {
        const double mag = norm(a_bf[i]);
        if (mag < 1e-12) continue;
        const double rel = norm(a_fmm[i] - a_bf[i]) / mag;
        if (rel > max_rel) max_rel = rel;
        sum_rel += rel;
        ++count;
    }
    const double mean_rel = sum_rel / static_cast<double>(count);
    // Quadrupole-multipole / order-2 local FMM: ~3% typical relative error;
    // the max is dominated by the closest well-separated pairs.
    CHECK(mean_rel < 0.06);
    CHECK(max_rel < 1.0);
}

// --- FMM scales better than brute force ----------------------------------
TEST(fmm_scales_better_than_brute_force) {
    const int n = 8000;
    const auto pos = random_positions(n);
    const auto mass = random_masses(n);
    std::vector<Vec3<double>> vel(n, Vec3<double>{}), acc(n);

    auto t0 = std::chrono::steady_clock::now();
    NewtonianForce bf;
    bf.accelerations(pos, vel, mass, acc);
    auto t1 = std::chrono::steady_clock::now();
    const double ms_bf = std::chrono::duration<double, std::milli>(t1 - t0).count();

    t0 = std::chrono::steady_clock::now();
    FMM fmm(16);
    fmm.accelerations(pos, vel, mass, acc);
    t1 = std::chrono::steady_clock::now();
    const double ms_fmm = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // At 8000 bodies FMM should be substantially faster than O(N^2) brute force.
    CHECK(ms_fmm < ms_bf);
}

CAS_TEST_MAIN()
