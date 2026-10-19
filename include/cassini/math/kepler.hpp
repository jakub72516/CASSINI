// Cassini — universal-variable Kepler propagation (Vallado's Algorithm 8).
//
// Given an initial state (r0, v0) and a propagation time dt under a central
// force mu, returns the new state (r, v). Handles elliptic, parabolic and
// hyperbolic orbits through the Stumpff functions without branch-specific
// anomaly logic. This is the "drift" map for the Wisdom-Holman integrator.
#pragma once

#include <cmath>

#include "cassini/core/constants.hpp"
#include "cassini/core/vec3.hpp"
#include "cassini/math/stumpff.hpp"

namespace cassini::math {

struct KeplerSolution {
    Vec3<double> r;
    Vec3<double> v;
    bool converged{false};
    int iterations{0};
};

inline KeplerSolution kepler_propagate(const Vec3<double>& r0,
                                       const Vec3<double>& v0,
                                       double dt,
                                       double mu) {
    KeplerSolution out;
    out.r = r0;
    out.v = v0;

    const double r0n = std::sqrt(dot(r0, r0));
    const double v0n = std::sqrt(dot(v0, v0));
    const double r0v0 = dot(r0, v0);
    const double smu = std::sqrt(mu);
    const double alpha = 2.0 / r0n - v0n * v0n / mu;  // = 1/a (specific energy form)

    // Initial guess for the universal anomaly chi.
    double chi;
    if (alpha > 0.0) {
        chi = smu * dt * alpha;
    } else if (alpha < 0.0) {
        chi = smu * dt * (-alpha);
    } else {
        chi = smu * dt / r0n;
    }

    constexpr double tol = 1.0e-12;
    constexpr int max_iter = 100;

    double psi = 0.0;
    double c2 = 0.0;
    double c3 = 0.0;

    for (int i = 0; i < max_iter; ++i) {
        psi = chi * chi * alpha;
        const Stumpff sf = stumpff(psi);
        c2 = sf.c2;
        c3 = sf.c3;

        // Universal Kepler equation residual and derivative.
        const double F = (r0v0 / smu) * chi * chi * c2
                         + (1.0 - alpha * r0n) * chi * chi * chi * c3
                         + r0n * chi - smu * dt;
        const double dF = (r0v0 / smu) * chi * (1.0 - psi * c3)
                          + (1.0 - alpha * r0n) * chi * chi * c2 + r0n;

        const double dchi = F / dF;
        chi -= dchi;
        if (std::abs(dchi) < tol) {
            out.converged = true;
            out.iterations = i + 1;
            break;
        }
        out.iterations = i + 1;
    }

    // Lagrange coefficients.
    const double f = 1.0 - (chi * chi / r0n) * c2;
    const double g = dt - (chi * chi * chi / smu) * c3;
    out.r = f * r0 + g * v0;

    const double rn = std::sqrt(dot(out.r, out.r));
    const double fdot = (smu / (rn * r0n)) * chi * (psi * c3 - 1.0);
    const double gdot = 1.0 - (chi * chi / rn) * c2;
    out.v = fdot * r0 + gdot * v0;

    return out;
}

}  // namespace cassini::math
