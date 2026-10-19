// Cassini — Wisdom-Holman symplectic integrator in democratic heliocentric
// (Jacobi) coordinates (WHFast formulation, Rein & Tamayo 2015).
//
// The N-body Hamiltonian is split as H = H_Kep + H_int:
//
//   H_Kep = sum_i [ P_i^2/(2 mu_i) - G m0 mi / Q_i ]     (N two-body problems)
//   H_int = sum_{i<j} P_i.P_j/m0 - sum_{i<j} G mi mj / Q_ij
//
// where Q_i are heliocentric positions and P_i barycentric momenta (i >= 1,
// Sun = 0). H_Kep is solved exactly per planet with the universal Kepler
// solver; H_int is applied as a kick. The result is a second-order symplectic
// map with no secular energy drift for near-Keplerian orbits.
#pragma once

#include <vector>

#include "cassini/core/constants.hpp"
#include "cassini/math/kepler.hpp"
#include "cassini/physics/integrator.hpp"
#include "cassini/physics/system.hpp"

namespace cassini::physics {

class WisdomHolman : public IIntegrator {
public:
    explicit WisdomHolman(double G = constants::G) : G_(G) {}

    void step(System& sys, double dt) override;

private:
    double G_;
};

inline void WisdomHolman::step(System& sys, double dt) {
    const std::size_t n = sys.size();
    if (n < 2) return;

    const std::size_t nplanets = n - 1;
    const double m0 = sys.mass[0];
    const double mtotal = sys.total_mass();

    // Convert inertial -> democratic heliocentric (assumes COM frame).
    std::vector<Vec3<double>> Q(nplanets);  // heliocentric positions
    std::vector<Vec3<double>> P(nplanets);  // barycentric momenta
    for (std::size_t i = 0; i < nplanets; ++i) {
        Q[i] = sys.pos[i + 1] - sys.pos[0];
        P[i] = sys.mass[i + 1] * sys.vel[i + 1];
    }

    const double hh = 0.5 * dt;

    // ---- Half kick 1 ----
    Vec3<double> Psum{};
    for (std::size_t i = 0; i < nplanets; ++i) Psum += P[i];
    for (std::size_t i = 0; i < nplanets; ++i) {
        // Position kick: dQ_i/dt = (Psum - P_i) / m0.
        Q[i] += hh * ((Psum - P[i]) * (1.0 / m0));
        // Velocity kick: direct planet-planet gravity.
        for (std::size_t j = 0; j < nplanets; ++j) {
            if (i == j) continue;
            const Vec3<double> dq = Q[j] - Q[i];
            const double r2 = dot(dq, dq);
            const double r = std::sqrt(r2);
            if (r == 0.0) continue;
            P[i] += hh * (G_ * sys.mass[i + 1] * sys.mass[j + 1] / (r2 * r)) * dq;
        }
    }

    // ---- Drift: exact Kepler propagation of each Sun-planet pair ----
    for (std::size_t i = 0; i < nplanets; ++i) {
        const double mi = sys.mass[i + 1];
        const double mu = mi * m0 / (mi + m0);          // reduced mass
        const Vec3<double> vrel = P[i] * (1.0 / mu);    // relative velocity
        const math::KeplerSolution sol = math::kepler_propagate(Q[i], vrel, dt, G_ * (m0 + mi));
        Q[i] = sol.r;
        P[i] = mu * sol.v;
    }

    // ---- Half kick 2 (same as kick 1) ----
    Psum = Vec3<double>{};
    for (std::size_t i = 0; i < nplanets; ++i) Psum += P[i];
    for (std::size_t i = 0; i < nplanets; ++i) {
        Q[i] += hh * ((Psum - P[i]) * (1.0 / m0));
        for (std::size_t j = 0; j < nplanets; ++j) {
            if (i == j) continue;
            const Vec3<double> dq = Q[j] - Q[i];
            const double r2 = dot(dq, dq);
            const double r = std::sqrt(r2);
            if (r == 0.0) continue;
            P[i] += hh * (G_ * sys.mass[i + 1] * sys.mass[j + 1] / (r2 * r)) * dq;
        }
    }

    // Convert back to inertial coordinates (COM frame).
    Vec3<double> mQ{};
    for (std::size_t i = 0; i < nplanets; ++i) mQ += sys.mass[i + 1] * Q[i];
    const Vec3<double> X0 = mQ * (-1.0 / mtotal);  // Sun position from COM constraint

    sys.pos[0] = X0;
    for (std::size_t i = 0; i < nplanets; ++i) {
        sys.pos[i + 1] = X0 + Q[i];
        sys.vel[i + 1] = P[i] * (1.0 / sys.mass[i + 1]);
    }
    // Sun velocity from momentum conservation: m0 v0 = -sum P_i.
    Vec3<double> psum{};
    for (std::size_t i = 0; i < nplanets; ++i) psum += P[i];
    sys.vel[0] = psum * (-1.0 / m0);

    sys.t += dt;
}

}  // namespace cassini::physics
