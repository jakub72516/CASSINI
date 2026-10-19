// Cassini — dynamical invariants: kinetic/potential energy, total energy,
// angular momentum, and the Lenz-Runge vector. Long-term drift of these is the
// primary correctness metric (target < 1e-12 relative over 1e6 orbits).
//
// The dd_real variants accumulate in double-double to expose floating-point
// cancellation that 53-bit doubles would hide.
#pragma once

#include <vector>

#include "cassini/core/constants.hpp"
#include "cassini/core/ddouble.hpp"
#include "cassini/core/vec3.hpp"
#include "cassini/physics/system.hpp"

namespace cassini::physics {

inline double kinetic_energy(const System& sys) {
    double ke = 0.0;
    for (std::size_t i = 0; i < sys.size(); ++i) {
        ke += 0.5 * sys.mass[i] * dot(sys.vel[i], sys.vel[i]);
    }
    return ke;
}

inline double potential_energy(const System& sys, double G = constants::G) {
    double pe = 0.0;
    for (std::size_t i = 0; i < sys.size(); ++i) {
        for (std::size_t j = i + 1; j < sys.size(); ++j) {
            const Vec3<double> dr = sys.pos[j] - sys.pos[i];
            const double r = std::sqrt(dot(dr, dr));
            if (r == 0.0) continue;
            pe -= G * sys.mass[i] * sys.mass[j] / r;
        }
    }
    return pe;
}

inline double total_energy(const System& sys, double G = constants::G) {
    return kinetic_energy(sys) + potential_energy(sys, G);
}

// Total energy with double-double accumulation for the potential sum.
inline dd_real total_energy_dd(const System& sys, double G = constants::G) {
    dd_real ke{0.0};
    for (std::size_t i = 0; i < sys.size(); ++i) {
        ke += 0.5 * sys.mass[i] * dot(sys.vel[i], sys.vel[i]);
    }
    dd_real pe{0.0};
    for (std::size_t i = 0; i < sys.size(); ++i) {
        for (std::size_t j = i + 1; j < sys.size(); ++j) {
            const Vec3<double> dr = sys.pos[j] - sys.pos[i];
            const double r = std::sqrt(dot(dr, dr));
            if (r == 0.0) continue;
            pe -= (G * sys.mass[i] * sys.mass[j]) / r;
        }
    }
    return ke + pe;
}

inline Vec3<double> angular_momentum(const System& sys) {
    Vec3<double> L{};
    for (std::size_t i = 0; i < sys.size(); ++i) {
        L += sys.mass[i] * cross(sys.pos[i], sys.vel[i]);
    }
    return L;
}

// Lenz-Runge (Laplace-Runge-Lenz) vector of body i about the most massive body.
// e = (v x h)/mu - r_hat  (constant for a Keplerian two-body orbit).
inline Vec3<double> lenz_runge(const System& sys, std::size_t i, double G = constants::G) {
    std::size_t central = 0;
    for (std::size_t k = 1; k < sys.size(); ++k) {
        if (sys.mass[k] > sys.mass[central]) central = k;
    }
    const Vec3<double> r = sys.pos[i] - sys.pos[central];
    const Vec3<double> v = sys.vel[i] - sys.vel[central];
    const double mu = G * (sys.mass[central] + sys.mass[i]);
    const Vec3<double> h = cross(r, v);
    const double rn = std::sqrt(dot(r, r));
    return cross(v, h) * (1.0 / mu) - r * (1.0 / rn);
}

}  // namespace cassini::physics
