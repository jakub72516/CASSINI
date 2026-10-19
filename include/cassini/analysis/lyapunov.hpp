// Cassini — Lyapunov exponent estimation for chaotic orbit divergence.
//
// Integrates a reference and a shadow trajectory started an infinitesimal
// distance apart, periodically renormalizing the separation, and measures the
// mean exponential divergence rate. Used to validate Mercury's Lyapunov time
// against the literature (~5 Myr, Laskar 1989 / Sussman & Wisdom).
#pragma once

#include <cmath>

#include "cassini/physics/integrator.hpp"
#include "cassini/physics/system.hpp"

namespace cassini::analysis {

inline System clone_system(const System& sys) {
    System c;
    c.mass = sys.mass;
    c.pos = sys.pos;
    c.vel = sys.vel;
    c.radius = sys.radius;
    c.j2 = sys.j2;
    c.j2_ref = sys.j2_ref;
    c.t = sys.t;
    return c;
}

inline double phase_distance(const System& a, const System& b) {
    double d2 = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const Vec3<double> dr = a.pos[i] - b.pos[i];
        const Vec3<double> dv = a.vel[i] - b.vel[i];
        d2 += dot(dr, dr) + dot(dv, dv) * 1.0e4;  // weight velocity
    }
    return std::sqrt(d2);
}

// Perturb `sys` by shifting each body's position by `eps` (in a fixed direction).
inline void perturb(System& sys, double eps) {
    for (std::size_t i = 0; i < sys.size(); ++i) {
        sys.pos[i].x += eps;
        sys.pos[i].y += eps * 0.7;
        sys.pos[i].z += eps * 0.3;
    }
}

// Estimate the largest Lyapunov exponent [1/day] by shadow-trajectory
// divergence with periodic renormalization.
inline double lyapunov_exponent(IIntegrator& integrator,
                                System reference,
                                double total_time,
                                double renormalize_every,
                                double d0 = 1.0e-10) {
    System shadow = clone_system(reference);
    perturb(shadow, d0);

    double sum = 0.0;
    double t = 0.0;
    const int steps_per_renorm = 200;
    const double dt = renormalize_every / steps_per_renorm;

    while (t < total_time) {
        for (int k = 0; k < steps_per_renorm; ++k) {
            integrator.step(reference, dt);
            integrator.step(shadow, dt);
        }
        t += renormalize_every;

        const double d = phase_distance(reference, shadow);
        if (d > 0.0) {
            sum += std::log(d / d0);
            // Renormalize the shadow back to distance d0 along the separation.
            const double scale = d0 / d;
            for (std::size_t i = 0; i < shadow.size(); ++i) {
                shadow.pos[i] = reference.pos[i] + (shadow.pos[i] - reference.pos[i]) * scale;
                shadow.vel[i] = reference.vel[i] + (shadow.vel[i] - reference.vel[i]) * scale;
            }
        }
    }
    return sum / t;
}

}  // namespace cassini::analysis
