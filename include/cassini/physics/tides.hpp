// Cassini — equilibrium tidal dissipation (constant time-lag model, leading
// order). The tidal bulge raised on the primary by the secondary lags the
// line of centers by an angle ~ (n - Omega) tau, producing a torque that
// circularizes and shrinks the orbit. See whitepaper §Tides.
#pragma once

#include <cmath>

#include "cassini/core/vec3.hpp"

namespace cassini::physics {

// Tidal acceleration on the secondary (mass m2) due to the tide raised on the
// primary (mass m1, radius R1, love number k2, lag tau, spin rate Omega).
// r, v: relative separation/velocity (primary -> secondary), G in AU^3/Msun/day^2.
inline Vec3<double> tidal_acceleration(const Vec3<double>& r,
                                       const Vec3<double>& v,
                                       double m1,
                                       double m2,
                                       double R1,
                                       double k2,
                                       double tau,
                                       double omega_spin,
                                       double G) {
    const double r2 = dot(r, r);
    const double rn = std::sqrt(r2);
    if (rn == 0.0 || m2 == 0.0) return {};

    const Vec3<double> rhat = r * (1.0 / rn);
    const Vec3<double> h = cross(r, v);             // specific angular momentum
    const double hmag = std::sqrt(dot(h, h));
    const double n = (hmag > 0.0) ? hmag / r2 : 0.0;  // mean motion

    // Tangential unit vector (in the instantaneous orbital plane, along v).
    Vec3<double> that;
    if (hmag > 0.0) {
        that = cross(h, r) * (1.0 / (hmag * rn));
    }

    const double ratio = R1 / rn;
    const double factor = 3.0 * k2 * (m1 / m2) * ratio * ratio * ratio * ratio * ratio *
                          (G * m2 / r2);

    // Conservative radial term + dissipative tangential term (lagged).
    const Vec3<double> a_radial = -factor * rhat;
    const Vec3<double> a_tangential = -factor * 2.0 * (n - omega_spin) * tau * that;
    return a_radial + a_tangential;
}

}  // namespace cassini::physics
