// Cassini — diurnal Yarkovsky effect for small rotating bodies (asteroids).
//
// Anisotropic thermal re-emission from a rotating body produces a small
// along-track acceleration that secularly drifts the semi-major axis. The
// leading-order diurnal term scales with the incident radiation-pressure
// acceleration times a thermal-lag efficiency. Documented in whitepaper §Yarkovsky.
#pragma once

#include <cmath>

#include "cassini/core/constants.hpp"
#include "cassini/core/vec3.hpp"

namespace cassini::physics {

// Yarkovsky acceleration [AU/day^2] on a body of radius R [AU] at heliocentric
// distance r with orbital velocity v, due to solar radiation.
// albedo = Bond albedo, rho = density [Msun/AU^3], spin = rotation rate [rad/day],
// efficiency = thermal-lag factor (dimensionless, 0..1).
inline Vec3<double> yarkovsky_acceleration(const Vec3<double>& r,
                                           const Vec3<double>& v,
                                           double R,
                                           double albedo,
                                           double rho,
                                           double spin,
                                           double efficiency,
                                           double L_sun = constants::L_SUN,
                                           double c = constants::C) {
    const double r2 = dot(r, r);
    const double rn = std::sqrt(r2);
    if (rn == 0.0 || R == 0.0 || rho == 0.0) return {};

    const double vmag = std::sqrt(dot(v, v));
    if (vmag == 0.0) return {};

    // Radiation-pressure scale (momentum flux times cross-section / mass).
    const double a_rp = 3.0 * (1.0 - albedo) * L_sun / (16.0 * 3.141592653589793 * rho * R * c * r2);
    const double a_mag = a_rp * efficiency * (spin >= 0.0 ? 1.0 : -1.0);
    return (v * (1.0 / vmag)) * a_mag;
}

}  // namespace cassini::physics
