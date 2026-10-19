// Cassini — post-Newtonian (PN) acceleration for a test particle in the
// Schwarzschild field of a central mass, derived from the harmonic-gauge
// Lagrangian through 2PN (Damour-Deruelle, test-particle limit).
//
// With r = separation vector, r = |r|, v = relative velocity, mu = G*M:
//
//   1PN:  a = (mu/(c^2 r^3)) [ (4 mu/r - v^2) r + 4 (r.v) v ]
//   2PN:  a = (mu^2/(c^4 r^4)) [ -(1/2)(5 v^2 + 7 mu/r) r + 5 (r.v) v ]
//
// The 1PN term reproduces Mercury's perihelion advance of 43.03 arcsec/century;
// the 2PN term adds ~5e-6 arcsec/century (seven orders of magnitude smaller).
// Units: [AU], [AU/day], G in AU^3/(Msun day^2), c in AU/day.
#pragma once

#include "cassini/core/vec3.hpp"

namespace cassini::physics {

inline Vec3<double> pn_acceleration_1pn(const Vec3<double>& r,
                                        const Vec3<double>& v,
                                        double mu,
                                        double c) {
    const double r2 = dot(r, r);
    const double rn = std::sqrt(r2);
    const double v2 = dot(v, v);
    const double rv = dot(r, v);
    const double k = mu / (c * c * r2 * rn);
    return k * ((4.0 * mu / rn - v2) * r + 4.0 * rv * v);
}

inline Vec3<double> pn_acceleration_2pn(const Vec3<double>& r,
                                        const Vec3<double>& v,
                                        double mu,
                                        double c) {
    const double r2 = dot(r, r);
    const double rn = std::sqrt(r2);
    const double v2 = dot(v, v);
    const double rv = dot(r, v);
    const double k = mu * mu / (c * c * c * c * r2 * r2);
    return k * (-0.5 * (5.0 * v2 + 7.0 * mu / rn) * r + 5.0 * rv * v);
}

// Total PN acceleration through the requested order (1 or 2).
inline Vec3<double> pn_acceleration(const Vec3<double>& r,
                                    const Vec3<double>& v,
                                    double mu,
                                    double c,
                                    int order) {
    Vec3<double> a = pn_acceleration_1pn(r, v, mu, c);
    if (order >= 2) {
        a += pn_acceleration_2pn(r, v, mu, c);
    }
    return a;
}

}  // namespace cassini::physics
