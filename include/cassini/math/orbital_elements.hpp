// Cassini — Keplerian orbital elements -> state vector (r, v) conversion.
// Used to seed the solar system from approximate J2000 mean elements and to
// generate analytic reference orbits for tests.
#pragma once

#include <cmath>

#include "cassini/core/vec3.hpp"

namespace cassini::math {

struct OrbitalElements {
    double a{1.0};          // semi-major axis [AU]
    double e{0.0};          // eccentricity
    double inc{0.0};        // inclination [rad]
    double Omega{0.0};      // longitude of ascending node [rad]
    double omega{0.0};      // argument of perihelion [rad]
    double M{0.0};          // mean anomaly [rad]
};

struct StateVector {
    Vec3<double> r;
    Vec3<double> v;
};

// Convert (a, e, i, Omega, omega, M) to heliocentric position/velocity.
inline StateVector elements_to_state(const OrbitalElements& el, double mu) {
    // Solve Kepler's equation M = E - e sin E (Newton iteration).
    double E = el.M;
    for (int k = 0; k < 50; ++k) {
        const double dE = (E - el.e * std::sin(E) - el.M) / (1.0 - el.e * std::cos(E));
        E -= dE;
        if (std::fabs(dE) < 1.0e-14) break;
    }

    const double cosE = std::cos(E);
    const double sinE = std::sin(E);
    const double sq = std::sqrt(1.0 - el.e * el.e);

    // True anomaly.
    const double nu = std::atan2(sq * sinE, cosE - el.e);
    const double r = el.a * (1.0 - el.e * cosE);

    // Perifocal frame (x toward perihelion).
    const double xp = r * std::cos(nu);
    const double yp = r * std::sin(nu);

    const double p = el.a * (1.0 - el.e * el.e);
    const double h = std::sqrt(mu * p);  // specific angular momentum
    const double vxp = -mu / h * std::sin(nu);
    const double vyp = mu / h * (el.e + std::cos(nu));

    // Rotation perifocal -> reference frame via Rz(Omega) Rx(i) Rz(omega).
    const double cO = std::cos(el.Omega), sO = std::sin(el.Omega);
    const double ci = std::cos(el.inc), si = std::sin(el.inc);
    const double cw = std::cos(el.omega), sw = std::sin(el.omega);

    // Direction cosines of the perifocal axes in the reference frame.
    const double Px = cO * cw - sO * sw * ci;
    const double Py = sO * cw + cO * sw * ci;
    const double Pz = sw * si;
    const double Qx = -cO * sw - sO * cw * ci;
    const double Qy = -sO * sw + cO * cw * ci;
    const double Qz = cw * si;

    StateVector out;
    out.r = {xp * Px + yp * Qx, xp * Py + yp * Qy, xp * Pz + yp * Qz};
    out.v = {vxp * Px + vyp * Qx, vxp * Py + vyp * Qy, vxp * Pz + vyp * Qz};
    return out;
}

}  // namespace cassini::math
