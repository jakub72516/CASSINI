// Cassini — solar system initial conditions from approximate J2000 mean
// elements (JPL "Keplerian elements for approximate positions", valid 1800-2050).
//
// Positions are barycentric in the J2000 ecliptic frame; the Sun is placed at
// the barycenter with recoil velocity so total momentum is zero. Masses are in
// solar masses, distances in AU, velocities in AU/day.
#pragma once

#include <cmath>
#include <vector>

#include "cassini/core/constants.hpp"
#include "cassini/math/orbital_elements.hpp"
#include "cassini/physics/system.hpp"

namespace cassini::physics {

struct PlanetElements {
    const char* name;
    double mass;    // [Msun]
    double a;       // [AU]
    double e;
    double inc;     // [deg]
    double Omega;   // [deg]
    double omega;   // [deg]
    double M;       // [deg]
};

// Approximate J2000 mean elements (JPL SSD table 1).
inline const std::vector<PlanetElements>& j2000_planets() {
    static const std::vector<PlanetElements> p = {
        {"Mercury", 1.66012e-7, 0.38709927, 0.20563593, 7.00497902, 48.33076593, 29.12469078, 174.79252722},
        {"Venus",   2.44784e-6, 0.72333566, 0.00677672, 3.39467605, 76.67984255, 54.92262463, 50.37563235},
        {"Earth",   3.00349e-6, 1.00000261, 0.01671123, -0.00001531, 0.0,        102.93768193, -2.47311027},
        {"Mars",    3.22715e-7, 1.52371034, 0.09339410, 1.84969142, 49.55953891, 19.38839055, 19.39019755},
        {"Jupiter", 9.54592e-4, 5.20288700, 0.04838624, 1.30439695, 100.47390909, 86.11945985, 19.66796066},
        {"Saturn",  2.85857e-4, 9.53667594, 0.05386179, 2.48599187, 113.66242448, -21.06484636, 47.34736592},
        {"Uranus",  4.36625e-5, 19.18916464, 0.04725744, 0.77263783, 74.01692503, 96.93735508, 142.28382820},
        {"Neptune", 5.15139e-5, 30.06992276, 0.00859048, 1.77004347, 131.78422574, 78.28573585, -100.08179194},
    };
    return p;
}

constexpr double DEG = 3.14159265358979323846 / 180.0;

// Build a full solar system (Sun + 8 planets) in the barycentric COM frame.
inline System build_solar_system() {
    System sys;
    const double msun = 1.0;
    sys.add(msun, Vec3<double>(0, 0, 0), Vec3<double>(0, 0, 0));

    // Heliocentric state per planet.
    std::vector<Vec3<double>> r(8), v(8);
    double mtot = msun;
    Vec3<double> rcom{}, vcom{};

    const auto& planets = j2000_planets();
    for (std::size_t i = 0; i < planets.size(); ++i) {
        const PlanetElements& pe = planets[i];
        math::OrbitalElements el;
        el.a = pe.a;
        el.e = pe.e;
        el.inc = pe.inc * DEG;
        el.Omega = pe.Omega * DEG;
        el.omega = pe.omega * DEG;
        el.M = pe.M * DEG;

        const math::StateVector sv = math::elements_to_state(el, constants::G * (msun + pe.mass));
        r[i] = sv.r;
        v[i] = sv.v;
        mtot += pe.mass;
        rcom += pe.mass * sv.r;
        vcom += pe.mass * sv.v;
    }

    // Sun at barycenter (COM frame).
    const Vec3<double> sun_r = rcom * (-1.0 / mtot);
    const Vec3<double> sun_v = vcom * (-1.0 / mtot);
    sys.pos[0] = sun_r;
    sys.vel[0] = sun_v;

    for (std::size_t i = 0; i < planets.size(); ++i) {
        sys.add(planets[i].mass, sun_r + r[i], sun_v + v[i]);
    }
    return sys;
}

}  // namespace cassini::physics
