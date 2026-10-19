// Cassini — adaptive step-size controller based on the local dynamical
// timescale. For symplectic integrators (Wisdom-Holman) the step must remain
// constant, so this is used to *choose* that constant from the shortest
// relevant orbital timescale; for IAS15/RK45 it feeds their internal adapters.
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

#include "cassini/core/constants.hpp"
#include "cassini/physics/system.hpp"

namespace cassini::physics {

struct StepController {
    double eta{0.02};  // fraction of the local dynamical time to use

    // Minimum local dynamical timescale tau = sqrt(r^3 / (G m_central)) over
    // all non-central bodies (rel. to the most massive body).
    double dynamical_time(const System& sys, double G = constants::G) const {
        std::size_t central = 0;
        for (std::size_t i = 1; i < sys.size(); ++i) {
            if (sys.mass[i] > sys.mass[central]) central = i;
        }
        double tmin = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < sys.size(); ++i) {
            if (i == central) continue;
            const Vec3<double> dr = sys.pos[i] - sys.pos[central];
            const double r = std::sqrt(dot(dr, dr));
            if (r == 0.0) continue;
            const double mu = G * (sys.mass[central] + sys.mass[i]);
            const double tau = std::sqrt(r * r * r / mu);
            if (tau < tmin) tmin = tau;
        }
        return tmin;
    }

    double suggest_step(const System& sys, double G = constants::G) const {
        const double tau = dynamical_time(sys, G);
        if (!std::isfinite(tau)) return 1.0;
        return eta * tau;
    }
};

}  // namespace cassini::physics
