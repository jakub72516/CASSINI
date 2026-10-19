// Cassini — Stumpff functions c0..c3 used by the universal-variable Kepler
// solver. Both the elliptic (z > 0) and hyperbolic (z < 0) branches are
// provided, with a power-series branch for |z| small to avoid catastrophic
// cancellation in 1-cos and x-sin forms.
#pragma once

#include <cmath>

namespace cassini::math {

struct Stumpff {
    double c0{0};
    double c1{0};
    double c2{0};
    double c3{0};
};

inline Stumpff stumpff(double z) {
    Stumpff s;
    if (std::abs(z) < 0.1) {
        // Power series in z (valid for both signs).
        const double z2 = z * z;
        const double z3 = z2 * z;
        const double z4 = z3 * z;
        const double z5 = z4 * z;
        s.c0 = 1.0 - z / 2.0 + z2 / 24.0 - z3 / 720.0 + z4 / 40320.0 - z5 / 3628800.0;
        s.c1 = 1.0 - z / 6.0 + z2 / 120.0 - z3 / 5040.0 + z4 / 362880.0 - z5 / 39916800.0;
        s.c2 = 0.5 - z / 24.0 + z2 / 720.0 - z3 / 40320.0 + z4 / 3628800.0 - z5 / 479001600.0;
        s.c3 = 1.0 / 6.0 - z / 120.0 + z2 / 5040.0 - z3 / 362880.0 + z4 / 39916800.0
               - z5 / 6227020800.0;
    } else if (z > 0.0) {
        const double x = std::sqrt(z);
        s.c0 = std::cos(x);
        s.c1 = std::sin(x) / x;
        s.c2 = (1.0 - s.c0) / z;
        s.c3 = (x - std::sin(x)) / (z * x);
    } else {
        const double x = std::sqrt(-z);
        s.c0 = std::cosh(x);
        s.c1 = std::sinh(x) / x;
        s.c2 = (s.c0 - 1.0) / (-z);
        s.c3 = (std::sinh(x) - x) / (x * x * x);
    }
    return s;
}

}  // namespace cassini::math
