// Cassini — collision regularization via the Levi-Civita (2D) / Kustaanheimo-
// Stiefel (3D) transformation.
//
// The Kepler problem is singular at r=0; the LC transform maps the planar
// two-body problem to a 2D harmonic oscillator that is regular everywhere,
// including through close encounters. This module provides the transform and
// an exact regularized stepper (harmonic-oscillator map) for the planar case,
// and the 3D KS matrix as the direct generalization. See whitepaper §KS.
#pragma once

#include <cmath>

#include "cassini/core/vec3.hpp"

namespace cassini::physics {

// --- 2D Levi-Civita transform -------------------------------------------
struct LCState {
    double u1{0}, u2{0};   // position in u-space
    double w1{0}, w2{0};   // velocity in u-space (du/ds)
};

// r = (x,y) -> u, with x = u1^2 - u2^2, y = 2 u1 u2. Assumes x >= 0 branch.
inline LCState lc_forward(double x, double y, double vx, double vy) {
    const double r = std::sqrt(x * x + y * y);
    LCState s;
    if (x >= 0.0) {
        s.u1 = std::sqrt((r + x) / 2.0);
        s.u2 = (s.u1 > 0.0) ? y / (2.0 * s.u1) : 0.0;
    } else {
        s.u2 = std::sqrt((r - x) / 2.0);
        s.u1 = (s.u2 > 0.0) ? y / (2.0 * s.u2) : 0.0;
    }
    // u' = (1/2) L^T v, with L = [[u1, -u2],[u2, u1]].
    s.w1 = 0.5 * (s.u1 * vx + s.u2 * vy);
    s.w2 = 0.5 * (-s.u2 * vx + s.u1 * vy);
    return s;
}

inline void lc_inverse(const LCState& s, double& x, double& y, double& vx, double& vy) {
    x = s.u1 * s.u1 - s.u2 * s.u2;
    y = 2.0 * s.u1 * s.u2;
    const double r = s.u1 * s.u1 + s.u2 * s.u2;
    // v = 2 L u' / r.
    vx = 2.0 * (s.u1 * s.w1 - s.u2 * s.w2) / r;
    vy = 2.0 * (s.u2 * s.w1 + s.u1 * s.w2) / r;
}

// Advance the regularized harmonic oscillator by fictitious time ds (no time
// bookkeeping). Returns the new state.
inline LCState lc_advance(const LCState& s, double E, double ds) {
    LCState out;
    if (E < 0.0) {
        const double w = std::sqrt(-E / 2.0);
        const double c = std::cos(w * ds);
        const double sn = std::sin(w * ds);
        out.u1 = s.u1 * c + s.w1 * sn / w;
        out.u2 = s.u2 * c + s.w2 * sn / w;
        out.w1 = -s.u1 * w * sn + s.w1 * c;
        out.w2 = -s.u2 * w * sn + s.w2 * c;
    } else {
        const double w = std::sqrt(E / 2.0);
        const double c = std::cosh(w * ds);
        const double sn = std::sinh(w * ds);
        out.u1 = s.u1 * c + s.w1 * sn / w;
        out.u2 = s.u2 * c + s.w2 * sn / w;
        out.w1 = s.u1 * w * sn + s.w1 * c;
        out.w2 = s.u2 * w * sn + s.w2 * c;
    }
    return out;
}

// Exact regularized Kepler step: u'' = (E/2) u (harmonic oscillator). Advances
// fictitious time by ds and returns the physical time elapsed (dt = int r ds).
inline LCState lc_regularized_step(const LCState& s, double E, double ds, double& dt) {
    const LCState out = lc_advance(s, E, ds);
    const LCState mid = lc_advance(s, E, ds * 0.5);
    const double r0 = s.u1 * s.u1 + s.u2 * s.u2;
    const double rm = mid.u1 * mid.u1 + mid.u2 * mid.u2;
    const double r1 = out.u1 * out.u1 + out.u2 * out.u2;
    dt = ds / 6.0 * (r0 + 4.0 * rm + r1);
    return out;
}

// --- 3D Kustaanheimo-Stiefel matrix (position map u -> r) ----------------
// r = L(u) u with L as below. |u|^2 = |r|.
inline Vec3<double> ks_position(double u1, double u2, double u3, double u4) {
    return {u1 * u1 - u2 * u2 - u3 * u3 + u4 * u4,
            2.0 * (u1 * u2 - u3 * u4),
            2.0 * (u1 * u3 + u2 * u4)};
}

inline Vec3<double> ks_velocity(double u1, double u2, double u3, double u4,
                                double w1, double w2, double w3, double w4) {
    const double r = u1 * u1 + u2 * u2 + u3 * u3 + u4 * u4;
    // v = 2 L(u) w / r.
    return {2.0 * (u1 * w1 - u2 * w2 - u3 * w3 + u4 * w4) / r,
            2.0 * (u2 * w1 + u1 * w2 - u4 * w3 - u3 * w4) / r,
            2.0 * (u3 * w1 + u4 * w2 + u1 * w3 + u2 * w4) / r};
}

}  // namespace cassini::physics
