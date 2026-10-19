// Cassini — Dormand-Prince RK5(4)7M adaptive integrator (cross-validation).
//
// Propagates the first-order system  d/dt (x, v) = (v, a(x, v))  with an
// embedded 4th-order error estimator and PI step-size control.
#pragma once

#include <algorithm>
#include <vector>

#include "cassini/physics/force_model.hpp"
#include "cassini/physics/integrator.hpp"
#include "cassini/physics/system.hpp"

namespace cassini::physics {

class RK45 : public IIntegrator {
public:
    explicit RK45(const ForceModel& force, double tol = 1.0e-12)
        : force_(force), tol_(tol) {}

    void step(System& sys, double dt) override;

private:
    void step_fixed(const System& sys, double h, System& out, double& err);

    const ForceModel& force_;
    double tol_;
    std::vector<Vec3<double>> kp_[7];  // position derivative stages
    std::vector<Vec3<double>> kv_[7];  // velocity derivative stages
    std::vector<Vec3<double>> acc_;
    std::vector<Vec3<double>> tp_;
    std::vector<Vec3<double>> tv_;

    void ensure(std::size_t n) {
        if (acc_.size() != n) {
            acc_.resize(n);
            tp_.resize(n);
            tv_.resize(n);
            for (int s = 0; s < 7; ++s) {
                kp_[s].resize(n);
                kv_[s].resize(n);
            }
        }
    }
};

namespace detail {
// Dormand-Prince 5(4)7M coefficients (classic).
constexpr double c2 = 1.0 / 5.0;
constexpr double c3 = 3.0 / 10.0;
constexpr double c4 = 4.0 / 5.0;
constexpr double c5 = 8.0 / 9.0;
constexpr double c6 = 1.0;
constexpr double c7 = 1.0;

constexpr double a21 = 1.0 / 5.0;
constexpr double a31 = 3.0 / 40.0;
constexpr double a32 = 9.0 / 40.0;
constexpr double a41 = 44.0 / 45.0;
constexpr double a42 = -56.0 / 15.0;
constexpr double a43 = 32.0 / 9.0;
constexpr double a51 = 19372.0 / 6561.0;
constexpr double a52 = -25360.0 / 2187.0;
constexpr double a53 = 64448.0 / 6561.0;
constexpr double a54 = -212.0 / 729.0;
constexpr double a61 = 9017.0 / 3168.0;
constexpr double a62 = -355.0 / 33.0;
constexpr double a63 = 46732.0 / 5247.0;
constexpr double a64 = 49.0 / 176.0;
constexpr double a65 = -5103.0 / 18656.0;
constexpr double a71 = 35.0 / 384.0;
constexpr double a73 = 500.0 / 1113.0;
constexpr double a74 = 125.0 / 192.0;
constexpr double a75 = -2187.0 / 6784.0;
constexpr double a76 = 11.0 / 84.0;

constexpr double b1 = 35.0 / 384.0;
constexpr double b3 = 500.0 / 1113.0;
constexpr double b4 = 125.0 / 192.0;
constexpr double b5 = -2187.0 / 6784.0;
constexpr double b6 = 11.0 / 84.0;

constexpr double d1 = 5179.0 / 57600.0;
constexpr double d3 = 7571.0 / 16695.0;
constexpr double d4 = 393.0 / 640.0;
constexpr double d5 = -92097.0 / 339200.0;
constexpr double d6 = 187.0 / 2100.0;
constexpr double d7 = 1.0 / 40.0;
}  // namespace detail

inline void RK45::step_fixed(const System& sys, double h, System& out, double& err) {
    const std::size_t n = sys.size();
    ensure(n);
    const auto& pos = sys.pos;
    const auto& vel = sys.vel;
    const auto& mass = sys.mass;

    // Stage 1.
    force_.accelerations(pos, vel, mass, acc_);
    for (std::size_t i = 0; i < n; ++i) {
        kp_[0][i] = vel[i];
        kv_[0][i] = acc_[i];
    }

    auto stage = [&](int s, double c, double a1, double a2, double a3, double a4, double a5,
                     double a6) {
        for (std::size_t i = 0; i < n; ++i) {
            tp_[i] = pos[i] + h * (a1 * kp_[0][i] + a2 * kp_[1][i] + a3 * kp_[2][i]
                                   + a4 * kp_[3][i] + a5 * kp_[4][i] + a6 * kp_[5][i]);
            tv_[i] = vel[i] + h * (a1 * kv_[0][i] + a2 * kv_[1][i] + a3 * kv_[2][i]
                                   + a4 * kv_[3][i] + a5 * kv_[4][i] + a6 * kv_[5][i]);
        }
        (void)c;
        force_.accelerations(tp_, tv_, mass, acc_);
        for (std::size_t i = 0; i < n; ++i) {
            kp_[s][i] = tv_[i];
            kv_[s][i] = acc_[i];
        }
    };

    stage(1, detail::c2, detail::a21, 0, 0, 0, 0, 0);
    stage(2, detail::c3, detail::a31, detail::a32, 0, 0, 0, 0);
    stage(3, detail::c4, detail::a41, detail::a42, detail::a43, 0, 0, 0);
    stage(4, detail::c5, detail::a51, detail::a52, detail::a53, detail::a54, 0, 0);
    stage(5, detail::c6, detail::a61, detail::a62, detail::a63, detail::a64, detail::a65, 0);
    stage(6, detail::c7, detail::a71, 0, detail::a73, detail::a74, detail::a75, detail::a76);

    out.pos.resize(n);
    out.vel.resize(n);
    out.mass = mass;
    out.radius = sys.radius;
    out.j2 = sys.j2;
    out.j2_ref = sys.j2_ref;
    out.t = sys.t + h;

    for (std::size_t i = 0; i < n; ++i) {
        // 5th-order solution.
        out.pos[i] = pos[i] + h * (detail::b1 * kp_[0][i] + detail::b3 * kp_[2][i]
                                   + detail::b4 * kp_[3][i] + detail::b5 * kp_[4][i]
                                   + detail::b6 * kp_[5][i]);
        out.vel[i] = vel[i] + h * (detail::b1 * kv_[0][i] + detail::b3 * kv_[2][i]
                                   + detail::b4 * kv_[3][i] + detail::b5 * kv_[4][i]
                                   + detail::b6 * kv_[5][i]);
        // Error (5th - 4th order).
        const Vec3<double> ep = h * ((detail::b1 - detail::d1) * kp_[0][i]
                                     + (detail::b3 - detail::d3) * kp_[2][i]
                                     + (detail::b4 - detail::d4) * kp_[3][i]
                                     + (detail::b5 - detail::d5) * kp_[4][i]
                                     + (detail::b6 - detail::d6) * kp_[5][i]
                                     - detail::d7 * kp_[6][i]);
        const Vec3<double> ev = h * ((detail::b1 - detail::d1) * kv_[0][i]
                                     + (detail::b3 - detail::d3) * kv_[2][i]
                                     + (detail::b4 - detail::d4) * kv_[3][i]
                                     + (detail::b5 - detail::d5) * kv_[4][i]
                                     + (detail::b6 - detail::d6) * kv_[5][i]
                                     - detail::d7 * kv_[6][i]);
        err = std::max(err, norm(ep) + norm(ev));
    }
}

inline void RK45::step(System& sys, double dt) {
    if (dt == 0.0) return;

    System trial;
    double t = 0.0;
    double h = std::min(dt, 0.1);  // conservative initial step; controller adjusts
    const double safety = 0.9;
    double prev_err = 0.0;

    while (t < dt) {
        if (t + h > dt) h = dt - t;

        double err = 0.0;
        step_fixed(sys, h, trial, err);

        if (err <= tol_ || h <= 1e-14) {
            sys = trial;
            t += h;
            prev_err = err;
        } else {
            // Reject and shrink.
            h = safety * h * std::pow(tol_ / err, 0.2);
            continue;
        }

        // Grow for next step (PI control).
        const double scale = (prev_err > 0.0) ? std::pow(tol_ / prev_err, 0.2) : 1.5;
        h = std::min(safety * h * scale, 2.0 * h);
    }
}

}  // namespace cassini::physics
