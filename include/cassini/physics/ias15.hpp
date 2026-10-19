// Cassini — IAS15 integrator (Integrator with Adaptive Step-size control,
// 15th order), a Gauss-Radau predictor-corrector (Everhart 1985; Rein &
// Spiegel 2015). Excellent for close encounters and high-accuracy orbit
// integration. The scheme is a modified 15th-order Runge-Kutta with a
// dynamically-determined number of predictor-corrector iterations.
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "cassini/physics/force_model.hpp"
#include "cassini/physics/integrator.hpp"
#include "cassini/physics/system.hpp"

namespace cassini::physics {

namespace ias15_detail {
// Gauss-Radau spacings.
inline constexpr double h[8] = {0.0,
                                0.0562625605369221464656521910318,
                                0.180240691736892364987579942780,
                                0.352624717113169637373907769648,
                                0.547153626330555383001448554766,
                                0.734210177215410531523210605558,
                                0.885320946839095768090359771030,
                                0.977520613561287501891174488626};
inline constexpr double rr[28] = {
    0.0562625605369221464656522, 0.1802406917368923649875799, 0.1239781311999702185219278,
    0.3526247171131696373739078, 0.2963621565762474909082556, 0.1723840253762772723863278,
    0.5471536263305553830014486, 0.4908910657936332365357964, 0.3669129345936630180138686,
    0.1945289092173857456275408, 0.7342101772154105315232106, 0.6779476166784883850575584,
    0.5539694854785181665356307, 0.3815854601022408941493028, 0.1870565508848551485217621,
    0.8853209468390957680903598, 0.8290583863021736216247076, 0.7050802551022034031027798,
    0.5326962297259261307164520, 0.3381673205085403850889112, 0.1511107696236852365671492,
    0.9775206135612875018911745, 0.9212580530243653554255223, 0.7972799218243951369035945,
    0.6248958964481178645172667, 0.4303669872307321188897259, 0.2433104363458769703679639,
    0.0921996667221917338008147};
inline constexpr double c[21] = {
    -0.0562625605369221464656522, 0.0101408028300636299864818, -0.2365032522738145114532321,
    -0.0035758977292516175949345, 0.0935376952594620658957485, -0.5891279693869841488271399,
    0.0019565654099472210769006, -0.0547553868890686864408084, 0.4158812000823068616886219,
    -1.1362815957175395318285885, -0.0014365302363708915424460, 0.0421585277212687077072973,
    -0.3600995965020568122897665, 1.2501507118406910258505441, -1.8704917729329500633517991,
    0.0012717903090268677492943, -0.0387603579159067703699046, 0.3609622434528459832253398,
    -1.4668842084004269643701553, 2.9061362593084293014237913, -2.7558127197720458314421588};
inline constexpr double d[21] = {
    0.0562625605369221464656522, 0.0031654757181708292499905, 0.2365032522738145114532321,
    0.0001780977692217433881125, 0.0457929855060279188954539, 0.5891279693869841488271399,
    0.0000100202365223291272096, 0.0084318571535257015445000, 0.2535340690545692665214616,
    1.1362815957175395318285885, 0.0000005637641639318207610, 0.0015297840025004658189490,
    0.0978342365324440053653648, 0.8752546646840910912297246, 1.8704917729329500633517991,
    0.0000000317188154017613665, 0.0002762930909826476593130, 0.0360285539837364596003871,
    0.5767330002770787313544596, 2.2485887607691597933926895, 2.7558127197720458314421588};

// Machine-independent 7th root (used by the adaptive timestep controller).
inline double sqrt7(double a) {
    double scale = 1.0;
    while (a < 1e-7 && std::isnormal(a)) {
        scale *= 0.1;
        a *= 1e7;
    }
    while (a > 1e2 && std::isnormal(a)) {
        scale *= 10.0;
        a *= 1e-7;
    }
    double x = 1.0;
    for (int k = 0; k < 20; ++k) {
        const double x6 = x * x * x * x * x * x;
        x += (a / x6 - x) / 7.0;
    }
    return x * scale;
}
}  // namespace ias15_detail

class IAS15 : public IIntegrator {
public:
    explicit IAS15(const ForceModel& force, double epsilon = 1.0e-12)
        : force_(force), epsilon_(epsilon) {}

    // Advance by dt using one (or more, if adaptive) IAS15 steps.
    void step(System& sys, double dt) override;

private:
    const ForceModel& force_;
    double epsilon_;
    double dt_last_done_{0.0};
    bool initialized_{false};

    std::vector<double> x0_, v0_, a0_, at_;
    std::vector<double> g_, b_, e_, br_, er_;
    std::vector<double> csx_, csv_, csb_;
    std::vector<double> px_, pv_;

    std::size_t n3_{0};

    void alloc(std::size_t n3) {
        if (n3_ == n3) return;
        n3_ = n3;
        x0_.assign(n3, 0.0);
        v0_.assign(n3, 0.0);
        a0_.assign(n3, 0.0);
        at_.assign(n3, 0.0);
        g_.assign(7 * n3, 0.0);
        b_.assign(7 * n3, 0.0);
        e_.assign(7 * n3, 0.0);
        br_.assign(7 * n3, 0.0);
        er_.assign(7 * n3, 0.0);
        csx_.assign(n3, 0.0);
        csv_.assign(n3, 0.0);
        csb_.assign(7 * n3, 0.0);
        px_.assign(n3, 0.0);
        pv_.assign(n3, 0.0);
    }

    double* bp(int j) { return b_.data() + static_cast<std::size_t>(j) * n3_; }
    double* gp(int j) { return g_.data() + static_cast<std::size_t>(j) * n3_; }

    void compute_accelerations(System& sys) {
        std::vector<Vec3<double>> pos(sys.size()), vel(sys.size()), acc(sys.size());
        const std::size_t n = sys.size();
        for (std::size_t i = 0; i < n; ++i) {
            pos[i] = sys.pos[i];
            vel[i] = sys.vel[i];
        }
        force_.accelerations(pos, vel, sys.mass, acc);
        for (std::size_t i = 0; i < n; ++i) {
            at_[3 * i + 0] = acc[i].x;
            at_[3 * i + 1] = acc[i].y;
            at_[3 * i + 2] = acc[i].z;
        }
    }

    void write_state(System& sys, const double* x, const double* v) {
        const std::size_t n = sys.size();
        for (std::size_t i = 0; i < n; ++i) {
            sys.pos[i] = {x[3 * i + 0], x[3 * i + 1], x[3 * i + 2]};
            sys.vel[i] = {v[3 * i + 0], v[3 * i + 1], v[3 * i + 2]};
        }
    }

    void predict_next_step(double ratio) {
        const std::size_t N3 = n3_;
        if (ratio > 20.0) {
            std::fill(e_.begin(), e_.end(), 0.0);
            std::fill(b_.begin(), b_.end(), 0.0);
            return;
        }
        const double q1 = ratio, q2 = q1 * q1, q3 = q1 * q2, q4 = q2 * q2, q5 = q2 * q3,
                     q6 = q3 * q3, q7 = q3 * q4;
        for (std::size_t k = 0; k < N3; ++k) {
            const double b0 = b_[0 * N3 + k] - er_[0 * N3 + k];
            const double b1 = b_[1 * N3 + k] - er_[1 * N3 + k];
            const double b2 = b_[2 * N3 + k] - er_[2 * N3 + k];
            const double b3 = b_[3 * N3 + k] - er_[3 * N3 + k];
            const double b4 = b_[4 * N3 + k] - er_[4 * N3 + k];
            const double b5 = b_[5 * N3 + k] - er_[5 * N3 + k];
            const double b6 = b_[6 * N3 + k] - er_[6 * N3 + k];
            const double B0 = br_[0 * N3 + k], B1 = br_[1 * N3 + k], B2 = br_[2 * N3 + k],
                         B3 = br_[3 * N3 + k], B4 = br_[4 * N3 + k], B5 = br_[5 * N3 + k],
                         B6 = br_[6 * N3 + k];

            e_[0 * N3 + k] = q1 * (B6 * 7.0 + B5 * 6.0 + B4 * 5.0 + B3 * 4.0 + B2 * 3.0 +
                                   B1 * 2.0 + B0);
            e_[1 * N3 + k] = q2 * (B6 * 21.0 + B5 * 15.0 + B4 * 10.0 + B3 * 6.0 + B2 * 3.0 + B1);
            e_[2 * N3 + k] = q3 * (B6 * 35.0 + B5 * 20.0 + B4 * 10.0 + B3 * 4.0 + B2);
            e_[3 * N3 + k] = q4 * (B6 * 35.0 + B5 * 15.0 + B4 * 5.0 + B3);
            e_[4 * N3 + k] = q5 * (B6 * 21.0 + B5 * 6.0 + B4);
            e_[5 * N3 + k] = q6 * (B6 * 7.0 + B5);
            e_[6 * N3 + k] = q7 * B6;

            b_[0 * N3 + k] = e_[0 * N3 + k] + b0;
            b_[1 * N3 + k] = e_[1 * N3 + k] + b1;
            b_[2 * N3 + k] = e_[2 * N3 + k] + b2;
            b_[3 * N3 + k] = e_[3 * N3 + k] + b3;
            b_[4 * N3 + k] = e_[4 * N3 + k] + b4;
            b_[5 * N3 + k] = e_[5 * N3 + k] + b5;
            b_[6 * N3 + k] = e_[6 * N3 + k] + b6;
        }
    }

    // One IAS15 step of size dt. Returns true on success (always, fixed-step).
    void step_once(System& sys, double dt) {
        const std::size_t n = sys.size();
        const std::size_t N3 = 3 * n;
        alloc(N3);

        if (!initialized_) {
            for (std::size_t i = 0; i < n; ++i) {
                x0_[3 * i + 0] = sys.pos[i].x;
                x0_[3 * i + 1] = sys.pos[i].y;
                x0_[3 * i + 2] = sys.pos[i].z;
                v0_[3 * i + 0] = sys.vel[i].x;
                v0_[3 * i + 1] = sys.vel[i].y;
                v0_[3 * i + 2] = sys.vel[i].z;
            }
            compute_accelerations(sys);
            std::copy(at_.begin(), at_.end(), a0_.begin());
            initialized_ = true;
        } else {
            for (std::size_t i = 0; i < n; ++i) {
                x0_[3 * i + 0] = sys.pos[i].x;
                x0_[3 * i + 1] = sys.pos[i].y;
                x0_[3 * i + 2] = sys.pos[i].z;
                v0_[3 * i + 0] = sys.vel[i].x;
                v0_[3 * i + 1] = sys.vel[i].y;
                v0_[3 * i + 2] = sys.vel[i].z;
            }
            compute_accelerations(sys);
            std::copy(at_.begin(), at_.end(), a0_.begin());
        }

        // Predictor init: g from b.
        for (std::size_t k = 0; k < N3; ++k) {
            const double* B = b_.data() + k;
            g_[0 * N3 + k] = B[6 * N3] * ias15_detail::d[15] + B[5 * N3] * ias15_detail::d[10] +
                             B[4 * N3] * ias15_detail::d[6] + B[3 * N3] * ias15_detail::d[3] +
                             B[2 * N3] * ias15_detail::d[1] + B[1 * N3] * ias15_detail::d[0] + B[0];
            g_[1 * N3 + k] = B[6 * N3] * ias15_detail::d[16] + B[5 * N3] * ias15_detail::d[11] +
                             B[4 * N3] * ias15_detail::d[7] + B[3 * N3] * ias15_detail::d[4] +
                             B[2 * N3] * ias15_detail::d[2] + B[1 * N3];
            g_[2 * N3 + k] = B[6 * N3] * ias15_detail::d[17] + B[5 * N3] * ias15_detail::d[12] +
                             B[4 * N3] * ias15_detail::d[8] + B[3 * N3] * ias15_detail::d[5] + B[2 * N3];
            g_[3 * N3 + k] = B[6 * N3] * ias15_detail::d[18] + B[5 * N3] * ias15_detail::d[13] +
                             B[4 * N3] * ias15_detail::d[9] + B[3 * N3];
            g_[4 * N3 + k] = B[6 * N3] * ias15_detail::d[19] + B[5 * N3] * ias15_detail::d[14] + B[4 * N3];
            g_[5 * N3 + k] = B[6 * N3] * ias15_detail::d[20] + B[5 * N3];
            g_[6 * N3 + k] = B[6 * N3];
        }

        double pc_error = 1e300, pc_error_last = 2.0;
        int iterations = 0;
        while (true) {
            if (pc_error < 1e-16) break;
            if (iterations > 2 && pc_error_last <= pc_error) break;
            if (iterations >= 12) break;
            pc_error_last = pc_error;
            pc_error = 0.0;
            ++iterations;

            for (int nn = 1; nn < 8; ++nn) {
                const double hn = ias15_detail::h[nn];
                // Predict positions and velocities at substep nn.
                for (std::size_t k = 0; k < N3; ++k) {
                    const double* B = b_.data() + k;
                    const double b0 = B[0], b1 = B[1 * N3], b2 = B[2 * N3], b3 = B[3 * N3],
                                 b4 = B[4 * N3], b5 = B[5 * N3], b6 = B[6 * N3];
                    const double xk = ((((((((b6 * 7.0 * hn / 9.0 + b5) * 3.0 * hn / 4.0 + b4) *
                                               5.0 * hn / 7.0 +
                                           b3) *
                                              2.0 * hn / 3.0 +
                                          b2) *
                                             3.0 * hn / 5.0 +
                                         b1) *
                                            hn / 2.0 +
                                        b0) *
                                           hn / 3.0 +
                                       a0_[k]) *
                                          dt * hn / 2.0 +
                                      v0_[k]) *
                                     dt * hn;
                    const double vk = (((((((b6 * 7.0 * hn / 8.0 + b5) * 6.0 * hn / 7.0 + b4) *
                                               5.0 * hn / 6.0 +
                                           b3) *
                                              4.0 * hn / 5.0 +
                                          b2) *
                                             3.0 * hn / 4.0 +
                                         b1) *
                                           2.0 * hn / 3.0 +
                                        b0) *
                                           hn / 2.0 +
                                       a0_[k]) *
                                      dt * hn;
                    px_[k] = x0_[k] + xk;
                    pv_[k] = v0_[k] + vk;
                }
                write_state(sys, px_.data(), pv_.data());
                compute_accelerations(sys);

                // Corrector.
                double maxak = 0.0, maxb6 = 0.0;
                for (std::size_t k = 0; k < N3; ++k) {
                    const double gk = at_[k] - a0_[k];
                    double tmp;
                    double* B = b_.data() + k;
                    double* G = g_.data() + k;
                    switch (nn) {
                        case 1:
                            tmp = G[0];
                            G[0] = gk / ias15_detail::rr[0];
                            B[0] += G[0] - tmp;
                            break;
                        case 2:
                            tmp = G[1 * N3];
                            G[1 * N3] = (gk / ias15_detail::rr[1] - G[0]) / ias15_detail::rr[2];
                            tmp = G[1 * N3] - tmp;
                            B[0] += tmp * ias15_detail::c[0];
                            B[1 * N3] += tmp;
                            break;
                        case 3:
                            tmp = G[2 * N3];
                            G[2 * N3] = ((gk / ias15_detail::rr[3] - G[0]) / ias15_detail::rr[4] -
                                         G[1 * N3]) /
                                        ias15_detail::rr[5];
                            tmp = G[2 * N3] - tmp;
                            B[0] += tmp * ias15_detail::c[1];
                            B[1 * N3] += tmp * ias15_detail::c[2];
                            B[2 * N3] += tmp;
                            break;
                        case 4:
                            tmp = G[3 * N3];
                            G[3 * N3] = (((gk / ias15_detail::rr[6] - G[0]) / ias15_detail::rr[7] -
                                          G[1 * N3]) /
                                             ias15_detail::rr[8] -
                                         G[2 * N3]) /
                                        ias15_detail::rr[9];
                            tmp = G[3 * N3] - tmp;
                            B[0] += tmp * ias15_detail::c[3];
                            B[1 * N3] += tmp * ias15_detail::c[4];
                            B[2 * N3] += tmp * ias15_detail::c[5];
                            B[3 * N3] += tmp;
                            break;
                        case 5:
                            tmp = G[4 * N3];
                            G[4 * N3] = ((((gk / ias15_detail::rr[10] - G[0]) / ias15_detail::rr[11] -
                                           G[1 * N3]) /
                                              ias15_detail::rr[12] -
                                          G[2 * N3]) /
                                             ias15_detail::rr[13] -
                                         G[3 * N3]) /
                                        ias15_detail::rr[14];
                            tmp = G[4 * N3] - tmp;
                            B[0] += tmp * ias15_detail::c[6];
                            B[1 * N3] += tmp * ias15_detail::c[7];
                            B[2 * N3] += tmp * ias15_detail::c[8];
                            B[3 * N3] += tmp * ias15_detail::c[9];
                            B[4 * N3] += tmp;
                            break;
                        case 6:
                            tmp = G[5 * N3];
                            G[5 * N3] = (((((gk / ias15_detail::rr[15] - G[0]) / ias15_detail::rr[16] -
                                            G[1 * N3]) /
                                               ias15_detail::rr[17] -
                                           G[2 * N3]) /
                                              ias15_detail::rr[18] -
                                          G[3 * N3]) /
                                             ias15_detail::rr[19] -
                                         G[4 * N3]) /
                                        ias15_detail::rr[20];
                            tmp = G[5 * N3] - tmp;
                            B[0] += tmp * ias15_detail::c[10];
                            B[1 * N3] += tmp * ias15_detail::c[11];
                            B[2 * N3] += tmp * ias15_detail::c[12];
                            B[3 * N3] += tmp * ias15_detail::c[13];
                            B[4 * N3] += tmp * ias15_detail::c[14];
                            B[5 * N3] += tmp;
                            break;
                        case 7:
                            tmp = G[6 * N3];
                            G[6 * N3] = ((((((gk / ias15_detail::rr[21] - G[0]) /
                                                  ias15_detail::rr[22] -
                                              G[1 * N3]) /
                                                 ias15_detail::rr[23] -
                                             G[2 * N3]) /
                                                ias15_detail::rr[24] -
                                            G[3 * N3]) /
                                               ias15_detail::rr[25] -
                                           G[4 * N3]) /
                                              ias15_detail::rr[26] -
                                          G[5 * N3]) /
                                         ias15_detail::rr[27];
                            tmp = G[6 * N3] - tmp;
                            B[0] += tmp * ias15_detail::c[15];
                            B[1 * N3] += tmp * ias15_detail::c[16];
                            B[2 * N3] += tmp * ias15_detail::c[17];
                            B[3 * N3] += tmp * ias15_detail::c[18];
                            B[4 * N3] += tmp * ias15_detail::c[19];
                            B[5 * N3] += tmp * ias15_detail::c[20];
                            B[6 * N3] += tmp;
                            {
                                const double ak = std::fabs(at_[k]);
                                if (std::isnormal(ak) && ak > maxak) maxak = ak;
                                const double b6k = std::fabs(tmp);
                                if (std::isnormal(b6k) && b6k > maxb6) maxb6 = b6k;
                            }
                            break;
                        default:
                            break;
                    }
                }
                if (nn == 7) {
                    pc_error = (maxak > 0.0) ? maxb6 / maxak : 0.0;
                }
            }
        }

        // Final position/velocity update.
        for (std::size_t k = 0; k < N3; ++k) {
            const double* B = b_.data() + k;
            x0_[k] += (B[6 * N3] / 72.0 + B[5 * N3] / 56.0 + B[4 * N3] / 42.0 + B[3 * N3] / 30.0 +
                       B[2 * N3] / 20.0 + B[1 * N3] / 12.0 + B[0] / 6.0) *
                          dt * dt +
                      a0_[k] / 2.0 * dt * dt + v0_[k] * dt;
            v0_[k] += (B[6 * N3] / 8.0 + B[5 * N3] / 7.0 + B[4 * N3] / 6.0 + B[3 * N3] / 5.0 +
                       B[2 * N3] / 4.0 + B[1 * N3] / 3.0 + B[0] / 2.0) *
                          dt +
                      a0_[k] * dt;
        }
        write_state(sys, x0_.data(), v0_.data());
        sys.t += dt;

        // Save b/e and predict next step's coefficients.
        std::copy(e_.begin(), e_.end(), er_.begin());
        std::copy(b_.begin(), b_.end(), br_.begin());
        predict_next_step(1.0);
    }
};

inline void IAS15::step(System& sys, double dt) {
    step_once(sys, dt);
}

}  // namespace cassini::physics
