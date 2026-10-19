// Cassini — O(N^2) brute-force force model with optional J2 oblateness and
// post-Newtonian corrections. Serves as the exact reference against which the
// Barnes-Hut / FMM approximations are validated.
#pragma once

#include <vector>

#include "cassini/core/constants.hpp"
#include "cassini/physics/force_model.hpp"
#include "cassini/physics/pn.hpp"

namespace cassini::physics {

struct ForceOptions {
    double G{constants::G};
    int pn_order{0};        // 0 = none, 1 = 1PN, 2 = 1PN+2PN
    double c{constants::C};
    bool j2_enabled{true};  // apply J2 oblateness for bodies with j2 != 0
};

class NewtonianForce : public ForceModel {
public:
    explicit NewtonianForce(ForceOptions opts = {}) : opts_(opts) {}

    void set_j2(const std::vector<double>& j2, const std::vector<double>& ref) {
        j2_ = j2;
        j2_ref_ = ref;
    }

    void accelerations(const std::vector<Vec3<double>>& pos,
                       const std::vector<Vec3<double>>& vel,
                       const std::vector<double>& mass,
                       std::vector<Vec3<double>>& acc) const override {
        const std::size_t n = pos.size();
        std::fill(acc.begin(), acc.end(), Vec3<double>{});

        std::size_t central = 0;
        for (std::size_t i = 1; i < n; ++i) {
            if (mass[i] > mass[central]) central = i;
        }

        const bool has_j2 = !j2_.empty() && j2_.size() == n;

        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                if (i == j) continue;
                const Vec3<double> dr = pos[j] - pos[i];
                const double r2 = dot(dr, dr);
                const double r = std::sqrt(r2);
                if (r == 0.0) continue;
                const double inv_r3 = 1.0 / (r2 * r);
                acc[i] += (opts_.G * mass[j] * inv_r3) * dr;

                if (opts_.j2_enabled && has_j2 && j2_ref_[j] > 0.0 && j2_[j] != 0.0) {
                    const double rr = j2_ref_[j];
                    const double z2 = dr.z * dr.z;
                    const double j2acc = 1.5 * j2_[j] * rr * rr / r2
                                         * (5.0 * z2 / r2 - 1.0);
                    acc[i] += (opts_.G * mass[j] * inv_r3) * j2acc * dr;
                }
            }

            if (opts_.pn_order >= 1 && i != central && n > 1) {
                const Vec3<double> dr = pos[i] - pos[central];
                const Vec3<double> dv = vel[i] - vel[central];
                const double mu = opts_.G * (mass[central] + mass[i]);
                const Vec3<double> pn = pn_acceleration(dr, dv, mu, opts_.c, opts_.pn_order);
                acc[i] += pn;
                acc[central] -= (mass[i] / mass[central]) * pn;
            }
        }
    }

private:
    ForceOptions opts_;
    std::vector<double> j2_;
    std::vector<double> j2_ref_;
};

}  // namespace cassini::physics
