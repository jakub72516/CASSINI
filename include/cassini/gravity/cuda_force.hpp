// Cassini — CUDA ForceModel wrapper. The actual kernel lives in
// src/gravity/cuda/cuda_gravity.cu and is linked via the cassini_cuda target.
// Declared separately so host-only translation units can use it without nvcc.
#pragma once

#include <vector>

#include "cassini/core/constants.hpp"
#include "cassini/physics/force_model.hpp"

namespace cassini::gravity {

extern "C" void cassini_cuda_gravity(const double* pos, const double* mass, double* acc, int n,
                                     double G);

class CudaForce : public ForceModel {
public:
    explicit CudaForce(double G = constants::G) : G_(G) {}

    void accelerations(const std::vector<Vec3<double>>& pos,
                       const std::vector<Vec3<double>>& vel,
                       const std::vector<double>& mass,
                       std::vector<Vec3<double>>& acc) const override {
        (void)vel;
        const std::size_t n = pos.size();
        acc.resize(n);
        pos_flat_.resize(3 * n);
        mass_flat_.resize(n);
        acc_flat_.resize(3 * n);
        for (std::size_t i = 0; i < n; ++i) {
            pos_flat_[3 * i] = pos[i].x;
            pos_flat_[3 * i + 1] = pos[i].y;
            pos_flat_[3 * i + 2] = pos[i].z;
            mass_flat_[i] = mass[i];
        }
        cassini_cuda_gravity(pos_flat_.data(), mass_flat_.data(), acc_flat_.data(),
                             static_cast<int>(n), G_);
        for (std::size_t i = 0; i < n; ++i) {
            acc[i] = {acc_flat_[3 * i], acc_flat_[3 * i + 1], acc_flat_[3 * i + 2]};
        }
    }

private:
    double G_;
    mutable std::vector<double> pos_flat_;
    mutable std::vector<double> mass_flat_;
    mutable std::vector<double> acc_flat_;
};

}  // namespace cassini::gravity
