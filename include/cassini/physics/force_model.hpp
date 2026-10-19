// Cassini — force model interface. Concrete implementations (brute force,
// Barnes-Hut, FMM) compute accelerations; integrators depend only on this
// abstraction so they can be swapped between O(N^2) and tree codes unchanged.
//
// Velocities are passed because post-Newtonian accelerations are velocity
// dependent; Newtonian-only solvers simply ignore them.
#pragma once

#include <vector>

#include "cassini/core/vec3.hpp"

namespace cassini {

class ForceModel {
public:
    virtual ~ForceModel() = default;

    // Compute gravitational acceleration [AU/day^2] for every body.
    // `acc` must be pre-sized to pos.size().
    virtual void accelerations(const std::vector<Vec3<double>>& pos,
                               const std::vector<Vec3<double>>& vel,
                               const std::vector<double>& mass,
                               std::vector<Vec3<double>>& acc) const = 0;

    // Hook for per-step setup (tree construction, multipole build, etc.).
    virtual void prepare(const std::vector<Vec3<double>>& pos,
                         const std::vector<double>& mass) {
        (void)pos;
        (void)mass;
    }
};

}  // namespace cassini
