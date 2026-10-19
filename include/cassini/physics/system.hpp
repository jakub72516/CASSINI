// Cassini — system state and body definitions.
//
// State is stored structure-of-arrays (SoA) so the gravity solvers can stream
// contiguous memory through SIMD / CUDA kernels without gather overhead.
#pragma once

#include <cstddef>
#include <vector>

#include "cassini/core/vec3.hpp"

namespace cassini {

struct System {
    std::vector<double> mass;         // [Msun]
    std::vector<Vec3<double>> pos;    // [AU]
    std::vector<Vec3<double>> vel;    // [AU/day]
    std::vector<double> radius;       // physical radius [AU] (0 = point mass)
    std::vector<double> j2;           // oblateness J2 (0 = none)
    std::vector<double> j2_ref;       // reference radius for J2 [AU]
    double t{0.0};                    // [day]

    std::size_t size() const { return mass.size(); }

    void reserve(std::size_t n) {
        mass.reserve(n);
        pos.reserve(n);
        vel.reserve(n);
        radius.reserve(n);
        j2.reserve(n);
        j2_ref.reserve(n);
    }

    void add(double m, Vec3<double> p, Vec3<double> v,
             double rad = 0.0, double j2v = 0.0, double j2r = 0.0) {
        mass.push_back(m);
        pos.push_back(p);
        vel.push_back(v);
        radius.push_back(rad);
        j2.push_back(j2v);
        j2_ref.push_back(j2r);
    }

    // Total mass and barycenter (used by Wisdom-Holman Jacobi coordinates).
    double total_mass() const {
        double m = 0.0;
        for (double mi : mass) m += mi;
        return m;
    }
};

}  // namespace cassini
