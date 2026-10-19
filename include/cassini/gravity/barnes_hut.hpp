// Cassini — Barnes-Hut octree gravity solver with multipole expansion up to
// quadrupole. OpenMP-parallel tree traversal. Implements ForceModel so it
// drops into the integrators as a drop-in replacement for brute force.
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "cassini/core/constants.hpp"
#include "cassini/physics/force_model.hpp"

namespace cassini::gravity {

struct Mat3 {
    double m[3][3]{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
};

struct OctNode {
    Vec3<double> com{};
    Vec3<double> center{};
    double mass{0.0};
    double size{0.0};
    Mat3 quad;               // traceless quadrupole about com
    int children[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
    int body{-1};            // leaf body index, else -1
};

class BarnesHut : public ForceModel {
public:
    explicit BarnesHut(double theta = 0.5, double G = constants::G) : theta_(theta), G_(G) {}

    void accelerations(const std::vector<Vec3<double>>& pos,
                       const std::vector<Vec3<double>>& vel,
                       const std::vector<double>& mass,
                       std::vector<Vec3<double>>& acc) const override {
        (void)vel;
        const std::size_t n = pos.size();
        acc.resize(n);
        build(pos, mass);
#ifdef _OPENMP
#pragma omp parallel for if (n > 512)
#endif
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i) {
            acc[static_cast<std::size_t>(i)] = traverse(pos[static_cast<std::size_t>(i)], 0);
        }
    }

private:
    double theta_;
    double G_;
    mutable std::vector<OctNode> nodes_;
    mutable std::vector<Vec3<double>> pos_;
    mutable std::vector<double> mass_;

    void build(const std::vector<Vec3<double>>& pos, const std::vector<double>& mass) const {
        const std::size_t n = pos.size();
        pos_ = pos;
        mass_ = mass;
        nodes_.clear();
        if (n == 0) return;

        Vec3<double> mn = pos[0], mx = pos[0];
        for (std::size_t i = 1; i < n; ++i) {
            mn.x = std::min(mn.x, pos[i].x); mx.x = std::max(mx.x, pos[i].x);
            mn.y = std::min(mn.y, pos[i].y); mx.y = std::max(mx.y, pos[i].y);
            mn.z = std::min(mn.z, pos[i].z); mx.z = std::max(mx.z, pos[i].z);
        }
        const Vec3<double> center = 0.5 * (mn + mx);
        const double size = std::max({mx.x - mn.x, mx.y - mn.y, mx.z - mn.z, 1e-9});

        std::vector<int> idx(n);
        for (std::size_t i = 0; i < n; ++i) idx[i] = static_cast<int>(i);

        nodes_.push_back(OctNode{});
        nodes_[0].center = center;
        nodes_[0].size = size;
        build_recursive(idx, 0, n, center, size, 0, 0);
        accumulate(0);
    }

    void build_recursive(std::vector<int>& idx, std::size_t lo, std::size_t hi,
                         Vec3<double> center, double size, int node, int depth) const {
        const std::size_t count = hi - lo;
        if (count == 1) {
            nodes_[static_cast<std::size_t>(node)].body = idx[lo];
            return;
        }
        if (count == 0 || depth > 64) return;

        // Partition bodies into 8 octants.
        const double half = 0.5 * size;
        std::size_t start[9]{};
        std::vector<int> bucket(count);
        for (std::size_t k = lo; k < hi; ++k) {
            const Vec3<double>& p = pos_[static_cast<std::size_t>(idx[k])];
            int o = (p.x >= center.x ? 4 : 0) | (p.y >= center.y ? 2 : 0) |
                    (p.z >= center.z ? 1 : 0);
            bucket[k - lo] = o;
        }
        // Count.
        std::vector<std::size_t> cnt(8, 0);
        for (std::size_t k = 0; k < count; ++k) cnt[bucket[k]]++;
        start[0] = 0;
        for (int o = 1; o <= 8; ++o) start[o] = start[o - 1] + cnt[o - 1];
        std::vector<int> sorted(count);
        std::vector<std::size_t> pos_in(8);
        for (int o = 0; o < 8; ++o) pos_in[o] = start[o];
        for (std::size_t k = lo; k < hi; ++k) {
            const int o = bucket[k - lo];
            sorted[pos_in[o]++] = idx[k];
        }
        for (std::size_t k = 0; k < count; ++k) idx[lo + k] = sorted[k];

        // Recurse into non-empty octants.
        for (int o = 0; o < 8; ++o) {
            const std::size_t s = start[o], e = start[o + 1];
            if (s == e) continue;
            const Vec3<double> cc{
                center.x + (o & 4 ? half : -half),
                center.y + (o & 2 ? half : -half),
                center.z + (o & 1 ? half : -half),
            };
            nodes_.push_back(OctNode{});
            const int child = static_cast<int>(nodes_.size() - 1);
            nodes_[static_cast<std::size_t>(node)].children[o] = child;
            nodes_[static_cast<std::size_t>(child)].center = cc;
            nodes_[static_cast<std::size_t>(child)].size = half;
            build_recursive(idx, lo + s, lo + e, cc, half, child, depth + 1);
        }
    }

    // Bottom-up multipole accumulation.
    void accumulate(int node) const {
        OctNode& nd = nodes_[static_cast<std::size_t>(node)];
        if (nd.body >= 0) {
            nd.mass = mass_[static_cast<std::size_t>(nd.body)];
            nd.com = pos_[static_cast<std::size_t>(nd.body)];
            nd.quad = Mat3{};
            return;
        }
        double m = 0.0;
        Vec3<double> com{};
        for (int o = 0; o < 8; ++o) {
            const int c = nd.children[o];
            if (c < 0) continue;
            accumulate(c);
            m += nodes_[static_cast<std::size_t>(c)].mass;
            com += nodes_[static_cast<std::size_t>(c)].mass * nodes_[static_cast<std::size_t>(c)].com;
        }
        if (m > 0.0) com = com * (1.0 / m);
        nd.mass = m;
        nd.com = com;

        // Quadrupole via parallel-axis shift.
        Mat3 q;
        for (int o = 0; o < 8; ++o) {
            const int c = nd.children[o];
            if (c < 0) continue;
            const OctNode& ch = nodes_[static_cast<std::size_t>(c)];
            const Vec3<double> d = ch.com - com;
            const double d2 = dot(d, d);
            const double mc = ch.mass;
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b) {
                    double shift = 3.0 * (a == 0 ? d.x : a == 1 ? d.y : d.z) *
                                           (b == 0 ? d.x : b == 1 ? d.y : d.z);
                    if (a == b) shift -= d2;
                    q.m[a][b] += ch.quad.m[a][b] + mc * shift;
                }
            }
        }
        nd.quad = q;
    }

    Vec3<double> traverse(const Vec3<double>& p, int node) const {
        const OctNode& nd = nodes_[static_cast<std::size_t>(node)];
        const Vec3<double> d = p - nd.com;
        const double r2 = dot(d, d);
        const double r = std::sqrt(r2);

        // Leaf: direct interaction.
        if (nd.body >= 0) {
            if (r == 0.0) return {};
            return (-G_ * nd.mass / (r2 * r)) * d;
        }

        // Far enough: multipole (monopole + quadrupole).
        if (nd.size / r < theta_ || r == 0.0) {
            if (r == 0.0) return {};
            const Vec3<double> n = d * (1.0 / r);
            // Monopole.
            Vec3<double> a = (-G_ * nd.mass / r2) * n;
            // Quadrupole: Q n and n.Q.n.
            const double qn[3] = {
                nd.quad.m[0][0] * n.x + nd.quad.m[0][1] * n.y + nd.quad.m[0][2] * n.z,
                nd.quad.m[1][0] * n.x + nd.quad.m[1][1] * n.y + nd.quad.m[1][2] * n.z,
                nd.quad.m[2][0] * n.x + nd.quad.m[2][1] * n.y + nd.quad.m[2][2] * n.z,
            };
            const double nqn = qn[0] * n.x + qn[1] * n.y + qn[2] * n.z;
            const double k = G_ / (2.0 * r2 * r2);  // G/(2 r^4)
            a += k * (Vec3<double>(2.0 * qn[0] - 5.0 * nqn * n.x,
                                   2.0 * qn[1] - 5.0 * nqn * n.y,
                                   2.0 * qn[2] - 5.0 * nqn * n.z));
            return a;
        }

        // Descend.
        Vec3<double> a{};
        for (int o = 0; o < 8; ++o) {
            const int c = nd.children[o];
            if (c >= 0) a += traverse(p, c);
        }
        return a;
    }
};

}  // namespace cassini::gravity
