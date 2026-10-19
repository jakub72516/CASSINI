// Cassini — Fast Multipole Method (FMM) gravity solver with O(N) scaling.
//
// Level octree, monopole+quadrupole multipoles and a second-order Cartesian
// Taylor local expansion. For each leaf, a single top-down traversal partitions
// all other cells into a well-separated *interaction list* (handled by M2L) and
// an adjacent *near field* (handled by direct summation), with no gap or
// overlap, giving O(N) work per force evaluation.
//
// Reference: Greengard & Rokhlin (1987); Dehnen (2000) for Cartesian FMM.
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "cassini/core/constants.hpp"
#include "cassini/gravity/barnes_hut.hpp"  // Mat3
#include "cassini/physics/force_model.hpp"

namespace cassini::gravity {

struct FmmCell {
    Vec3<double> com{};
    Vec3<double> center{};
    double mass{0.0};
    double size{0.0};
    Mat3 quad{};
    int children[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
    std::vector<int> bodies;  // leaf bodies
    std::vector<int> ilist;   // well-separated cells (M2L sources)
    std::vector<int> near;    // near-field bodies (direct sum)
    double L[3]{0, 0, 0};     // local expansion (dipole)
    double Lq[3][3]{{0}};     // local expansion (quadrupole, symmetric)
};

class FMM : public ForceModel {
public:
    explicit FMM(int leaf_max = 16, double sep = 3.0, double G = constants::G)
        : leaf_max_(leaf_max), sep_(sep), G_(G) {}

    void accelerations(const std::vector<Vec3<double>>& pos,
                       const std::vector<Vec3<double>>& vel,
                       const std::vector<double>& mass,
                       std::vector<Vec3<double>>& acc) const override {
        (void)vel;
        const std::size_t n = pos.size();
        acc.assign(n, Vec3<double>{});
        if (n == 0) return;
        build(pos, mass);
        upward(0);
        build_lists(0);
        downward();
        evaluate(pos, acc);
    }

private:
    int leaf_max_;
    double sep_;
    double G_;
    mutable std::vector<FmmCell> cells_;
    mutable std::vector<Vec3<double>> pos_;
    mutable std::vector<double> mass_;
    mutable std::vector<int> all_;

    void build(const std::vector<Vec3<double>>& pos, const std::vector<double>& mass) const {
        const std::size_t n = pos.size();
        pos_ = pos;
        mass_ = mass;
        cells_.clear();

        Vec3<double> mn = pos[0], mx = pos[0];
        for (std::size_t i = 1; i < n; ++i) {
            mn.x = std::min(mn.x, pos[i].x); mx.x = std::max(mx.x, pos[i].x);
            mn.y = std::min(mn.y, pos[i].y); mx.y = std::max(mx.y, pos[i].y);
            mn.z = std::min(mn.z, pos[i].z); mx.z = std::max(mx.z, pos[i].z);
        }
        const Vec3<double> center = 0.5 * (mn + mx);
        const double size = std::max({mx.x - mn.x, mx.y - mn.y, mx.z - mn.z, 1e-9});

        all_.resize(n);
        for (std::size_t i = 0; i < n; ++i) all_[i] = static_cast<int>(i);

        cells_.push_back(FmmCell{});
        cells_[0].center = center;
        cells_[0].size = size;
        subdivide(0, all_, 0);
    }

    void subdivide(int cell, const std::vector<int>& idx, int depth) const {
        const Vec3<double> center = cells_[static_cast<std::size_t>(cell)].center;
        const double size = cells_[static_cast<std::size_t>(cell)].size;
        if (idx.size() <= static_cast<std::size_t>(leaf_max_) || depth > 40) {
            cells_[static_cast<std::size_t>(cell)].bodies = idx;
            return;
        }
        const double half = 0.5 * size;
        std::vector<std::vector<int>> oct(8);
        for (int b : idx) {
            const Vec3<double>& p = pos_[static_cast<std::size_t>(b)];
            int o = (p.x >= center.x ? 4 : 0) | (p.y >= center.y ? 2 : 0) |
                    (p.z >= center.z ? 1 : 0);
            oct[static_cast<std::size_t>(o)].push_back(b);
        }
        for (int o = 0; o < 8; ++o) {
            if (oct[static_cast<std::size_t>(o)].empty()) continue;
            const Vec3<double> cc{center.x + (o & 4 ? half : -half),
                                  center.y + (o & 2 ? half : -half),
                                  center.z + (o & 1 ? half : -half)};
            cells_.push_back(FmmCell{});
            const int child = static_cast<int>(cells_.size() - 1);
            cells_[static_cast<std::size_t>(child)].center = cc;
            cells_[static_cast<std::size_t>(child)].size = half;
            cells_[static_cast<std::size_t>(cell)].children[o] = child;
            subdivide(child, oct[static_cast<std::size_t>(o)], depth + 1);
        }
    }

    void upward(int cell) const {
        FmmCell& c = cells_[static_cast<std::size_t>(cell)];
        if (!c.bodies.empty()) {
            double m = 0.0;
            Vec3<double> com{};
            for (int b : c.bodies) {
                m += mass_[static_cast<std::size_t>(b)];
                com += mass_[static_cast<std::size_t>(b)] * pos_[static_cast<std::size_t>(b)];
            }
            if (m > 0.0) com = com * (1.0 / m);
            c.mass = m;
            c.com = com;
            Mat3 q;
            for (int b : c.bodies) {
                const Vec3<double> d = pos_[static_cast<std::size_t>(b)] - com;
                const double d2 = dot(d, d);
                const double mb = mass_[static_cast<std::size_t>(b)];
                for (int a = 0; a < 3; ++a)
                    for (int bb = 0; bb < 3; ++bb) {
                        const double dv = (a == 0 ? d.x : a == 1 ? d.y : d.z);
                        const double dw = (bb == 0 ? d.x : bb == 1 ? d.y : d.z);
                        q.m[a][bb] += mb * (3.0 * dv * dw - (a == bb ? d2 : 0.0));
                    }
            }
            c.quad = q;
            return;
        }
        double m = 0.0;
        Vec3<double> com{};
        for (int o = 0; o < 8; ++o) {
            const int ch = c.children[o];
            if (ch < 0) continue;
            upward(ch);
            m += cells_[static_cast<std::size_t>(ch)].mass;
            com += cells_[static_cast<std::size_t>(ch)].mass * cells_[static_cast<std::size_t>(ch)].com;
        }
        if (m > 0.0) com = com * (1.0 / m);
        c.mass = m;
        c.com = com;
        Mat3 q;
        for (int o = 0; o < 8; ++o) {
            const int ch = c.children[o];
            if (ch < 0) continue;
            const FmmCell& cc = cells_[static_cast<std::size_t>(ch)];
            const Vec3<double> d = cc.com - com;
            const double d2 = dot(d, d);
            for (int a = 0; a < 3; ++a)
                for (int bb = 0; bb < 3; ++bb) {
                    const double dv = (a == 0 ? d.x : a == 1 ? d.y : d.z);
                    const double dw = (bb == 0 ? d.x : bb == 1 ? d.y : d.z);
                    q.m[a][bb] += cc.quad.m[a][bb] + cc.mass * (3.0 * dv * dw - (a == bb ? d2 : 0.0));
                }
        }
        c.quad = q;
    }

    bool well_separated(const FmmCell& a, const FmmCell& b) const {
        const Vec3<double> d = a.center - b.center;
        return std::sqrt(dot(d, d)) >= sep_ * std::max(a.size, b.size);
    }

    void build_lists(int cell) const {
        FmmCell& c = cells_[static_cast<std::size_t>(cell)];
        if (!c.bodies.empty()) {
            partition(0, cell, c.ilist, c.near);
        }
        for (int o = 0; o < 8; ++o)
            if (c.children[o] >= 0) build_lists(c.children[o]);
    }

    // Single top-down traversal: well-separated cells -> ilist, else near bodies.
    void partition(int node, int target, std::vector<int>& ilist, std::vector<int>& near) const {
        if (node == target) {
            const FmmCell& c = cells_[static_cast<std::size_t>(node)];
            near.insert(near.end(), c.bodies.begin(), c.bodies.end());
            return;
        }
        const FmmCell& nd = cells_[static_cast<std::size_t>(node)];
        const FmmCell& t = cells_[static_cast<std::size_t>(target)];
        if (well_separated(nd, t)) {
            ilist.push_back(node);
            return;
        }
        if (!nd.bodies.empty()) {
            near.insert(near.end(), nd.bodies.begin(), nd.bodies.end());
            return;
        }
        for (int o = 0; o < 8; ++o)
            if (nd.children[o] >= 0) partition(nd.children[o], target, ilist, near);
    }

    void downward() const { downward_rec(0); }

    void downward_rec(int cell) const {
        FmmCell& c = cells_[static_cast<std::size_t>(cell)];
        for (int src : c.ilist) m2l(src, cell);
        for (int o = 0; o < 8; ++o) {
            const int ch = c.children[o];
            if (ch < 0) continue;
            FmmCell& child = cells_[static_cast<std::size_t>(ch)];
            const Vec3<double> delta = child.center - c.center;
            for (int k = 0; k < 3; ++k) {
                double Lk = c.L[k];
                Lk += c.Lq[k][0] * delta.x + c.Lq[k][1] * delta.y + c.Lq[k][2] * delta.z;
                child.L[k] += Lk;
                child.Lq[k][0] += c.Lq[k][0];
                child.Lq[k][1] += c.Lq[k][1];
                child.Lq[k][2] += c.Lq[k][2];
            }
            downward_rec(ch);
        }
    }

    void m2l(int src, int dst) const {
        const FmmCell& s = cells_[static_cast<std::size_t>(src)];
        FmmCell& d = cells_[static_cast<std::size_t>(dst)];
        const Vec3<double> u = d.center - s.com;
        const double r2 = dot(u, u);
        const double rn = std::sqrt(r2);
        if (rn == 0.0) return;
        const double inv_r3 = 1.0 / (r2 * rn);
        const double inv_r5 = inv_r3 / r2;
        const double inv_r7 = inv_r5 / r2;
        const double inv_r9 = inv_r7 / r2;

        const double ux = u.x, uy = u.y, uz = u.z;
        // Monopole M2L.
        d.L[0] += G_ * s.mass * ux * inv_r3;
        d.L[1] += G_ * s.mass * uy * inv_r3;
        d.L[2] += G_ * s.mass * uz * inv_r3;
        d.Lq[0][0] += -G_ * s.mass * (3.0 * ux * ux * inv_r5 - inv_r3);
        d.Lq[0][1] += -G_ * s.mass * (3.0 * ux * uy * inv_r5);
        d.Lq[0][2] += -G_ * s.mass * (3.0 * ux * uz * inv_r5);
        d.Lq[1][1] += -G_ * s.mass * (3.0 * uy * uy * inv_r5 - inv_r3);
        d.Lq[1][2] += -G_ * s.mass * (3.0 * uy * uz * inv_r5);
        d.Lq[2][2] += -G_ * s.mass * (3.0 * uz * uz * inv_r5 - inv_r3);

        // Quadrupole M2L.
        const double qx = s.quad.m[0][0] * ux + s.quad.m[0][1] * uy + s.quad.m[0][2] * uz;
        const double qy = s.quad.m[1][0] * ux + s.quad.m[1][1] * uy + s.quad.m[1][2] * uz;
        const double qz = s.quad.m[2][0] * ux + s.quad.m[2][1] * uy + s.quad.m[2][2] * uz;
        const double rQr = qx * ux + qy * uy + qz * uz;
        d.L[0] += -G_ * (qx * inv_r5 - 2.5 * rQr * ux * inv_r7);
        d.L[1] += -G_ * (qy * inv_r5 - 2.5 * rQr * uy * inv_r7);
        d.L[2] += -G_ * (qz * inv_r5 - 2.5 * rQr * uz * inv_r7);
        d.Lq[0][0] += -G_ * (s.quad.m[0][0] * inv_r5 - 10.0 * qx * ux * inv_r7 - 2.5 * rQr * inv_r7 + 17.5 * rQr * ux * ux * inv_r9);
        d.Lq[0][1] += -G_ * (s.quad.m[0][1] * inv_r5 - 5.0 * qx * uy * inv_r7 - 5.0 * qy * ux * inv_r7 + 17.5 * rQr * ux * uy * inv_r9);
        d.Lq[0][2] += -G_ * (s.quad.m[0][2] * inv_r5 - 5.0 * qx * uz * inv_r7 - 5.0 * qz * ux * inv_r7 + 17.5 * rQr * ux * uz * inv_r9);
        d.Lq[1][1] += -G_ * (s.quad.m[1][1] * inv_r5 - 10.0 * qy * uy * inv_r7 - 2.5 * rQr * inv_r7 + 17.5 * rQr * uy * uy * inv_r9);
        d.Lq[1][2] += -G_ * (s.quad.m[1][2] * inv_r5 - 5.0 * qy * uz * inv_r7 - 5.0 * qz * uy * inv_r7 + 17.5 * rQr * uy * uz * inv_r9);
        d.Lq[2][2] += -G_ * (s.quad.m[2][2] * inv_r5 - 10.0 * qz * uz * inv_r7 - 2.5 * rQr * inv_r7 + 17.5 * rQr * uz * uz * inv_r9);
    }

    void evaluate(const std::vector<Vec3<double>>& pos, std::vector<Vec3<double>>& acc) const {
        for (std::size_t b = 0; b < pos.size(); ++b) {
            const int leaf = find_leaf(0, pos[b]);
            const FmmCell& c = cells_[static_cast<std::size_t>(leaf)];
            const Vec3<double> dx = pos[b] - c.center;
            Vec3<double> a{};
            a.x -= c.L[0] + c.Lq[0][0] * dx.x + c.Lq[0][1] * dx.y + c.Lq[0][2] * dx.z;
            a.y -= c.L[1] + c.Lq[1][0] * dx.x + c.Lq[1][1] * dx.y + c.Lq[1][2] * dx.z;
            a.z -= c.L[2] + c.Lq[2][0] * dx.x + c.Lq[2][1] * dx.y + c.Lq[2][2] * dx.z;
            for (int b2 : c.near) {
                if (b2 == static_cast<int>(b)) continue;
                const Vec3<double> dr = pos[static_cast<std::size_t>(b2)] - pos[b];
                const double r2 = dot(dr, dr);
                const double rn = std::sqrt(r2);
                if (rn == 0.0) continue;
                a += (G_ * mass_[static_cast<std::size_t>(b2)] / (r2 * rn)) * dr;
            }
            acc[b] = a;
        }
    }

    int find_leaf(int node, const Vec3<double>& p) const {
        const FmmCell& c = cells_[static_cast<std::size_t>(node)];
        if (!c.bodies.empty()) return node;
        int o = (p.x >= c.center.x ? 4 : 0) | (p.y >= c.center.y ? 2 : 0) |
                (p.z >= c.center.z ? 1 : 0);
        if (c.children[o] < 0) return node;
        return find_leaf(c.children[o], p);
    }
};

}  // namespace cassini::gravity
