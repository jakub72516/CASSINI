// Cassini — minimal SPICE SPK (DAF) reader/writer for JPL DE440 ephemerides.
//
// Implements the DAF container (header + linked summary records) and SPK data
// types 2 and 3 (Chebyshev position, optionally with velocity). Positions are
// returned in the segment's native units (km / km/s for DE440); callers convert
// to AU / AU-day. Velocities for Type 2 are obtained by differentiating the
// Chebyshev polynomials.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "cassini/core/vec3.hpp"

namespace cassini::data {

enum class Endian { Little, Big };

inline Endian endian_from_string(const char* s) {
    // "LTL-IEEE" -> little, "BIG-IEEE" -> big.
    return (s[0] == 'B') ? Endian::Big : Endian::Little;
}

// Byte-level readers (offset in bytes).
inline double read_f64(const std::vector<std::uint8_t>& b, std::size_t off, Endian e) {
    std::uint64_t u = 0;
    if (e == Endian::Little) {
        for (int i = 7; i >= 0; --i) u = (u << 8) | b[off + i];
    } else {
        for (int i = 0; i < 8; ++i) u = (u << 8) | b[off + i];
    }
    double d;
    std::memcpy(&d, &u, 8);
    return d;
}

inline std::int32_t read_i32(const std::vector<std::uint8_t>& b, std::size_t off, Endian e) {
    if (e == Endian::Little) {
        return static_cast<std::int32_t>(b[off] | (b[off + 1] << 8) | (b[off + 2] << 16) |
                                         (b[off + 3] << 24));
    }
    return static_cast<std::int32_t>((b[off] << 24) | (b[off + 1] << 16) | (b[off + 2] << 8) |
                                     b[off + 3]);
}

inline std::uint64_t pack_i32_i32(std::int32_t lo, std::int32_t hi, Endian e) {
    std::uint32_t a = static_cast<std::uint32_t>(lo);
    std::uint32_t b = static_cast<std::uint32_t>(hi);
    std::uint64_t u = (e == Endian::Little) ? (static_cast<std::uint64_t>(b) << 32 | a)
                                            : (static_cast<std::uint64_t>(a) << 32 | b);
    return u;
}

inline void unpack_i32_i32(std::uint64_t u, std::int32_t& lo, std::int32_t& hi, Endian e) {
    if (e == Endian::Little) {
        lo = static_cast<std::int32_t>(u & 0xffffffffULL);
        hi = static_cast<std::int32_t>(u >> 32);
    } else {
        hi = static_cast<std::int32_t>(u & 0xffffffffULL);
        lo = static_cast<std::int32_t>(u >> 32);
    }
}

struct SpkSegment {
    int target{0};
    int center{0};
    int frame{0};
    int type{0};
    double begin{0.0};   // start epoch (ET seconds past J2000)
    double end{0.0};     // end epoch
    double init{0.0};    // initial epoch of first record
    double intlen{0.0};  // seconds per record
    int rsize{0};        // doubles per record
    int n{0};            // number of records
    int degree{0};
    std::vector<double> rec;  // n * rsize coefficients (records)
};

class SpkFile {
public:
    bool load_from_memory(const std::vector<std::uint8_t>& bytes);
    bool load_from_file(const std::string& path);

    const std::vector<SpkSegment>& segments() const { return segments_; }

    // State of `target` relative to `center` at ET (seconds past J2000 TDB).
    // Returns false if no matching segment covers `et`.
    bool state(int target, int center, double et, Vec3<double>& r, Vec3<double>& v) const;

private:
    Endian endian_{Endian::Little};
    std::vector<SpkSegment> segments_;
};

// --- Reader --------------------------------------------------------------
inline bool SpkFile::load_from_memory(const std::vector<std::uint8_t>& b) {
    segments_.clear();
    if (b.size() < 1024) return false;

    // File record header.
    char id[9]{};
    std::memcpy(id, b.data(), 8);
    if (std::string(id) != "DAF/SPK " && std::string(id) != "NAIF/DAF") return false;

    char fmt[9]{};
    std::memcpy(fmt, b.data() + 88, 8);
    endian_ = endian_from_string(fmt);

    const int nd = read_i32(b, 8, endian_);
    const int ni = read_i32(b, 12, endian_);
    const int fward = read_i32(b, 76, endian_);  // first summary record (1-based)
    (void)nd;
    (void)ni;

    const int summary_size = 2 + (6 + 1) / 2;  // 5 doubles per SPK summary
    constexpr int rec_bytes = 1024;

    // Walk the summary linked list.
    int rec_no = fward;
    while (rec_no != 0) {
        const std::size_t off = static_cast<std::size_t>(rec_no - 1) * rec_bytes;
        if (off + rec_bytes > b.size()) break;

        const double next = read_f64(b, off, endian_);
        const double nsum = read_f64(b, off + 16, endian_);
        const int count = static_cast<int>(nsum);

        const std::size_t sum_off = off + 24;
        for (int s = 0; s < count; ++s) {
            const std::size_t base = sum_off + static_cast<std::size_t>(s) * summary_size * 8;
            SpkSegment seg;
            seg.begin = read_f64(b, base, endian_);
            seg.end = read_f64(b, base + 8, endian_);

            const std::uint64_t dc2 = [&] {
                double d = read_f64(b, base + 16, endian_);
                std::uint64_t u;
                std::memcpy(&u, &d, 8);
                return u;
            }();
            const std::uint64_t dc3 = [&] {
                double d = read_f64(b, base + 24, endian_);
                std::uint64_t u;
                std::memcpy(&u, &d, 8);
                return u;
            }();
            const std::uint64_t dc4 = [&] {
                double d = read_f64(b, base + 32, endian_);
                std::uint64_t u;
                std::memcpy(&u, &d, 8);
                return u;
            }();

            std::int32_t lo, hi;
            unpack_i32_i32(dc2, lo, hi, endian_);
            seg.target = lo;
            seg.center = hi;
            unpack_i32_i32(dc3, lo, hi, endian_);
            seg.frame = lo;
            seg.type = hi;
            unpack_i32_i32(dc4, lo, hi, endian_);
            const int begin_addr = lo;  // 1-based DAF address
            const int end_addr = hi;

            // Read segment data.
            const std::size_t data_off = static_cast<std::size_t>(begin_addr - 1) * 8;
            const std::size_t n_doubles = static_cast<std::size_t>(end_addr - begin_addr + 1);

            // Directory is the last 4 doubles: INIT, INTLEN, RSIZE, N.
            seg.init = read_f64(b, data_off + (n_doubles - 4) * 8, endian_);
            seg.intlen = read_f64(b, data_off + (n_doubles - 3) * 8, endian_);
            seg.rsize = static_cast<int>(read_f64(b, data_off + (n_doubles - 2) * 8, endian_));
            seg.n = static_cast<int>(read_f64(b, data_off + (n_doubles - 1) * 8, endian_));

            const int comps = (seg.type == 3) ? 6 : 3;
            seg.degree = (seg.rsize - 2) / comps - 1;

            seg.rec.resize(static_cast<std::size_t>(seg.rsize) * seg.n);
            for (std::size_t k = 0; k < seg.rec.size(); ++k) {
                seg.rec[k] = read_f64(b, data_off + k * 8, endian_);
            }
            segments_.push_back(std::move(seg));
        }

        const int nrec = static_cast<int>(next);
        rec_no = nrec;
    }
    return !segments_.empty();
}

inline bool SpkFile::load_from_file(const std::string& path) {
    std::vector<std::uint8_t> bytes;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    const long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (sz > 0) {
        bytes.resize(static_cast<std::size_t>(sz));
        const std::size_t got = std::fread(bytes.data(), 1, bytes.size(), f);
        if (got != bytes.size()) {
            std::fclose(f);
            return false;
        }
    }
    std::fclose(f);
    return load_from_memory(bytes);
}

namespace detail {
// Evaluate Chebyshev series sum_{k=0}^n c[k] T_k(x), returning value and
// derivative w.r.t. x (via direct recurrence — clear and O(n)).
inline void chebyshev(const double* c, int n, double x, double& val, double& der) {
    if (n == 0) {
        val = c[0];
        der = 0.0;
        return;
    }
    double t0 = 1.0, t1 = x;         // T_0, T_1
    double d0 = 0.0, d1 = 1.0;       // T_0', T_1'
    val = c[0] * t0 + c[1] * t1;
    der = c[0] * d0 + c[1] * d1;
    for (int k = 2; k <= n; ++k) {
        const double t2 = 2.0 * x * t1 - t0;
        const double d2 = 2.0 * t1 + 2.0 * x * d1 - d0;
        val += c[k] * t2;
        der += c[k] * d2;
        t0 = t1; t1 = t2;
        d0 = d1; d1 = d2;
    }
}
}  // namespace detail

inline bool SpkFile::state(int target, int center, double et, Vec3<double>& r,
                           Vec3<double>& v) const {
    for (const auto& seg : segments_) {
        if (seg.target != target || seg.center != center) continue;
        if (et < seg.begin || et > seg.end) continue;

        int j = static_cast<int>((et - seg.init) / seg.intlen);
        if (j < 0) j = 0;
        if (j >= seg.n) j = seg.n - 1;

        const double* rec = seg.rec.data() + static_cast<std::size_t>(j) * seg.rsize;
        const double mid = rec[0];
        const double radius = rec[1];
        const double s = (et - mid) / radius;

        const int np = seg.degree + 1;
        if (seg.type == 3) {
            const double* px = rec + 2;
            const double* py = px + np;
            const double* pz = py + np;
            const double* vx = pz + np;
            const double* vy = vx + np;
            const double* vz = vy + np;
            double junk;
            double rx, ry, rz, vvx, vvy, vvz;
            detail::chebyshev(px, seg.degree, s, rx, junk);
            detail::chebyshev(py, seg.degree, s, ry, junk);
            detail::chebyshev(pz, seg.degree, s, rz, junk);
            detail::chebyshev(vx, seg.degree, s, vvx, junk);
            detail::chebyshev(vy, seg.degree, s, vvy, junk);
            detail::chebyshev(vz, seg.degree, s, vvz, junk);
            r = {rx, ry, rz};
            v = {vvx, vvy, vvz};
        } else {
            const double* px = rec + 2;
            const double* py = px + np;
            const double* pz = py + np;
            double dx, dy, dz;
            double rx, ry, rz;
            detail::chebyshev(px, seg.degree, s, rx, dx);
            detail::chebyshev(py, seg.degree, s, ry, dy);
            detail::chebyshev(pz, seg.degree, s, rz, dz);
            r = {rx, ry, rz};
            const double dsdt = 1.0 / radius;  // s in [-1,1] over 2*radius sec
            v = {dx * dsdt, dy * dsdt, dz * dsdt};
        }
        return true;
    }
    return false;
}

// --- Writer (Type 2, for export and round-trip tests) --------------------
inline void write_f64(std::vector<std::uint8_t>& b, std::size_t off, double d, Endian e) {
    std::uint64_t u;
    std::memcpy(&u, &d, 8);
    if (e == Endian::Little) {
        for (int i = 0; i < 8; ++i) b[off + i] = static_cast<std::uint8_t>(u >> (8 * i));
    } else {
        for (int i = 0; i < 8; ++i) b[off + i] = static_cast<std::uint8_t>(u >> (8 * (7 - i)));
    }
}

inline void write_i32(std::vector<std::uint8_t>& b, std::size_t off, std::int32_t v, Endian e) {
    const std::uint32_t u = static_cast<std::uint32_t>(v);
    if (e == Endian::Little) {
        for (int i = 0; i < 4; ++i) b[off + i] = static_cast<std::uint8_t>(u >> (8 * i));
    } else {
        for (int i = 0; i < 4; ++i) b[off + i] = static_cast<std::uint8_t>(u >> (8 * (3 - i)));
    }
}

inline void write_u64(std::vector<std::uint8_t>& b, std::size_t off, std::uint64_t u, Endian e) {
    if (e == Endian::Little) {
        for (int i = 0; i < 8; ++i) b[off + i] = static_cast<std::uint8_t>(u >> (8 * i));
    } else {
        for (int i = 0; i < 8; ++i) b[off + i] = static_cast<std::uint8_t>(u >> (8 * (7 - i)));
    }
}

// Build a minimal single-segment Type 2 SPK file (little-endian).
// `records` holds n * rsize doubles; each record is [MID, RADIUS, X.., Y.., Z..].
inline std::vector<std::uint8_t> make_type2_spk(int target, int center, int frame,
                                                double begin, double end, double init,
                                                double intlen, int rsize,
                                                const std::vector<double>& records,
                                                Endian e = Endian::Little) {
    const std::size_t rec_bytes = 1024;
    const std::size_t n_records = records.size() / static_cast<std::size_t>(rsize);
    const std::size_t n_doubles = records.size() + 4;  // + directory
    const std::size_t total_doubles = 128 + 128 + n_doubles;
    std::vector<std::uint8_t> b(total_doubles * 8, 0);

    // Header record.
    const char* idword = "DAF/SPK ";
    std::memcpy(b.data(), idword, 8);
    write_i32(b, 8, 2, e);    // ND
    write_i32(b, 12, 6, e);   // NI
    std::memcpy(b.data() + 16, "CASSINI", 7);
    write_i32(b, 76, 2, e);   // FWARD (summary = record 2)
    write_i32(b, 80, 2, e);   // BWARD
    const std::int32_t free_addr = static_cast<std::int32_t>(total_doubles + 1);
    write_i32(b, 84, free_addr, e);  // FREE
    const char* fmt = (e == Endian::Little) ? "LTL-IEEE" : "BIG-IEEE";
    std::memcpy(b.data() + 88, fmt, 8);

    // Summary record.
    const std::size_t sbase = rec_bytes;
    write_f64(b, sbase, 0.0, e);         // NEXT
    write_f64(b, sbase + 8, 0.0, e);     // PREV
    write_f64(b, sbase + 16, 1.0, e);    // NSUM
    const std::size_t sum = sbase + 24;
    write_f64(b, sum, begin, e);
    write_f64(b, sum + 8, end, e);
    const std::int32_t begin_addr = 257;
    const std::int32_t end_addr = 257 + static_cast<std::int32_t>(n_doubles) - 1;
    {
        write_u64(b, sum + 16, pack_i32_i32(target, center, e), e);
        write_u64(b, sum + 24, pack_i32_i32(frame, 2, e), e);
        write_u64(b, sum + 32, pack_i32_i32(begin_addr, end_addr, e), e);
    }

    // Segment data.
    const std::size_t doff = static_cast<std::size_t>(begin_addr - 1) * 8;
    for (std::size_t k = 0; k < records.size(); ++k) {
        write_f64(b, doff + k * 8, records[k], e);
    }
    const std::size_t dir = doff + records.size() * 8;
    write_f64(b, dir, init, e);
    write_f64(b, dir + 8, intlen, e);
    write_f64(b, dir + 16, static_cast<double>(rsize), e);
    write_f64(b, dir + 24, static_cast<double>(n_records), e);

    return b;
}

}  // namespace cassini::data
