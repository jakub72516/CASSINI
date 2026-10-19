// Cassini — versioned binary snapshot format with explicit endianness and a
// CRC32 integrity check. Used for checkpoint/restart; round-trips are bit-exact
// across x86-64 and ARM64 because all scalars are serialized as raw IEEE-754
// bits in an explicit byte order (no arithmetic on load/save).
#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

#include "cassini/physics/system.hpp"

namespace cassini::data {

inline constexpr std::uint32_t SNAPSHOT_MAGIC = 0x4341534e;  // "CASN"
inline constexpr std::uint32_t SNAPSHOT_VERSION = 1;
inline constexpr std::uint32_t ENDIAN_LITTLE = 1;
inline constexpr std::uint32_t ENDIAN_BIG = 2;

struct SnapshotHeader {
    std::uint32_t magic{SNAPSHOT_MAGIC};
    std::uint32_t version{SNAPSHOT_VERSION};
    std::uint32_t endian{ENDIAN_LITTLE};
    std::uint32_t flags{0};
    std::uint64_t n_bodies{0};
    double time{0.0};
    std::uint32_t crc{0};
    std::uint32_t reserved{0};
};

// --- CRC32 (IEEE 802.3) --------------------------------------------------
inline std::uint32_t crc32(const std::uint8_t* data, std::size_t len) {
    std::uint32_t crc = 0xFFFFFFFFu;
    for (std::size_t i = 0; i < len; ++i) {
        crc ^= data[i];
        for (int k = 0; k < 8; ++k) {
            crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
        }
    }
    return ~crc;
}

// --- Serialization helpers -----------------------------------------------
namespace detail {
inline bool little_endian_host() {
    const std::uint16_t v = 1;
    return *reinterpret_cast<const std::uint8_t*>(&v) == 1;
}

inline void put_u32(std::vector<std::uint8_t>& b, std::size_t off, std::uint32_t v, bool le) {
    for (int i = 0; i < 4; ++i) {
        b[off + i] = static_cast<std::uint8_t>(v >> (8 * (le ? i : 3 - i)));
    }
}
inline std::uint32_t get_u32(const std::vector<std::uint8_t>& b, std::size_t off, bool le) {
    std::uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= static_cast<std::uint32_t>(b[off + i]) << (8 * (le ? i : 3 - i));
    return v;
}
inline void put_u64(std::vector<std::uint8_t>& b, std::size_t off, std::uint64_t v, bool le) {
    for (int i = 0; i < 8; ++i) b[off + i] = static_cast<std::uint8_t>(v >> (8 * (le ? i : 7 - i)));
}
inline std::uint64_t get_u64(const std::vector<std::uint8_t>& b, std::size_t off, bool le) {
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= static_cast<std::uint64_t>(b[off + i]) << (8 * (le ? i : 7 - i));
    return v;
}
inline void put_f64(std::vector<std::uint8_t>& b, std::size_t off, double d, bool le) {
    std::uint64_t u;
    std::memcpy(&u, &d, 8);
    put_u64(b, off, u, le);
}
inline double get_f64(const std::vector<std::uint8_t>& b, std::size_t off, bool le) {
    const std::uint64_t u = get_u64(b, off, le);
    double d;
    std::memcpy(&d, &u, 8);
    return d;
}
}  // namespace detail

// --- Snapshot read/write -------------------------------------------------
inline std::size_t snapshot_body_size(std::size_t n) {
    // Per body: mass, 3 pos, 3 vel, radius, j2, j2_ref = 10 doubles.
    return n * 10 * sizeof(double);
}

// Write the snapshot into `out`. Always little-endian output (portable).
inline void write_snapshot(const cassini::System& sys, std::vector<std::uint8_t>& out) {
    const std::size_t n = sys.size();
    const std::size_t header_sz = sizeof(SnapshotHeader);
    const std::size_t body_sz = snapshot_body_size(n);
    out.assign(header_sz + body_sz, 0);
    const bool le = true;

    const std::size_t off = 0;
    detail::put_u32(out, off + 0, SNAPSHOT_MAGIC, le);
    detail::put_u32(out, off + 4, SNAPSHOT_VERSION, le);
    detail::put_u32(out, off + 8, ENDIAN_LITTLE, le);
    detail::put_u32(out, off + 12, 0, le);
    detail::put_u64(out, off + 16, static_cast<std::uint64_t>(n), le);
    detail::put_f64(out, off + 24, sys.t, le);

    std::size_t p = header_sz;
    for (std::size_t i = 0; i < n; ++i) {
        detail::put_f64(out, p, sys.mass[i], le); p += 8;
        detail::put_f64(out, p, sys.pos[i].x, le); p += 8;
        detail::put_f64(out, p, sys.pos[i].y, le); p += 8;
        detail::put_f64(out, p, sys.pos[i].z, le); p += 8;
        detail::put_f64(out, p, sys.vel[i].x, le); p += 8;
        detail::put_f64(out, p, sys.vel[i].y, le); p += 8;
        detail::put_f64(out, p, sys.vel[i].z, le); p += 8;
        detail::put_f64(out, p, sys.radius[i], le); p += 8;
        detail::put_f64(out, p, sys.j2[i], le); p += 8;
        detail::put_f64(out, p, sys.j2_ref[i], le); p += 8;
    }

    const std::uint32_t crc = crc32(out.data(), body_sz);
    detail::put_u32(out, off + 28, crc, le);
}

// Read a snapshot into `sys`. Returns false on any corruption/mismatch.
inline bool read_snapshot(cassini::System& sys, const std::vector<std::uint8_t>& in) {
    const std::size_t header_sz = sizeof(SnapshotHeader);
    if (in.size() < header_sz) return false;
    const bool le = true;

    if (detail::get_u32(in, 0, le) != SNAPSHOT_MAGIC) return false;
    if (detail::get_u32(in, 4, le) != SNAPSHOT_VERSION) return false;

    const std::uint64_t n = detail::get_u64(in, 16, le);
    const double t = detail::get_f64(in, 24, le);
    const std::uint32_t stored_crc = detail::get_u32(in, 28, le);
    const std::size_t body_sz = snapshot_body_size(static_cast<std::size_t>(n));
    if (in.size() != header_sz + body_sz) return false;
    if (crc32(in.data(), body_sz) != stored_crc) return false;

    sys.mass.resize(static_cast<std::size_t>(n));
    sys.pos.resize(static_cast<std::size_t>(n));
    sys.vel.resize(static_cast<std::size_t>(n));
    sys.radius.resize(static_cast<std::size_t>(n));
    sys.j2.resize(static_cast<std::size_t>(n));
    sys.j2_ref.resize(static_cast<std::size_t>(n));
    sys.t = t;

    std::size_t p = header_sz;
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
        sys.mass[i] = detail::get_f64(in, p, le); p += 8;
        sys.pos[i].x = detail::get_f64(in, p, le); p += 8;
        sys.pos[i].y = detail::get_f64(in, p, le); p += 8;
        sys.pos[i].z = detail::get_f64(in, p, le); p += 8;
        sys.vel[i].x = detail::get_f64(in, p, le); p += 8;
        sys.vel[i].y = detail::get_f64(in, p, le); p += 8;
        sys.vel[i].z = detail::get_f64(in, p, le); p += 8;
        sys.radius[i] = detail::get_f64(in, p, le); p += 8;
        sys.j2[i] = detail::get_f64(in, p, le); p += 8;
        sys.j2_ref[i] = detail::get_f64(in, p, le); p += 8;
    }
    return true;
}

}  // namespace cassini::data
