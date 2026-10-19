// Cassini — double-double (approx. 106-bit) floating-point arithmetic.
//
// Implements the classic "QD" two-sum / two-prod / quick-two-sum algorithms
// (Dekker, Knuth, Bailey). Dekker splitting is used for exact product errors so
// that results are FMA-independent and bit-identical across x86-64 and ARM64.
//
// This type is used for *critical accumulations* (energy, angular momentum,
// Lenz-Runge vector) where 53-bit doubles lose ~15 digits of cancellation over
// 1e8-year integrations. It is deliberately small and dependency-free.
#pragma once

#include <cmath>
#include <cstdint>
#include <iosfwd>
#include <limits>

namespace cassini {

struct dd_real {
    double hi{0.0};  // most significant part
    double lo{0.0};  // residual

    dd_real() = default;
    dd_real(double h, double l) : hi(h), lo(l) {}
    dd_real(double x) : hi(x), lo(0.0) {}          // NOLINT(hicpp-explicit-conversions)
    dd_real(int x) : hi(static_cast<double>(x)), lo(0.0) {}
    dd_real(std::int64_t x) : hi(static_cast<double>(x)), lo(0.0) {}
};

// ---------------------------------------------------------------------------
// Exact primitives (error-free transformations)
// ---------------------------------------------------------------------------
namespace detail {
// (s, e) = a + b exactly, where s = fl(a+b).
inline void two_sum(double a, double b, double& s, double& e) {
    s = a + b;
    const double bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}

// Requires |a| >= |b|.
inline void quick_two_sum(double a, double b, double& s, double& e) {
    s = a + b;
    e = b - (s - a);
}

// (p, e) = a * b exactly via Dekker splitting (FMA-independent).
inline void two_prod(double a, double b, double& p, double& e) {
    constexpr double split = 134217729.0;  // 2^27 + 1
    p = a * b;
    const double ah = a * split;
    const double ahi = ah - (ah - a);
    const double alo = a - ahi;
    const double bh = b * split;
    const double bhi = bh - (bh - b);
    const double blo = b - bhi;
    e = ((ahi * bhi - p) + ahi * blo + alo * bhi) + alo * blo;
}

inline dd_real dd_add(double ah, double al, double bh, double bl) {
    double s1, s2;
    two_sum(ah, bh, s1, s2);
    s2 += al + bl;
    double hi, lo;
    quick_two_sum(s1, s2, hi, lo);
    return {hi, lo};
}

inline dd_real dd_mul(double ah, double al, double bh, double bl) {
    double p1, p2;
    two_prod(ah, bh, p1, p2);
    p2 += ah * bl + al * bh;
    double hi, lo;
    quick_two_sum(p1, p2, hi, lo);
    return {hi, lo};
}
}  // namespace detail

// ---------------------------------------------------------------------------
// Arithmetic operators
// ---------------------------------------------------------------------------
inline dd_real operator+(dd_real a, dd_real b) {
    return detail::dd_add(a.hi, a.lo, b.hi, b.lo);
}
inline dd_real operator-(dd_real a, dd_real b) {
    return detail::dd_add(a.hi, a.lo, -b.hi, -b.lo);
}
inline dd_real operator*(dd_real a, dd_real b) {
    return detail::dd_mul(a.hi, a.lo, b.hi, b.lo);
}
inline dd_real operator/(dd_real a, dd_real b) {
    // One Newton refinement of q1 = a.hi / b.hi gives ~106 correct bits.
    const double q1 = a.hi / b.hi;
    const dd_real r = a - b * q1;
    const double q2 = r.hi / b.hi;
    return detail::dd_add(q1, 0.0, q2, 0.0);
}
inline dd_real operator-(dd_real a) { return {-a.hi, -a.lo}; }
inline dd_real operator+(dd_real a) { return a; }

inline dd_real& operator+=(dd_real& a, dd_real b) { a = a + b; return a; }
inline dd_real& operator-=(dd_real& a, dd_real b) { a = a - b; return a; }
inline dd_real& operator*=(dd_real& a, dd_real b) { a = a * b; return a; }
inline dd_real& operator/=(dd_real& a, dd_real b) { a = a / b; return a; }

// ---------------------------------------------------------------------------
// Comparisons
// ---------------------------------------------------------------------------
inline bool operator==(dd_real a, dd_real b) { return a.hi == b.hi && a.lo == b.lo; }
inline bool operator!=(dd_real a, dd_real b) { return !(a == b); }
inline bool operator<(dd_real a, dd_real b) { return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo); }
inline bool operator>(dd_real a, dd_real b) { return b < a; }
inline bool operator<=(dd_real a, dd_real b) { return !(b < a); }
inline bool operator>=(dd_real a, dd_real b) { return !(a < b); }

// ---------------------------------------------------------------------------
// Mathematical functions
// ---------------------------------------------------------------------------
inline dd_real abs(dd_real a) { return (a.hi < 0.0) ? -a : a; }
inline bool isfinite(dd_real a) { return std::isfinite(a.hi); }
inline double to_double(dd_real a) { return a.hi; }

inline dd_real sqrt(dd_real a) {
    if (a.hi <= 0.0) {
        return {std::sqrt(a.hi), 0.0};
    }
    const double x0 = std::sqrt(a.hi);
    // Refine: x = x0 + (a - x0^2) / (2 x0)
    const dd_real t = (a - x0 * x0);
    const double corr = t.hi / (2.0 * x0);
    return detail::dd_add(x0, 0.0, corr, 0.0);
}

// ---------------------------------------------------------------------------
// Free functions to make dd_real usable as a generic scalar in templates
// ---------------------------------------------------------------------------
inline dd_real sqr(dd_real a) { return a * a; }
inline dd_real sqr(double a) { return a * a; }
inline double sqrt(double a) { return std::sqrt(a); }
inline double abs(double a) { return std::fabs(a); }

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------
inline std::ostream& operator<<(std::ostream& os, dd_real a) {
    return os << a.hi << '+' << a.lo;
}

}  // namespace cassini

// ---------------------------------------------------------------------------
// numeric_limits specialization
// ---------------------------------------------------------------------------
namespace std {
template <>
class numeric_limits<cassini::dd_real> {
public:
    static constexpr bool is_specialized = true;
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr int radix = 2;
    static constexpr int digits = 106;
    static constexpr int max_exponent = numeric_limits<double>::max_exponent;
    static constexpr int min_exponent = numeric_limits<double>::min_exponent;
    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = false;
    static constexpr float_round_style round_style = round_to_nearest;

    static cassini::dd_real epsilon() { return {std::ldexp(1.0, -104), 0.0}; }
    static cassini::dd_real min() { return {std::numeric_limits<double>::min(), 0.0}; }
    static cassini::dd_real max() { return {std::numeric_limits<double>::max(), 0.0}; }
    static cassini::dd_real infinity() { return {std::numeric_limits<double>::infinity(), 0.0}; }
    static cassini::dd_real quiet_NaN() { return {std::numeric_limits<double>::quiet_NaN(), 0.0}; }
};
}  // namespace std
