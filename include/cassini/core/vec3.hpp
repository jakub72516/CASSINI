// Cassini — minimal 3-vector with scalar type templating (double / dd_real).
#pragma once

#include <cmath>

namespace cassini {

template <typename T>
struct Vec3 {
    T x{0};
    T y{0};
    T z{0};

    Vec3() = default;
    Vec3(T vx, T vy, T vz) : x(vx), y(vy), z(vz) {}

    Vec3 operator-() const { return {-x, -y, -z}; }
    Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    Vec3& operator-=(const Vec3& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
    Vec3& operator*=(T s) { x *= s; y *= s; z *= s; return *this; }
    Vec3& operator/=(T s) { x /= s; y /= s; z /= s; return *this; }
};

template <typename T>
inline Vec3<T> operator+(Vec3<T> a, const Vec3<T>& b) { a += b; return a; }
template <typename T>
inline Vec3<T> operator-(Vec3<T> a, const Vec3<T>& b) { a -= b; return a; }
template <typename T>
inline Vec3<T> operator*(Vec3<T> a, T s) { a *= s; return a; }
template <typename T>
inline Vec3<T> operator*(T s, Vec3<T> a) { a *= s; return a; }
template <typename T>
inline Vec3<T> operator/(Vec3<T> a, T s) { a /= s; return a; }

template <typename T>
inline T dot(const Vec3<T>& a, const Vec3<T>& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
template <typename T>
inline Vec3<T> cross(const Vec3<T>& a, const Vec3<T>& b) {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}
template <typename T>
inline T norm_sq(const Vec3<T>& a) { return dot(a, a); }
template <typename T>
inline double norm(const Vec3<T>& a) { return std::sqrt(static_cast<double>(norm_sq(a))); }

}  // namespace cassini
