// Cassini — small deterministic RNG (PCG32) for reproducible perturbed-orbit
// and chaos-test ensembles. A single 64-bit seed fully determines a run, which
// is required for bit-exact cross-architecture reproducibility of tests.
#pragma once

#include <cstdint>

namespace cassini {

class Pcg32 {
public:
    explicit Pcg32(std::uint64_t seed = 0x853c49e6748fea9bULL) : state_(seed + inc_) {}

    std::uint32_t next() {
        const std::uint64_t old = state_;
        state_ = old * 6364136223846793005ULL + inc_;
        const std::uint32_t xorshifted = static_cast<std::uint32_t>(((old >> 18u) ^ old) >> 27u);
        const std::uint32_t rot = static_cast<std::uint32_t>(old >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31u));
    }

    // Uniform in [0, 1).
    double uniform() {
        return static_cast<double>(next()) * (1.0 / 4294967296.0);
    }

    // Uniform in [a, b).
    double uniform(double a, double b) { return a + (b - a) * uniform(); }

private:
    std::uint64_t state_;
    static constexpr std::uint64_t inc_ = 1442695040888963407ULL;
};

}  // namespace cassini
