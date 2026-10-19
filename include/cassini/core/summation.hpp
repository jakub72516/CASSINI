// Cassini — compensated summation (Kahan and Neumaier).
//
// Neumaier's variant is preferred: it is exact when |sum| >= |term| and only
// accumulates a bounded error otherwise, making it strictly better than Kahan
// for mixed-sign long-running reductions (the N-body energy drift case).
#pragma once

#include "cassini/core/ddouble.hpp"

namespace cassini {

// Neumaier compensated summation over an iterator range.
// Returns a double-double pair for the most precise single-pass result.
template <typename It>
dd_real neumaier_sum(It first, It last) {
    double sum = 0.0;
    double c = 0.0;  // running correction
    for (; first != last; ++first) {
        const double t = sum + *first;
        if (std::abs(sum) >= std::abs(*first)) {
            c += (sum - t) + *first;
        } else {
            c += (*first - t) + sum;
        }
        sum = t;
    }
    return {sum, c};
}

// Kahan summation (simpler, slightly weaker guarantee). Kept for reference and
// for the property tests that cross-check both algorithms.
template <typename It>
double kahan_sum(It first, It last) {
    double sum = 0.0;
    double c = 0.0;
    for (; first != last; ++first) {
        const double y = *first - c;
        const double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
}

}  // namespace cassini
