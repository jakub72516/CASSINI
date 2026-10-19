// Cassini — minimal header-only test framework (no external dependency).
// Provides TEST registration, CHECK/CHECK_NEAR/CHECK_THROWS and a main runner.
#pragma once

#include <cmath>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace cassini_test {

struct TestCase {
    std::string name;
    std::function<void()> fn;
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> cases;
    return cases;
}

struct Registrar {
    Registrar(const char* name, std::function<void()> fn) {
        registry().push_back({name, std::move(fn)});
    }
};

inline int& failures() {
    static int f = 0;
    return f;
}

inline int run_all() {
    int passed = 0;
    for (auto& tc : registry()) {
        const int before = failures();
        try {
            tc.fn();
        } catch (const std::exception& e) {
            std::printf("  [FAIL] %s : unexpected exception: %s\n", tc.name.c_str(), e.what());
            ++failures();
        }
        if (failures() == before) {
            std::printf("  [PASS] %s\n", tc.name.c_str());
            ++passed;
        } else {
            std::printf("  [FAIL] %s\n", tc.name.c_str());
        }
    }
    std::printf("\n%d/%zu tests passed\n", passed, registry().size());
    return failures() == 0 ? 0 : 1;
}

}  // namespace cassini_test

#define CAS_TEST_CAT_(a, b) a##b
#define CAS_TEST_CAT(a, b) CAS_TEST_CAT_(a, b)
#define CAS_TEST_ANON(name) CAS_TEST_CAT(name, __LINE__)

#define TEST(name)                                                    \
    static void CAS_TEST_ANON(cassini_test_fn_)(void);                \
    static ::cassini_test::Registrar CAS_TEST_ANON(cassini_reg_)(     \
        #name, CAS_TEST_ANON(cassini_test_fn_));                      \
    static void CAS_TEST_ANON(cassini_test_fn_)(void)

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            ::cassini_test::failures()++;                                   \
            std::printf("    check failed: %s (%s:%d)\n", #cond, __FILE__,  \
                        __LINE__);                                          \
        }                                                                   \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                 \
    do {                                                                      \
        const double _a = (a);                                                \
        const double _b = (b);                                                \
        if (!(std::fabs(_a - _b) <= (tol))) {                                 \
            ::cassini_test::failures()++;                                     \
            std::printf("    check failed: |%s - %s| = %g > %g (%s:%d)\n",    \
                        #a, #b, std::fabs(_a - _b), (double)(tol), __FILE__,  \
                        __LINE__);                                            \
        }                                                                     \
    } while (0)

#define CHECK_THROWS(expr)                                                    \
    do {                                                                      \
        bool _threw = false;                                                  \
        try {                                                                 \
            (void)(expr);                                                     \
        } catch (...) {                                                       \
            _threw = true;                                                    \
        }                                                                     \
        if (!_threw) {                                                        \
            ::cassini_test::failures()++;                                     \
            std::printf("    expected throw: %s (%s:%d)\n", #expr, __FILE__,  \
                        __LINE__);                                            \
        }                                                                     \
    } while (0)

#define CAS_TEST_MAIN() \
    int main() { return ::cassini_test::run_all(); }
