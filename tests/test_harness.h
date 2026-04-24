// Minimal test harness so we don't need GoogleTest as a hard dep in this
// skeleton. Each test file registers itself; test_main runs them all.
#ifndef IERR_TEST_HARNESS_H
#define IERR_TEST_HARNESS_H

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ierr_test {

struct TestCase { std::string name; std::function<void()> fn; };
inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> r;
    return r;
}
struct Register {
    Register(const char* name, std::function<void()> fn) {
        registry().push_back({name, std::move(fn)});
    }
};

#define IERR_TEST(NAME)                                                     \
    static void NAME();                                                     \
    static ::ierr_test::Register _reg_##NAME(#NAME, &NAME);                 \
    static void NAME()

#define IERR_EXPECT(COND)                                                   \
    do { if (!(COND)) {                                                     \
        std::fprintf(stderr, "  EXPECT failed: %s @ %s:%d\n",               \
                     #COND, __FILE__, __LINE__);                            \
        throw std::runtime_error("expectation failed");                     \
    } } while (0)

#define IERR_EXPECT_EQ(A, B)                                                \
    do { auto _a = (A); auto _b = (B);                                      \
        if (!(_a == _b)) {                                                  \
            std::fprintf(stderr, "  EXPECT_EQ failed @ %s:%d\n",            \
                         __FILE__, __LINE__);                               \
            throw std::runtime_error("expectation failed");                 \
        } } while (0)

#define IERR_EXPECT_NEAR(A, B, EPS)                                         \
    do { double _a = (A); double _b = (B); double _e = (EPS);               \
        double _d = _a - _b; if (_d < 0) _d = -_d;                          \
        if (_d > _e) {                                                      \
            std::fprintf(stderr,                                            \
                "  EXPECT_NEAR failed: %g vs %g (diff %g > %g) @ %s:%d\n",  \
                _a, _b, _d, _e, __FILE__, __LINE__);                        \
            throw std::runtime_error("expectation failed");                 \
        } } while (0)

} // namespace ierr_test

#endif
