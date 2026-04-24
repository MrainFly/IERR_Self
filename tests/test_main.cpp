#include <cstdio>
#include <stdexcept>

#include "test_harness.h"

int main() {
    int passed = 0, failed = 0;
    for (auto& t : ierr_test::registry()) {
        std::printf("[ RUN      ] %s\n", t.name.c_str());
        try {
            t.fn();
            std::printf("[       OK ] %s\n", t.name.c_str());
            ++passed;
        } catch (const std::exception& e) {
            std::printf("[  FAILED  ] %s: %s\n", t.name.c_str(), e.what());
            ++failed;
        }
    }
    std::printf("[==========] %d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
