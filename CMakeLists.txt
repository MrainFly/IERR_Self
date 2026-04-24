cmake_minimum_required(VERSION 3.16)
project(ierr LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

option(IERR_BUILD_TESTS "Build IERR unit tests" ON)

# ---- core runtime library ----
add_library(ierr_runtime STATIC
    src/runtime/runtime.cpp
    src/runtime/graph.cpp
    src/runtime/op_registry.cpp
    src/runtime/logging.cpp
    src/hal/sim/sim_hal.cpp
    src/kernels/cpu/add.cpp
    src/kernels/cpu/matmul.cpp
    src/kernels/cpu/relu.cpp
)
target_include_directories(ierr_runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_include_directories(ierr_runtime PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
find_package(Threads REQUIRED)
target_link_libraries(ierr_runtime PUBLIC Threads::Threads)

# ---- model_runner CLI ----
add_executable(model_runner tools/model_runner/main.cpp)
target_link_libraries(model_runner PRIVATE ierr_runtime)

# ---- tests ----
if(IERR_BUILD_TESTS)
    enable_testing()
    add_executable(ierr_tests
        tests/test_hal_sim.cpp
        tests/test_kernels.cpp
        tests/test_runtime.cpp
        tests/test_main.cpp
    )
    target_include_directories(ierr_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )
    target_link_libraries(ierr_tests PRIVATE ierr_runtime)
    add_test(NAME ierr_tests COMMAND ierr_tests)
endif()
