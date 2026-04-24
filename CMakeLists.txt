cmake_minimum_required(VERSION 3.21)

# This repository hosts two independent runtimes that share the project name:
#
#   1. ierr  – a C++17 inference runtime with a PC-simulator HAL and CPU
#              kernels (Add / ReLU / MatMul). Always built.
#   2. iree_arm_runtime – a thin C wrapper around the upstream IREE runtime
#              for ARM (AArch64 / ARMv7). Optional, requires IREE; disabled
#              by default so the default build stays dependency-free.
#
# Enable the ARM/IREE wrapper with -DIERR_BUILD_IREE_ARM=ON (and either
# -DIREE_SOURCE_DIR=... or -DIREE_DIR=...).

project(ierr
  VERSION 1.0.0
  DESCRIPTION "IERR runtime + optional ARM IREE wrapper"
  LANGUAGES C CXX
)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(IERR_BUILD_TESTS    "Build IERR unit tests"            ON)
option(IERR_BUILD_IREE_ARM "Build the optional ARM IREE wrapper" OFF)

if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ============================================================================
# ierr – simulator-backed inference runtime (C++17)
# ============================================================================
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

add_executable(model_runner tools/model_runner/main.cpp)
target_link_libraries(model_runner PRIVATE ierr_runtime)

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

# ============================================================================
# iree_arm_runtime – optional ARM IREE wrapper (C)
# ============================================================================
if(IERR_BUILD_IREE_ARM)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)")
        message(STATUS "Target: AArch64 – enabling NEON / crypto extensions")
        add_compile_options(-march=armv8-a+crypto -mtune=cortex-a55)
        add_compile_definitions(IREE_ARM_ARCH_AARCH64=1)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^arm")
        message(STATUS "Target: ARMv7 – enabling NEON")
        add_compile_options(-march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard)
        add_compile_definitions(IREE_ARM_ARCH_ARMV7=1)
    else()
        message(STATUS "Target: ${CMAKE_SYSTEM_PROCESSOR} (non-ARM host build)")
    endif()

    set(IREE_SOURCE_DIR "" CACHE PATH "Path to the IREE source directory (optional)")

    if(IREE_SOURCE_DIR)
        message(STATUS "Using IREE from source: ${IREE_SOURCE_DIR}")
        set(IREE_BUILD_COMPILER     OFF CACHE BOOL "" FORCE)
        set(IREE_BUILD_TESTS        OFF CACHE BOOL "" FORCE)
        set(IREE_BUILD_SAMPLES      OFF CACHE BOOL "" FORCE)
        set(IREE_HAL_DRIVER_LOCAL_TASK ON  CACHE BOOL "" FORCE)
        add_subdirectory(${IREE_SOURCE_DIR} iree_build EXCLUDE_FROM_ALL)
        set(IREE_RUNTIME_LIBS
            iree_runtime_runtime
            iree_hal_drivers_local_task_registration
            iree_hal_local_loaders_embedded_elf_loader
            iree_hal_local_loaders_vmvx_module_loader
            iree_task_api
        )
    else()
        find_package(iree REQUIRED)
        set(IREE_RUNTIME_LIBS
            iree::runtime
            iree::hal::drivers::local_task
        )
    endif()

    add_library(iree_arm_runtime STATIC src/iree_arm_runtime.c)
    target_include_directories(iree_arm_runtime
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
            $<INSTALL_INTERFACE:include>
    )
    target_link_libraries(iree_arm_runtime PUBLIC ${IREE_RUNTIME_LIBS})

    add_executable(iree_arm_example src/main.c)
    target_link_libraries(iree_arm_example PRIVATE iree_arm_runtime)

    include(GNUInstallDirs)
    install(TARGETS iree_arm_runtime iree_arm_example
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
    install(FILES src/iree_arm_runtime.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()
