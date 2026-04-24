# AArch64 (ARM 64-bit) Linux cross-compilation toolchain for IREE runtime
#
# Usage:
#   cmake -DCMAKE_TOOLCHAIN_FILE=toolchains/aarch64-linux.cmake \
#         -DIREE_SOURCE_DIR=/path/to/iree \
#         -B build-aarch64
#
# Prerequisites (Debian/Ubuntu):
#   sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

set(CMAKE_SYSTEM_NAME      Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# ── Compiler selection ───────────────────────────────────────────────────────
# Override with environment variables or CMake cache entries if your toolchain
# lives in a non-standard location.
set(TOOLCHAIN_PREFIX "aarch64-linux-gnu-"
    CACHE STRING "Cross-compiler prefix")

find_program(CMAKE_C_COMPILER   ${TOOLCHAIN_PREFIX}gcc  REQUIRED)
find_program(CMAKE_CXX_COMPILER ${TOOLCHAIN_PREFIX}g++  REQUIRED)
find_program(CMAKE_AR           ${TOOLCHAIN_PREFIX}ar   REQUIRED)
find_program(CMAKE_RANLIB       ${TOOLCHAIN_PREFIX}ranlib REQUIRED)
find_program(CMAKE_STRIP        ${TOOLCHAIN_PREFIX}strip  REQUIRED)

# ── Sysroot (optional) ───────────────────────────────────────────────────────
# Point CMAKE_SYSROOT at the target device's root filesystem when you have one.
# Leaving it unset works for bare-metal or simple Linux targets.
# set(CMAKE_SYSROOT /path/to/sysroot)

# ── Search path policies ─────────────────────────────────────────────────────
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# ── ARM architecture flags ───────────────────────────────────────────────────
set(CMAKE_C_FLAGS_INIT
    "-march=armv8-a+crypto -mtune=cortex-a55")
set(CMAKE_CXX_FLAGS_INIT "${CMAKE_C_FLAGS_INIT}")
