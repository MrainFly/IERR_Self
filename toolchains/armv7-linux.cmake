# ARMv7-A hard-float Linux cross-compilation toolchain for IREE runtime
#
# Usage:
#   cmake -DCMAKE_TOOLCHAIN_FILE=toolchains/armv7-linux.cmake \
#         -DIREE_SOURCE_DIR=/path/to/iree \
#         -B build-armv7
#
# Prerequisites (Debian/Ubuntu):
#   sudo apt-get install gcc-arm-linux-gnueabihf g++-arm-linux-gnueabihf

set(CMAKE_SYSTEM_NAME      Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

# ── Compiler selection ───────────────────────────────────────────────────────
set(TOOLCHAIN_PREFIX "arm-linux-gnueabihf-"
    CACHE STRING "Cross-compiler prefix")

find_program(CMAKE_C_COMPILER   ${TOOLCHAIN_PREFIX}gcc  REQUIRED)
find_program(CMAKE_CXX_COMPILER ${TOOLCHAIN_PREFIX}g++  REQUIRED)
find_program(CMAKE_AR           ${TOOLCHAIN_PREFIX}ar   REQUIRED)
find_program(CMAKE_RANLIB       ${TOOLCHAIN_PREFIX}ranlib REQUIRED)
find_program(CMAKE_STRIP        ${TOOLCHAIN_PREFIX}strip  REQUIRED)

# ── Sysroot (optional) ───────────────────────────────────────────────────────
# set(CMAKE_SYSROOT /path/to/sysroot)

# ── Search path policies ─────────────────────────────────────────────────────
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# ── ARM architecture flags ───────────────────────────────────────────────────
set(CMAKE_C_FLAGS_INIT
    "-march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard -mthumb")
set(CMAKE_CXX_FLAGS_INIT "${CMAKE_C_FLAGS_INIT}")
