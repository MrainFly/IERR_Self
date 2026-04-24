cmake_minimum_required(VERSION 3.21)

project(iree_arm_runtime
  VERSION 1.0.0
  DESCRIPTION "ARM-based IREE runtime implementation"
  LANGUAGES C CXX
)

# ── C / C++ standards ────────────────────────────────────────────────────────
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Build type ───────────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ── ARM architecture flags ───────────────────────────────────────────────────
# Detect whether we are targeting AArch64 or ARMv7.
# These flags are applied only when the target processor is ARM.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)")
  message(STATUS "Target: AArch64 – enabling NEON / crypto extensions")
  add_compile_options(
    -march=armv8-a+crypto
    -mtune=cortex-a55
  )
  add_compile_definitions(IREE_ARM_ARCH_AARCH64=1)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^arm")
  message(STATUS "Target: ARMv7 – enabling NEON")
  add_compile_options(
    -march=armv7-a
    -mfpu=neon-vfpv4
    -mfloat-abi=hard
  )
  add_compile_definitions(IREE_ARM_ARCH_ARMV7=1)
else()
  message(STATUS "Target: ${CMAKE_SYSTEM_PROCESSOR} (non-ARM host build)")
endif()

# ── IREE dependency ──────────────────────────────────────────────────────────
# Prefer a pre-installed IREE package.  Fall back to a source-tree build by
# pointing IREE_SOURCE_DIR at the IREE repo root.
option(IREE_SOURCE_DIR "" "Path to the IREE source directory (optional)")

if(IREE_SOURCE_DIR)
  # --------------------------------------------------------------------------
  # Source-tree build: pull IREE in as a sub-project.
  # Only the runtime + local-task HAL driver are enabled to keep build times
  # short.
  # --------------------------------------------------------------------------
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
  # --------------------------------------------------------------------------
  # Installed package build: use find_package.
  # Set IREE_DIR (or CMAKE_PREFIX_PATH) to the directory containing
  # iree-config.cmake produced by `cmake --install`.
  # --------------------------------------------------------------------------
  find_package(iree REQUIRED)

  set(IREE_RUNTIME_LIBS
    iree::runtime
    iree::hal::drivers::local_task
  )
endif()

# ── Library target ───────────────────────────────────────────────────────────
add_library(iree_arm_runtime STATIC
  src/iree_arm_runtime.c
)

target_include_directories(iree_arm_runtime
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(iree_arm_runtime
  PUBLIC
    ${IREE_RUNTIME_LIBS}
)

# ── Example executable ───────────────────────────────────────────────────────
add_executable(iree_arm_example
  src/main.c
)

target_link_libraries(iree_arm_example
  PRIVATE
    iree_arm_runtime
)

# ── Installation rules ───────────────────────────────────────────────────────
include(GNUInstallDirs)

install(TARGETS iree_arm_runtime iree_arm_example
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(FILES src/iree_arm_runtime.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
