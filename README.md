# IERR\_Self – ARM IREE Runtime

A lightweight C wrapper that embeds the [IREE](https://github.com/openxla/iree)
machine-learning runtime on **ARM** devices (AArch64 / ARMv7).  It exposes a
simple API for loading compiled `.vmfb` modules and invoking exported
functions, without requiring direct knowledge of IREE's internal C API.

---

## Features

| Feature | Notes |
|---------|-------|
| AArch64 (ARM64) support | NEON + crypto extensions, Cortex-A tuning |
| ARMv7 support | NEON VFPv4, hard-float ABI |
| Local-task CPU HAL | Multi-threaded CPU backend supplied by IREE |
| Simple embedding API | Create → load → push inputs → invoke → pull outputs |
| CMake cross-compilation | Bundled toolchain files for both ARM targets |

---

## Directory layout

```
IERR_Self/
├── CMakeLists.txt           # Top-level build
├── toolchains/
│   ├── aarch64-linux.cmake  # AArch64 cross-compilation toolchain
│   └── armv7-linux.cmake    # ARMv7 cross-compilation toolchain
└── src/
    ├── iree_arm_runtime.h   # Public API header
    ├── iree_arm_runtime.c   # Runtime implementation
    └── main.c               # Example / smoke-test executable
```

---

## Prerequisites

### Host tools
* CMake ≥ 3.21
* Ninja (recommended) or Make

### Cross-compiler (choose one)

| Target | Package (Debian/Ubuntu) |
|--------|-------------------------|
| AArch64 | `sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu` |
| ARMv7   | `sudo apt-get install gcc-arm-linux-gnueabihf g++-arm-linux-gnueabihf` |

### IREE runtime

Clone and build (or install) IREE for your target platform.  Only the
runtime and the local-task HAL driver are required:

```bash
git clone https://github.com/openxla/iree.git
cd iree && git submodule update --init --recursive

cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DIREE_BUILD_COMPILER=OFF \
  -DIREE_BUILD_TESTS=OFF \
  -DIREE_HAL_DRIVER_LOCAL_TASK=ON \
  -DCMAKE_INSTALL_PREFIX=$PWD/../iree-install \
  -B build && cmake --build build --target install
```

For cross-compilation add the matching toolchain file from IREE's own
`build_tools/cmake/` directory, or use the toolchain files in this repo as a
starting point.

---

## Building

### Native build (on the ARM device itself)

```bash
cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DIREE_SOURCE_DIR=/path/to/iree \
  -B build-native

cmake --build build-native
```

### Cross-compiled – AArch64

```bash
cmake -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=toolchains/aarch64-linux.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DIREE_SOURCE_DIR=/path/to/iree \
  -B build-aarch64

cmake --build build-aarch64
```

### Cross-compiled – ARMv7

```bash
cmake -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=toolchains/armv7-linux.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DIREE_SOURCE_DIR=/path/to/iree \
  -B build-armv7

cmake --build build-armv7
```

### Using an installed IREE package

Set `IREE_DIR` to the directory that contains `iree-config.cmake`:

```bash
cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DIREE_DIR=/path/to/iree-install/lib/cmake/iree \
  -B build-installed

cmake --build build-installed
```

---

## Usage

### API quick-reference

```c
#include "iree_arm_runtime.h"

/* 1. Create the runtime (VM instance + ARM CPU HAL device). */
iree_arm_runtime_t *rt = NULL;
iree_arm_runtime_create(&rt);

/* 2. Load a compiled .vmfb module. */
iree_arm_module_t *mod = NULL;
iree_arm_module_load_file(rt, "my_model.vmfb", &mod);

/* 3. Prepare a call for the exported function "main". */
iree_arm_call_t *call = NULL;
iree_arm_call_init(rt, mod, "main", &call);

/* 4. Push input tensors. */
float input[] = {1.0f, 2.0f, 3.0f, 4.0f};
iree_arm_call_push_f32_buffer(call, input, 4);

/* 5. Invoke. */
iree_arm_call_invoke(call);

/* 6. Read outputs. */
float output[4];
iree_arm_call_get_f32_result(call, 0, output, 4);

/* 7. Clean up. */
iree_arm_call_deinit(call);
iree_arm_module_release(mod);
iree_arm_runtime_destroy(rt);
```

### Running the example

```bash
# Smoke-test (no model needed):
./build-native/iree_arm_example

# With a real compiled model:
./build-native/iree_arm_example my_model.vmfb
```

Expected smoke-test output:

```
IREE ARM Runtime example
========================
iree_arm_runtime:
  device : local-task
  arch   : AArch64

=== API smoke test ===
  [OK] module_load_memory(NULL) -> INVALID
  [OK] call_init(NULL, ...) -> INVALID
  [OK] push_f32_buffer(NULL, ...) -> INVALID
  [OK] call_invoke(NULL) -> INVALID
  [OK] status_string(OK)="OK"
  [OK] status_string(IREE)="IREE_ERROR"
  [OK] NULL-safe release helpers
=== All smoke tests passed ===

Done (exit 0)
```

### Compiling a model to .vmfb

Use the IREE compiler (`iree-compile`) to produce a `.vmfb` targeting the ARM
CPU backend:

```bash
# AArch64
iree-compile my_model.mlir \
  --iree-hal-target-backends=llvm-cpu \
  --iree-llvmcpu-target-triple=aarch64-linux-gnu \
  -o my_model_aarch64.vmfb

# ARMv7
iree-compile my_model.mlir \
  --iree-hal-target-backends=llvm-cpu \
  --iree-llvmcpu-target-triple=armv7-linux-gnueabihf \
  --iree-llvmcpu-target-cpu-features="+neon,+vfp4" \
  -o my_model_armv7.vmfb
```

---

## API reference

| Function | Description |
|----------|-------------|
| `iree_arm_runtime_create` | Create runtime (VM + HAL device) |
| `iree_arm_runtime_destroy` | Release all runtime resources |
| `iree_arm_runtime_print_info` | Print device / arch summary |
| `iree_arm_module_load_file` | Load `.vmfb` from a file path |
| `iree_arm_module_load_memory` | Load `.vmfb` from a memory buffer |
| `iree_arm_module_release` | Release a loaded module |
| `iree_arm_call_init` | Prepare a call for a named function |
| `iree_arm_call_push_f32_buffer` | Append float32 input tensor |
| `iree_arm_call_push_i32_buffer` | Append int32 input tensor |
| `iree_arm_call_invoke` | Execute the function |
| `iree_arm_call_get_f32_result` | Read float32 output tensor |
| `iree_arm_call_get_i32_result` | Read int32 output tensor |
| `iree_arm_call_deinit` | Release call resources |

All functions return `iree_arm_status_t`; use `iree_arm_status_string()` to
get a human-readable description.

---

## License

This project is distributed under the Apache License 2.0 – the same license
as the upstream IREE project.
