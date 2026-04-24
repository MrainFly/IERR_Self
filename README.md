# IERR_Self

This repository hosts two complementary runtimes:

1. **`ierr`** — a small C++17 inference runtime targeting **ARM Host CPU +
   multi-core NPU**, with a **PC simulator backend** so you can develop and
   test on Windows / Linux without real hardware. *(Always built; the focus of
   this repo.)*
2. **`iree_arm_runtime`** — an optional thin C wrapper around the upstream
   [IREE](https://github.com/openxla/iree) runtime for ARM (AArch64 / ARMv7).
   Disabled by default; enable with `-DIERR_BUILD_IREE_ARM=ON`.

---

## 1. `ierr` runtime (default)

A small inference runtime targeting **ARM Host CPU + multi-core NPU**, with a
**PC simulator backend** so you can develop and test on Windows / Linux without
real hardware.

### Architecture

```
   ┌──────────────────── User App / model_runner CLI ────────────────────┐
   │                       Runtime Public API (C++17)                    │
   ├─────────────────────────────────────────────────────────────────────┤
   │  Host Runtime (runs on ARM, or on PC CPU in sim mode)               │
   │   · Graph IR + per-op kernel dispatch                               │
   │   · Tensor / device-memory management                               │
   │   · Scheduler: data-parallel fan-out across N cores                 │
   │   · Stream / Event / synchronize primitives                         │
   ├─────────────────────────────────────────────────────────────────────┤
   │                       HAL (hal.h, opaque handles)                   │
   │   device_open / mem_alloc / stream_submit / record_event / wait …  │
   ├──────────────────────────┬──────────────────────────────────────────┤
   │ HAL: NPU (stub)          │ HAL: Sim (this repo's main backend)      │
   │  · ioctl + DMA + IRQ     │  · std::thread pool = N virtual cores    │
   │                          │  · aligned heap = "device" memory        │
   │                          │  · cv + atomic = events / fences         │
   ├──────────────────────────┴──────────────────────────────────────────┤
   │             Op Library: CPU kernels (Add, ReLU, MatMul)             │
   └─────────────────────────────────────────────────────────────────────┘
```

#### Responsibility split

| Layer        | Knows about                          | Doesn't know about               |
|--------------|--------------------------------------|----------------------------------|
| Host Runtime | Graph, ops, scheduling, lifetimes    | NPU ISA, core count, DMA wires   |
| HAL          | Cores, memory, streams, events       | What ops are or what they compute|
| Kernels      | Math, tensor layout, parallel split  | How work is dispatched           |

The simulator HAL implements the same vtable as a real NPU driver would, so
the entire runtime + kernel stack runs unchanged on Windows / Linux PCs.

### Build

Requires a C++17 compiler and CMake ≥ 3.21. No third-party dependencies.

#### Linux / macOS
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/model_runner --m 32 --k 64 --n 16
```

#### Windows (MSVC)
```
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
ctest --test-dir build --output-on-failure -C Release
build\Release\model_runner.exe --m 32 --k 64 --n 16
```

### CLI demo

`model_runner` builds an in-memory graph `y = relu(a @ b + bias)`, runs it on
the simulator, and prints a checksum.

```
model_runner [--backend sim|npu] [--m M] [--k K] [--n N]
```

Environment variables:

| Variable         | Effect                                                   |
|------------------|----------------------------------------------------------|
| `IERR_SIM_CORES` | Override simulated NPU core count (default = HW threads) |
| `IERR_LOG`       | `debug` / `info` / `warn` / `error`                      |

### Layout

```
include/ierr/        # public API headers (ierr.h, hal.h, error.h)
src/runtime/         # host runtime: scheduler, graph, op registry, logging
src/hal/sim/         # PC simulator HAL (multi-core thread pool)
src/hal/npu/         # real NPU HAL (stub – returns NOT_IMPLEMENTED)
src/kernels/cpu/     # CPU kernels: Add, ReLU, MatMul
src/kernels/npu/     # NPU kernels (placeholder)
tools/model_runner/  # CLI demo
tests/               # zero-dep unit tests + end-to-end test
```

### Status / roadmap

Implemented (this skeleton):

- Public C++ runtime API + opaque-handle HAL
- PC simulator HAL with multi-core dispatch, streams, events, aligned memory
- Host Runtime scheduler with data-parallel fan-out
- CPU kernels: Add, ReLU, MatMul
- `model_runner` CLI demo
- Unit tests + end-to-end test, CI on Linux + Windows

Not yet:

- Real NPU HAL implementation (stub only)
- Disk model file format / parser
- Quantization, layout transforms (NCHW ↔ NC1HWC0)
- Profiling / Chrome-trace export
- More ops (Conv, Softmax, …)

---

## 2. `iree_arm_runtime` – optional ARM IREE wrapper

A lightweight C wrapper that embeds the [IREE](https://github.com/openxla/iree)
machine-learning runtime on **ARM** devices (AArch64 / ARMv7). It exposes a
simple API for loading compiled `.vmfb` modules and invoking exported
functions, without requiring direct knowledge of IREE's internal C API.

This component is **off by default**. Enable with `-DIERR_BUILD_IREE_ARM=ON`
and supply IREE either as source (`-DIREE_SOURCE_DIR=...`) or as an installed
package (`-DIREE_DIR=...`).

### Features

| Feature | Notes |
|---------|-------|
| AArch64 (ARM64) support | NEON + crypto extensions, Cortex-A tuning |
| ARMv7 support | NEON VFPv4, hard-float ABI |
| Local-task CPU HAL | Multi-threaded CPU backend supplied by IREE |
| Simple embedding API | Create → load → push inputs → invoke → pull outputs |
| CMake cross-compilation | Bundled toolchain files for both ARM targets |

### Layout

```
toolchains/
├── aarch64-linux.cmake  # AArch64 cross-compilation toolchain
└── armv7-linux.cmake    # ARMv7 cross-compilation toolchain
src/
├── iree_arm_runtime.h   # Public C API header
├── iree_arm_runtime.c   # Runtime implementation
└── main.c               # Example / smoke-test executable
```

### Prerequisites

* CMake ≥ 3.21, Ninja recommended.
* Cross-compiler:
  | Target | Package (Debian/Ubuntu) |
  |--------|-------------------------|
  | AArch64 | `sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu` |
  | ARMv7   | `sudo apt-get install gcc-arm-linux-gnueabihf g++-arm-linux-gnueabihf` |
* IREE: clone and build (or install) for your target. Only the runtime and
  the local-task HAL driver are required:

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

### Building

#### Native build (on the ARM device itself)
```bash
cmake -G Ninja \
  -DIERR_BUILD_IREE_ARM=ON \
  -DIREE_SOURCE_DIR=/path/to/iree \
  -B build-native
cmake --build build-native
```

#### Cross-compiled – AArch64
```bash
cmake -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=toolchains/aarch64-linux.cmake \
  -DIERR_BUILD_IREE_ARM=ON \
  -DIREE_SOURCE_DIR=/path/to/iree \
  -B build-aarch64
cmake --build build-aarch64
```

#### Cross-compiled – ARMv7
```bash
cmake -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=toolchains/armv7-linux.cmake \
  -DIERR_BUILD_IREE_ARM=ON \
  -DIREE_SOURCE_DIR=/path/to/iree \
  -B build-armv7
cmake --build build-armv7
```

#### Using an installed IREE package
```bash
cmake -G Ninja \
  -DIERR_BUILD_IREE_ARM=ON \
  -DIREE_DIR=/path/to/iree-install/lib/cmake/iree \
  -B build-installed
cmake --build build-installed
```

### Usage

```c
#include "iree_arm_runtime.h"

iree_arm_runtime_t *rt = NULL;
iree_arm_runtime_create(&rt);

iree_arm_module_t *mod = NULL;
iree_arm_module_load_file(rt, "my_model.vmfb", &mod);

iree_arm_call_t *call = NULL;
iree_arm_call_init(rt, mod, "main", &call);

float input[] = {1.0f, 2.0f, 3.0f, 4.0f};
iree_arm_call_push_f32_buffer(call, input, 4);
iree_arm_call_invoke(call);

float output[4];
iree_arm_call_get_f32_result(call, 0, output, 4);

iree_arm_call_deinit(call);
iree_arm_module_release(mod);
iree_arm_runtime_destroy(rt);
```

Compile a model to `.vmfb` with the IREE compiler, e.g.:

```bash
iree-compile my_model.mlir \
  --iree-hal-target-backends=llvm-cpu \
  --iree-llvmcpu-target-triple=aarch64-linux-gnu \
  -o my_model_aarch64.vmfb
```

### API reference

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
