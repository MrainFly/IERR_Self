# IERR_Self

A small inference runtime targeting **ARM Host CPU + multi-core NPU**, with a
**PC simulator backend** so you can develop and test on Windows / Linux without
real hardware.

## Architecture

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

### Responsibility split

| Layer        | Knows about                          | Doesn't know about               |
|--------------|--------------------------------------|----------------------------------|
| Host Runtime | Graph, ops, scheduling, lifetimes    | NPU ISA, core count, DMA wires   |
| HAL          | Cores, memory, streams, events       | What ops are or what they compute|
| Kernels      | Math, tensor layout, parallel split  | How work is dispatched           |

The simulator HAL implements the same vtable as a real NPU driver would, so
the entire runtime + kernel stack runs unchanged on Windows / Linux PCs.

## Build

Requires a C++17 compiler and CMake ≥ 3.16. No third-party dependencies.

### Linux / macOS
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/model_runner --m 32 --k 64 --n 16
```

### Windows (MSVC)
```
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
ctest --test-dir build --output-on-failure -C Release
build\Release\model_runner.exe --m 32 --k 64 --n 16
```

## CLI demo

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

## Layout

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

## Status / roadmap

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
