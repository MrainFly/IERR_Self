// IERR Hardware Abstraction Layer.
//
// The HAL is the single seam between the (hardware-agnostic) Host Runtime and
// any backend that can actually execute work. It is intentionally narrow and
// uses opaque handles so that:
//   * The simulator backend can implement every primitive with std::thread /
//     condition_variable / malloc on Windows or Linux.
//   * The real NPU backend can implement them via ioctl + DMA + IRQ on ARM.
//
// The Host Runtime never assumes a specific backend; it only talks through the
// vtable returned by ierr_hal_get(). This keeps responsibilities clean:
//   Runtime  -> graph, memory bookkeeping, scheduling
//   HAL      -> "where do I run a command, where does memory live"
//   Kernels  -> the actual math, picked per backend
#ifndef IERR_HAL_H
#define IERR_HAL_H

#include <stddef.h>
#include <stdint.h>
#include "ierr/error.h"

#ifdef __cplusplus
extern "C" {
#endif

// ---- opaque handles --------------------------------------------------------
typedef struct ierr_hal_device_s* ierr_hal_device_t;
typedef struct ierr_hal_stream_s* ierr_hal_stream_t;
typedef struct ierr_hal_event_s*  ierr_hal_event_t;
typedef struct ierr_hal_buffer_s* ierr_hal_buffer_t;

typedef enum ierr_memcpy_dir_t {
    IERR_MEMCPY_H2D = 0, // host -> device
    IERR_MEMCPY_D2H = 1, // device -> host
    IERR_MEMCPY_D2D = 2  // device -> device
} ierr_memcpy_dir_t;

typedef struct ierr_hal_caps_t {
    uint32_t num_cores;       // number of independently-schedulable cores
    uint32_t mem_align_bytes; // required alignment for device buffers
    const char* name;         // backend name, e.g. "sim" or "npu-v1"
} ierr_hal_caps_t;

// A unit of work submitted to the HAL. The HAL does not inspect `fn` – it just
// runs it on one of its cores. Multi-core data parallelism is expressed by the
// scheduler submitting multiple commands with different (begin,end) ranges.
typedef void (*ierr_hal_kernel_fn)(void* user, uint32_t core_id);

typedef struct ierr_hal_command_t {
    ierr_hal_kernel_fn fn;
    void* user;
    int32_t preferred_core; // -1 = any core
} ierr_hal_command_t;

// HAL vtable. Each backend implements one of these and returns it via the
// backend's `ierr_hal_get_<name>()` factory.
typedef struct ierr_hal_t {
    // device lifecycle
    ierr_status_t (*device_open)(ierr_hal_device_t* out_dev);
    ierr_status_t (*device_close)(ierr_hal_device_t dev);
    ierr_status_t (*device_query_caps)(ierr_hal_device_t dev, ierr_hal_caps_t* out);

    // memory
    ierr_status_t (*mem_alloc)(ierr_hal_device_t dev, size_t bytes, ierr_hal_buffer_t* out);
    ierr_status_t (*mem_free)(ierr_hal_buffer_t buf);
    // For sim backend the returned pointer is just a host pointer; for real NPU
    // it would be a mapped pointer. Either way the runtime treats it opaquely.
    void*         (*mem_map)(ierr_hal_buffer_t buf);
    size_t        (*mem_size)(ierr_hal_buffer_t buf);

    // streams + events
    ierr_status_t (*stream_create)(ierr_hal_device_t dev, ierr_hal_stream_t* out);
    ierr_status_t (*stream_destroy)(ierr_hal_stream_t s);
    ierr_status_t (*stream_submit)(ierr_hal_stream_t s, const ierr_hal_command_t* cmd);
    ierr_status_t (*stream_memcpy)(ierr_hal_stream_t s, ierr_memcpy_dir_t dir,
                                   ierr_hal_buffer_t dst, size_t dst_off,
                                   ierr_hal_buffer_t src, size_t src_off,
                                   size_t bytes);
    ierr_status_t (*stream_record_event)(ierr_hal_stream_t s, ierr_hal_event_t* out);
    ierr_status_t (*stream_wait_event) (ierr_hal_stream_t s, ierr_hal_event_t e);
    ierr_status_t (*stream_synchronize)(ierr_hal_stream_t s);

    ierr_status_t (*event_destroy)(ierr_hal_event_t e);
    ierr_status_t (*event_synchronize)(ierr_hal_event_t e);
} ierr_hal_t;

// Built-in backend factories. Real-NPU one is currently a stub that returns
// IERR_ERR_NOT_IMPLEMENTED on device_open – the Host Runtime can therefore be
// developed and tested entirely on the simulator.
const ierr_hal_t* ierr_hal_get_sim(void);
const ierr_hal_t* ierr_hal_get_npu(void);

#ifdef __cplusplus
}
#endif

#endif // IERR_HAL_H
