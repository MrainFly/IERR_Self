/**
 * @file iree_arm_runtime.h
 * @brief ARM-targeted IREE runtime wrapper
 *
 * This header exposes a thin, self-contained C API that wraps the IREE
 * embedding runtime for deployment on ARM (AArch64 / ARMv7) devices.
 *
 * Typical usage
 * -------------
 *   iree_arm_runtime_t *rt = NULL;
 *   iree_arm_runtime_create(&rt);
 *
 *   iree_arm_module_t *mod = NULL;
 *   iree_arm_module_load_file(rt, "my_model.vmfb", &mod);
 *
 *   // Build input list and invoke a function
 *   iree_arm_call_t call = {0};
 *   iree_arm_call_init(rt, mod, "main", &call);
 *
 *   float input_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
 *   iree_arm_call_push_f32_buffer(&call, input_data, 4);
 *   iree_arm_call_invoke(&call);
 *
 *   float output[4];
 *   iree_arm_call_get_f32_result(&call, 0, output, 4);
 *
 *   iree_arm_call_deinit(&call);
 *   iree_arm_module_release(mod);
 *   iree_arm_runtime_destroy(rt);
 */

#ifndef IREE_ARM_RUNTIME_H_
#define IREE_ARM_RUNTIME_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Status ────────────────────────────────────────────────────────────────── */

/** Return codes for all public API calls. */
typedef enum {
  IREE_ARM_OK             = 0,  /**< Success.                               */
  IREE_ARM_ERROR_INVALID  = 1,  /**< Invalid argument or internal state.    */
  IREE_ARM_ERROR_IO       = 2,  /**< File / I/O error.                      */
  IREE_ARM_ERROR_ALLOC    = 3,  /**< Allocation failure.                    */
  IREE_ARM_ERROR_IREE     = 4,  /**< Underlying IREE call failed.           */
  IREE_ARM_ERROR_NOT_FOUND = 5, /**< Requested resource not found.          */
} iree_arm_status_t;

/** Returns a human-readable string for @p status. */
const char *iree_arm_status_string(iree_arm_status_t status);

/* ── Runtime ───────────────────────────────────────────────────────────────── */

/**
 * Opaque runtime handle.
 *
 * One instance should be created per process.  It owns the IREE VM instance
 * and the local-task HAL device used for CPU execution on ARM.
 */
typedef struct iree_arm_runtime_t iree_arm_runtime_t;

/**
 * Create and initialise the IREE ARM runtime.
 *
 * @param[out] out_runtime  Receives the newly created runtime on success.
 * @return IREE_ARM_OK on success.
 */
iree_arm_status_t iree_arm_runtime_create(iree_arm_runtime_t **out_runtime);

/**
 * Destroy the runtime and free all associated resources.
 *
 * @param runtime  Handle obtained from iree_arm_runtime_create().
 *                 May be NULL (no-op).
 */
void iree_arm_runtime_destroy(iree_arm_runtime_t *runtime);

/* ── Module ────────────────────────────────────────────────────────────────── */

/**
 * Opaque module handle.
 *
 * A module corresponds to a single compiled IREE VM flat-buffer (.vmfb) that
 * may contain one or more exported functions.
 */
typedef struct iree_arm_module_t iree_arm_module_t;

/**
 * Load a .vmfb module from a file path.
 *
 * @param runtime     Initialised runtime.
 * @param path        NUL-terminated path to the .vmfb file.
 * @param[out] out    Receives the module handle on success.
 * @return IREE_ARM_OK on success.
 */
iree_arm_status_t iree_arm_module_load_file(
    iree_arm_runtime_t  *runtime,
    const char          *path,
    iree_arm_module_t  **out);

/**
 * Load a .vmfb module from an in-memory buffer.
 *
 * The caller retains ownership of @p data; the buffer must remain valid for
 * the lifetime of the returned module.
 *
 * @param runtime     Initialised runtime.
 * @param data        Pointer to the raw .vmfb bytes.
 * @param size        Size of the buffer in bytes.
 * @param[out] out    Receives the module handle on success.
 * @return IREE_ARM_OK on success.
 */
iree_arm_status_t iree_arm_module_load_memory(
    iree_arm_runtime_t  *runtime,
    const void          *data,
    size_t               size,
    iree_arm_module_t  **out);

/**
 * Release a module handle.
 *
 * @param mod  Handle to release.  May be NULL (no-op).
 */
void iree_arm_module_release(iree_arm_module_t *mod);

/* ── Function call ─────────────────────────────────────────────────────────── */

/**
 * Bundles the state needed to invoke a single exported function.
 *
 * Use iree_arm_call_init() / iree_arm_call_deinit() to manage lifetime.
 */
typedef struct iree_arm_call_t iree_arm_call_t;

/**
 * Allocate and prepare a call context for @p function_name in @p mod.
 *
 * @param runtime         Initialised runtime.
 * @param mod             Loaded module.
 * @param function_name   NUL-terminated exported function name.
 * @param[out] out_call   Receives the call handle on success.
 * @return IREE_ARM_OK on success.
 */
iree_arm_status_t iree_arm_call_init(
    iree_arm_runtime_t  *runtime,
    iree_arm_module_t   *mod,
    const char          *function_name,
    iree_arm_call_t    **out_call);

/**
 * Append a 1-D float32 tensor as the next positional input.
 *
 * @param call         Call context.
 * @param data         Pointer to float32 values.
 * @param element_count Number of float32 elements.
 * @return IREE_ARM_OK on success.
 */
iree_arm_status_t iree_arm_call_push_f32_buffer(
    iree_arm_call_t *call,
    const float     *data,
    size_t           element_count);

/**
 * Append a 1-D int32 tensor as the next positional input.
 *
 * @param call          Call context.
 * @param data          Pointer to int32_t values.
 * @param element_count Number of elements.
 * @return IREE_ARM_OK on success.
 */
iree_arm_status_t iree_arm_call_push_i32_buffer(
    iree_arm_call_t  *call,
    const int32_t    *data,
    size_t            element_count);

/**
 * Execute the function with the inputs that have been pushed.
 *
 * @param call  Call context with all inputs pushed.
 * @return IREE_ARM_OK on success.
 */
iree_arm_status_t iree_arm_call_invoke(iree_arm_call_t *call);

/**
 * Copy float32 output at position @p result_index into @p out_data.
 *
 * Must be called after a successful iree_arm_call_invoke().
 *
 * @param call          Call context.
 * @param result_index  Zero-based index into the function's output list.
 * @param out_data      Buffer that receives the float32 values.
 * @param element_count Capacity of @p out_data in elements.
 * @return IREE_ARM_OK on success.
 */
iree_arm_status_t iree_arm_call_get_f32_result(
    iree_arm_call_t *call,
    size_t           result_index,
    float           *out_data,
    size_t           element_count);

/**
 * Copy int32 output at position @p result_index into @p out_data.
 *
 * @param call          Call context.
 * @param result_index  Zero-based index into the function's output list.
 * @param out_data      Buffer that receives the int32_t values.
 * @param element_count Capacity of @p out_data in elements.
 * @return IREE_ARM_OK on success.
 */
iree_arm_status_t iree_arm_call_get_i32_result(
    iree_arm_call_t *call,
    size_t           result_index,
    int32_t         *out_data,
    size_t           element_count);

/**
 * Release the call context and all associated buffers.
 *
 * @param call  Handle to release.  May be NULL (no-op).
 */
void iree_arm_call_deinit(iree_arm_call_t *call);

/* ── Utilities ─────────────────────────────────────────────────────────────── */

/**
 * Print a concise summary of the runtime and loaded device to stdout.
 *
 * @param runtime  Initialised runtime.
 */
void iree_arm_runtime_print_info(const iree_arm_runtime_t *runtime);

#ifdef __cplusplus
}
#endif

#endif /* IREE_ARM_RUNTIME_H_ */
