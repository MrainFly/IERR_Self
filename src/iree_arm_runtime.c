/**
 * @file iree_arm_runtime.c
 * @brief ARM-targeted IREE runtime wrapper – implementation
 *
 * This file implements the API declared in iree_arm_runtime.h.  It uses the
 * IREE embedding C API to:
 *
 *   1. Create a VM instance.
 *   2. Register and create a "local-task" CPU HAL device (optimised for ARM
 *      via NEON / SVE when the IREE runtime was compiled with those targets).
 *   3. Load compiled .vmfb modules.
 *   4. Invoke exported functions and marshal inputs / outputs as buffer views.
 *
 * Dependencies
 * ------------
 *   iree/runtime/api.h   – high-level embedding runtime
 *   iree/hal/api.h       – hardware abstraction layer
 *   iree/vm/api.h        – virtual machine
 *   iree/hal/drivers/local_task/registration/driver_module.h
 *                        – registers the local-task CPU driver
 */

#include "iree_arm_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* IREE public headers */
#include "iree/runtime/api.h"
#include "iree/hal/api.h"
#include "iree/vm/api.h"
#include "iree/hal/drivers/local_task/registration/driver_module.h"

/* ── Internal helpers ───────────────────────────────────────────────────────── */

/** Convert an iree_status_t to our public status code. */
static iree_arm_status_t status_from_iree(iree_status_t s) {
  if (iree_status_is_ok(s)) {
    iree_status_ignore(s);
    return IREE_ARM_OK;
  }
  iree_status_fprint(stderr, s);
  iree_status_free(s);
  return IREE_ARM_ERROR_IREE;
}

/* Convenience macro – jumps to `cleanup` on failure. */
#define RETURN_IF_ERR(expr)                       \
  do {                                            \
    iree_arm_status_t _s = (expr);                \
    if (_s != IREE_ARM_OK) { status = _s; goto cleanup; } \
  } while (0)

#define RETURN_IREE_IF_ERR(expr)                  \
  do {                                            \
    iree_status_t _s = (expr);                    \
    if (!iree_status_is_ok(_s)) {                 \
      status = status_from_iree(_s);              \
      goto cleanup;                               \
    }                                             \
  } while (0)

/* ── iree_arm_runtime_t ─────────────────────────────────────────────────────── */

struct iree_arm_runtime_t {
  iree_runtime_instance_t *instance;
  iree_hal_device_t       *device;
};

const char *iree_arm_status_string(iree_arm_status_t status) {
  switch (status) {
    case IREE_ARM_OK:              return "OK";
    case IREE_ARM_ERROR_INVALID:   return "INVALID_ARGUMENT";
    case IREE_ARM_ERROR_IO:        return "IO_ERROR";
    case IREE_ARM_ERROR_ALLOC:     return "ALLOCATION_ERROR";
    case IREE_ARM_ERROR_IREE:      return "IREE_ERROR";
    case IREE_ARM_ERROR_NOT_FOUND: return "NOT_FOUND";
    default:                       return "UNKNOWN";
  }
}

iree_arm_status_t iree_arm_runtime_create(iree_arm_runtime_t **out_runtime) {
  if (!out_runtime) return IREE_ARM_ERROR_INVALID;

  iree_arm_status_t status = IREE_ARM_OK;
  iree_arm_runtime_t *rt = NULL;
  iree_hal_driver_registry_t *registry = NULL;
  iree_hal_driver_t          *driver   = NULL;

  rt = (iree_arm_runtime_t *)calloc(1, sizeof(*rt));
  if (!rt) { status = IREE_ARM_ERROR_ALLOC; goto cleanup; }

  /* 1 – Create a runtime instance (owns the VM). */
  iree_runtime_instance_options_t inst_opts;
  iree_runtime_instance_options_initialize(&inst_opts);
  iree_runtime_instance_options_use_all_available_drivers(&inst_opts);

  RETURN_IREE_IF_ERR(iree_runtime_instance_create(
      &inst_opts, iree_allocator_system(), &rt->instance));

  /* 2 – Register the local-task CPU driver (ARM NEON/SVE path). */
  registry = iree_hal_driver_registry_default();
  RETURN_IREE_IF_ERR(
      iree_hal_local_task_driver_module_register(registry));

  /* 3 – Create the default device from the local-task driver. */
  RETURN_IREE_IF_ERR(iree_hal_driver_registry_try_create(
      registry,
      iree_make_cstring_view("local-task"),
      iree_allocator_system(),
      &driver));

  RETURN_IREE_IF_ERR(iree_hal_driver_create_default_device(
      driver, iree_allocator_system(), &rt->device));

  iree_hal_driver_release(driver);
  driver = NULL;

  *out_runtime = rt;
  return IREE_ARM_OK;

cleanup:
  if (driver)  iree_hal_driver_release(driver);
  if (rt) {
    if (rt->device)   iree_hal_device_release(rt->device);
    if (rt->instance) iree_runtime_instance_release(rt->instance);
    free(rt);
  }
  return status;
}

void iree_arm_runtime_destroy(iree_arm_runtime_t *runtime) {
  if (!runtime) return;
  iree_hal_device_release(runtime->device);
  iree_runtime_instance_release(runtime->instance);
  free(runtime);
}

void iree_arm_runtime_print_info(const iree_arm_runtime_t *runtime) {
  if (!runtime) {
    printf("iree_arm_runtime: (null)\n");
    return;
  }
  printf("iree_arm_runtime:\n");
  printf("  device : %s\n",
         iree_hal_device_id(runtime->device).data
             ? iree_hal_device_id(runtime->device).data
             : "(unknown)");
#if defined(IREE_ARM_ARCH_AARCH64)
  printf("  arch   : AArch64\n");
#elif defined(IREE_ARM_ARCH_ARMV7)
  printf("  arch   : ARMv7\n");
#else
  printf("  arch   : host\n");
#endif
}

/* ── iree_arm_module_t ──────────────────────────────────────────────────────── */

struct iree_arm_module_t {
  iree_arm_runtime_t    *runtime;    /* back-pointer (not owned)          */
  iree_vm_module_t      *vm_module;  /* the compiled VMFB module          */
  iree_vm_context_t     *context;    /* execution context                 */
  void                  *file_data;  /* heap copy of the file (may be NULL) */
};

/**
 * Internal helper: given raw .vmfb bytes, create the vm_module and context.
 *
 * @p data must remain valid for the lifetime of the returned module when
 * @p owns_data is false.
 */
static iree_arm_status_t module_create_from_bytes(
    iree_arm_runtime_t  *runtime,
    const void          *data,
    size_t               size,
    void                *owned_data,   /* if non-NULL, freed on release */
    iree_arm_module_t  **out) {

  iree_arm_status_t status = IREE_ARM_OK;
  iree_arm_module_t *m = NULL;
  iree_vm_module_t  *hal_module  = NULL;

  m = (iree_arm_module_t *)calloc(1, sizeof(*m));
  if (!m) { return IREE_ARM_ERROR_ALLOC; }

  m->runtime   = runtime;
  m->file_data = owned_data;

  /* Create the bytecode VM module from the flat-buffer. */
  RETURN_IREE_IF_ERR(iree_vm_bytecode_module_create(
      iree_runtime_instance_vm_instance(runtime->instance),
      iree_make_const_byte_span(data, size),
      iree_allocator_null(),   /* data is externally owned */
      iree_allocator_system(),
      &m->vm_module));

  /* Create a HAL module so the program can allocate buffers. */
  RETURN_IREE_IF_ERR(iree_hal_module_create(
      iree_runtime_instance_vm_instance(runtime->instance),
      IREE_HAL_MODULE_FLAG_SYNCHRONOUS,
      /*device_count=*/1,
      &runtime->device,
      iree_allocator_system(),
      &hal_module));

  /* Assemble the execution context: [hal_module, vm_module]. */
  iree_vm_module_t *modules[] = { hal_module, m->vm_module };
  RETURN_IREE_IF_ERR(iree_vm_context_create_with_modules(
      iree_runtime_instance_vm_instance(runtime->instance),
      IREE_VM_CONTEXT_FLAG_NONE,
      IREE_ARRAYSIZE(modules), modules,
      iree_allocator_system(),
      &m->context));

  iree_vm_module_release(hal_module);

  *out = m;
  return IREE_ARM_OK;

cleanup:
  if (hal_module) iree_vm_module_release(hal_module);
  if (m) {
    if (m->context)   iree_vm_context_release(m->context);
    if (m->vm_module) iree_vm_module_release(m->vm_module);
    if (m->file_data) free(m->file_data);
    free(m);
  }
  return status;
}

iree_arm_status_t iree_arm_module_load_file(
    iree_arm_runtime_t  *runtime,
    const char          *path,
    iree_arm_module_t  **out) {

  if (!runtime || !path || !out) return IREE_ARM_ERROR_INVALID;

  FILE  *f    = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "iree_arm_runtime: cannot open '%s'\n", path);
    return IREE_ARM_ERROR_IO;
  }

  fseek(f, 0, SEEK_END);
  long file_size = ftell(f);
  rewind(f);

  if (file_size <= 0) { fclose(f); return IREE_ARM_ERROR_IO; }

  void *buf = malloc((size_t)file_size);
  if (!buf) { fclose(f); return IREE_ARM_ERROR_ALLOC; }

  if (fread(buf, 1, (size_t)file_size, f) != (size_t)file_size) {
    fclose(f);
    free(buf);
    return IREE_ARM_ERROR_IO;
  }
  fclose(f);

  iree_arm_status_t s = module_create_from_bytes(
      runtime, buf, (size_t)file_size, buf, out);
  if (s != IREE_ARM_OK) free(buf);
  return s;
}

iree_arm_status_t iree_arm_module_load_memory(
    iree_arm_runtime_t  *runtime,
    const void          *data,
    size_t               size,
    iree_arm_module_t  **out) {

  if (!runtime || !data || !size || !out) return IREE_ARM_ERROR_INVALID;
  return module_create_from_bytes(runtime, data, size, NULL, out);
}

void iree_arm_module_release(iree_arm_module_t *mod) {
  if (!mod) return;
  iree_vm_context_release(mod->context);
  iree_vm_module_release(mod->vm_module);
  free(mod->file_data);
  free(mod);
}

/* ── iree_arm_call_t ────────────────────────────────────────────────────────── */

struct iree_arm_call_t {
  iree_arm_runtime_t *runtime;   /* back-pointer (not owned) */
  iree_arm_module_t  *module;    /* back-pointer (not owned) */
  iree_vm_function_t  function;
  iree_vm_list_t     *inputs;
  iree_vm_list_t     *outputs;
};

iree_arm_status_t iree_arm_call_init(
    iree_arm_runtime_t  *runtime,
    iree_arm_module_t   *mod,
    const char          *function_name,
    iree_arm_call_t    **out_call) {

  if (!runtime || !mod || !function_name || !out_call)
    return IREE_ARM_ERROR_INVALID;

  iree_arm_status_t  status = IREE_ARM_OK;
  iree_arm_call_t   *call   = NULL;

  call = (iree_arm_call_t *)calloc(1, sizeof(*call));
  if (!call) return IREE_ARM_ERROR_ALLOC;

  call->runtime = runtime;
  call->module  = mod;

  /* Look up the exported function. */
  iree_status_t s = iree_vm_module_lookup_function_by_name(
      mod->vm_module,
      IREE_VM_FUNCTION_LINKAGE_EXPORT,
      iree_make_cstring_view(function_name),
      &call->function);
  if (!iree_status_is_ok(s)) {
    fprintf(stderr, "iree_arm_runtime: function '%s' not found\n",
            function_name);
    status = status_from_iree(s);
    goto cleanup;
  }

  /* Allocate input / output variant lists. */
  RETURN_IREE_IF_ERR(iree_vm_list_create(
      iree_vm_make_undefined_type_def(),
      /*initial_capacity=*/8,
      iree_allocator_system(),
      &call->inputs));

  RETURN_IREE_IF_ERR(iree_vm_list_create(
      iree_vm_make_undefined_type_def(),
      /*initial_capacity=*/8,
      iree_allocator_system(),
      &call->outputs));

  *out_call = call;
  return IREE_ARM_OK;

cleanup:
  if (call) {
    if (call->inputs)  iree_vm_list_release(call->inputs);
    if (call->outputs) iree_vm_list_release(call->outputs);
    free(call);
  }
  return status;
}

/** Helper: upload an element buffer to the HAL device and wrap it in a
 *  buffer-view, then append it to @p list. */
static iree_arm_status_t push_buffer_to_list(
    iree_arm_call_t       *call,
    const void            *data,
    size_t                 element_count,
    size_t                 element_size,
    iree_hal_element_type_t element_type,
    iree_vm_list_t        *list) {

  iree_arm_status_t status = IREE_ARM_OK;
  iree_hal_buffer_t      *buffer      = NULL;
  iree_hal_buffer_view_t *buffer_view = NULL;

  iree_hal_dim_t shape[] = { (iree_hal_dim_t)element_count };
  size_t byte_length = element_count * element_size;

  iree_hal_buffer_params_t params = {
      .type  = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
               IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
      .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
  };

  RETURN_IREE_IF_ERR(iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(call->runtime->device),
      params,
      byte_length,
      &buffer));

  RETURN_IREE_IF_ERR(iree_hal_buffer_map_write(
      buffer, /*byte_offset=*/0, data, byte_length));

  RETURN_IREE_IF_ERR(iree_hal_buffer_view_create(
      buffer,
      /*shape_rank=*/1, shape,
      element_type,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      iree_allocator_system(),
      &buffer_view));

  iree_vm_ref_t ref = iree_hal_buffer_view_move_ref(buffer_view);
  RETURN_IREE_IF_ERR(iree_vm_list_push_ref_move(list, &ref));

  iree_hal_buffer_release(buffer);
  return IREE_ARM_OK;

cleanup:
  if (buffer_view) iree_hal_buffer_view_release(buffer_view);
  if (buffer)      iree_hal_buffer_release(buffer);
  return status;
}

iree_arm_status_t iree_arm_call_push_f32_buffer(
    iree_arm_call_t *call,
    const float     *data,
    size_t           element_count) {

  if (!call || !data || !element_count) return IREE_ARM_ERROR_INVALID;
  return push_buffer_to_list(call, data, element_count,
                             sizeof(float),
                             IREE_HAL_ELEMENT_TYPE_FLOAT_32,
                             call->inputs);
}

iree_arm_status_t iree_arm_call_push_i32_buffer(
    iree_arm_call_t  *call,
    const int32_t    *data,
    size_t            element_count) {

  if (!call || !data || !element_count) return IREE_ARM_ERROR_INVALID;
  return push_buffer_to_list(call, data, element_count,
                             sizeof(int32_t),
                             IREE_HAL_ELEMENT_TYPE_SINT_32,
                             call->inputs);
}

iree_arm_status_t iree_arm_call_invoke(iree_arm_call_t *call) {
  if (!call) return IREE_ARM_ERROR_INVALID;

  /* Reset the output list before each invocation. */
  iree_vm_list_clear(call->outputs);

  iree_status_t s = iree_vm_invoke(
      call->module->context,
      call->function,
      IREE_VM_INVOCATION_FLAG_NONE,
      /*policy=*/NULL,
      call->inputs,
      call->outputs,
      iree_allocator_system());

  return status_from_iree(s);
}

/** Helper: read elements from the buffer-view at @p result_index. */
static iree_arm_status_t get_buffer_result(
    iree_arm_call_t *call,
    size_t           result_index,
    void            *out_data,
    size_t           element_count,
    size_t           element_size) {

  if (!call || !out_data) return IREE_ARM_ERROR_INVALID;

  iree_vm_ref_t ref = {0};
  iree_status_t s = iree_vm_list_get_ref_assign(
      call->outputs, result_index, &ref);
  if (!iree_status_is_ok(s)) return status_from_iree(s);

  iree_hal_buffer_view_t *bv =
      iree_hal_buffer_view_deref(ref);
  if (!bv) return IREE_ARM_ERROR_INVALID;

  iree_hal_buffer_t *buf = iree_hal_buffer_view_buffer(bv);
  size_t byte_length = element_count * element_size;

  s = iree_hal_buffer_map_read(buf, /*byte_offset=*/0,
                                out_data, byte_length);
  return status_from_iree(s);
}

iree_arm_status_t iree_arm_call_get_f32_result(
    iree_arm_call_t *call,
    size_t           result_index,
    float           *out_data,
    size_t           element_count) {
  return get_buffer_result(call, result_index,
                           out_data, element_count, sizeof(float));
}

iree_arm_status_t iree_arm_call_get_i32_result(
    iree_arm_call_t *call,
    size_t           result_index,
    int32_t         *out_data,
    size_t           element_count) {
  return get_buffer_result(call, result_index,
                           out_data, element_count, sizeof(int32_t));
}

void iree_arm_call_deinit(iree_arm_call_t *call) {
  if (!call) return;
  if (call->inputs)  iree_vm_list_release(call->inputs);
  if (call->outputs) iree_vm_list_release(call->outputs);
  free(call);
}
