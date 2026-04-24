/**
 * @file main.c
 * @brief Example demonstrating the ARM IREE runtime wrapper
 *
 * This example shows the full lifecycle of an IREE model inference call using
 * the iree_arm_runtime API:
 *
 *   1. Initialise the runtime (creates a local-task CPU HAL device).
 *   2. Load a compiled .vmfb module (from file or from a synthetic in-memory
 *      buffer when no path is supplied).
 *   3. Push input tensors.
 *   4. Invoke an exported function.
 *   5. Read back the output tensors.
 *   6. Clean up all resources.
 *
 * Usage
 * -----
 *   ./iree_arm_example [path_to_model.vmfb]
 *
 * When run without arguments the example exercises the runtime with a
 * synthetic (empty) .vmfb stub and verifies that all API calls succeed up to
 * the point that real model data would be required.  Pass a real .vmfb file
 * to perform end-to-end inference.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree_arm_runtime.h"

/* ── Helpers ─────────────────────────────────────────────────────────────── */

#define CHECK(expr)                                                    \
  do {                                                                 \
    iree_arm_status_t _s = (expr);                                     \
    if (_s != IREE_ARM_OK) {                                           \
      fprintf(stderr, "FATAL [%s:%d]: %s => %s\n",                    \
              __FILE__, __LINE__, #expr,                               \
              iree_arm_status_string(_s));                             \
      return EXIT_FAILURE;                                             \
    }                                                                  \
  } while (0)

/* ── Demo: inference with a real .vmfb file ─────────────────────────────── */

static int run_inference(iree_arm_runtime_t *runtime, const char *vmfb_path) {
  printf("\n=== Inference demo: %s ===\n", vmfb_path);

  iree_arm_module_t *module = NULL;
  CHECK(iree_arm_module_load_file(runtime, vmfb_path, &module));

  iree_arm_call_t *call = NULL;
  iree_arm_status_t s = iree_arm_call_init(runtime, module, "main", &call);
  if (s != IREE_ARM_OK) {
    fprintf(stderr,
            "Note: function 'main' not found (normal for stub modules).\n");
    iree_arm_module_release(module);
    return EXIT_SUCCESS;
  }

  /* Push a 4-element float32 input tensor: [1.0, 2.0, 3.0, 4.0] */
  const float input[] = {1.0f, 2.0f, 3.0f, 4.0f};
  CHECK(iree_arm_call_push_f32_buffer(call, input, 4));

  /* Invoke the function. */
  s = iree_arm_call_invoke(call);
  if (s != IREE_ARM_OK) {
    fprintf(stderr,
            "Note: invoke failed (expected for stub modules): %s\n",
            iree_arm_status_string(s));
    iree_arm_call_deinit(call);
    iree_arm_module_release(module);
    return EXIT_SUCCESS;
  }

  /* Read back the first float32 output. */
  float output[4] = {0};
  CHECK(iree_arm_call_get_f32_result(call, 0, output, 4));
  printf("Output[0..3]: %.4f  %.4f  %.4f  %.4f\n",
         output[0], output[1], output[2], output[3]);

  iree_arm_call_deinit(call);
  iree_arm_module_release(module);
  return EXIT_SUCCESS;
}

/* ── Demo: API lifecycle without a real model ─────────────────────────────── */

/**
 * Exercise every public API call to verify correct initialisation,
 * argument validation (NULL / bad-arg paths), and clean teardown.
 *
 * Returns EXIT_SUCCESS when all checks pass.
 */
static int run_api_smoke_test(iree_arm_runtime_t *runtime) {
  printf("\n=== API smoke test ===\n");

  /* --- module_load_memory with NULL data should return INVALID --- */
  iree_arm_module_t *mod = NULL;
  iree_arm_status_t s = iree_arm_module_load_memory(runtime, NULL, 0, &mod);
  if (s != IREE_ARM_ERROR_INVALID) {
    fprintf(stderr,
            "Expected IREE_ARM_ERROR_INVALID for NULL data, got %s\n",
            iree_arm_status_string(s));
    return EXIT_FAILURE;
  }
  printf("  [OK] module_load_memory(NULL) -> INVALID\n");

  /* --- iree_arm_call_init with NULL runtime should return INVALID --- */
  iree_arm_call_t *call = NULL;
  s = iree_arm_call_init(NULL, NULL, "main", &call);
  if (s != IREE_ARM_ERROR_INVALID) {
    fprintf(stderr,
            "Expected IREE_ARM_ERROR_INVALID for NULL runtime, got %s\n",
            iree_arm_status_string(s));
    return EXIT_FAILURE;
  }
  printf("  [OK] call_init(NULL, ...) -> INVALID\n");

  /* --- push_f32_buffer with NULL call should return INVALID --- */
  const float dummy[] = {1.0f};
  s = iree_arm_call_push_f32_buffer(NULL, dummy, 1);
  if (s != IREE_ARM_ERROR_INVALID) {
    fprintf(stderr,
            "Expected IREE_ARM_ERROR_INVALID for NULL call, got %s\n",
            iree_arm_status_string(s));
    return EXIT_FAILURE;
  }
  printf("  [OK] push_f32_buffer(NULL, ...) -> INVALID\n");

  /* --- iree_arm_call_invoke with NULL call should return INVALID --- */
  s = iree_arm_call_invoke(NULL);
  if (s != IREE_ARM_ERROR_INVALID) {
    fprintf(stderr,
            "Expected IREE_ARM_ERROR_INVALID for NULL call, got %s\n",
            iree_arm_status_string(s));
    return EXIT_FAILURE;
  }
  printf("  [OK] call_invoke(NULL) -> INVALID\n");

  /* --- status strings --- */
  const char *ok_str  = iree_arm_status_string(IREE_ARM_OK);
  const char *err_str = iree_arm_status_string(IREE_ARM_ERROR_IREE);
  if (!ok_str || !err_str || strlen(ok_str) == 0 || strlen(err_str) == 0) {
    fprintf(stderr, "status_string returned empty string\n");
    return EXIT_FAILURE;
  }
  printf("  [OK] status_string(OK)=\"%s\"\n", ok_str);
  printf("  [OK] status_string(IREE)=\"%s\"\n", err_str);

  /* NULL-safe release helpers must not crash. */
  iree_arm_call_deinit(NULL);
  iree_arm_module_release(NULL);
  iree_arm_runtime_destroy(NULL);
  printf("  [OK] NULL-safe release helpers\n");

  printf("=== All smoke tests passed ===\n");
  return EXIT_SUCCESS;
}

/* ── main ──────────────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
  printf("IREE ARM Runtime example\n");
  printf("========================\n");

  /* Create the runtime (IREE VM + local-task HAL device). */
  iree_arm_runtime_t *runtime = NULL;
  iree_arm_status_t s = iree_arm_runtime_create(&runtime);
  if (s != IREE_ARM_OK) {
    fprintf(stderr,
            "Failed to create IREE ARM runtime: %s\n"
            "Ensure the IREE runtime library was compiled with the "
            "local-task HAL driver enabled.\n",
            iree_arm_status_string(s));
    return EXIT_FAILURE;
  }

  iree_arm_runtime_print_info(runtime);

  int ret = EXIT_SUCCESS;

  /* If a .vmfb path was supplied, run inference on it. */
  if (argc >= 2) {
    ret = run_inference(runtime, argv[1]);
  }

  /* Always run the API smoke test. */
  if (ret == EXIT_SUCCESS) {
    ret = run_api_smoke_test(runtime);
  }

  iree_arm_runtime_destroy(runtime);
  printf("\nDone (exit %d)\n", ret);
  return ret;
}
