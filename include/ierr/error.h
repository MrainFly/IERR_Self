// IERR Runtime – common error codes.
// Pure C-compatible enum so the HAL boundary stays language-neutral.
#ifndef IERR_ERROR_H
#define IERR_ERROR_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum ierr_status_t {
    IERR_OK                   = 0,
    IERR_ERR_INVALID_ARG      = 1,
    IERR_ERR_OUT_OF_MEMORY    = 2,
    IERR_ERR_NOT_FOUND        = 3,
    IERR_ERR_NOT_IMPLEMENTED  = 4,
    IERR_ERR_DEVICE           = 5,
    IERR_ERR_TIMEOUT          = 6,
    IERR_ERR_INTERNAL         = 7
} ierr_status_t;

const char* ierr_status_str(ierr_status_t s);

#ifdef __cplusplus
}
#endif

#endif // IERR_ERROR_H
