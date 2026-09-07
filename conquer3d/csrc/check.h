/**
 * @file check.h
 * @brief CUDA error-checking utilities and PyTorch tensor assertion macros.
 */

#ifndef CHECK_H
#define CHECK_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

/**
 * @def CHECK_CUDA(x)
 * @brief Asserts that tensor `x` resides on a CUDA device.
 */
#define CHECK_CUDA(x) \
    TORCH_CHECK((x).device().is_cuda(), #x " must be a CUDA tensor")

/**
 * @def CHECK_CONTIGUOUS(x)
 * @brief Asserts that tensor `x` has contiguous memory layout in global device memory.
 */
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

/**
 * @def CHECK_INPUT(x)
 * @brief Asserts that tensor `x` is both residing on CUDA and memory-contiguous.
 */
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

/**
 * @brief Verifies the status code of a CUDA runtime call.
 * 
 * @param[in] code Return code from CUDA runtime function (`cudaError_t`).
 * @param[in] file Calling source file name (`__FILE__`).
 * @param[in] line Calling line number (`__LINE__`).
 * @return bool True if successful, False if an error occurred.
 */
inline bool check_cuda_result(cudaError_t code, const char *file, int line)
{
    if (code == cudaSuccess)
        return true;

    fprintf(stderr, "CUDA error %u: %s (%s:%d)\n", unsigned(code), cudaGetErrorString(code), file, line);
    return false;
}

/**
 * @brief Aborts with a descriptive message when a CUDA API call fails.
 * @details Wraps a runtime call, checks its status, and raises a Torch error naming the
 * file and line. Used for internal allocations and copies where a silent failure would
 * surface much later as corrupt output.
 */
#define CHECK_CUDA_INTERNAL(code) check_cuda_result((code), __FILE__, __LINE__)

#endif // CHECK_H