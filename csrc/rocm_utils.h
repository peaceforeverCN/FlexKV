#pragma once

#include <cstdint>

#if defined(FLEXKV_USE_ROCM)

// C++ sources are compiled by the host compiler in PyTorch extensions, so set
// the HIP platform explicitly before including the runtime headers.
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif
#include <hip/hip_runtime.h>

using cudaError_t = hipError_t;
using cudaEvent_t = hipEvent_t;
using cudaMemcpyKind = hipMemcpyKind;
using cudaStream_t = hipStream_t;

constexpr cudaError_t cudaSuccess = hipSuccess;
constexpr cudaError_t cudaErrorNotReady = hipErrorNotReady;
constexpr unsigned int cudaEventDisableTiming = hipEventDisableTiming;
constexpr unsigned int cudaHostAllocDefault = hipHostMallocDefault;
constexpr unsigned int cudaStreamNonBlocking = hipStreamNonBlocking;
constexpr cudaMemcpyKind cudaMemcpyHostToDevice = hipMemcpyHostToDevice;
constexpr cudaMemcpyKind cudaMemcpyDeviceToHost = hipMemcpyDeviceToHost;
constexpr cudaMemcpyKind cudaMemcpyDeviceToDevice = hipMemcpyDeviceToDevice;

#define cudaDeviceGetStreamPriorityRange hipDeviceGetStreamPriorityRange
#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy hipEventDestroy
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventQuery hipEventQuery
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaFree hipFree
#define cudaFreeHost hipHostFree
#define cudaGetDevice hipGetDevice
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaLaunchHostFunc hipLaunchHostFunc
#define cudaMalloc hipMalloc
#define cudaMallocHost hipHostMalloc
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#define cudaSetDevice hipSetDevice
#define cudaStreamCreate hipStreamCreate
#define cudaStreamCreateWithPriority hipStreamCreateWithPriority
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize

#ifndef CUDART_CB
#define CUDART_CB
#endif

using nvtxRangeId_t = std::uint64_t;
inline nvtxRangeId_t nvtxRangeStartA(const char *) { return 0; }
inline void nvtxRangeEnd(nvtxRangeId_t) {}
inline int nvtxRangePushA(const char *) { return 0; }
inline int nvtxRangePop() { return 0; }

#else

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#endif
