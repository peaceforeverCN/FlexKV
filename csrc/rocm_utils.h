#pragma once

#include <cstdint>

#if defined(FLEXKV_USE_ROCM)

// C++ sources are compiled by the host compiler in PyTorch extensions, so set
// the HIP platform explicitly before including the runtime headers.
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif
#include <hip/hip_runtime.h>

// PyTorch's ROCm build auto-generates a rocm_utils_hip.h shim that already
// #defines many CUDA→HIP symbol aliases (cudaSuccess, cudaMemcpyHostToDevice,
// cudaMalloc, etc.). Use #ifndef guards so we never conflict with that shim.
#ifndef cudaError_t
using cudaError_t = hipError_t;
#endif
#ifndef cudaEvent_t
using cudaEvent_t = hipEvent_t;
#endif
#ifndef cudaMemcpyKind
using cudaMemcpyKind = hipMemcpyKind;
#endif
#ifndef cudaStream_t
using cudaStream_t = hipStream_t;
#endif

#ifndef cudaSuccess
#define cudaSuccess hipSuccess
#endif
#ifndef cudaErrorNotReady
#define cudaErrorNotReady hipErrorNotReady
#endif
#ifndef cudaEventDisableTiming
#define cudaEventDisableTiming hipEventDisableTiming
#endif
#ifndef cudaHostAllocDefault
#define cudaHostAllocDefault hipHostMallocDefault
#endif
#ifndef cudaStreamNonBlocking
#define cudaStreamNonBlocking hipStreamNonBlocking
#endif
#ifndef cudaMemcpyHostToDevice
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#endif
#ifndef cudaMemcpyDeviceToHost
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#endif
#ifndef cudaMemcpyDeviceToDevice
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#endif

#ifndef cudaDeviceGetStreamPriorityRange
#define cudaDeviceGetStreamPriorityRange hipDeviceGetStreamPriorityRange
#endif
#ifndef cudaEventCreate
#define cudaEventCreate hipEventCreate
#endif
#ifndef cudaEventCreateWithFlags
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#endif
#ifndef cudaEventDestroy
#define cudaEventDestroy hipEventDestroy
#endif
#ifndef cudaEventElapsedTime
#define cudaEventElapsedTime hipEventElapsedTime
#endif
#ifndef cudaEventQuery
#define cudaEventQuery hipEventQuery
#endif
#ifndef cudaEventRecord
#define cudaEventRecord hipEventRecord
#endif
#ifndef cudaEventSynchronize
#define cudaEventSynchronize hipEventSynchronize
#endif
#ifndef cudaFree
#define cudaFree hipFree
#endif
#ifndef cudaFreeHost
#define cudaFreeHost hipHostFree
#endif
#ifndef cudaGetDevice
#define cudaGetDevice hipGetDevice
#endif
#ifndef cudaGetErrorString
#define cudaGetErrorString hipGetErrorString
#endif
#ifndef cudaGetLastError
#define cudaGetLastError hipGetLastError
#endif
#ifndef cudaLaunchHostFunc
#define cudaLaunchHostFunc hipLaunchHostFunc
#endif
#ifndef cudaMalloc
#define cudaMalloc hipMalloc
#endif
#ifndef cudaMallocHost
#define cudaMallocHost hipHostMalloc
#endif
#ifndef cudaMemcpyAsync
#define cudaMemcpyAsync hipMemcpyAsync
#endif
#ifndef cudaMemcpy2DAsync
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#endif
#ifndef cudaSetDevice
#define cudaSetDevice hipSetDevice
#endif
#ifndef cudaStreamCreate
#define cudaStreamCreate hipStreamCreate
#endif
#ifndef cudaStreamCreateWithPriority
#define cudaStreamCreateWithPriority hipStreamCreateWithPriority
#endif
#ifndef cudaStreamDestroy
#define cudaStreamDestroy hipStreamDestroy
#endif
#ifndef cudaStreamSynchronize
#define cudaStreamSynchronize hipStreamSynchronize
#endif

#ifndef CUDART_CB
#define CUDART_CB
#endif

// PyTorch's hipify maps nvtxRangeStartA -> roctxRangeStartA etc., so we
// include the ROCm tracer header to provide those symbols. Do NOT define
// no-op stubs here: hipify would transform their names and break the code
// (e.g. nvtxRangeId_t -> int turns "using nvtxRangeId_t = ..." into
// "using int = ...").
#include <roctracer/roctx.h>

#else

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#endif
