import ctypes
import weakref
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from flexkv.common.debug import flexkv_logger
from flexkv.storage.allocator import (
    alloc_hugepage_tensor,
    free_hugepage_tensor,
)

_gpu_runtime = None
_gpu_runtime_load_error: Optional[OSError] = None


def _is_rocm() -> bool:
    return bool(getattr(torch.version, "hip", None))


def _get_gpu_runtime():
    global _gpu_runtime
    global _gpu_runtime_load_error

    if _gpu_runtime is None and _gpu_runtime_load_error is None:
        library_names = ("libamdhip64.so", "libamdhip64.so.6") if _is_rocm() else ("libcudart.so",)
        for library_name in library_names:
            try:
                _gpu_runtime = ctypes.CDLL(library_name)
                break
            except OSError as exc:
                _gpu_runtime_load_error = exc

    if _gpu_runtime is None:
        runtime_name = "libamdhip64.so" if _is_rocm() else "libcudart.so"
        raise RuntimeError(f"{runtime_name} is unavailable: {_gpu_runtime_load_error}")
    return _gpu_runtime


def _runtime_symbol(cuda_name: str, hip_name: str):
    return getattr(_get_gpu_runtime(), hip_name if _is_rocm() else cuda_name)


def cuda_host_registration_available() -> bool:
    try:
        _get_gpu_runtime()
    except RuntimeError:
        return False
    return True


# Portable + Mapped is retained for CUDA compatibility. CE-only ROCm uses the
# same host-registration flags with HIP runtime entry points.
CUDA_HOST_REGISTER_PORTABLE = 0x01
CUDA_HOST_REGISTER_MAPPED = 0x02
CUDA_HOST_ALLOC_PORTABLE = 0x01
CUDA_HOST_ALLOC_MAPPED = 0x02


def cudaHostRegister(tensor: torch.Tensor) -> None:
    runtime_register = _runtime_symbol("cudaHostRegister", "hipHostRegister")
    ptr = tensor.data_ptr()
    size = tensor.numel() * tensor.element_size()
    flags = CUDA_HOST_REGISTER_PORTABLE | CUDA_HOST_REGISTER_MAPPED
    ret = runtime_register(
        ctypes.c_void_p(ptr), ctypes.c_size_t(size), ctypes.c_uint(flags)
    )
    if ret != 0:
        api = "hipHostRegister" if _is_rocm() else "cudaHostRegister"
        raise RuntimeError(f"{api} failed with error code {ret}")


def cudaHostUnregister(tensor: torch.Tensor) -> None:
    runtime_unregister = _runtime_symbol("cudaHostUnregister", "hipHostUnregister")
    ptr = tensor.data_ptr()
    ret = runtime_unregister(ctypes.c_void_p(ptr))
    if ret != 0:
        api = "hipHostUnregister" if _is_rocm() else "cudaHostUnregister"
        raise RuntimeError(f"{api} failed with error code {ret}")


@dataclass
class HostBufferHandle:
    tensor: torch.Tensor
    is_hugepage: bool = False
    is_cuda_registered: bool = False

    def __post_init__(self) -> None:
        if self.is_cuda_registered and not self.is_hugepage:
            raise ValueError("CUDA-registered host buffer must be HugePage-backed")

    @classmethod
    def pinned(cls, tensor: torch.Tensor) -> "HostBufferHandle":
        return cls(tensor=tensor)

    @classmethod
    def hugepage(cls, tensor: torch.Tensor) -> "HostBufferHandle":
        return cls(tensor=tensor, is_hugepage=True, is_cuda_registered=True)

    def release(self) -> None:
        if not self.is_hugepage:
            return

        if self.is_cuda_registered:
            try:
                cudaHostUnregister(self.tensor)
            except Exception as e:
                flexkv_logger.warning(
                    f"[host_buffer] release hugepage host buffer: cuda unregister failed ({e})"
                )
            self.is_cuda_registered = False

        free_hugepage_tensor(self.tensor)
        flexkv_logger.info("[host_buffer] release hugepage host buffer")
        self.is_hugepage = False


def alloc_mapped_host_tensor(num_elements: int, dtype: torch.dtype) -> torch.Tensor:
    """Allocate portable mapped host memory through CUDA or HIP runtime."""
    if num_elements <= 0:
        raise ValueError("num_elements must be positive")
    num_bytes = num_elements * dtype.itemsize
    host_ptr = ctypes.c_void_p()
    flags = CUDA_HOST_ALLOC_PORTABLE | CUDA_HOST_ALLOC_MAPPED
    runtime_alloc = _runtime_symbol("cudaHostAlloc", "hipHostMalloc")
    runtime_free = _runtime_symbol("cudaFreeHost", "hipHostFree")
    err = runtime_alloc(
        ctypes.byref(host_ptr),
        ctypes.c_size_t(num_bytes),
        ctypes.c_uint(flags),
    )
    if err != 0 or not host_ptr.value:
        api = "hipHostMalloc" if _is_rocm() else "cudaHostAlloc"
        raise RuntimeError(f"{api}(mapped) failed with error code {err}")

    buf_type = ctypes.c_uint8 * num_bytes
    raw = buf_type.from_address(host_ptr.value)
    np_arr = np.frombuffer(raw, dtype=np.uint8, count=num_bytes)
    tensor = (
        torch.frombuffer(np_arr, dtype=torch.uint8, count=num_bytes)
        .view(dtype)[:num_elements]
    )
    weakref.finalize(tensor, runtime_free, host_ptr)
    return tensor


def _allocate_pinned_cpu_tensor(num_elements: int, dtype: torch.dtype) -> HostBufferHandle:
    return HostBufferHandle.pinned(alloc_mapped_host_tensor(num_elements, dtype))


def _fallback_to_pinned(
    num_elements: int,
    dtype: torch.dtype,
    reason: Exception,
) -> HostBufferHandle:
    flexkv_logger.warning(
        f"[host_buffer] fallback to pinned host buffer ({reason})"
    )
    return _allocate_pinned_cpu_tensor(num_elements, dtype)


def allocate_host_buffer(
    num_elements: int,
    dtype: torch.dtype,
    use_hugepage: bool,
    hugepage_size_bytes: int,
) -> HostBufferHandle:
    if not use_hugepage:
        return _allocate_pinned_cpu_tensor(num_elements, dtype)

    flexkv_logger.info("[host_buffer] attempt hugepage host buffer")

    hugepage_buf = None
    try:
        hugepage_buf = alloc_hugepage_tensor(
            num_elements=num_elements,
            dtype=dtype,
            page_size_bytes=hugepage_size_bytes,
        )
        cudaHostRegister(hugepage_buf)
    except Exception as e:
        if hugepage_buf is not None:
            free_hugepage_tensor(hugepage_buf)
        return _fallback_to_pinned(num_elements, dtype, e)

    flexkv_logger.info(
        f"[host_buffer] hugepage host buffer ready: "
        f"{hugepage_buf.numel() * hugepage_buf.element_size() / (1024 ** 3):.3f} GB"
    )
    return HostBufferHandle.hugepage(hugepage_buf)