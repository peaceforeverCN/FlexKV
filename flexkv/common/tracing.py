"""Optional tracing facade that never requires NVIDIA NVTX on ROCm."""

import torch


class _NoopNvtx:
    @staticmethod
    def start_range(*args, **kwargs):
        return None

    @staticmethod
    def end_range(*args, **kwargs) -> None:
        return None

    @staticmethod
    def push_range(*args, **kwargs) -> None:
        return None

    @staticmethod
    def pop_range(*args, **kwargs) -> None:
        return None


def _load_nvtx():
    if getattr(torch.version, "hip", None):
        return _NoopNvtx()
    try:
        import nvtx
        return nvtx
    except ImportError:
        return _NoopNvtx()


nvtx = _load_nvtx()
