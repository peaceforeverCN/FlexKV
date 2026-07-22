import pytest
import torch


pytestmark = pytest.mark.skipif(
    not bool(getattr(torch.version, "hip", None)), reason="requires ROCm PyTorch"
)


def test_rocm_backend_is_ce_only():
    from flexkv import c_ext
    from flexkv.common.config import GLOBAL_CONFIG_FROM_ENV

    assert GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_h2d
    assert GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_d2h
    assert not GLOBAL_CONFIG_FROM_ENV.enable_ce_memcpy2d
    assert not hasattr(c_ext, "GDSManager")
    assert not hasattr(c_ext, "ANSTransferContext")


def test_rocm_host_runtime_selection():
    from flexkv.transfer import host_buffer

    assert host_buffer._is_rocm()
