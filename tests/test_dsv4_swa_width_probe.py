"""The host SWA width is measured off the GPU pool, not assumed.

``cache_config.swa.bytes_per_token_per_layer`` is built before any GPU pool
exists, so it can only guess a DSv4 KV layout (the V4 one: 584 B/token padded
to 585 by the 576-byte page alignment). sglang picks the layout from
``SGLANG_DSV4_KV_LAYOUT`` and the GPU arch, and the V4.1 layouts are narrower,
so the guess has to be replaced by a measurement before ``StorageEngine`` turns
it into the host/SSD/remote stride.

The numbers below are the real geometries of DeepSeek-V4.1-Flash on SM120 with
``SGLANG_DSV4_KV_LAYOUT=v41`` + ``SGLANG_SM120_FLASHMLA_BACKEND=flashinfer``:

======================  ==========  ===========  =============
pool                    KVLayout    B/token      page bytes
======================  ==========  ===========  =============
SWA (page 256)          V41         528          135168
ratio-1 KV (page 256)   V41_FP4     288          73728
ratio-2 KV (page 128)   V41_FP4     288          36864
indexer (page 64)       fp4         68           4352
======================  ==========  ===========  =============

For reference the V4 layout, which the bootstrap estimate assumes, is 584
logical B/token and 585 once ``ceil_div(256 * 584, 576) * 576 = 149760`` page
bytes are spread back over the 256 tokens.
"""

import ast
import inspect
from types import SimpleNamespace

import pytest
import torch

from flexkv.common.pool import PoolId
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.integration.sglang.connector import FlexKVConnector
from flexkv.transfer.workers.gpu_cpu import GPUCPUTransferWorker

# DSv4 SWA page geometries, as sglang's KVLayout.page_bytes() computes them:
# ceil_div(page_size * bytes_per_token, page_align) * page_align.
V4_PAGE_BYTES = 149760  # V4:      ceil_div(256 * 584, 576) * 576 -> 585 B/token
V41_PAGE_BYTES = 135168  # V41:     256 * 528, already 512-aligned
V41_FP4_PAGE_BYTES = 73728  # V41_FP4: 256 * 288, already 256-aligned

V4_WIDTH = 585
V41_WIDTH = 528
V41_FP4_WIDTH = 288

PAGE_SIZE = 256


class _FakeSWAPool:
    """The 2D ``[num_pages, page_bytes]`` uint8 rows every DSv4 pool exposes."""

    def __init__(self, page_bytes, *, page_size=PAGE_SIZE, n_layers=2, n_pages=4):
        self.page_size = page_size
        self.kv_buffer = [
            torch.zeros(n_pages, page_bytes, dtype=torch.uint8)
            for _ in range(n_layers)
        ]


def _connector(swa_pool, *, configured):
    connector = FlexKVConnector.__new__(FlexKVConnector)
    connector._swa_kv_pool = swa_pool
    connector._label = "[test]"
    connector.cache_config = SimpleNamespace(
        swa=SimpleNamespace(bytes_per_token_per_layer=configured)
    )
    return connector


def _swa_layout(width, *, tokens_per_block=PAGE_SIZE, num_block=4, num_layer=2):
    """An SWA layout as ``StorageEngine`` / the connector build it: one
    single-head single-region plane of ``width`` bytes per token per layer."""
    return KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=num_layer,
        num_block=num_block,
        tokens_per_block=tokens_per_block,
        num_head=1,
        head_size=width,
        kv_dim=1,
        num_kv_heads=1,
    )


# --------------------------------------------------------------------------
# _measure_swa_bytes_per_token
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("page_bytes", "expected"),
    [
        (V4_PAGE_BYTES, V4_WIDTH),
        (V41_PAGE_BYTES, V41_WIDTH),
        (V41_FP4_PAGE_BYTES, V41_FP4_WIDTH),
    ],
    ids=["v4", "v41", "v41_fp4"],
)
def test_measure_reports_each_dsv4_layout_width(page_bytes, expected):
    pool = _FakeSWAPool(page_bytes)
    assert FlexKVConnector._measure_swa_bytes_per_token(pool) == expected


def test_measure_divides_by_the_pools_own_page_size():
    # SM120 direct-SWA gives the GPU pool 64-token pages; the per-token width
    # must come out the same as with 256-token pages.
    pool = _FakeSWAPool(V41_PAGE_BYTES // 4, page_size=64)
    assert FlexKVConnector._measure_swa_bytes_per_token(pool) == V41_WIDTH


def test_measure_returns_none_without_a_usable_buffer():
    assert FlexKVConnector._measure_swa_bytes_per_token(SimpleNamespace()) is None
    assert (
        FlexKVConnector._measure_swa_bytes_per_token(
            SimpleNamespace(kv_buffer=[], page_size=PAGE_SIZE)
        )
        is None
    )
    # A 1D buffer is not the paged layout this derivation is valid for.
    assert (
        FlexKVConnector._measure_swa_bytes_per_token(
            SimpleNamespace(
                kv_buffer=[torch.zeros(8, dtype=torch.uint8)], page_size=PAGE_SIZE
            )
        )
        is None
    )
    assert (
        FlexKVConnector._measure_swa_bytes_per_token(
            SimpleNamespace(kv_buffer=[torch.zeros(4, 8, dtype=torch.uint8)], page_size=0)
        )
        is None
    )


def test_measure_rejects_a_stride_that_is_not_a_whole_number_of_tokens():
    pool = _FakeSWAPool(V41_PAGE_BYTES + 1)
    with pytest.raises(RuntimeError, match="not divisible"):
        FlexKVConnector._measure_swa_bytes_per_token(pool)


# --------------------------------------------------------------------------
# _sync_swa_bytes_per_token
# --------------------------------------------------------------------------


def test_sync_replaces_the_v4_estimate_with_the_measured_v41_width():
    # The regression this guards: the bootstrap estimate is the V4 585, the GPU
    # pool is V41. Keeping 585 over-allocates every host SWA slot by 10% and
    # ships an uninitialised 14592 B tail per slot to SSD / remote / mooncake.
    connector = _connector(_FakeSWAPool(V41_PAGE_BYTES), configured=V4_WIDTH)
    connector._sync_swa_bytes_per_token()
    assert connector.cache_config.swa.bytes_per_token_per_layer == V41_WIDTH


def test_sync_is_a_noop_when_the_estimate_already_matches():
    connector = _connector(_FakeSWAPool(V4_PAGE_BYTES), configured=V4_WIDTH)
    connector._sync_swa_bytes_per_token()
    assert connector.cache_config.swa.bytes_per_token_per_layer == V4_WIDTH


def test_sync_keeps_the_configured_width_when_the_pool_is_not_measurable():
    connector = _connector(SimpleNamespace(), configured=V4_WIDTH)
    connector._sync_swa_bytes_per_token()
    assert connector.cache_config.swa.bytes_per_token_per_layer == V4_WIDTH


def test_sync_does_nothing_without_an_swa_pool_or_config():
    connector = _connector(None, configured=V4_WIDTH)
    connector._sync_swa_bytes_per_token()
    assert connector.cache_config.swa.bytes_per_token_per_layer == V4_WIDTH

    connector = _connector(_FakeSWAPool(V41_PAGE_BYTES), configured=V4_WIDTH)
    connector.cache_config.swa = None
    connector._sync_swa_bytes_per_token()  # must not raise
    assert connector.cache_config.swa is None


def test_sync_agrees_with_the_width_the_gpu_registration_derives():
    """The host pool and the GPU registration must not drift apart: both take
    the width from ``_measure_swa_bytes_per_token`` (connector.py registers it
    as the SWA layout's ``head_size``)."""
    pool = _FakeSWAPool(V41_PAGE_BYTES)
    connector = _connector(pool, configured=V4_WIDTH)
    connector._sync_swa_bytes_per_token()

    registered_head_size = FlexKVConnector._measure_swa_bytes_per_token(pool)
    assert (
        connector.cache_config.swa.bytes_per_token_per_layer == registered_head_size
    )


# --------------------------------------------------------------------------
# GPUCPUTransferWorker._check_uniform_chunk_widths
# --------------------------------------------------------------------------


def _check(cpu_width, gpu_width, *, pool_id=PoolId.SWA, gpu_tokens=PAGE_SIZE):
    cpu_layout = _swa_layout(cpu_width)
    gpu_layout = _swa_layout(gpu_width, tokens_per_block=gpu_tokens)
    # uint8: itemsize 1, so chunk elements are chunk bytes.
    GPUCPUTransferWorker._check_uniform_chunk_widths(
        pool_id,
        [gpu_layout],
        [gpu_layout.get_chunk_size()],
        cpu_layout,
        cpu_layout.get_chunk_size(),
        pool_id.name.lower(),
    )


def test_check_accepts_a_host_mirror_as_wide_as_the_gpu_pool():
    _check(V41_WIDTH, V41_WIDTH)
    _check(V4_WIDTH, V4_WIDTH)
    _check(V41_FP4_WIDTH, V41_FP4_WIDTH)


def test_check_accepts_differing_page_sizes_at_the_same_width():
    # SM120 direct-SWA: 64-token GPU pages against the host's 256. Only the
    # per-token width has to agree.
    _check(V41_WIDTH, V41_WIDTH, gpu_tokens=64)


def test_check_rejects_a_host_mirror_wider_than_the_gpu_pool():
    # Exactly the stale-estimate case: host still V4 585, GPU already V41 528.
    with pytest.raises(ValueError, match="pool geometry mismatch"):
        _check(V4_WIDTH, V41_WIDTH)


def test_check_rejects_a_host_mirror_narrower_than_the_gpu_pool():
    # The dangerous direction: cpp would copy the GPU's chunk size out of a
    # host base computed with the smaller stride, overrunning the slot.
    with pytest.raises(ValueError, match="pool geometry mismatch"):
        _check(V41_FP4_WIDTH, V41_WIDTH)


def test_check_error_names_both_widths_and_the_pool():
    with pytest.raises(ValueError) as excinfo:
        _check(V4_WIDTH, V41_WIDTH)
    message = str(excinfo.value)
    assert "swa pool geometry mismatch" in message
    assert str(V4_WIDTH) in message
    assert str(V41_WIDTH) in message


def test_check_ignores_the_full_kv_pool():
    # The main pool's CPU layout carries the node-wide head count while each
    # GPU layout carries this rank's shard, so its chunk sizes legitimately
    # differ under TP and must not be compared.
    _check(V4_WIDTH, V41_WIDTH, pool_id=PoolId.FULL_KV)


def test_check_passes_for_the_real_v41_swa_geometry():
    """End to end on the deployed numbers: a measured host pool and the GPU
    pool it was measured from always agree, whatever the layout."""
    for page_bytes in (V4_PAGE_BYTES, V41_PAGE_BYTES, V41_FP4_PAGE_BYTES):
        pool = _FakeSWAPool(page_bytes)
        connector = _connector(pool, configured=V4_WIDTH)
        connector._sync_swa_bytes_per_token()
        measured = connector.cache_config.swa.bytes_per_token_per_layer

        cpu_layout = _swa_layout(measured)
        gpu_layout = _swa_layout(measured)
        GPUCPUTransferWorker._check_uniform_chunk_widths(
            PoolId.SWA,
            [gpu_layout],
            [gpu_layout.get_chunk_size()],
            cpu_layout,
            cpu_layout.get_chunk_size(),
            "swa",
        )
        # The host slot is exactly the GPU page, with no padding tail.
        assert cpu_layout.get_chunk_size() == page_bytes


# --------------------------------------------------------------------------
# Call sites
#
# Both helpers only do their job from one specific place, and neither place is
# reachable from a test: ``FlexKVConnector.__init__`` wants a live SGLang KV
# cache and a KVManager, ``_init_uniform`` wants imported IPC handles and a
# CUDA context. Losing a call would leave every assertion above passing while
# the host pool silently keeps a stale width, so the call sites are asserted on
# the parsed source instead.
# --------------------------------------------------------------------------


def _method_calls(func):
    """Names of the ``self.<name>(...)`` calls in *func*'s body."""
    tree = ast.parse(inspect.cleandoc(inspect.getsource(func)))
    return {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }


def test_connector_init_syncs_the_swa_width():
    # Must run before KVManager starts: StorageEngine turns the value into the
    # CPU / SSD / REMOTE SWA strides, and radixshmem publishes it as the slot
    # size.
    assert "_sync_swa_bytes_per_token" in _method_calls(FlexKVConnector.__init__)


def test_init_uniform_checks_the_pool_geometry():
    assert "_check_uniform_chunk_widths" in _method_calls(
        GPUCPUTransferWorker._init_uniform
    )
