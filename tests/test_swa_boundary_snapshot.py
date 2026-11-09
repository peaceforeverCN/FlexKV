# SPDX-License-Identifier: Apache-2.0
"""SWA snapshot at a page boundary interior to an existing node.

Scenario (DSv4 SWA branch point): request A+B was stored, so the tree has one
node [A|B] with an SWA snapshot only at its end. A later request A+C repairs
the SWA window at |A| on the GPU tree and stores prefix A. All Full blocks of A
are already cached, so before this change the PUT was dropped as empty and A
stayed unreachable from FlexKV (Full hit without SWA is discarded whole).

With FLEXKV_SWA_BOUNDARY_SNAPSHOT the PUT carries only the trailing SWA page
and commit splits [A|B] into [A]->[B], mounting the snapshot on [A].
"""
import numpy as np
import pytest
import torch

pytest.importorskip("flexkv.c_ext")

from flexkv.cache.cache_engine import GlobalCacheEngine  # noqa: E402
from flexkv.cache.radixtree import RadixTreeIndex  # noqa: E402
from flexkv.common.block import SequenceMeta  # noqa: E402
from flexkv.common.config import CacheConfig, ModelConfig, SWAPoolConfig  # noqa: E402
from flexkv.common.debug import flexkv_logger  # noqa: E402
from flexkv.common.transfer import TransferType  # noqa: E402

pytestmark = pytest.mark.smoke

TPB = 16


@pytest.fixture(autouse=True)
def _silence_flexkv_logger():
    prev_level = flexkv_logger.logger.level
    prev_enabled = flexkv_logger.enabled
    flexkv_logger.set_level("OFF")
    yield
    flexkv_logger.logger.setLevel(prev_level)
    flexkv_logger.enabled = prev_enabled


def _engine(num_slots=256):
    cc = CacheConfig(
        tokens_per_block=TPB,
        enable_cpu=True, enable_ssd=False, enable_remote=False,
        num_cpu_blocks=4096,
    )
    cc.swa = SWAPoolConfig(
        enabled=True, num_slots=num_slots, num_swa_layers=1,
        bytes_per_token_per_layer=64,
    )
    cc.enable_swa_transfer = True
    mc = ModelConfig(
        num_layers=4, num_kv_heads=1, head_size=128,
        kv_dim=1, dtype=torch.bfloat16, tp_size=1, dp_size=1,
    )
    return GlobalCacheEngine(cc, mc)


def _tokens(n_blocks, base):
    rs = np.random.RandomState(base)
    return rs.randint(0, 30000, size=n_blocks * TPB, dtype=np.int64)


def _put(eng, rid, tok):
    mask = np.ones_like(tok, dtype=np.int64)
    sm = np.arange(tok.shape[0], dtype=np.int64)
    return eng.put(rid, tok, mask, sm, dp_client_id=0)


def _complete(op_cb, cb):
    for c in op_cb.values():
        c()
    cb()


def _match(eng, tok):
    seq = SequenceMeta(token_ids=tok, tokens_per_block=TPB)
    seq.gen_hashes()
    return eng.cpu_cache_engine.match(seq)


def _swa_ops(graph):
    return [op for op in graph._op_map.values() if getattr(op, "is_swa", False)]


def _full_ops(graph):
    return [op for op in graph._op_map.values()
            if not getattr(op, "is_swa", False)
            and op.transfer_type != TransferType.VIRTUAL]


def _seed_branch(eng):
    a, b, c = _tokens(3, 1), _tokens(2, 2), _tokens(2, 3)
    g, _rm, cb, op_cb, _e = _put(eng, 1, np.concatenate([a, b]))
    _complete(op_cb, cb)
    before = _match(eng, np.concatenate([a, c]))
    assert before.num_matched_blocks == 3
    assert before.swa_hit_blocks == 0, "no snapshot at |A| yet"
    return a, b, c


def test_swa_only_put_mounts_snapshot_mid_node():
    eng = _engine()
    a, b, c = _seed_branch(eng)
    used = eng.cpu_cache_engine.swa_pool.num_used

    graph, return_mask, cb, op_cb, _e = _put(eng, 2, a)
    assert graph.num_ops > 0, "SWA-only PUT must not be dropped"
    assert not return_mask.any(), "no Full block is transferred"
    assert _full_ops(graph) == []
    swa = _swa_ops(graph)
    assert len(swa) == 1 and swa[0].transfer_type == TransferType.D2H
    assert swa[0].op_id in graph._swa_gpu_transfer_op_id
    assert eng.cpu_cache_engine.swa_pool.num_used == used + 1

    # Hidden until the graph drains (same contract as the regular PUT).
    for f in op_cb.values():
        f()
    assert _match(eng, np.concatenate([a, c])).swa_hit_blocks == 0
    cb()

    hit = _match(eng, np.concatenate([a, c]))
    assert hit.num_matched_blocks == 3
    assert hit.swa_hit_blocks == 3
    # The original tail snapshot survives the split.
    full = _match(eng, np.concatenate([a, b]))
    assert full.num_matched_blocks == 5
    assert full.swa_hit_blocks == 5
    assert eng.cpu_cache_engine.swa_pool.num_used == used + 1


def test_swa_only_put_binds_tail_gpu_slot_with_empty_full_mapping():
    """Launch binds the GPU SWA page even though no Full block is mapped."""
    eng = _engine()
    a, _b, _c = _seed_branch(eng)
    graph, return_mask, _cb, _op_cb, _e = _put(eng, 2, a)
    assert not return_mask.any()
    full_ids = eng.slot_mapping_to_block_ids(np.array([], dtype=np.int64), TPB)
    swa_mapping = np.arange(5 * TPB, 8 * TPB, dtype=np.int64)
    swa_ids = eng.slot_mapping_to_block_ids(swa_mapping, TPB)
    graph.set_gpu_blocks(full_ids, swa_ids)
    swa = _swa_ops(graph)[0]
    assert swa.src_block_ids.tolist() == [int(swa_ids[-1])]


def test_match_deeper_than_put_end_is_not_swa_only():
    """A trailing-masked PUT whose match runs past its end is left empty."""
    eng = _engine()
    a, b, _c = _seed_branch(eng)
    tok = np.concatenate([a, b])
    mask = np.ones_like(tok, dtype=np.int64)
    mask[3 * TPB:] = 0
    used = eng.cpu_cache_engine.swa_pool.num_used
    graph, _rm, _cb, _op_cb, _e = eng.put(
        2, tok, mask, np.arange(3 * TPB, dtype=np.int64), dp_client_id=0)
    assert graph.num_ops == 0
    assert eng.cpu_cache_engine.swa_pool.num_used == used


def test_boundary_with_snapshot_stays_empty_put():
    eng = _engine()
    a, b, _c = _seed_branch(eng)
    _g, _rm, cb, op_cb, _e = _put(eng, 2, a)
    _complete(op_cb, cb)
    used = eng.cpu_cache_engine.swa_pool.num_used
    graph, _rm, _cb, _op_cb, _e = _put(eng, 3, a)
    assert graph.num_ops == 0
    assert eng.cpu_cache_engine.swa_pool.num_used == used
    # The exact end of an existing node is also already covered.
    graph, _rm, _cb, _op_cb, _e = _put(eng, 4, np.concatenate([a, b]))
    assert graph.num_ops == 0


def test_switch_off_keeps_previous_behavior():
    eng = _engine()
    eng.swa_boundary_snapshot = False
    a, _b, c = _seed_branch(eng)
    graph, _rm, _cb, _op_cb, _e = _put(eng, 2, a)
    assert graph.num_ops == 0
    assert _match(eng, np.concatenate([a, c])).swa_hit_blocks == 0


def test_no_free_swa_slot_returns_empty():
    eng = _engine(num_slots=1)
    a, _b, c = _seed_branch(eng)
    # The only slot is mounted on the [A|B] leaf, which the PUT protects.
    graph, _rm, cb, op_cb, _e = _put(eng, 2, a)
    if graph.num_ops:
        _complete(op_cb, cb)
    # Either the slot was reclaimed from the old tail or the PUT was dropped;
    # the pool never over-commits.
    assert eng.cpu_cache_engine.swa_pool.num_used <= 1


def test_python_index_split_node_preserves_leaf_and_swa():
    idx = RadixTreeIndex(tokens_per_block=TPB)
    tok = _tokens(5, 7)
    seq = SequenceMeta(token_ids=tok, tokens_per_block=TPB)
    seq.gen_hashes()
    leaf = idx.insert(seq, np.arange(5, dtype=np.int64))
    idx.set_swa(leaf, 0)
    assert idx.split_node(leaf, 0) is None
    assert idx.split_node(leaf, 5) is None
    prefix = idx.split_node(leaf, 3)
    assert prefix is not None and prefix.size() == 3 and leaf.size() == 2
    assert leaf.parent is prefix and not prefix.has_swa() and leaf.has_swa()
    assert leaf.head_hash() in idx.leaf_nodes
    assert prefix.head_hash() not in idx.leaf_nodes
    idx.set_swa(prefix, 1)
    short = SequenceMeta(token_ids=np.concatenate([tok[:3 * TPB], _tokens(2, 8)]),
                         tokens_per_block=TPB)
    m = idx.match_prefix(short)
    assert m.num_matched_blocks == 3 and m.swa_hit_blocks == 3
    assert idx.match_prefix(seq).swa_hit_blocks == 5
