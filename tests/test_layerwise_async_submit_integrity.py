"""Pipelined layerwise transfers must land correct data, not just land early.

The ring tests prove batch N+1 can arm while batch N is unconsumed, and the
IPC tests prove the markers cross processes. Neither would notice the failure
mode that actually matters here: an async submit whose source blocks are
recycled, or reused, while the copy engine is still reading them. That
corrupts KV silently -- no exception, no timeout, just wrong tokens.

So this drives the real C++ path (``submit_layerwise_async`` /
``poll_layer_completion`` on a real ``RegionBatchGroup``) with distinctive
per-block payloads and checks the bytes. Three scenarios:

  1. a single async op lands the right data;
  2. two ops in flight at once do not interleave into each other's blocks --
     this is the concurrency the pipeline introduces;
  3. a completion token is not reported done before the data is actually
     there (the check that would catch "poll always returns true").

Single process, single GPU, no KVManager: the question is whether the async
submit path is sound, not whether the scheduler wires it up.

Run (needs one GPU):
    python tests/test_layerwise_async_submit_integrity.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NUM_LAYERS = 6
NUM_BLOCKS = 8
TOKENS_PER_BLOCK = 16
HEADS = 4
HEAD_DIM = 32
KV_DIM = 2

ELEMS_PER_BLOCK_LAYER = KV_DIM * TOKENS_PER_BLOCK * HEADS * HEAD_DIM


def build(device: int):
    """One region spanning NUM_LAYERS, layout [layer][kv][block][tok][h][d]."""
    import torch

    from flexkv.c_ext import RegionBatchGroup, RegionDesc

    itemsize = 2  # float16
    gpu = torch.zeros(
        (NUM_LAYERS, KV_DIM, NUM_BLOCKS, TOKENS_PER_BLOCK, HEADS, HEAD_DIM),
        dtype=torch.float16, device=f"cuda:{device}",
    )
    cpu = torch.zeros(
        (NUM_BLOCKS, NUM_LAYERS, KV_DIM, TOKENS_PER_BLOCK, HEADS, HEAD_DIM),
        dtype=torch.float16,
    ).pin_memory()

    region = RegionDesc()
    region.name = "probe"
    region.num_layers = NUM_LAYERS
    region.kv_dim = KV_DIM
    region.num_kv_heads = HEADS
    region.num_tensors_per_gpu = 1
    region.cpu_ptr = cpu.data_ptr()
    # CPU is block-first: block, then layer, then kv.
    region.cpu_block_stride_in_bytes = (
        NUM_LAYERS * KV_DIM * TOKENS_PER_BLOCK * HEADS * HEAD_DIM * itemsize)
    region.cpu_layer_stride_in_bytes = (
        KV_DIM * TOKENS_PER_BLOCK * HEADS * HEAD_DIM * itemsize)
    region.cpu_kv_stride_in_bytes = TOKENS_PER_BLOCK * HEADS * HEAD_DIM * itemsize
    region.cpu_tp_stride_in_bytes = 0
    # GPU is layer-first.
    region.gpu_block_ptrs_flat = [gpu.data_ptr()]
    region.gpu_layer_strides_in_bytes = [
        KV_DIM * NUM_BLOCKS * TOKENS_PER_BLOCK * HEADS * HEAD_DIM * itemsize]
    region.gpu_kv_strides_in_bytes = [
        NUM_BLOCKS * TOKENS_PER_BLOCK * HEADS * HEAD_DIM * itemsize]
    region.gpu_block_strides_in_bytes = [
        TOKENS_PER_BLOCK * HEADS * HEAD_DIM * itemsize]
    region.gpu_chunk_sizes_in_bytes = [
        TOKENS_PER_BLOCK * HEADS * HEAD_DIM * itemsize]

    batch = RegionBatchGroup([device], [region])
    return batch, gpu, cpu


def make_requests(batch, cpu_ids, gpu_ids, layers):
    """One request per layer, tagged with the layer it completes.

    Built through the project's own ``make_requests`` so the field defaults
    (backend, share mode, CE selection) match what the worker really submits;
    only ``milestone_layer`` is set here, since that is what the notifier
    keys completion on.
    """
    import torch

    from flexkv.transfer.region_batch import make_requests as mk

    out = []
    for layer in layers:
        reqs = mk(
            num_regions=1,
            gpu_block_ids=torch.tensor(gpu_ids, dtype=torch.int64).pin_memory(),
            cpu_block_ids=torch.tensor(cpu_ids, dtype=torch.int64).pin_memory(),
            is_host_to_device=True,
            layer_id=layer,
            layer_granularity=1,
        )
        for req in reqs:
            req.milestone_layer = layer
        out.extend(reqs)
    return out


def fill_cpu(cpu, blocks, tag: float) -> None:
    """Give every (block, layer) a distinguishable value."""
    for b in blocks:
        for layer in range(NUM_LAYERS):
            cpu[b, layer] = tag + layer * 0.01 + b * 0.0001


def check_gpu(gpu, blocks, tag: float, label: str, failures: list) -> None:
    import torch

    for b in blocks:
        for layer in range(NUM_LAYERS):
            want = tag + layer * 0.01 + b * 0.0001
            got = gpu[layer, :, b]
            if not torch.allclose(
                got, torch.full_like(got, want), atol=1e-2, rtol=0
            ):
                failures.append(
                    f"{label}: block {b} layer {layer} wrong: "
                    f"expected ~{want:.4f}, got min={got.min().item():.4f} "
                    f"max={got.max().item():.4f}"
                )
                return


def main() -> int:
    import torch

    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return 0

    device = 0
    torch.cuda.set_device(device)
    failures: list[str] = []

    batch, gpu, cpu = build(device)
    if not hasattr(batch, "submit_layerwise_async"):
        print("SKIP: extension lacks submit_layerwise_async (stale build?)")
        return 0

    layers = list(range(NUM_LAYERS))

    # --- 1. one async op lands correct data ----------------------------------
    blocks_a = [0, 1, 2]
    fill_cpu(cpu, blocks_a, tag=1.0)
    reqs = make_requests(batch, blocks_a, blocks_a, layers)
    token = batch.submit_layerwise_async(reqs, [], 0)

    deadline = time.monotonic() + 60
    done = ok = False
    err = ""
    while time.monotonic() < deadline:
        done, ok, err = batch.poll_layer_completion(token)
        if done:
            break
        time.sleep(0.0005)
    if not done:
        print("FAIL single: async op never completed")
        return 1
    if not ok:
        print(f"FAIL single: async op failed: {err}")
        return 1
    torch.cuda.synchronize()
    check_gpu(gpu, blocks_a, 1.0, "single", failures)
    if not failures:
        print("PASS single: one async layerwise op landed correct data")

    # --- 2. two ops in flight must not corrupt each other --------------------
    gpu.zero_()
    blocks_b = [3, 4]
    blocks_c = [5, 6]
    fill_cpu(cpu, blocks_b, tag=2.0)
    fill_cpu(cpu, blocks_c, tag=3.0)

    reqs_b = make_requests(batch, blocks_b, blocks_b, layers)
    reqs_c = make_requests(batch, blocks_c, blocks_c, layers)

    # Both submitted before either is drained: the situation the serialised
    # worker could never reach, and the one the pipeline relies on.
    token_b = batch.submit_layerwise_async(reqs_b, [], 0)
    token_c = batch.submit_layerwise_async(reqs_c, [], 1)
    inflight = batch.inflight_completions
    if inflight < 2:
        failures.append(
            f"only {inflight} op(s) in flight after two async submits; the "
            "worker is still serialising and the pipeline cannot engage"
        )
    else:
        print(f"PASS concurrent: {inflight} ops in flight simultaneously")

    for name, tok in (("b", token_b), ("c", token_c)):
        deadline = time.monotonic() + 60
        done = False
        while time.monotonic() < deadline:
            done, ok, err = batch.poll_layer_completion(tok)
            if done:
                break
            time.sleep(0.0005)
        if not done:
            failures.append(f"concurrent op {name} never completed")
        elif not ok:
            failures.append(f"concurrent op {name} failed: {err}")

    torch.cuda.synchronize()
    before = len(failures)
    check_gpu(gpu, blocks_b, 2.0, "concurrent-b", failures)
    check_gpu(gpu, blocks_c, 3.0, "concurrent-c", failures)
    if len(failures) == before:
        print("PASS integrity: two concurrent ops both landed correct data")

    # --- 3. completion must not be claimed before the data is there ----------
    gpu.zero_()
    blocks_d = [7]
    fill_cpu(cpu, blocks_d, tag=4.0)
    reqs_d = make_requests(batch, blocks_d, blocks_d, layers)
    token_d = batch.submit_layerwise_async(reqs_d, [], 2)

    saw_pending = False
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        done, ok, err = batch.poll_layer_completion(token_d)
        if not done:
            saw_pending = True
        else:
            break
        time.sleep(0.0002)
    if not done:
        failures.append("op d never completed")
    else:
        # No explicit sync: a token reported done must mean the data is
        # visible, otherwise callers cannot release the source blocks on it.
        before = len(failures)
        check_gpu(gpu, blocks_d, 4.0, "post-completion", failures)
        if len(failures) == before:
            print("PASS contract: data is visible as soon as the token reports done"
                  + (" (and it did report pending first)" if saw_pending else ""))

    if batch.inflight_completions != 0:
        failures.append(
            f"{batch.inflight_completions} completion(s) still tracked after "
            "all were reaped; tokens are leaking"
        )
    else:
        print("PASS cleanup: no completion tokens leaked")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nALL PASS: async layerwise submits are sound under concurrency")
    return 0


if __name__ == "__main__":
    sys.exit(main())
