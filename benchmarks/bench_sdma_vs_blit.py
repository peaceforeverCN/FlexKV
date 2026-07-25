#!/usr/bin/env python3
"""Compare SDMA vs HIP compute kernel for small per-layer transfers.

Tests the hypothesis that a HIP compute copy kernel is faster than SDMA
(hipMemcpyAsync) for layerwise per-layer transfers, where data sizes are
small and SDMA launch overhead dominates.
"""
import argparse
import statistics
import time

import torch

from flexkv import c_ext


def bench_transfer(num_blocks, num_layers, iters, is_h2d, is_mla, chunk_size,
                   mode, kernel_threshold):
    """Benchmark a single transfer_kv_blocks call."""
    kv_dim = 1 if is_mla else 2
    block_stride = chunk_size
    layer_stride = (num_blocks + 10) * block_stride
    kv_stride = chunk_size

    block_ids = torch.arange(num_blocks, dtype=torch.int64)

    # Allocate GPU tensors (one per layer for VLLM backend, gpu_block_type=0)
    gpu_tensors = [
        torch.empty(layer_stride * kv_dim // 8, dtype=torch.int64, device="cuda")
        for _ in range(num_layers)
    ]
    gpu_tensor_ptrs = torch.tensor(
        [t.data_ptr() for t in gpu_tensors], dtype=torch.int64
    )

    # CPU pinned tensor — must hold all layers' data
    total_cpu_elems = layer_stride * kv_dim * num_layers // 8
    cpu_tensor = torch.empty(total_cpu_elems, dtype=torch.int64, pin_memory=True)

    force_path = 5 if mode == "compute_kernel" else -1
    ce_kernel_threshold = kernel_threshold if mode == "auto" else 0

    common_kwargs = dict(
        gpu_block_id_tensor=block_ids,
        gpu_tensor_ptrs_tensor=gpu_tensor_ptrs,
        gpu_kv_stride_in_bytes=kv_stride,
        gpu_block_stride_in_bytes=block_stride,
        gpu_layer_stride_in_bytes=layer_stride,
        cpu_block_id_tensor=block_ids,
        cpu_tensor=cpu_tensor,
        cpu_kv_stride_in_bytes=kv_stride,
        cpu_layer_stride_in_bytes=layer_stride,
        cpu_block_stride_in_bytes=block_stride,
        chunk_size_in_bytes=chunk_size,
        start_layer_id=0,
        num_layers=num_layers,
        transfer_num_cta=4,
        is_host_to_device=is_h2d,
        use_ce_transfer=True,
        is_mla=is_mla,
        gpu_block_type=0,
        sync=True,
        ce_path_opt=True,
        ce_segment_threshold=8,
        ce_force_path=force_path,
        ce_enable_memcpy2d=False,
        is_blockfirst=False,
        ce_kernel_threshold=ce_kernel_threshold,
    )

    # Warmup
    for _ in range(10):
        c_ext.transfer_kv_blocks(**common_kwargs)

    torch.cuda.synchronize()

    # Timed runs
    times_us = []
    for _ in range(iters):
        t0 = time.perf_counter()
        c_ext.transfer_kv_blocks(**common_kwargs)
        torch.cuda.synchronize()
        times_us.append((time.perf_counter() - t0) * 1e6)

    return statistics.median(times_us), min(times_us), max(times_us)


def main():
    parser = argparse.ArgumentParser(
        description="SDMA vs HIP compute kernel benchmark")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--mode", choices=["sdma", "compute_kernel", "auto"],
                        default="sdma")
    parser.add_argument("--kernel-threshold", type=int, default=32768)
    args = parser.parse_args()

    mode_name = {"sdma": "SDMA", "compute_kernel": "COMPUTE_KERNEL",
                 "auto": f"AUTO(thr={args.kernel_threshold})"}[args.mode]
    print(f"\n{'='*70}")
    print(f"  Transfer Mode: {mode_name}")
    print(f"  Iterations: {args.iters}")
    print(f"{'='*70}\n")

    # Single-layer small transfers (layerwise per-layer scenario)
    test_cases = [
        (4, 1, True, 1024, "4 blk, 1 lyr (4 KB)"),
        (8, 1, True, 1024, "8 blk, 1 lyr (8 KB)"),
        (16, 1, True, 1024, "16 blk, 1 lyr (16 KB)"),
        (32, 1, True, 1024, "32 blk, 1 lyr (32 KB)"),
        (64, 1, True, 1024, "64 blk, 1 lyr (64 KB)"),
        (128, 1, True, 1024, "128 blk, 1 lyr (128 KB)"),
        (256, 1, True, 1024, "256 blk, 1 lyr (256 KB)"),
    ]

    print(f"{'Config':<30} {'H2D med':>10} {'H2D min':>10} {'D2H med':>10} {'D2H min':>10}")
    print("-" * 75)

    for num_blocks, num_layers, is_mla, chunk_size, label in test_cases:
        h2d_med, h2d_min, h2d_max = bench_transfer(
            num_blocks, num_layers, args.iters, True, is_mla, chunk_size,
            args.mode, args.kernel_threshold)
        d2h_med, d2h_min, d2h_max = bench_transfer(
            num_blocks, num_layers, args.iters, False, is_mla, chunk_size,
            args.mode, args.kernel_threshold)
        print(f"{label:<30} {h2d_med:>10.1f} {h2d_min:>10.1f} {d2h_med:>10.1f} {d2h_min:>10.1f}")

    print(f"\n{'='*70}\n")
    return 0


if __name__ == "__main__":
    main()
