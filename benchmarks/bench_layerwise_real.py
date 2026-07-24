#!/usr/bin/env python3
"""Real layerwise benchmark: per-layer transfer with sync=false.

Simulates the true layerwise protocol: layer_granularity=1, one
transfer_kv_blocks call per layer, async (sync=false), with cudaEvent
measuring each layer's completion. This is what LayerwiseTransferGroup
actually does in production.
"""
import argparse
import statistics
import time

import torch

from flexkv import c_ext


def bench_layerwise(num_blocks, num_layers, iters, is_h2d, chunk_size, mode):
    """Per-layer transfer, sync=false, measured via cudaEvent."""
    kv_dim = 1  # MLA
    block_stride = chunk_size
    layer_stride = (num_blocks + 10) * block_stride

    block_ids = torch.arange(num_blocks, dtype=torch.int64, pin_memory=True)
    # One GPU tensor per layer (VLLM backend, gpu_block_type=0)
    gpu_tensors = [
        torch.empty(layer_stride * kv_dim // 8, dtype=torch.int64, device="cuda")
        for _ in range(num_layers)
    ]
    # Pin the ptrs array so GPU can read it directly (like LayerwiseTransferGroup
    # which uses cudaMallocHost for gpu_blocks_)
    gpu_tensor_ptrs = torch.tensor(
        [t.data_ptr() for t in gpu_tensors], dtype=torch.int64, pin_memory=True)
    total_cpu_elems = layer_stride * kv_dim * num_layers // 8
    cpu_tensor = torch.empty(total_cpu_elems, dtype=torch.int64, pin_memory=True)

    force_path = 5 if mode == "compute_kernel" else -1
    ce_kt = 32768 if mode == "auto" else 0

    # Create cuda events for per-layer timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    kwargs_template = dict(
        gpu_block_id_tensor=block_ids,
        gpu_tensor_ptrs_tensor=gpu_tensor_ptrs,
        gpu_kv_stride_in_bytes=chunk_size,
        gpu_block_stride_in_bytes=block_stride,
        gpu_layer_stride_in_bytes=layer_stride,
        cpu_block_id_tensor=block_ids,
        cpu_tensor=cpu_tensor,
        cpu_kv_stride_in_bytes=chunk_size,
        cpu_layer_stride_in_bytes=layer_stride,
        cpu_block_stride_in_bytes=block_stride,
        chunk_size_in_bytes=chunk_size,
        transfer_num_cta=4,
        is_host_to_device=is_h2d,
        use_ce_transfer=True,
        is_mla=True,
        gpu_block_type=0,
        ce_path_opt=True,
        ce_segment_threshold=8,
        ce_force_path=force_path,
        ce_enable_memcpy2d=False,
        is_blockfirst=False,
        ce_kernel_threshold=ce_kt,
    )

    # Warmup
    for _ in range(5):
        for layer in range(num_layers):
            c_ext.transfer_kv_blocks(
                start_layer_id=layer, num_layers=1,
                sync=False, **kwargs_template)
        torch.cuda.synchronize()

    # Timed runs: measure full layerwise loop (all layers, async)
    times_us = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start_event.record()

        for layer in range(num_layers):
            c_ext.transfer_kv_blocks(
                start_layer_id=layer, num_layers=1,
                sync=False, **kwargs_template)

        end_event.record()
        torch.cuda.synchronize()
        times_us.append(start_event.elapsed_time(end_event) * 1e3)  # ms->us

    return statistics.median(times_us), min(times_us)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=50)
    args = p.parse_args()

    # Realistic layerwise scenarios: layer_granularity=1
    # Per-layer data = num_blocks * chunk_size (MLA, 1024B chunk)
    cases = [
        (4, 32, "4 blk, 32 lyr (4KB/lyr, 128KB total)"),
        (8, 32, "8 blk, 32 lyr (8KB/lyr, 256KB total)"),
        (16, 32, "16 blk, 32 lyr (16KB/lyr, 512KB total)"),
        (32, 32, "32 blk, 32 lyr (32KB/lyr, 1MB total)"),
        (64, 32, "64 blk, 32 lyr (64KB/lyr, 2MB total)"),
        (4, 80, "4 blk, 80 lyr (4KB/lyr, 320KB total)"),
        (8, 80, "8 blk, 80 lyr (8KB/lyr, 640KB total)"),
        (16, 80, "16 blk, 80 lyr (16KB/lyr, 1.25MB total)"),
    ]

    results = {}
    for mode in ["sdma", "compute_kernel", "auto"]:
        name = {"sdma": "SDMA", "compute_kernel": "COMPUTE_KERNEL",
                "auto": "AUTO(thr=32768)"}[mode]
        print(f"\n{'='*80}")
        print(f"  Mode: {name}  (layer_granularity=1, sync=false, async)")
        print(f"{'='*80}")
        print(f"{'Config':<45} {'H2D med':>10} {'H2D min':>10} {'D2H med':>10} {'D2H min':>10}")
        print("-" * 80)
        results[mode] = {}
        for nb, nl, label in cases:
            try:
                h2d_med, h2d_min = bench_layerwise(
                    nb, nl, args.iters, True, 1024, mode)
                d2h_med, d2h_min = bench_layerwise(
                    nb, nl, args.iters, False, 1024, mode)
                results[mode][label] = (h2d_med, d2h_med)
                print(f"{label:<45} {h2d_med:>10.1f} {h2d_min:>10.1f} {d2h_med:>10.1f} {d2h_min:>10.1f}")
            except Exception as e:
                results[mode][label] = (0, 0)
                print(f"{label:<45} ERROR: {e}")
            torch.cuda.empty_cache()

    # Print comparison
    print(f"\n{'='*80}")
    print(f"  Comparison (layer_granularity=1, async, pinned metadata)")
    print(f"{'='*80}")
    print(f"{'Config':<45} {'SDMA H2D':>10} {'CK H2D':>10} {'speedup':>8} {'Auto H2D':>10} {'Auto/SDMA':>10}")
    print("-" * 100)
    for _, _, label in cases:
        sdma_h2d = results["sdma"].get(label, (0,0))[0]
        ck_h2d = results["compute_kernel"].get(label,(0,0))[0]
        auto_h2d = results["auto"].get(label, (0,0))[0]
        h2d_sp = sdma_h2d / ck_h2d if ck_h2d > 0 else 0
        auto_sp = sdma_h2d / auto_h2d if auto_h2d > 0 else 0
        print(f"{label:<45} {sdma_h2d:>10.1f} {ck_h2d:>10.1f} {h2d_sp:>7.2f}x {auto_h2d:>10.1f} {auto_sp:>9.2f}x")
    print()


if __name__ == "__main__":
    main()
