"""Benchmark: layerwise per-GPU pinned worker pool (Phase 1) before/after.

Measures end-to-end H2D wall time for the two layerwise hot paths that the
pinned-pool optimization targets:

  - single-group ``layerwise_transfer``      (all available GPUs)
  - multi-group + SWA ``layerwise_transfer_multi_group`` (DSv4-like, M>=2)

Both are exercised across notify modes (hostfunc / polling) and engines
(cuda kernel / CE).  Output: per-config min/p50/p90/max/mean/stdev + JSON.

== Before / after comparison ==
The pinned-pool change lives in ``csrc/layerwise.{h,cpp}``.  To compare:

  # after (this branch)
  FLEXKV_USE_ROCM=1 python setup.py build_ext --inplace
  python benchmarks/bench_layerwise_pinned_pool.py --json after.json

  # before (baseline, pre-migration)
  git stash            # or checkout the baseline commit of csrc/layerwise.*
  FLEXKV_USE_ROCM=1 python setup.py build_ext --inplace
  python benchmarks/bench_layerwise_pinned_pool.py --json before.json

  # diff the two JSON files (same machine, same GPU config, same iters).

Run:
  python benchmarks/bench_layerwise_pinned_pool.py
  python benchmarks/bench_layerwise_pinned_pool.py --paths single --notify hostfunc
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch

from flexkv.c_ext import LayerwiseTransferGroup
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType

DTYPE = torch.float16
ES = DTYPE.itemsize

# (num_layers, num_blocks, tpb, head_dim)
SCALES = {
    "small": (8, 16, 16, 512),
    "medium": (32, 64, 16, 512),
    "large": (80, 256, 16, 512),
}
NOTIFY_MODES = ["hostfunc", "polling"]
ENGINES = {"cuda": False, "ce": True}
WARMUP_ITERS = 3
BENCH_ITERS = 20


# --------------------------------------------------------------------------- #
# stats helper
# --------------------------------------------------------------------------- #
def _stats(times: List[float]) -> Dict[str, float]:
    s = sorted(times)
    n = len(s)

    def _pct(p: float) -> float:
        if n == 0:
            return float("nan")
        k = (n - 1) * p / 100.0
        f = int(k)
        c = min(f + 1, n - 1)
        return s[f] + (s[c] - s[f]) * (k - f)

    return {
        "min": s[0] if n else float("nan"),
        "p50": _pct(50),
        "p90": _pct(90),
        "max": s[-1] if n else float("nan"),
        "mean": statistics.mean(times) if n else float("nan"),
        "stdev": statistics.stdev(times) if n > 1 else 0.0,
        "n": n,
    }


# --------------------------------------------------------------------------- #
# single-group path (all GPUs) — mirrors microbenchmark_notify_mode.py
# --------------------------------------------------------------------------- #
def _make_single_group_tensors(num_layers, num_blocks, tpb, head_dim, num_gpus):
    heads_per_rank = 1
    kv_dim = 1
    gpu_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST, num_layer=num_layers,
        num_block=num_blocks, tokens_per_block=tpb, num_head=heads_per_rank,
        head_size=head_dim, is_mla=True)
    cpu_layout = KVCacheLayout(
        type=KVCacheLayoutType.BLOCKFIRST, num_layer=num_layers,
        num_block=num_blocks, tokens_per_block=tpb, num_head=1,
        head_size=head_dim, is_mla=True)
    all_gpu = []
    for g in range(num_gpus):
        full = torch.zeros(
            (num_layers, kv_dim, num_blocks, tpb, heads_per_rank, head_dim),
            dtype=DTYPE, device=f"cuda:{g}")
        all_gpu.append([full[i] for i in range(num_layers)])
    cpu_kv = torch.zeros(tuple(cpu_layout.kv_shape), dtype=DTYPE, pin_memory=True)
    return gpu_layout, cpu_layout, all_gpu, cpu_kv


def _make_single_group(cpu_kv, all_gpu, num_gpus, gpu_layout, num_layers):
    def strides_t(getter):
        return torch.tensor([getter() * ES] * num_gpus, dtype=torch.int64)
    return LayerwiseTransferGroup(
        num_gpus, all_gpu, cpu_kv, {}, num_layers,
        strides_t(gpu_layout.get_kv_stride),
        strides_t(gpu_layout.get_block_stride),
        strides_t(gpu_layout.get_layer_stride),
        strides_t(gpu_layout.get_chunk_size),
        0, 0, torch.empty(0, dtype=torch.int32), num_gpus,
        is_mla=True, is_blockfirst=True)


def bench_single_group(num_layers, num_blocks, tpb, head_dim, num_gpus,
                       use_ce, notify_mode,
                       warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    gpu_layout, cpu_layout, all_gpu, cpu_kv = _make_single_group_tensors(
        num_layers, num_blocks, tpb, head_dim, num_gpus)
    block_ids = torch.arange(num_blocks, dtype=torch.int64).pin_memory()
    cpu_stride_kv = cpu_layout.get_kv_stride() * ES
    cpu_stride_layer = cpu_layout.get_layer_stride() * ES
    cpu_stride_block = cpu_layout.get_block_stride() * ES
    cpu_stride_tp = cpu_stride_block // num_gpus
    chunk_size = gpu_layout.get_chunk_size() * ES

    lw = _make_single_group(cpu_kv, all_gpu, num_gpus, gpu_layout, num_layers)
    empty_ids = torch.empty(0, dtype=torch.int64)

    def run_once():
        lw.layerwise_transfer(
            ssd_block_ids=empty_ids, cpu_block_ids_d2h=empty_ids,
            ssd_layer_stride_in_bytes=0, ssd_kv_stride_in_bytes=0,
            num_blocks_per_file=0, round_robin=0, num_threads_per_device=0,
            gpu_block_id_tensor=block_ids, cpu_block_id_tensor=block_ids,
            cpu_kv_stride_in_bytes=cpu_stride_kv,
            cpu_layer_stride_in_bytes=cpu_stride_layer,
            cpu_block_stride_in_bytes=cpu_stride_block,
            cpu_chunk_size_in_bytes=chunk_size,
            h2d_cpu_kv_stride_in_bytes=cpu_stride_kv,
            h2d_cpu_layer_stride_in_bytes=cpu_stride_layer,
            cpu_tp_stride_in_bytes=cpu_stride_tp,
            transfer_cta_num=4, use_ce_transfer=use_ce, num_layers=num_layers,
            layer_granularity=1, is_mla=True, counter_id=0,
            mla_d2h_mode="sharded", notify_mode=notify_mode)

    for _ in range(warmup):
        run_once()
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_once()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    del lw
    return _stats(times)


# --------------------------------------------------------------------------- #
# multi-group + SWA path — reuses the test fixture helpers
# --------------------------------------------------------------------------- #
def _import_mg_fixture():
    """Import multi-group+SWA fixture builders from the test modules."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, os.path.join(repo_root, "tests"))
    # test_layerwise_eventfd_timing re-exports the SWA fixture helpers from
    # test_layerwise_multi_group_swa.
    import test_layerwise_eventfd_timing as t  # noqa: F401
    return (t._build_fixture_with_eventfds, t._run_h2d,
            t._dsv4_like_groups, t._seed_all_layers, t._drain_eventfd_units,
            t._layer_fds)


def bench_multi_group_swa(num_c4_layers: int = 4,
                          warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    (build_fx, run_h2d, dsv4_groups, seed_all,
     _drain, _layer_fds) = _import_mg_fixture()

    num_layers = num_c4_layers * 2
    groups = dsv4_groups(num_c4_layers)
    fx, owned = build_fx(groups, num_layers, has_swa=True)
    seed_all(fx, with_swa=True)

    # warmup
    for _ in range(warmup):
        run_h2d(fx, with_swa=True)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        run_h2d(fx, with_swa=True)
        times.append((time.perf_counter() - t0) * 1000.0)
    return _stats(times)


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
def _print_table(rows: List[Dict], title: str):
    print(f"\n== {title} ==")
    print(f"  {'path':<14} {'engine':<7} {'notify':<9} {'scale':<7} "
          f"{'p50_ms':>8} {'min_ms':>8} {'p90_ms':>8} {'max_ms':>8} "
          f"{'stdev':>7}")
    for r in rows:
        print(f"  {r['path']:<14} {r.get('engine','-'):<7} "
              f"{r.get('notify','-'):<9} {r.get('scale','-'):<7} "
              f"{r['p50']:8.2f} {r['min']:8.2f} {r['p90']:8.2f} "
              f"{r['max']:8.2f} {r['stdev']:7.2f}")


def main():
    ap = argparse.ArgumentParser(
        description="Benchmark layerwise per-GPU pinned worker pool.")
    ap.add_argument("--paths", nargs="+", default=["single", "multigroup"],
                    choices=["single", "multigroup"])
    ap.add_argument("--scales", nargs="+", default=list(SCALES),
                    choices=list(SCALES))
    ap.add_argument("--notify", nargs="+", default=NOTIFY_MODES,
                    choices=NOTIFY_MODES)
    ap.add_argument("--engines", nargs="+", default=list(ENGINES),
                    choices=list(ENGINES))
    ap.add_argument("--iters", type=int, default=BENCH_ITERS)
    ap.add_argument("--warmup", type=int, default=WARMUP_ITERS)
    ap.add_argument("--json", metavar="PATH", default=None,
                    help="write machine-readable results to PATH")
    args = ap.parse_args()

    num_gpus = torch.cuda.device_count()
    if num_gpus < 1:
        print("No GPU available")
        sys.exit(1)
    print(f"FlexKV layerwise pinned-pool benchmark  GPUs={num_gpus}  "
          f"warmup={args.warmup} iters={args.iters} dtype={str(DTYPE)}")

    results: List[Dict] = []

    if "single" in args.paths:
        for scale in args.scales:
            num_layers, num_blocks, tpb, head_dim = SCALES[scale]
            for engine in args.engines:
                use_ce = ENGINES[engine]
                for notify in args.notify:
                    try:
                        st = bench_single_group(
                            num_layers, num_blocks, tpb, head_dim, num_gpus,
                            use_ce, notify, args.warmup, args.iters)
                        row = {"path": "single", "engine": engine,
                               "notify": notify, "scale": scale, **st}
                        results.append(row)
                    except Exception as e:
                        print(f"[single/{scale}/{engine}/{notify}] ERROR: {e}")

    if "multigroup" in args.paths:
        try:
            st = bench_multi_group_swa(num_c4_layers=4,
                                       warmup=args.warmup, iters=args.iters)
            row = {"path": "multigroup+swa", "engine": "ce", "notify": "hostfunc",
                   "scale": "dsv4-8L", **st}
            results.append(row)
        except Exception as e:
            print(f"[multigroup+swa] ERROR: {e}")

    _print_table(results, "layerwise H2D wall time")

    if args.json:
        out = {
            "config": {"num_gpus": num_gpus, "warmup": args.warmup,
                       "iters": args.iters, "dtype": str(DTYPE)},
            "results": results,
        }
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"-> results written to {args.json}")


if __name__ == "__main__":
    main()
