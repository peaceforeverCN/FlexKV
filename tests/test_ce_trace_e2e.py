#!/usr/bin/env python3
"""End-to-end verification for CE trace: generation, JSON parsing, and replay.

Usage:
    python tests/test_ce_trace_e2e.py
"""
import json
import os
import sys
import tempfile
import faulthandler

faulthandler.enable()

import torch

# ---- 1. Enable CE tracing via env var BEFORE importing flexkv ----
TRACE_FILE = tempfile.mktemp(suffix=".jsonl")
os.environ["FLEXKV_CE_TRACE"] = "1"
os.environ["FLEXKV_CE_TRACE_FILE"] = TRACE_FILE
os.environ["FLEXKV_CE_TRACE_MAX_BLOCKS"] = "0"  # no truncation for replay

from flexkv import c_ext  # noqa: E402

print(f"[setup] trace file: {TRACE_FILE}")
print(f"[setup] ce_trace_enabled = {c_ext.ce_trace_enabled()}")
print(f"[setup] ce_trace_file_path = {c_ext.ce_trace_file_path()}")
print(f"[setup] ce_trace_max_blocks = {c_ext.ce_trace_max_blocks()}")

assert c_ext.ce_trace_enabled(), "CE trace should be enabled"
assert c_ext.ce_trace_max_blocks() == 0, "max_blocks should be 0 (no limit)"

# ---- 2. Build a minimal H2D transfer call ----
# Parameters: 4 blocks, 1 layer, non-MLA, chunk_size=1024 bytes
NUM_BLOCKS = 4
NUM_LAYERS = 1
CHUNK_SIZE = 1024  # bytes (128 int64 elements)
GPU_BLOCK_STRIDE = 1024
GPU_KV_STRIDE = 1024
GPU_LAYER_STRIDE = 8 * GPU_BLOCK_STRIDE  # enough for 8 blocks
CPU_BLOCK_STRIDE = 1024
CPU_KV_STRIDE = 1024
CPU_LAYER_STRIDE = 8 * CPU_BLOCK_STRIDE

# Contiguous block IDs [0, 1, 2, 3] — should pick CONTIG_DIRECT
gpu_block_ids = torch.arange(NUM_BLOCKS, dtype=torch.int64)
cpu_block_ids = torch.arange(NUM_BLOCKS, dtype=torch.int64)

# GPU tensor: VLLM needs num_layers pointers (one per layer)
gpu_tensors = [
    torch.empty(GPU_LAYER_STRIDE // 8, dtype=torch.int64, device="cuda")
    for _ in range(NUM_LAYERS)
]
gpu_tensor_ptrs = torch.tensor(
    [t.data_ptr() for t in gpu_tensors], dtype=torch.int64,
)
cpu_tensor = torch.empty(
    8 * CPU_BLOCK_STRIDE // 8, dtype=torch.int64, pin_memory=True
)

print("\n[step1] Triggering H2D transfer (contiguous, should pick CONTIG_DIRECT)...")
c_ext.transfer_kv_blocks(
    gpu_block_id_tensor=gpu_block_ids,
    gpu_tensor_ptrs_tensor=gpu_tensor_ptrs,
    gpu_kv_stride_in_bytes=GPU_KV_STRIDE,
    gpu_block_stride_in_bytes=GPU_BLOCK_STRIDE,
    gpu_layer_stride_in_bytes=GPU_LAYER_STRIDE,
    cpu_block_id_tensor=cpu_block_ids,
    cpu_tensor=cpu_tensor,
    cpu_kv_stride_in_bytes=CPU_KV_STRIDE,
    cpu_layer_stride_in_bytes=CPU_LAYER_STRIDE,
    cpu_block_stride_in_bytes=CPU_BLOCK_STRIDE,
    chunk_size_in_bytes=CHUNK_SIZE,
    start_layer_id=0,
    num_layers=NUM_LAYERS,
    transfer_num_cta=4,
    is_host_to_device=True,
    use_ce_transfer=True,
    is_mla=False,
    gpu_block_type=0,  # VLLM
    sync=True,
    ce_path_opt=True,
    ce_segment_threshold=8,
    ce_force_path=-1,
    ce_enable_memcpy2d=False,
    is_blockfirst=False,
)
print("[step1] H2D transfer completed.")

# ---- 3. Trigger a scattered H2D transfer (should pick SEGMENT_* or GATHER_*) ----
scattered_ids = torch.tensor([0, 2, 5, 7], dtype=torch.int64)
gpu_tensors2 = [
    torch.empty(16 * GPU_BLOCK_STRIDE // 8, dtype=torch.int64, device="cuda")
    for _ in range(NUM_LAYERS)
]
gpu_tensor_ptrs2 = torch.tensor(
    [t.data_ptr() for t in gpu_tensors2], dtype=torch.int64,
)
cpu_tensor2 = torch.empty(
    16 * CPU_BLOCK_STRIDE // 8, dtype=torch.int64, pin_memory=True
)

print("\n[step2] Triggering H2D transfer (scattered, should pick SEGMENT/GATHER)...")
c_ext.transfer_kv_blocks(
    gpu_block_id_tensor=scattered_ids,
    gpu_tensor_ptrs_tensor=gpu_tensor_ptrs2,
    gpu_kv_stride_in_bytes=GPU_KV_STRIDE,
    gpu_block_stride_in_bytes=GPU_BLOCK_STRIDE,
    gpu_layer_stride_in_bytes=16 * GPU_BLOCK_STRIDE,
    cpu_block_id_tensor=scattered_ids,
    cpu_tensor=cpu_tensor2,
    cpu_kv_stride_in_bytes=CPU_KV_STRIDE,
    cpu_layer_stride_in_bytes=16 * CPU_BLOCK_STRIDE,
    cpu_block_stride_in_bytes=CPU_BLOCK_STRIDE,
    chunk_size_in_bytes=CHUNK_SIZE,
    start_layer_id=0,
    num_layers=NUM_LAYERS,
    transfer_num_cta=4,
    is_host_to_device=True,
    use_ce_transfer=True,
    is_mla=False,
    gpu_block_type=0,
    sync=True,
    ce_path_opt=True,
    ce_segment_threshold=8,
    ce_force_path=-1,
    ce_enable_memcpy2d=False,
    is_blockfirst=False,
)
print("[step2] Scattered H2D transfer completed.")

# ---- 4. Flush spdlog async buffer (wait briefly for background thread) ----
print("\n[step3] Waiting for spdlog async flush...")
import time
time.sleep(0.5)  # give the async logger time to write

# ---- 5. Verify trace file exists and JSON is parseable ----
print(f"\n[step4] Verifying trace file: {TRACE_FILE}")
assert os.path.exists(TRACE_FILE), f"Trace file not created: {TRACE_FILE}"

with open(TRACE_FILE, "r") as f:
    lines = [l.strip() for l in f if l.strip()]

print(f"[step4] Found {len(lines)} trace entries")
assert len(lines) >= 2, f"Expected >= 2 trace entries, got {len(lines)}"

entries = []
for i, line in enumerate(lines):
    try:
        entry = json.loads(line)
        entries.append(entry)
        print(f"  [entry {i}] trace_id={entry['trace_id']} "
              f"direction={entry['direction']} "
              f"ce_path={entry['ce_path']} "
              f"batch_id={entry['ce_config']['batch_id']}")
    except json.JSONDecodeError as e:
        print(f"  [entry {i}] JSON PARSE FAILED: {e}")
        sys.exit(1)

print(f"\n[step4] All {len(entries)} entries parsed successfully.")

# Verify key fields exist
required_fields = [
    "trace_id", "ts_ns", "tid", "direction", "backend", "num_blocks",
    "start_layer_id", "num_layers", "is_mla", "kv_dim", "chunk_size_in_bytes",
    "strides", "offsets", "ce_config", "ce_analysis", "ce_path", "ce_path_id",
    "gpu_block_ids", "cpu_block_ids", "block_ids_truncated",
]
for i, entry in enumerate(entries):
    for field in required_fields:
        assert field in entry, f"Entry {i} missing field: {field}"
    # Verify ce_config has batch_id
    assert "batch_id" in entry["ce_config"], f"Entry {i} missing ce_config.batch_id"
    # Verify strides sub-fields
    for s in ["gpu_kv_stride", "gpu_block_stride", "gpu_layer_stride",
              "cpu_kv_stride", "cpu_layer_stride", "cpu_block_stride"]:
        assert s in entry["strides"], f"Entry {i} missing strides.{s}"

print("[step4] All required fields present in all entries.")

# ---- 6. Replay verification ----
print(f"\n[step5] Replay verification via CETraceReplayer...")
from flexkv.transfer.ce_replay import CETraceReplayer

replayer = CETraceReplayer(TRACE_FILE)
summary = replayer.summary()
print(f"  Summary: {summary}")
assert summary["total_entries"] >= 2, "Replayer should load >= 2 entries"

# Replay entry 0 with original strategy
print("\n  Replaying entry 0 (original strategy)...")
result0 = replayer.replay(entry_idx=0)
print(f"    trace_id={result0.trace_id} "
      f"path={result0.replayed_path} "
      f"success={result0.success} "
      f"elapsed={result0.elapsed_us:.1f}us "
      f"bw={result0.bandwidth_gbs:.2f}GB/s")
if not result0.success:
    print(f"    ERROR: {result0.error}")
    sys.exit(1)

# Replay entry 1 with forced PER_BLOCK strategy (force_path=-1 + path_opt=False)
print("\n  Replaying entry 1 (forced PER_BLOCK)...")
result1 = replayer.replay(entry_idx=1, path_opt=False)
print(f"    trace_id={result1.trace_id} "
      f"path={result1.replayed_path} "
      f"success={result1.success} "
      f"elapsed={result1.elapsed_us:.1f}us "
      f"bw={result1.bandwidth_gbs:.2f}GB/s")
if not result1.success:
    print(f"    ERROR: {result1.error}")
    sys.exit(1)

# Replay entry 0 with forced CONTIG_DIRECT (force_path=0)
print("\n  Replaying entry 0 (forced CONTIG_DIRECT)...")
result2 = replayer.replay(entry_idx=0, force_path=0)
print(f"    trace_id={result2.trace_id} "
      f"path={result2.replayed_path} "
      f"success={result2.success} "
      f"elapsed={result2.elapsed_us:.1f}us "
      f"bw={result2.bandwidth_gbs:.2f}GB/s")
if not result2.success:
    print(f"    ERROR: {result2.error}")
    sys.exit(1)

print("\n" + "=" * 60)
print("  ALL VERIFICATIONS PASSED")
print("=" * 60)
print(f"  1. Trace generation:     {len(entries)} entries written")
print(f"  2. JSON parsing:         all entries valid")
print(f"  3. Replay:               3 replays succeeded")
print("=" * 60)

# Flush spdlog before exit to avoid core dump on async thread cleanup
c_ext.ce_trace_shutdown()
