/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * CE transfer tracing: structured JSONL logging of data-layout analysis and
 * strategy selection for H2D/D2H transfers. Enabled at runtime via
 * FLEXKV_CE_TRACE=1 (zero overhead when disabled). Trace records contain all
 * parameters needed to replay the transfer_kv_blocks() call from Python.
 */
#pragma once

#include "ce_transfer.h"

#include <cstdint>
#include <string>

namespace flexkv {

/// Runtime switch — checks a std::atomic<bool> cached from FLEXKV_CE_TRACE.
/// First call reads the env var; subsequent calls do a single atomic load.
bool ce_trace_enabled();

/// Set the trace flag at runtime (pybind callable). Overrides the env var.
void ce_trace_set_enabled(bool enabled);

/// Flush and shut down the spdlog async logger.  Must be called before
/// process exit to ensure all queued trace entries are written to disk.
void ce_trace_shutdown();

/// Trace file path (cached from FLEXKV_CE_TRACE_FILE, default
/// ./flexkv_ce_trace.jsonl).
const std::string &ce_trace_file_path();

/// Max block IDs to log per entry (cached from FLEXKV_CE_TRACE_MAX_BLOCKS,
/// default 256).  0 = no limit (log all block IDs for full replay).
int ce_trace_max_blocks();

/// Convert CEPath to string name for JSON output.
const char *ce_path_name(CEPath path);

/// Convert BackendType int to string name.
const char *backend_type_name(int backend_type);

/// Main trace entry point — called after choose_path() in transfer_kv_blocks().
///
/// All pointer params (gpu_block_ids, cpu_block_ids) must be host-accessible
/// (CPU pinned memory).  The function is safe to call from GPU worker threads.
///
/// When tracing is disabled, this function returns immediately after a single
/// atomic<bool> load (< 1 ns).
void ce_trace_log(
    int backend_type,               ///< 0=VLLM, 1=TRTLLM, 2=SGLANG
    bool is_host_to_device,         ///< true=H2D, false=D2H
    int num_blocks,
    int start_layer_id,
    int num_layers,
    bool is_mla,
    int kv_dim,
    int64_t chunk_size_in_bytes,
    int64_t gpu_kv_stride_in_bytes,
    int64_t gpu_block_stride_in_bytes,
    int64_t gpu_layer_stride_in_bytes,
    int64_t cpu_kv_stride_in_bytes,
    int64_t cpu_layer_stride_in_bytes,
    int64_t cpu_block_stride_in_bytes,
    int64_t gpu_startoff_inside_chunks,
    int64_t cpu_startoff_inside_chunks,
    const CETransferConfig &ce_config,
    const CEAnalysis &ce_analysis,
    CEPath ce_path,
    const int64_t *gpu_block_ids,
    const int64_t *cpu_block_ids);

}  // namespace flexkv
