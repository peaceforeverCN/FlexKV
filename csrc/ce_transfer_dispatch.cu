#include "transfer.cuh"

#include <torch/extension.h>

#include "ce_trace.h"
#include "ce_transfer.h"
#include "rocm_utils.h"

namespace flexkv {

// ROCm CE-only implementation. The CUDA PTX kernel in transfer.cu is excluded
// from this target; all transfers are dispatched through CE memcpy/staging paths.
template <BackendType Type>
void transfer_kv_blocks(
    int num_blocks, int start_layer_id, int num_layers, int64_t *gpu_block_ids,
    GTensorHandler gpu_tensor_handler, int64_t gpu_startoff_inside_chunks,
    int64_t *cpu_block_ids, void *cpu_ptr, int64_t cpu_kv_stride_in_bytes,
    int64_t cpu_layer_stride_in_bytes, int64_t cpu_block_stride_in_bytes,
    int64_t cpu_startoff_inside_chunks, int64_t chunk_size_in_bytes,
    cudaStream_t stream, int transfer_num_cta, bool is_host_to_device,
    bool use_ce_transfer, bool is_mla, int64_t gpu_block_stride_in_bytes,
    bool sync, const CETransferConfig &ce_config) {
  (void)transfer_num_cta;
  TORCH_CHECK(use_ce_transfer,
              "ROCm builds support only the CE transfer path; enable "
              "FLEXKV_USE_CE_TRANSFER_H2D and FLEXKV_USE_CE_TRANSFER_D2H");
  TORCH_CHECK(num_blocks >= 0 && num_layers >= 0,
              "num_blocks and num_layers must be non-negative");
  TORCH_CHECK(chunk_size_in_bytes > 0 &&
                  chunk_size_in_bytes % static_cast<int64_t>(sizeof(int64_t)) == 0,
              "chunk_size_in_bytes must be a positive multiple of 8");
  TORCH_CHECK(gpu_startoff_inside_chunks % static_cast<int64_t>(sizeof(int64_t)) == 0 &&
                  cpu_startoff_inside_chunks % static_cast<int64_t>(sizeof(int64_t)) == 0,
              "GPU and CPU transfer offsets must be multiples of 8");
  TORCH_CHECK(cpu_kv_stride_in_bytes % static_cast<int64_t>(sizeof(int64_t)) == 0 &&
                  cpu_layer_stride_in_bytes % static_cast<int64_t>(sizeof(int64_t)) == 0 &&
                  cpu_block_stride_in_bytes % static_cast<int64_t>(sizeof(int64_t)) == 0,
              "CPU strides must be multiples of 8");

  if (num_blocks == 0 || num_layers == 0) {
    return;
  }
  TORCH_CHECK(gpu_block_ids != nullptr && cpu_block_ids != nullptr && cpu_ptr != nullptr,
              "transfer buffers must not be null");

  const int kv_dim = is_mla ? 1 : 2;
  int64_t *cpu_ptr_int64 = reinterpret_cast<int64_t *>(cpu_ptr);
  const int64_t cpu_kv_stride_int64 =
      cpu_kv_stride_in_bytes / static_cast<int64_t>(sizeof(int64_t));
  const int64_t cpu_block_stride_int64 =
      cpu_block_stride_in_bytes / static_cast<int64_t>(sizeof(int64_t));
  const int64_t cpu_layer_stride_int64 =
      cpu_layer_stride_in_bytes / static_cast<int64_t>(sizeof(int64_t));
  const int64_t cpu_startoff_inside_chunks_int64 =
      cpu_startoff_inside_chunks / static_cast<int64_t>(sizeof(int64_t));
  const int64_t gpu_startoff_inside_chunks_int64 =
      gpu_startoff_inside_chunks / static_cast<int64_t>(sizeof(int64_t));

  CETransferConfig rocm_ce_config = ce_config;
  // ROCm CE-only validation starts with the portable staging path. The CUDA
  // memcpy2D fast path is intentionally disabled until independently tuned.
  rocm_ce_config.enable_memcpy2d = false;

  const CEAnalysis analysis = analyze_ce_transfer(
      gpu_block_ids, cpu_block_ids, num_blocks, cpu_block_stride_in_bytes,
      chunk_size_in_bytes, gpu_block_stride_in_bytes);

  if (!rocm_ce_config.path_opt_enabled) {
    ce_trace_log(static_cast<int>(Type), is_host_to_device,
                 num_blocks, start_layer_id, num_layers, is_mla, kv_dim,
                 chunk_size_in_bytes,
                 gpu_tensor_handler.gpu_kv_stride * (int64_t)sizeof(int64_t),
                 gpu_block_stride_in_bytes,
                 gpu_tensor_handler.gpu_layer_stride * (int64_t)sizeof(int64_t),
                 cpu_kv_stride_in_bytes, cpu_layer_stride_in_bytes,
                 cpu_block_stride_in_bytes,
                 gpu_startoff_inside_chunks, cpu_startoff_inside_chunks,
                 rocm_ce_config, analysis, CEPath::PER_BLOCK,
                 gpu_block_ids, cpu_block_ids);
    ce_transfer_per_block<Type>(
        num_blocks, start_layer_id, num_layers, kv_dim, gpu_block_ids,
        gpu_tensor_handler, gpu_startoff_inside_chunks_int64, cpu_block_ids,
        cpu_ptr_int64, cpu_kv_stride_int64, cpu_layer_stride_int64,
        cpu_block_stride_int64, cpu_startoff_inside_chunks_int64,
        chunk_size_in_bytes, stream, is_host_to_device);
  } else {
    CEPath path;
    if (rocm_ce_config.force_path >= 0) {
      TORCH_CHECK(rocm_ce_config.force_path <= 5,
                  "force_path out of range [0,5]: ",
                  rocm_ce_config.force_path);
      path = static_cast<CEPath>(rocm_ce_config.force_path);
    } else {
      const bool is_full_block =
          static_cast<int64_t>(num_layers) * kv_dim * chunk_size_in_bytes ==
          cpu_block_stride_in_bytes;
      path = choose_path(analysis, rocm_ce_config, chunk_size_in_bytes,
                         is_host_to_device, is_full_block);
      // Compute-kernel bypass (BENCHMARK/DEBUG ONLY — disabled by default):
      // When total_bytes >= kernel_threshold, switch to a single HIP compute
      // copy kernel. Pure-transfer benchmarks show 1.3x speedup at num_blocks
      // >= 24, BUT in real inference this path competes with attention/MLP
      // compute kernels for CUs, destroying the transfer/compute overlap that
      // CE (SDMA, dedicated copy engine) provides for both D2H (async offload)
      // and H2D (layerwise). Default kernel_threshold=0 keeps CE. See
      // docs/hip_compute_kernel_perf.md for the full analysis.
      if (rocm_ce_config.kernel_threshold > 0) {
        const int64_t total_bytes =
            static_cast<int64_t>(num_blocks) * num_layers * kv_dim *
            chunk_size_in_bytes;
        if (total_bytes >= rocm_ce_config.kernel_threshold) {
          path = CEPath::COMPUTE_KERNEL;
        }
      }
    }

    ce_trace_log(static_cast<int>(Type), is_host_to_device,
                 num_blocks, start_layer_id, num_layers, is_mla, kv_dim,
                 chunk_size_in_bytes,
                 gpu_tensor_handler.gpu_kv_stride * (int64_t)sizeof(int64_t),
                 gpu_block_stride_in_bytes,
                 gpu_tensor_handler.gpu_layer_stride * (int64_t)sizeof(int64_t),
                 cpu_kv_stride_in_bytes, cpu_layer_stride_in_bytes,
                 cpu_block_stride_in_bytes,
                 gpu_startoff_inside_chunks, cpu_startoff_inside_chunks,
                 rocm_ce_config, analysis, path,
                 gpu_block_ids, cpu_block_ids);

    switch (path) {
      case CEPath::CONTIG_DIRECT:
        ce_transfer_contig_direct<Type>(
            num_blocks, start_layer_id, num_layers, kv_dim, gpu_block_ids,
            gpu_tensor_handler, gpu_startoff_inside_chunks_int64, cpu_block_ids,
            cpu_ptr_int64, cpu_kv_stride_int64, cpu_layer_stride_int64,
            cpu_block_stride_int64, cpu_startoff_inside_chunks_int64,
            chunk_size_in_bytes, stream, is_host_to_device);
        break;
      case CEPath::SEGMENT_DIRECT:
        ce_transfer_segment_direct<Type>(
            num_blocks, start_layer_id, num_layers, kv_dim, gpu_block_ids,
            gpu_tensor_handler, gpu_startoff_inside_chunks_int64, cpu_block_ids,
            cpu_ptr_int64, cpu_kv_stride_int64, cpu_layer_stride_int64,
            cpu_block_stride_int64, cpu_startoff_inside_chunks_int64,
            chunk_size_in_bytes, stream, is_host_to_device, analysis,
            rocm_ce_config);
        break;
      case CEPath::SEGMENT_SCATTER:
        ce_transfer_segment_scatter<Type>(
            num_blocks, start_layer_id, num_layers, kv_dim, gpu_block_ids,
            gpu_tensor_handler, gpu_startoff_inside_chunks_int64, cpu_block_ids,
            cpu_ptr_int64, cpu_kv_stride_int64, cpu_layer_stride_int64,
            cpu_block_stride_int64, cpu_startoff_inside_chunks_int64,
            chunk_size_in_bytes, stream, is_host_to_device, analysis,
            rocm_ce_config);
        break;
      case CEPath::GATHER_SCATTER:
        ce_transfer_gather_scatter<Type>(
            num_blocks, start_layer_id, num_layers, kv_dim, gpu_block_ids,
            gpu_tensor_handler, gpu_startoff_inside_chunks_int64, cpu_block_ids,
            cpu_ptr_int64, cpu_kv_stride_int64, cpu_layer_stride_int64,
            cpu_block_stride_int64, cpu_startoff_inside_chunks_int64,
            chunk_size_in_bytes, stream, is_host_to_device, analysis,
            rocm_ce_config);
        break;
      case CEPath::GATHER_DIRECT:
        ce_transfer_gather_direct<Type>(
            num_blocks, start_layer_id, num_layers, kv_dim, gpu_block_ids,
            gpu_tensor_handler, gpu_startoff_inside_chunks_int64, cpu_block_ids,
            cpu_ptr_int64, cpu_kv_stride_int64, cpu_layer_stride_int64,
            cpu_block_stride_int64, cpu_startoff_inside_chunks_int64,
            chunk_size_in_bytes, stream, is_host_to_device, analysis,
            rocm_ce_config);
        break;
      case CEPath::COMPUTE_KERNEL:
        ce_compute_kernel_transfer<Type>(
            num_blocks, start_layer_id, num_layers, kv_dim, gpu_block_ids,
            gpu_tensor_handler, gpu_startoff_inside_chunks_int64, cpu_block_ids,
            cpu_ptr_int64, cpu_kv_stride_int64, cpu_layer_stride_int64,
            cpu_block_stride_int64, cpu_startoff_inside_chunks_int64,
            chunk_size_in_bytes, stream, is_host_to_device);
        break;
    }
  }

  if (sync) {
    TORCH_CHECK(cudaStreamSynchronize(stream) == cudaSuccess,
                "ROCm CE stream synchronization failed");
  }
}

template void transfer_kv_blocks<BackendType::VLLM>(
    int, int, int, int64_t *, GTensorHandler, int64_t, int64_t *, void *,
    int64_t, int64_t, int64_t, int64_t, int64_t, cudaStream_t, int, bool, bool,
    bool, int64_t, bool, const CETransferConfig &);
template void transfer_kv_blocks<BackendType::TRTLLM>(
    int, int, int, int64_t *, GTensorHandler, int64_t, int64_t *, void *,
    int64_t, int64_t, int64_t, int64_t, int64_t, cudaStream_t, int, bool, bool,
    bool, int64_t, bool, const CETransferConfig &);
template void transfer_kv_blocks<BackendType::SGLANG>(
    int, int, int, int64_t *, GTensorHandler, int64_t, int64_t *, void *,
    int64_t, int64_t, int64_t, int64_t, int64_t, cudaStream_t, int, bool, bool,
    bool, int64_t, bool, const CETransferConfig &);

} // namespace flexkv
