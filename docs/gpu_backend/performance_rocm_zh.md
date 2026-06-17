# FlexKV 在 AMD ROCm 上的 D2H / H2D 性能记录与优化分析

> 本文档记录 FlexKV 在 AMD GPU + ROCm 环境下 KV-cache 在 GPU↔CPU 之间传输(D2H / H2D)的实测性能,定位瓶颈,并给出可落地的优化方向。
>
> 关联文档:
> - 安装与分阶段验证: [`verification_rocm_zh.md`](./verification_rocm_zh.md)
> - 设计/抽象层全景: [`README_zh.md`](./README_zh.md)

---

## 1. 测试环境

| 项 | 值 |
| -- | -- |
| 机器 | 8× AMD MI 系列 (gfx942) |
| ROCm | 7.2.26015 (`/opt/rocm-7.2.0`) |
| PyTorch | `2.9.1+rocm7.2.0`,`torch.version.hip='7.2.26015'`,`cuda=None` |
| Python venv | `/opt/venv` |
| FlexKV 分支 | `star/gpu_backend-on-swa2` (HEAD `aa6bdac`) |
| C 扩展 | `flexkv/c_ext.so` (hipcc 编译,见 verification 文档 Stage 1) |
| 传输模式 | **CE transfer** (`FLEXKV_USE_CE_TRANSFER_H2D=1` + `FLEXKV_USE_CE_TRANSFER_D2H=1`,ROCm 必须,见 verification §7.8.2) |

运行期必备环境变量:

```bash
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
export FLEXKV_ENABLE_MPS=0
export FLEXKV_USE_CE_TRANSFER_H2D=1
export FLEXKV_USE_CE_TRANSFER_D2H=1
```

---

## 2. 测试方法

使用官方微基准 `benchmarks/benchmark_single_batch.py`(`put` 走 **D2H** GPU→CPU,`get` 走 **H2D** CPU→GPU),配置 `benchmarks/example_config.yml`:

| 模型参数 | 值 |
| -- | -- |
| num_layers | 64 |
| num_kv_heads | 8 |
| head_size | 128 |
| dtype | bfloat16 |
| tokens_per_block | 16 |
| tp_size / dp_size | 1 / 1 |
| cpu_cache_gb / ssd_cache_gb | 8 / 16 |

派生量:
- **每 token** = `64 layers × 8 heads × 128 × 2(K,V) × 2 bytes` = **256 KiB**
- **每个传输 chunk**(单 layer × 单 K或V × 单 block) = `16 tokens × 8 × 128 × 2 bytes` = **32 KiB**

> **复现注意**:官方脚本用默认 `fork` 起 `tp_client` 子进程,父进程已初始化 CUDA → `Cannot re-initialize CUDA in forked subprocess` 直接卡死。需用 `spawn`(测试代码均如此)。本次用一个 `mp.set_start_method('spawn', force=True)` 的包装脚本复用官方 `benchmark_flexkv()`,未改动仓库文件。完整复现见 §6。

---

## 3. 当前性能数据

### 3.1 FlexKV 端到端(CE transfer 模式,单 batch,含 Python 编排)

| 传输量 | D2H `put` (GPU→CPU) | H2D `get` (CPU→GPU) | 备注 |
| -- | -- | -- | -- |
| 2 GB | **2.99 GB/s** | **2.66 GB/s** | 纯 CPU↔GPU |
| 4 GB | 2.95 GB/s | 2.72 GB/s | 纯 CPU↔GPU |
| 6 GB | 3.08 GB/s | 1.68 GB/s | get 已溢出 SSD(见下) |
| 8 GB | 3.06 GB/s | 1.51 GB/s | get 已溢出 SSD |

- **D2H 稳定在 ~3.0 GB/s**,与传输量无关。
- **H2D 在 2–4GB 为 ~2.7 GB/s**;6–8GB 下降是因为 CPU cache 仅 8GB,大于此的数据溢出到 SSD,`get` 有一部分变成 SSD→CPU→GPU(非纯 H2D),不代表 H2D 本身变慢。
- **纯 CPU↔GPU 的代表值**:**D2H ≈ 3.0 GB/s,H2D ≈ 2.7 GB/s**。

### 3.2 裸硬件上限对照(torch pinned 单次 2GB 连续 `copy_`)

| 方向 | 带宽 |
| -- | -- |
| H2D (pinned→GPU) | **53.7 GB/s** |
| D2H (GPU→pinned) | **45.2 GB/s** |

### 3.3 结论

FlexKV 实测仅达到裸硬件带宽的 **约 6%**(3.0 / 45–54 GB/s)。**瓶颈不是 PCIe/xGMI 带宽,而是传输被拆成海量小块拷贝后的 per-call 提交开销**(launch-bound)。

---

## 4. 瓶颈分析

### 4.1 根因:CE transfer 路径按 (layer × kv × block) 逐块发 `gpuMemcpyAsync`

`csrc/gpu_backend/nvidia/transfer.cu`(ROCm 编译的就是它 hipify 后的版本)的 CE 路径是**三重嵌套循环,每个最小 chunk 发一次异步拷贝**:

```cpp
// csrc/gpu_backend/nvidia/transfer.cu:126-159 (use_ce_transfer 分支)
for (int i = 0; i < num_layers; i++)          // 64
  for (int j = 0; j < kv_dim; j++)            // 2 (K,V)
    for (int k = 0; k < num_blocks; k++) {    // 512 (for 2GB)
      ...
      gpuMemcpyAsync(dst, src, chunk_size_in_bytes, dir, stream);  // 32 KiB
    }
// 最后 gpuStreamSynchronize(stream)
```

代入本次配置(2GB 传输):

| 量 | 值 |
| -- | -- |
| 拷贝次数 | `64 × 2 × 512` = **65536** 次 |
| 每次大小 | **32 KiB** |
| 实测 D2H 用时 | ~669 ms |
| **平均每次 `hipMemcpyAsync`** | **~10.2 µs** |

32 KiB 在 PCIe Gen4/5 上的纯传输时间 < 1 µs,而实测每次 ~10 µs → **~9 µs 全是每次调用的提交/排队固定开销**。这是典型的 **launch-bound**:增大总传输量(更多 block)带宽不变(§3.1 D2H 恒为 ~3GB/s 印证了这一点——固定开销随次数线性增长,被传输量完全抵消)。

### 4.2 为什么 ROCm 不能用更快的 kernel 路径

`use_ce_transfer=false` 时走 `transfer_kv_blocks_kernel`,**一次 kernel launch** 由 GPU 线程直接读写 host pinned 内存(zero-copy)。但 ROCm 上 host pinned 内存默认不建立 device-side mapping,GPU shader deref host 地址会 `Memory access fault`(verification §7.8.2)。所以 ROCm 被强制留在 CE 模式 → 落进 §4.1 的小块拷贝陷阱。

### 4.3 次要开销:单线程单流串行 + 每次调用的 Python/pinned 开销

- 65536 次拷贝由**单个 host 线程**在**单条 stream** 上串行 enqueue,无并行(`flexkv/transfer/worker.py` 的 `_do_transfer` 单次 `transfer_kv_blocks` 调用)。
- Python 侧每次传输还有 `gpu_ptrs.contiguous().pin_memory()` 等准备开销(`worker.py:544`),以及单 batch 同步 `wait(completely=True)` 的调度往返。
- ROCm fallback 丢掉了 NVPTX 的 cache-hint(`ld.global.nc`/`st.global.cg`),改用 plain float4 load/store(§7.2)——但这在 CE 模式下不生效(CE 不走 kernel),仅影响 kernel 模式。

---

## 5. 优化方向(按性价比排序)

### 5.1 ⭐ 合并连续 block 的拷贝(最高性价比,纯 C++,不依赖驱动特性)

CE 路径里,如果相邻 block 在 CPU 端和 GPU 端**地址都连续**(同一 layer/kv 下 block_stride 连续、block_id 连续),就可以把它们合并成**一次大 `gpuMemcpyAsync`**,而不是每 block 一次。

- 预期收益:把 65536 次 → 在 block 连续场景下可降到 `64 × 2`=128 次(每次 16MiB),launch 次数降 3 个数量级,有望逼近硬件带宽。
- 实现:在 `transfer.cu` CE 分支里对 `k` 维做 run-length 合并(检测 `cpu_block_idx`/`gpu_block_idx` 连续段),连续段用单次 memcpy。
- 风险:block_id 通常不连续(cache 分配是离散的),合并率取决于实际负载;但即使只能合并 layer 内同 kv 的部分段也有收益。

### 5.2 ⭐ 用 2D/3D 批量拷贝 API 折叠 layer/kv 维(中等改动,确定收益)

即使 block 不连续,**layer 维和 kv 维的 stride 是规则的**。可用 `hipMemcpy2DAsync`(甚至构造 3D)把"对同一组 block、跨所有 layer"的拷贝**用一次带 stride 的拷贝**表达,把内层 `i`(layer)循环折叠掉。

- 预期收益:拷贝次数从 `L×kv×B` 降到 `kv×B`(本例 65536→1024),launch 开销降 64×。
- 实现:`gpuMemcpy2DAsync(dst, dpitch, src, spitch, width=chunk, height=num_layers, ...)`,其中 pitch=layer_stride。需保证 layer_stride 在两端都规则(当前 layout 满足)。

### 5.3 ⭐⭐ 修复 ROCm kernel zero-copy 路径(根治,改动较大)

让 ROCm 也能走 `use_ce_transfer=false` 的单次 kernel 路径(verification §7.8.2 的长期方案):

- 把 worker 里给 kernel 用的 host 指针表/CPU pinned 缓冲改成对 device 可见:`hipHostRegister(..., hipHostRegisterMapped)` + `hipHostGetDevicePointer`,把 device 端别名传给 kernel;
- 或在 RocmBackend 注册 host tensor 时统一用 mapped 方式。
- 预期收益:整次传输 **1 次 kernel launch** 完成,彻底消除 §4.1 的 65536 次提交开销,理论可达 kernel 访存上限。
- 代价:需要改 `host_buffer.py` / `worker.py` 的 host 内存注册路径 + `transfer.cu` kernel 在 ROCm 下用 device 别名;并验证 zero-copy 在 MI 上的实际带宽(host zero-copy 走 PCIe,未必比 CE 的 DMA 快,需实测对比 5.1/5.2)。

### 5.4 多流并行 + 多 CTA(低成本叠加项)

- CE 路径目前单 stream。把 65536(或合并后的)拷贝**分散到 N 条 stream** 并行 enqueue,可重叠提交开销与 DMA。
- kernel 路径已有 `transfer_num_cta`(默认 4),可调大以提高并行度(仅 kernel 模式生效)。
- 预期收益:与 5.1/5.2 叠加,进一步隐藏延迟;单独使用对 launch-bound 也有 2–4× 改善空间。

### 5.5 启用 layerwise / pipelined 传输(架构层)

NVIDIA 上有 `LayerwiseTransferGroup` 做 layer 粒度的计算/传输流水线,ROCm wheel 当前不含(verification §2)。在推理集成场景按 layer 边算边传可隐藏传输延迟。属于集成层优化,微基准看不出,但端到端 TTFT 受益。

### 5.6 减少 Python 编排开销(小优化)

- `worker.py:544` 每次传输都 `gpu_ptrs.contiguous().pin_memory()`,可缓存/复用 pinned 指针表(指针表本身很小,但高频调用累积可观)。
- 微基准的单 batch 同步 `wait(completely=True)` 不代表真实并发吞吐;真实负载下多 op pipeline 会摊薄调度开销。评估真实吞吐应跑 `benchmark_reuse_serving.py` 类多请求场景。

### 5.7 增大 tokens_per_block(配置层,立即可试)

`tokens_per_block` 从 16 增大到 64/128,直接让每个 chunk 从 32KiB 增大到 128/256KiB,**拷贝次数同比下降**,在 launch-bound 下近似线性提升带宽。代价是 KV-cache 块粒度变粗、前缀复用命中率可能下降——需在带宽与命中率间权衡。

---

## 6. 复现步骤

```bash
source /opt/venv/bin/activate
cd /sgl-workspace/FlexKV
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
export FLEXKV_ENABLE_MPS=0 FLEXKV_USE_CE_TRANSFER_H2D=1 FLEXKV_USE_CE_TRANSFER_D2H=1
mkdir -p ssd_cache1 ssd_cache2
rm -f /tmp/flexkv_*

# 包装脚本(强制 spawn,复用官方 benchmark_flexkv;避免 fork+CUDA 卡死)
cat > /tmp/run_bench.py <<'PY'
import sys, argparse, multiprocessing as mp
sys.path.insert(0, "/sgl-workspace/FlexKV/benchmarks")
def main():
    from utils import load_config
    import benchmark_single_batch as bsb
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/sgl-workspace/FlexKV/benchmarks/example_config.yml")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--sequence-length", type=int, default=1024)
    ap.add_argument("--cache-ratio", type=float, default=1.0)
    a = ap.parse_args()
    mc, cc = load_config(a.config)
    bc = bsb.BenchmarkConfig(-1, a.batch_size, a.sequence_length, a.cache_ratio, False)
    tpb = cc.tokens_per_block
    bc.sequence_length = ((bc.sequence_length - 1)//tpb + 1)*tpb
    bsb.benchmark_flexkv(mc, cc, bc)
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True); main()
PY

# 2GB / 4GB 干净 CPU↔GPU 档
python /tmp/run_bench.py --batch-size 4 --sequence-length 2048   # 2GB
python /tmp/run_bench.py --batch-size 8 --sequence-length 2048   # 4GB
```

裸硬件上限对照:

```bash
python - <<'PY'
import torch, time
torch.cuda.set_device(0)
N = 2*1024**3
h = torch.empty(N, dtype=torch.uint8).pin_memory()
d = torch.empty(N, dtype=torch.uint8, device='cuda')
def bench(src, dst, tag):
    dst.copy_(src, non_blocking=True); torch.cuda.synchronize()
    t=time.time()
    for _ in range(5): dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize(); dt=(time.time()-t)/5
    print(f"{tag}: {dt*1000:.2f}ms  {N/1024**3/dt:.1f} GB/s")
bench(h, d, "H2D"); bench(d, h, "D2H")
PY
```

---

## 8. 优化实现与 A/B 结果(2026-06-16)

实现位置:`csrc/gpu_backend/nvidia/transfer.cu` 的 CE transfer 分支(ROCm 编译用其 hipify 产物)。在每个 `(layer, kv)` 内检测 **CPU 与 GPU block id 同时 +1 递增的最大连续 run**,把原来 `run_len` 次逐块 `gpuMemcpyAsync` 合并为:

- **§5.1(1D 合并)**:若两端都 gap-less(`chunk_size == block_stride`),run 在物理上连续 → 一次 `gpuMemcpyAsync(run_len × chunk)`。
- **§5.2(2D 跨步)**:否则 run 是规则跨步 → 一次 `gpuMemcpy2DAsync(width=chunk, height=run_len, pitch=block_stride)`,同时天然处理 gap。

由环境变量 `FLEXKV_CE_COALESCE` 控制(`0`=baseline 逐块,`1`=§5.1,`2`=§5.2,`3`=both,**默认 3**),便于 A/B。新增 `gpuMemcpy2DAsync` 跨厂商宏(`csrc/gpu_backend/gpu_types.h`)。

### 8.1 A/B 场景 1:`benchmark_single_batch.py`(VLLM 布局,**gapped**,纯 CPU↔GPU)

| 模式 | D2H `put` | H2D `get` |
| -- | -- | -- |
| 0 baseline(逐块) | 2.35–2.83 GB/s | 2.65–2.73 GB/s |
| 1 §5.1(1D) | 2.35–2.83 GB/s(**无变化**) | 2.65–2.73 GB/s(**无变化**) |
| 2 §5.2(2D) | 3.79 GB/s | **31.5 GB/s** |
| 3 both | 3.56–3.81 GB/s | **22.7–31.7 GB/s** |

此布局 `chunk_size ≠ block_stride`(block 间有 gap),1D 合并条件不成立 → §5.1 无效;§5.2(2D 跨步)生效,H2D ~8–11×。

### 8.2 A/B 场景 2:`flexkv_transfer_microbench.py`(MLA/SGLANG 布局,**gapless**,直接调 `transfer_kv_blocks`)

来自 zhjc1124 fork commit `94ea329`,78 层 / chunk 16 KiB / 512 block / MLA(kv_dim=1)。`chunk == block_stride`(gapless)。单位 GiB/s,5 iter mean:

| 方向 / 模式 | 0 baseline | 1 §5.1(1D) | 2 §5.2(2D) | 3 both |
| -- | -- | -- | -- | -- |
| D2H path0(全连续) | 0.99 | **8.82** | 7.99 | 8.82 |
| D2H path1(2 段+gap) | 0.99 | **5.63** | 5.26 | 5.70 |
| D2H scatter(cpu 跳步) | 0.98 | 0.99 | 0.98 | 0.98 |
| H2D path1 | 3.20 | 5.48 | **5.72** | 5.55 |

gapless 布局下 **§5.1(1D)生效且略优于 2D**(D2H path0 8.82 vs 7.99);scatter(block id 非 +1 连续)两者都不触发(属 fork `path2` staging 处理的场景,本实现未做)。

### 8.2b A/B 场景 3:`flexkv_tp8_transfer_microbench.py`(TP8 sharded D2H,走 `TPTransferThreadGroup`)

来自同一 fork。`TPTransferThreadGroup::tp_group_transfer` 内部对**每个 GPU** 调用我们改过的 `transfer_kv_blocks<Type>`,所以本优化对 TP 真实 worker 路径同样生效。8 卡并行、每卡传 shard(16 KiB)、`cpu_block_stride=total_chunk(128 KiB)` → **gapped**。聚合带宽(8 GPU):

| pattern / 模式 | 0 baseline | 1 §5.1(1D) | 2 §5.2(2D) | 3 both |
| -- | -- | -- | -- | -- |
| path1(2 段+gap) | 22.2 | 21.8 | **163.1** | 161.7 GiB/s |
| path0(全连续) | 22.4 | 22.1 | **161.4** | 161.3 GiB/s |

§5.2 提速 **~7.2–7.3×**(22 → ~163 GiB/s 聚合,约 20 GiB/s/卡);§5.1 不触发(sharded 布局 gapped),符合预期。

### 8.3 结论

- **§5.2(2D 跨步合并)= 完成 ✅**。在 gapped 布局(VLLM)下 H2D 提升 ~8–11×(2.7 → 23–31 GB/s),D2H ~1.3–1.5×;gapless 布局下同样有效。
- **§5.1(1D 连续合并)= 完成 ✅**(补测后修正前述结论)。在 **gapless 布局**(MLA/SGLANG microbench)下 D2H 提升 **5.7×(path1)~8.9×(path0)**,且单次大块 1D 拷贝略快于 2D。之前 single_batch(VLLM)无收益仅因该布局 `chunk ≠ block_stride` 有 gap。
- **mode 3(both,默认)= 最优**:gapless 时走 1D(最快),否则走 2D(最通用),取两者之优。`put` 提升较小是因为合并后传输已非瓶颈,转为受 CPU 侧 bookkeeping(哈希 / radix 插入)限制。
- 正确性:`test_kvmanager.py -k test_config0-cache_config0`(含多 tp 变体)**17 passed / 8 skipped**,数据完整性通过(VLLM 路径);实现基于裸指针/stride,与 backend layout 无关。

> 关键认知:① launch 次数从 `L×kv×B` 降到 `L×kv`(场景 2 即 39936 → 78)是提速根因;② "连续内存 1D 合并"(§5.1)只在 gapless 布局成立且最快,"带 pitch 的 2D 跨步"(§5.2)更通用、处理 gap,二者互补;③ 两端 block id 必须**同时 +1**才合并,scatter/gather 这类单边跳步的场景需要 staging(未来工作)。

---

## 7. 一句话总结

ROCm 上 FlexKV 基线的 D2H≈2.7 / H2D≈2.7 GB/s 远低于硬件的 45–54 GB/s,**根因是 CE transfer 把一次传输拆成数万个 32KiB 小拷贝(launch-bound)**。**已落地并经 A/B 验证的优化是 §5.2(`gpuMemcpy2DAsync` 跨步合并),H2D 提升 ~8–11× 至 23–31 GB/s,D2H ~1.5×,正确性通过**(§8)。§5.1(1D 连续合并)因 FlexKV 布局存在 block stride gap 而无收益,不计完成。后续:根治方案 §5.3(ROCm kernel zero-copy),`put` 侧 bookkeeping 与多流(§5.4)可进一步提升,配置层可试 §5.7 增大 tokens_per_block。
