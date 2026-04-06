# llama-cpp-moe-flash

Implementing "LLM in a Flash" style SSD-streaming inference for MoE models in llama.cpp,
targeting AMD Ryzen AI 365 (Strix Halo) on Linux with Vulkan.

## Status (2026-04-03)

**Production image: latest main** on b8664 -- Persistent buffer pool + force-offload.

Patches applied: 0001 (core MoE flash + persistent buffer pool + sync skip + force-offload guard), 0014 (vec-path aliasing check), 0017 (auto-detect + fit disable).

### What works well (models that fit in GTT)

All MoE models up to 120 GB (the GTT limit) run at full GPU speed. Auto-detect
clears CPU_MOE, no special configuration needed. These are production-ready.

- Qwen3-235B Q2_K (80 GB): **20 t/s**, coherent, full GPU
- GLM-4-7-Flash (17 GB): **~50 t/s**, coherent, full GPU

### What's improved but still limited (models exceeding GTT)

- DeepSeek-R1-0528 Q2_K (228 GB): **~4 t/s** with CPU MoE (can't test force-offload -- 128 GB RAM too small for 228 GB model copy buffers)
- Qwen3-235B Q4_K_M (133 GB): **1.4 t/s** with force-offload (slower than CPU MoE ~6-7 t/s)

**Force-offload finding:** The persistent buffer pool has **0% cache hit rate** with
the current per-projection design. 282 unique weight tensors (94 layers x 3 projections:
gate/up/down) overwhelm the 9-entry pool. Within-layer sharing doesn't work because
gate, up, and down are different tensors with different `input->data` keys.

**The real bottleneck is CPU MoE matmul.** With `--cpu-moe`, expert matrix multiplications
run on CPU (AVX-512 on Zen 5). This is inherently slower than GPU compute. For comparison,
KTransformers achieves 28 t/s on DeepSeek with Intel AMX-optimized kernels -- 7x faster
expert matmul than our AVX-512 path. The bottleneck is not I/O, not copies, not sync,
not graph splits -- it is the CPU compute for expert weights.

### What didn't work as hoped

- **Buffer pool (per-projection and per-layer)**: Both have 0% hit rate. Per-projection:
  282 tensors with 9 entries. Per-layer: 94 layers with 15 entries. Sequential execution
  (layer 0,1,...,93) defeats LRU — would need P=94 (~141 GB) to cache all layers.
- **Force-offload at 1.4 t/s**: Slower than CPU MoE (6-7 t/s) due to pool thrashing.
  Expert copy bandwidth dominates: 8 experts x 3 projections x 94 layers x 3.5 MB = 7.9 GB/token.
- **I11 slot buffer / dynamic expert import**: Infrastructure built but the Vulkan
  MUL_MAT_ID shader can't handle slot-remapped expert IDS. Patches 0015/0016/0019
  removed from production. Upstream #20757 (two-tier cache) is the right solution.
- **Graph split reduction**: The 284 splits come from CPU<->GPU backend transitions
  for attention vs MoE, not from expert weight copying. Can't be reduced without
  changing how the scheduler handles hybrid compute.
- **Flash-moe async prefetch for DeepSeek**: Reads from disk each token instead of
  using cached GPU buffers. Slower than standard path with expert cache (2.3 vs 4.1 t/s).

### What would actually help

1. **Upstream two-tier expert cache (#20757)**: Proper GPU expert matmul for >GTT models with shader support. Expected 10-15 t/s.
3. **Better CPU kernels**: ik_llama.cpp's FlashMLA + fused MoE FFN, or Intel AMX support. Could give 3-5x CPU speedup.
4. **Rebase to newer llama.cpp**: As upstream MoE work lands (expert caching, better CPU kernels).
5. **Hardware**: Faster NVMe (Gen5), more RAM, or a system with AMX support.

---

## Goal

Enable running MoE models **larger than available GTT** (120 GB on this hardware) by streaming
expert weights from NVMe on demand rather than requiring the full model in memory. Secondary
goal: reduce cold-start latency when models are paged back in after scale-to-zero.

## Reference

- **flash-moe** (inspiration): `danveloper/flash-moe` — runs Qwen3.5-397B at 4.4 tok/s on a
  48 GB MacBook by streaming 209 GB of expert weights from a 17.5 GB/s Apple SSD using
  parallel `pread()` + Metal compute. Documented 58 experiments.
- **"LLM in a Flash"** (Apple paper): theoretical foundation for windowed weight streaming.

## Hardware

| Property | Value |
|---|---|
| Node | `shadow` (MSI Prestige) |
| CPU | AMD Ryzen AI 385+ (Zen 5, 24 threads) |
| RAM | 125 GB system RAM |
| GPU | AMD Radeon 8060S (Strix Halo iGPU, **gfx1151** / GC_11_5_0, 40 CUs, 80 SIMDs) |
| PCI Device | `0x1586` |
| GTT pool | 120 GB (`amdgpu.gttsize=122880` — already set in kernel cmdline) |
| Swap | 32 GB |
| NVMe | Gen4, ~7 GB/s cold sequential read |
| OS | Talos Linux (kernel 6.18.15-talos) |
| Backend | Vulkan via RADV (via `llamacpp-vulkan-moe` InferenceBackend) |

## Current Model Inventory

Models currently deployed in the `inference` namespace (all `ScaledToZero`):

| Model | Memory | Backend | Fits in GTT? |
|---|---|---|---|
| qwen35-reap-212b-a17b | 110 Gi | llamacpp-vulkan-moe | Barely (110/120) |
| minimax-m25-reap | 100 Gi | llamacpp-vulkan-moe | Yes |
| qwen3-235b-a22b | 80 Gi | llamacpp-vulkan-moe | Yes |
| devstral-2-123b | 85 Gi | llamacpp-vulkan-moe | Yes |
| nemotron-3-super-120b | 85 Gi | llamacpp-vulkan-moe | Yes |
| qwen35-reap-212b-a17b | 110 Gi | llamacpp-vulkan-moe | Barely |

All current models fit. Flash streaming is needed for models > 120 GB or for multiple
concurrent models exceeding the GTT budget.

## I/O Budget Reality Check

Per-token I/O for streaming (cold NVMe read, no page cache):

| Model | Expert I/O / token | @ 7 GB/s cold | @ 30 GB/s warm |
|---|---|---|---|
| Qwen3-235B Q2_K (128 exp, K=8) | 4.3 GB | ~634 ms/tok | ~148 ms/tok |
| Qwen3.5-REAP-212B IQ4_XS (est K=8) | 6.4 GB | ~933 ms/tok | ~218 ms/tok |

**Takeaway**: Flash streaming only makes sense when the OS page cache is warm (repeated
generation), or with a much faster NVMe. With a warm cache these models are 1-5 tok/s
territory — viable but not fast. This matches flash-moe's 4.4 tok/s on 17.5 GB/s SSD.

## Results (Updated 2026-04-03)

| Model | Size | Fits GTT? | Config | Gen t/s | Status |
|---|---|---|---|---|---|
| GLM-4-7-Flash | 17 GB | Yes | Full GPU (auto-detect) | **~50** | Production-ready |
| Qwen3-235B Q2_K | 80 GB | Yes | Full GPU (auto-detect) | **20.0** | Production-ready |
| Qwen3-235B Q4_K_M | 133 GB | No | Force-offload (persistent pool) | **1.4** | Pool thrashing (0% hit) |
| Qwen3-235B Q4_K_M | 133 GB | No | CPU MoE | **~6-7** | Coherent, CPU-bottlenecked |
| DeepSeek-R1-0528 Q2_K | 228 GB | No | CPU MoE + expert GPU cache | **~4** | Coherent, CPU-bottlenecked |

**Key insight**: Models that fit in GTT (<=120 GB) are production-ready at full GPU speed.
Models exceeding GTT are bottlenecked by CPU expert matmul, not by I/O or memory copies.
Force-offload with per-projection pool is slower than CPU MoE due to 0% cache hit rate.

---

### Patch Status (latest main on b8664)

| Patch | Status | Purpose |
|---|---|---|
| 0001 | Applied | Core MoE flash + persistent buffer pool + sync skip + force-offload guard |
| 0014 | Applied | Vec-path runtime byte-range overlap check |
| 0017 | Applied | Disable upstream -fit + auto-detect CPU_MOE |

**Key finding**: Buffer pool has 0% hit rate regardless of grouping strategy. Per-projection
(282 tensors, 9 entries) and per-layer (94 layers, 15 entries) both fail because sequential
execution defeats LRU. Force-offload (1.4 t/s) is slower than CPU MoE (6-7 t/s).

## Documents

- [`docs/testing-guide.md`](docs/testing-guide.md) — **Testing guide for I14 + I10b optimizations** ← Start here
- [`docs/plan.md`](docs/plan.md) — implementation plan and task tracking
- [`docs/next-investigations.md`](docs/next-investigations.md) — roadmap for 2026-Q2 investigations
- [`docs/I14-iouring-polish.md`](docs/I14-iouring-polish.md) — io_uring performance optimizations (SINGLE_ISSUER, MADV_HUGEPAGE)
- [`docs/I12-ik-llama-benchmark.md`](docs/I12-ik-llama-benchmark.md) — I12: ik_llama.cpp CPU-only benchmark (complete)
- [`docs/I10b-findings.md`](docs/I10b-findings.md) — GPU MoE slot buffer investigation complete
- [`docs/I10b-option-b-force-offload.md`](docs/I10b-option-b-force-offload.md) — Force-offload testing for >GTT models
- [`docs/I11-async-prefetch-summary.md`](docs/I11-async-prefetch-summary.md) — I11 async expert prefetch implementation
- [`docs/test-results.md`](docs/test-results.md) — verified test results (2026-03-31)
- [`docs/measurements.md`](docs/measurements.md) — all benchmark results and analysis
- [`docs/findings.md`](docs/findings.md) — key lessons from flash-moe's 58 experiments
- [`docs/architecture.md`](docs/architecture.md) — llama.cpp internals
- [`docs/design.md`](docs/design.md) — io_uring expert prefetcher design (original)

## Container Image

```bash
# Pre-built image with io_uring MoE flash streaming (use short SHA from CI):
docker pull ghcr.io/cecil-the-coder/llama-cpp-moe-flash:<sha>
```

Built from `docker.io/kyuz0/amd-strix-halo-toolboxes:vulkan-radv` + our patch.
Tags: short git SHA (e.g. `a1b2c3d`) for deployments, `latest` as convenience alias.

## Usage

The flash MoE module is controlled by environment variables:

### Basic Enable (recommended)
```bash
LLAMA_FLASH_MOE_ENABLED=1 \
llama-server -m /models/model.gguf --n-gpu-layers all ...
```

### Async Expert Prefetch (NEW - I11)
Enable async prefetch to load next layer's experts while current layer computes:
```bash
LLAMA_FLASH_MOE_ENABLED=1 \
LLAMA_FLASH_MOE_MODE=async_prefetch \
LLAMA_FLASH_MOE_GGUF_PATH=/models/model.gguf \
llama-server -m /models/model.gguf --n-gpu-layers all ...
```

**What it does:**
- Registers a callback that triggers on every MoE layer execution
- Parses layer ID from tensor names (handles `ffn_moe_gate-N` format)
- Prefetches ALL experts in layer N+1 using `posix_fadvise(WILLNEED)`
- Works with multi-shard GGUF files (automatically detects shards)

**Requirements:**
- Linux kernel with POSIX_FADVISE support
- GGUF file path must be accessible (uses `LLAMA_FLASH_MOE_GGUF_PATH` or falls back to `HF_SOURCE`)

### With io_uring (if compiled with GGML_IOURING=ON)
```bash
LLAMA_FLASH_MOE_ENABLED=1 \
LLAMA_FLASH_MOE_IOURING=1 \
LLAMA_FLASH_MOE_GGUF_PATH=/models/model.gguf \
llama-server -m /models/model.gguf --n-gpu-layers all ...
```

### Alternative: fadvise fallback (no io_uring/IPC_LOCK needed)
```bash
LLAMA_FLASH_MOE_ENABLED=1 \
LLAMA_FLASH_MOE_FADVISE=1 \
LLAMA_FLASH_MOE_EXPERTS_DIR=/models/experts/ \
llama-server -m /models/model.gguf ...
```

### Debug: log expert routing decisions
```bash
LLAMA_FLASH_MOE_ENABLED=1 \
LLAMA_FLASH_MOE_LOG_ROUTING=1 \
llama-server -m /models/model.gguf ...
```

**Requirements for io_uring mode**: `IPC_LOCK` capability, liburing, built with `-DGGML_IOURING=ON`.

## Building from Source

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout b8298
git apply ../patches/0001-moe-flash-complete.patch
cmake -B build -DGGML_VULKAN=ON -DGGML_IOURING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target llama-server
```
