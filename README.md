# llama-cpp-moe-flash

Implementing "LLM in a Flash" style SSD-streaming inference for MoE models in llama.cpp,
targeting AMD Ryzen AI 365 (Strix Halo) on Linux with Vulkan.

## Status (2026-04-03) — Research Complete

**Production image: `7937441`** on b8664 -- 9-patch stack delivering **4x baseline** on >GTT MoE models.

**MILESTONE: 8 MoE models validated across full performance spectrum (13-50 t/s for in-GTT, 7-10 t/s for >GTT, 1.6-2.9 t/s CPU-only for 744B).**

**Research phase complete.** Auto-detect works for all models that fit in GTT -- no manual configuration needed. Slot remapping (N_SLOTS=64) only benefits >GTT models. At N_SLOTS=64 on 128 GB UMA, expert copy is 3.5ms/token (optimized from 530ms -- 150x reduction) while GPU compute is 85ms/token (hardware-limited). 7-10 t/s is near the hardware ceiling for >GTT models.

### Patch Stack (9 patches, each separate for upstream potential)

| # | Patch | Purpose |
|---|-------|---------|
| 1 | 0001a | Core MoE flash module (prefetch, io_uring, metrics) |
| 2 | 0001b | Force-offload guard (MUL_MAT_ID GPU routing) |
| 3 | 0001c | Persistent pool + slot remap (LRU cache, ne[2] override) |
| 4 | 0014 | Vec-path aliasing check (byte-range overlap) |
| 5 | 0017 | Auto-detect CPU_MOE (-fit disable) |
| 6 | 0021 | MoE split merging (282->96 splits) |
| 7 | 0022 | Speculative prefetch (pre-seed next layer) |
| 8 | 0023 | Least-stale eviction (layer-aware LRU) |
| 9 | 0024 | Batch pool allocation (one sync, all buffers) |

### Multi-Model Test Results

All MoE models that fit in GTT run at full GPU speed with auto-detect -- no manual
configuration needed. Slot remapping only benefits models exceeding GTT (120 GB).

| Model | Size | Experts | K | t/s | Path |
|-------|------|---------|---|-----|------|
| GLM-4-7-Flash | 17 GB | 64 | ? | ~50 | Full GPU |
| Minimax-M25-REAP | 100 GB | ? | ? | **28.4** | Full GPU (auto-detect) |
| Qwen3-235B Q2_K | 80 GB | 128 | 8 | **20** | Full GPU (auto-detect) |
| Qwen3.5-REAP-212B | 110 GB | 267 | 10 | **18** | Full GPU (auto-detect) |
| Nemotron-3-Super-120B | 85 GB | ? | ? | **13.2** | Full GPU (auto-detect) |
| Qwen3-235B Q4_K_M | 133 GB | 128 | 8 | **7-10** | GPU MoE slot remap (N_SLOTS=64) |
| DeepSeek-R1-0528 | 228 GB | 256 | 8 | ~4 | CPU MoE |
| GLM-5.1 UD-Q2_K_XL | 252 GB | 256 | 8 | 1.6-2.9 | CPU MoE (GPU pending driver fix) |

### Key findings

1. **Auto-detect works for all <=GTT models** -- no manual configuration needed
2. **Slot remapping only benefits >GTT models** (Q4_K_M at 133 GB)
3. **For <=GTT models that fit, full GPU is always fastest** (no slot overhead)
4. **Batch allocation (0024) fixes N_SLOTS startup stall** -- seconds instead of minutes
5. **Memory trade-off available**: slot remapping at N_SLOTS=64 uses 4x less GTT but runs 3x slower. Useful for running multiple models concurrently.

### Slot remapping breakthrough (models exceeding GTT)

- Qwen3-235B Q4_K_M (133 GB, >GTT): **7-10 t/s** with GPU MoE via 64-slot remapping
- Expert hit rate: **94.6%** with speculative prefetch (64 slots for 128 experts, K=8)
- ~60 GB persistent GPU buffers, stable on 128 GB node
- Configurable via `GGML_MOE_N_SLOTS` environment variable

**What makes slot remapping work:**
- `ne[2]=N_SLOTS` override flows correctly to `n_as=N_SLOTS` in all three shaders (batch, vec, count_experts)
- Persistent pool outside gallocr -- buffers survive across tokens
- LRU slot eviction with bidirectional mapping
- IDS rewrite with deferred write -- prevents overwrite by input copy loop
- Original IDS cache -- prevents gate/up/down cross-contamination
- No shader modifications needed

**N_SLOTS=64 is the stable production config for 128 GB node.** N_SLOTS=96 delivers higher throughput but crashes on 128 GB (90 GB pool + mmap exceeds RAM).

**N_SLOTS tuning results (Qwen3-235B Q4_K_M, 128 experts, K=8):**

| N_SLOTS | Hit Rate | t/s | Memory | Notes |
|---------|----------|-----|--------|-------|
| 32 | 74.9% | 3.5-4.1 | ~35 GB | Default |
| 64 | 94.6% | 7-10 | ~60 GB | **Stable production (128 GB)** |
| 96 | 97.1% | 10.4-11.1 | ~90 GB | Unstable (needs 192+ GB RAM) |
| 128 | -- | OOM | ~123 GB | Exceeds RADV/UMA limits |

**Optimization journey (Q4_K_M 133 GB):**

| Step | t/s | Improvement |
|------|-----|-------------|
| Baseline CPU MoE | 1.8 | -- |
| b8664 rebase | 2.75 | +53% |
| Slot remapping (N=32) | 3.5-4.1 | +49% |
| N_SLOTS=64 | 7.5-9.4 | +100% |
| Split merging (0021) | ~same | marginal |
| Speculative prefetch (0022) | 7-10 | marginal |
| **N_SLOTS=96 (unstable)** | **10.4-11.1** | **+17% (needs 192+ GB RAM)** |

### Research Conclusions

All viable software optimizations for this hardware have been explored:

| Investigation | Status | Outcome |
|---------------|--------|---------|
| KHR_coopmat | Already active | No free performance available |
| Least-stale eviction (0023) | **DONE** | Equivalent to LRU at full pool size |
| AMDVLK driver | Not available | Would need separate build image |
| APEX requant | Skipped | Too risky for marginal gain |
| MoEpic / KTransformers / PuzzleMoE | Skipped | Too complex for marginal gains on UMA |
| D: Graph split reduction | **DONE** | Patch 0021, 282->96 splits, marginal t/s |
| E: Speculative expert prefetch | **DONE** | Patch 0022, marginal t/s |
| F: Adaptive N_SLOTS per layer | **SKIP** | Homogeneous experts; upgrade RAM instead |
| G: Expert routing prediction | **DEFER** | <0.1 t/s gain, GPU-compute-limited |
| A: DeepSeek on slot remapping | Blocked | Needs 256 GB node |

At 94.6% hit rate, GPU compute (~85ms) dominates over expert copy (~3.5ms) and sync (~7ms). The 8-patch stack is production-ready. Further gains require hardware changes.

### Previous approaches that didn't work

- **Buffer pool (per-projection and per-layer)**: Both have 0% hit rate. Per-projection:
  282 tensors with 9 entries. Per-layer: 94 layers with 15 entries. Sequential execution
  (layer 0,1,...,93) defeats LRU -- would need P=94 (~141 GB) to cache all layers.
- **Force-offload at 1.4 t/s**: Slower than CPU MoE (6-7 t/s) due to pool thrashing.
- **Graph split reduction**: The 284 splits come from CPU<->GPU backend transitions
  for attention vs MoE, not from expert weight copying.
- **Flash-moe async prefetch for DeepSeek**: Reads from disk each token instead of
  using cached GPU buffers. Slower than standard path with expert cache (2.3 vs 4.1 t/s).

### Remaining paths (all hardware-dependent)

1. **More RAM (192+ GB)**: Enables N_SLOTS=96 for 10.4-11.1 t/s (+17%)
2. **Upstream two-tier expert cache (#20757)**: Proper GPU expert matmul with shader support. 14 t/s PoC.
3. **Newer llama.cpp**: Upstream Vulkan shader optimizations reduce the 85ms GPU compute
4. **Better hardware**: Intel AMX (28 t/s per KTransformers), faster discrete GPU

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
| Minimax-M25-REAP | 100 GB | Yes | Full GPU (auto-detect) | **28.4** | Production-ready |
| Qwen3-235B Q2_K | 80 GB | Yes | Full GPU (auto-detect) | **20** | Production-ready |
| Qwen3.5-REAP-212B | 110 GB | Yes | Full GPU (auto-detect) | **18** | Production-ready |
| Nemotron-3-Super-120B | 85 GB | Yes | Full GPU (auto-detect) | **13.2** | Production-ready |
| **Qwen3-235B Q4_K_M** | **133 GB** | **No** | **GPU MoE, N_SLOTS=64** | **7-10** | **Production-ready** |
| DeepSeek-R1-0528 Q2_K | 228 GB | No | CPU MoE (can't test -- 128 GB RAM limit) | **~4** | CPU-bottlenecked |
| GLM-5.1 UD-Q2_K_XL | 252 GB | No | CPU MoE (Vulkan driver error) | **1.6-2.9** | Warming up, ~4 expected w/ GPU attn |

**Key insight**: Auto-detect works for all <=GTT models (no configuration needed).
Slot remapping with N_SLOTS=64 delivers **4x speedup** over baseline CPU MoE
(1.8 -> 7-10 t/s) for >GTT models with 94.6% expert hit rate + speculative prefetch.
GPU compute is the dominant cost, not expert copy bandwidth. Configurable via
`GGML_MOE_N_SLOTS` env var. N_SLOTS=64 (~60 GB pool) is the stable production config
for 128 GB nodes.

---

### Patch Status (latest main on b8664)

| Patch | Status | Purpose |
|---|---|---|
| 0001a | Applied | Core MoE flash (prefetch, io_uring, metrics) |
| 0001b | Applied | Force-offload MUL_MAT_ID guard |
| 0001c | Applied | Persistent buffer pool + slot remapping |
| 0014 | Applied | Vec-path byte-range aliasing check (safety net) |
| 0017 | Applied | Auto-detect CPU_MOE + disable upstream -fit hang |
| 0021 | Applied | Merge MoE splits within layer (282->96 splits) |
| 0022 | Applied | Speculative expert prefetch (pre-seed next layer's slots) |
| 0023 | Applied | Least-stale eviction policy |
| 0024 | Applied | Batch pool allocation (one sync, all buffers) |

**Research COMPLETE**: 9-patch stack validated across 8 MoE models (13-50 t/s for
in-GTT, 7-10 t/s for >GTT). Auto-detect works for all <=GTT models. Slot remapping
only needed for >GTT. Batch allocation (0024) fixes N_SLOTS startup stall.
GPU compute (85ms) is the hardware ceiling. N_SLOTS=64 is the stable production
config for 128 GB nodes (~60 GB buffer memory).

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
