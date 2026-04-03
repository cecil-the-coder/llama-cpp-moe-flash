# llama-cpp-moe-flash

Implementing "LLM in a Flash" style SSD-streaming inference for MoE models in llama.cpp,
targeting AMD Ryzen AI 365 (Strix Halo) on Linux with Vulkan.

## ✅ Status Update (2026-04-03)

**I18 Cache Hit Tracking COMPLETE**: Fixed cache hit metrics to include cross-layer expert sharing.
- Cache hit rate now properly reflects expert reuse between layers
- Metrics work in all modes (prefetch, io_uring, cache)
- See [`docs/I18-cache-hit-fix.md`](docs/I18-cache-hit-fix.md) for details
- **Production image**: `ghcr.io/cecil-the-coder/llama-cpp-moe-flash:c791e75`

**I17 Prometheus Metrics COMPLETE**: Full observability with Prometheus + Grafana.
- HTTP metrics server on port 9090
- 8-panel Grafana dashboard
- Tracks requests, cache hits, I/O savings
- See [`docs/I17-prometheus-complete.md`](docs/I17-prometheus-complete.md) for details

**I11 Async Expert Prefetch COMPLETE**: Fully functional async prefetch with `posix_fadvise` is now working.
- Callback triggers on every MoE layer execution
- Automatically prefetches next layer's experts to page cache
- 3 GGUF shards detected and mapped for prefetching
- See [`docs/I11-async-prefetch-summary.md`](docs/I11-async-prefetch-summary.md) for details
- **Production image**: `ghcr.io/cecil-the-coder/llama-cpp-moe-flash:86982b0`

**I10b Investigation COMPLETE**: GPU MoE expert matmul with fixed-size slot buffer is **working**.
- 3× speedup for models ≤ GTT (120 GB): 6 t/s → 18-20 t/s
- Auto-detect logic routes models to optimal backend (GPU or CPU)
- Slot buffer code ready for >GTT models (future activation)
- **Production image**: `ghcr.io/cecil-the-coder/llama-cpp-moe-flash:ce76b8d`

**Key Finding**: "Defensive" patches (0006) were causing crashes, not the slot buffer code itself.
Consolidated patch (0001-0015 without 0006) is stable and production-ready.

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

## Results (Updated 2026-03-31)

| Model | Size | RAM | Config | Gen t/s | Status |
|---|---|---|---|---|---|
| glm-4-7-flash | 17 GB | 125 GB | Full GPU (auto-detect) | **50.57** | ✅ I10b Option A |
| qwen3-235b-a22b Q2_K | 80 GB | 125 GB | Full GPU (auto-detect) | **20.21-20.77** | ✅ I10b Option A |
| qwen3-235b-a22b-q4km | 133 GB | 125 GB | Full GPU (auto-detect) | **18.0** | ✅ I10b Option A |
| DeepSeek-R1-0528 Q2_K | 228 GB | 125 GB | mmap-wrap + CPU MoE | **1.37-1.8** | ✅ Working |

**I10b Achievement**: Auto-detect `llama_params_fit` now clears `CPU_MOE` for models ≤ 120 GB GTT,
enabling 3× faster GPU expert matmul. Models exceeding GTT use stable mmap-wrap fallback.

### I10b Investigation Summary

**Option A: Full GPU Offload (≤GTT Models)** — ✅ WORKING
- All models ≤ 120 GB automatically use full GPU (18-50 t/s)
- No manual backend selection needed
- Verified: glm-4-7-flash, qwen3-235b variants

**Option B: Slot Buffer for >GTT Models** — 🔄 READY FOR ACTIVATION
- Slot buffer code present in consolidated patch (0014-0015)
- Currently disabled ( `--cpu-moe` keeps matmul on CPU)
- Can be activated with `LLAMA_FLASH_MOE_FORCE_OFFLOAD=1` flag
- Target: DeepSeek 228 GB → 6-10 t/s (vs current 1.8 t/s)

**Root Cause**: Patch 0006 "defensive checks" were causing silent KV cache failures.
Removed from consolidated patch. All tests now pass.

---

### Patch Status

| Patch | Status | Purpose |
|---|---|---|
| 0001-0005 | ✅ Included | Core MoE flash, io_uring, TQ2 KV |
| 0006 | ❌ REMOVED | Defensive checks were causing crashes |
| 0007-0015 | ✅ Included | Debug logging, buffer fixes, slot buffer |
| **Consolidated** | ✅ **ce76b8d** | Production-ready patch |

## Documents

- [`docs/testing-guide.md`](docs/testing-guide.md) — **Testing guide for I14 + I10b optimizations** ← Start here
- [`docs/plan.md`](docs/plan.md) — implementation plan and task tracking
- [`docs/next-investigations.md`](docs/next-investigations.md) — roadmap for 2026-Q2 investigations
- [`docs/I14-iouring-polish.md`](docs/I14-iouring-polish.md) — io_uring performance optimizations (SINGLE_ISSUER, MADV_HUGEPAGE)
- [`docs/I12-ik-llama-benchmark.md`](docs/I12-ik-llama-benchmark.md) — **I12: ik_llama.cpp CPU-only benchmark** (in progress)
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
