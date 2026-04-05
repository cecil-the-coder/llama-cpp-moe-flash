# llama-cpp-moe-flash

Implementing "LLM in a Flash" style SSD-streaming inference for MoE models in llama.cpp,
targeting AMD Ryzen AI 365 (Strix Halo) on Linux with Vulkan.

## Status (2026-04-03)

**Production image: `7e64a82`** -- DeepSeek fix complete, all models coherent.

Patches applied: 0001 (with force-offload host buffer guard), 0014 (vec-path aliasing check), 0017 (auto-detect + fit disable).
Patches NOT applied: 0015, 0016, 0019 (slot buffer infrastructure -- kept in repo for future use).

**Root causes found and fixed (I11)**:
1. Force-offload in patch 0001 unconditionally pushed MUL_MAT_ID to GPU even when expert weights were on CPU (`--cpu-moe`). Fixed with `ggml_backend_buffer_is_host()` guard.
2. Patches 0015/0016 (`ggml_set_input/output` on `selected_experts`) changed gallocr allocation and corrupted output for ALL models. Removed from build.
3. Broken YAML in Qwen3 CRD (duplicate `value:` key) blocked Flux reconciliation for hours. Fixed.

**I12 benchmark (complete)**:
- Stock Vulkan: 20.7 t/s (Qwen3-235B Q2_K)
- moe-flash: 20.0 t/s (Qwen3), 2.05 t/s (DeepSeek)
- ik_llama.cpp CPU-only: 11.5 t/s (Qwen3), 1.5 t/s (DeepSeek without flash_attn)

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

| Model | Size | RAM | Config | Gen t/s | Status |
|---|---|---|---|---|---|
| GLM-4-7-Flash | 17 GB | 125 GB | Full GPU (auto-detect) | **~50** | Coherent |
| Qwen3-235B Q2_K | 80 GB | 125 GB | Full GPU (auto-detect) | **20.0** | Coherent |
| DeepSeek-R1-0528 Q2_K | 228 GB | 125 GB | CPU MoE path | **2.05** | Coherent |

Auto-detect clears CPU_MOE for models that fit in GTT (120 GB). DeepSeek uses
CPU MoE path with mmap-wrap and partial prefetch.

---

### Patch Status (image `7e64a82`)

| Patch | Status | Purpose |
|---|---|---|
| 0001 | Applied | Core MoE flash + force-offload host buffer guard |
| 0014 | Applied | Vec-path runtime aliasing check (byte-range overlap) |
| 0017 | Applied | Auto-detect CPU_MOE + disable upstream `llama_params_fit` |
| 0015, 0016, 0019 | NOT applied | Slot buffer infrastructure (kept in repo for future use) |

**Lesson learned**: Broken YAML in Flux CRDs (duplicate `value:` key) silently blocks
reconciliation. Always validate CRD YAML before debugging backend issues.

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
