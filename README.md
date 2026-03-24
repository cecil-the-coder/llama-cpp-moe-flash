# llama-cpp-moe-flash

Implementing "LLM in a Flash" style SSD-streaming inference for MoE models in llama.cpp,
targeting AMD Ryzen AI 365 (Strix Halo) on Linux with Vulkan.

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

## Documents

- [`docs/findings.md`](docs/findings.md) — key lessons from flash-moe's 58 experiments and
  what transfers to Linux/io_uring
- [`docs/architecture.md`](docs/architecture.md) — llama.cpp internals: how MoE execution
  works, where the hook points are
- [`docs/design.md`](docs/design.md) — full io_uring expert prefetcher design
- [`docs/plan.md`](docs/plan.md) — implementation plan and task tracking

## Container Image

```bash
# Pre-built image with io_uring MoE flash streaming (use short SHA from CI):
docker pull ghcr.io/cecil-the-coder/llama-cpp-moe-flash:<sha>
```

Built from `docker.io/kyuz0/amd-strix-halo-toolboxes:vulkan-radv` + our patch.
Tags: short git SHA (e.g. `a1b2c3d`) for deployments, `latest` as convenience alias.

## Usage

The flash MoE module is controlled by environment variables:

```bash
# Enable with GGUF page cache warming (recommended, no split files needed):
LLAMA_FLASH_MOE_ENABLED=1 \
LLAMA_FLASH_MOE_IOURING=1 \
LLAMA_FLASH_MOE_GGUF_PATH=/models/model.gguf \
llama-server -m /models/model.gguf --n-gpu-layers all ...

# Alternative: fadvise fallback (no io_uring/IPC_LOCK needed):
LLAMA_FLASH_MOE_ENABLED=1 \
LLAMA_FLASH_MOE_FADVISE=1 \
LLAMA_FLASH_MOE_EXPERTS_DIR=/models/experts/ \
llama-server -m /models/model.gguf ...

# Debug: log expert routing decisions:
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
