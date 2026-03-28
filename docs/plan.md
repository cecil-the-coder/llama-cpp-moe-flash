# MoE Flash — Implementation Plan

## Current State

**Production image: `6c04589`** on shadow node (AMD Strix Halo, 125 GB RAM, Radeon 8060S)

Single-backend architecture with auto-detect `--cpu-moe`:
- All models use `moe-flash-cpumoe` backend (`CPU_MOE=1` default)
- `llama_params_fit` checks if full model fits in device memory without the override
- If it fits → clears override → full GPU (20-50 t/s)
- If not → keeps override → mmap-wrap with partial prefetch (1.8-6.3 t/s)

| Model | Size | TPS | Loading |
|---|---|---|---|
| glm-4-7-flash | 17 GB | **50.57** | standard (full GPU offload) |
| qwen3-235b Q2_K | 80 GB | **20.21-20.77** | standard (full GPU offload) |
| qwen3-235b Q4_K_M | 133 GB | 2.72→**6.0-6.3** | patched: mmap experts + partial prefetch |
| deepseek-r1-0528 | 228 GB | 1.63→**1.8** | patched: mmap experts + partial prefetch |

---

## What the Patch Does

**Patch 0002** (456 lines, 5 files) adds to llama.cpp b8298:

| Feature | File | Lines | Purpose |
|---|---|---|---|
| Auto-detect `--cpu-moe` | `src/llama.cpp` | 32 | Single backend for all model sizes |
| Non-host expert copy | `ggml/src/ggml-backend.cpp` | 11 | Staging buffer for Vulkan_Host buffers |
| Vulkan_Host CPU interface | `ggml/src/ggml-vulkan.cpp` | ~30 | SIGSEGV fix, nullptr guard, alignment |
| Hybrid pinned/mmap alloc | `src/llama-model.cpp` | ~120 | mmap-wrap for >RAM, pinned for ≤RAM |
| Partial prefetch + madvise | `src/llama-model.cpp` | ~30 | WILLNEED up to MemAvail-8G, RANDOM+HUGEPAGE |
| Callback-only mode guard | `src/llama-moe-flash.cpp` | 4 | Prevent 3× slowdown from sync readback |

---

## Novelty Analysis

### What's standard (same as stock llama.cpp)
- Full GPU offload for models ≤ GTT — `ggml_vk_create_buffer_device` + suballocation
- `--cpu-moe` expert offloading to CPU — upstream feature
- mmap for model loading — Justine Tunney's original llama.cpp optimization

### What's novel (within llama.cpp ecosystem)
1. **mmap-wrap for >RAM expert data** — stock `--cpu-moe` mallocs expert tensors → OOM
   for models exceeding RAM. We wrap mmap directly as CPU buffer via
   `ggml_backend_cpu_buffer_from_ptr`, keeping experts demand-paged. This makes
   133-228 GB models loadable on 125 GB RAM.

2. **Partial prefetch with safety margin** — `MADV_WILLNEED` up to MemAvailable minus
   8 GiB. Stock llama.cpp does all-or-nothing prefetch. We prefetch what fits and
   leave the rest demand-paged.

3. **Auto-detect `--cpu-moe`** — checks projected memory in `llama_params_fit` and
   auto-clears the override when model fits. Stock requires manual backend assignment.
   llama.cpp discussion #18049 describes similar intent but is not merged.

4. **io_uring page cache warming** — SQPOLL + registered buffers for zero-copy expert
   prefetch. No published LLM inference system uses io_uring. (Background thread won
   over callback mode; io_uring code is in patch 0001.)

### What others do better

| System | Approach | vs Ours |
|---|---|---|
| **KTransformers** | AMX-optimized CPU expert kernels | 28 t/s DeepSeek vs our 1.8 — **15× faster** expert matmul |
| **Fate** | Cross-layer gate prediction (97% accuracy) | Targeted prefetch vs our blind background sweep |
| **HOBBIT** | Mixed-precision: cache-miss experts at lower quant | Reduces I/O for cold experts |
| **flash-moe** | pread() + GCD on Apple Silicon | 4.4 t/s on 397B/48GB MacBook |
| **Fiddler** | Per-expert CPU-vs-GPU profiling | Optimal placement, not all-or-nothing |
| **PreScope** | Async I/O decoupled from compute | 141% throughput improvement |

---

## Completed Investigations

### F1. AVX512 Expert Kernels — DONE (+134% cold start)

Enabled AVX512F + VNNI + BF16 in Dockerfile (`GGML_NATIVE=OFF` required for
cross-compilation). VNNI accelerates the quantized dot product inner loop
(`_mm256_dpbusd_epi32` replaces 2-step `maddubs + madd`).

**Result**: q4km cold start 2.72 → 6.36 t/s (+134%). Warm unchanged (I/O bound).

### F2. Prediction-Based Expert Prefetch — DONE (+12% deepseek warm)

Scheduler callback (`ggml_backend_sched_expert_copy_callback`) fires when expert
IDs are read during MUL_MAT_ID selective copy. Prefetches next-layer experts via
`posix_fadvise(WILLNEED)` — 8/256 = 3% of data for deepseek vs 100% with blind
thread. Stops background thread to avoid I/O contention.

**Result**: deepseek warm 1.8 → 2.01 t/s (+12%). q4km warm unchanged (~6.2 t/s).

### F3. 1 GB Hugepages — NOT VIABLE

Hugepages cannot be paged to disk. Our >RAM models (q4km 133 GB, deepseek 228 GB
on 125 GB RAM) rely on demand paging from NVMe. `MAP_HUGETLB` only works with
anonymous/hugetlbfs mmap, not file-backed. `MADV_HUGEPAGE` has no effect on
file-backed mmap (THP only works for anonymous memory). The 10× claim from
llama.cpp #12444 was startup time only, not inference speed.

### F4. RADV GPU Page Fault — CONFIRMED, NOT FIXABLE

Re-tested with image `7736644` on Mesa 25.3.6 / kernel 6.18.15 / GFX1151.
Exit code 139 (SIGSEGV) when compute shaders read from pinned host memory
(`ggml_vk_host_malloc`). This is NOT the firmware bug (GCVM_L2_PROTECTION_FAULT,
fixed in linux-firmware 20260110) — it's a fundamental RADV limitation.
`supports_buft` for Vulkan_Host must remain disabled.

---

## Investigation Queue

### I1. Thread Count Tuning — DONE (+7% warm)

Tested 12, 16, 24 threads on 16-core/32-thread Zen 5.

| Threads | q4km cold | q4km warm (avg) |
|---|---|---|
| 12 | 4.43 | 6.51 |
| **16** | **6.70** | **6.93** |
| 24 | 6.43 | 6.56 |

**16 threads is optimal**. 24 regresses from memory bandwidth contention (CPU and
GPU share the same memory controller on UMA). Set permanently in backend config.

### I2. NVMe Direct I/O with io_uring — BLOCKED (SIGSEGV)

Implemented io_uring async reads in `moe_flash_expert_copy_cb` (fire-and-forget
page cache warming with 128KB scratch buffer). Build succeeds and io_uring ring
initializes correctly. However, Q2K (GPU-only path) crashes with SIGSEGV (exit 139)
during inference despite guards for null buffers and GPU-only tensors.

The crash occurs after model loading succeeds (graph splits=1) — during the first
inference request. Root cause not yet identified. Possibly the callback fires during
`sched_reserve` with invalid tensor state, or the io_uring ring conflicts with the
existing moe-flash ring from patch 0001.

**Attempted fixes**: is_host guard, null buffer check — both insufficient.
**Rollback**: Production on `f58e4c6` (posix_fadvise only, no io_uring in callback).

**Status**: blocked — needs debugging with GDB or Vulkan validation layer

### I3. Prefetch Lookahead Depth

Currently prefetch layer N+1 from layer N's routing. Could prefetch N+2 or N+3
as well. Marginal cost of extra fadvise calls is near-zero. Deeper lookahead
gives more time for kernel readahead to complete before the data is needed.

**Test**: Add `lookahead` parameter to `moe_flash_expert_copy_cb`, test 1/2/3.

**Status**: not started

### I4. AMDVLK vs RADV

Community benchmarks show AMDVLK is 16% faster than RADV for token generation
on Strix Halo. Our Docker image uses RADV. Switching could give a free boost.
AMDVLK might also not have the pinned memory GPU page fault.

**Test**: Build image with AMDVLK driver, run full benchmark suite.

**Status**: not started

### I5. Expert Frequency Caching

For deepseek (256 experts, 8 selected per token), some experts are "hot"
(selected frequently). Track expert selection frequency per layer and pin hot
experts in RAM via `mlock()`. Cold experts page-fault from NVMe.

**Test**: Log expert frequencies during inference, identify hot/cold distribution.
If skewed, implement frequency-based pinning.

**Status**: not started

### I6. TQ2_KV Testing

Patches 0004+0005 add TQ2_KV 2-bit KV cache quantization. Not yet tested on
real inference. Should reduce KV cache 7.5× (3 GB → 0.4 GB at 8K context).

**Test**: Run Q2K with `-ctk tq2_kv -ctv tq2_kv`, verify quality and memory.

**Status**: not started

### I7. Context Size Scaling

With TQ2_KV freeing memory, push Q2K to 32K or 64K context. Measure quality
impact of 2-bit KV at long context lengths.

**Test**: Benchmark at 8K/16K/32K with TQ2_KV vs F16 KV cache.

**Status**: not started

### I8. Batch Size Tuning

Current config: batch=4096, ubatch=1024. For single-user, smaller batch may
reduce latency. For throughput, larger ubatch may help GPU utilization.

**Test**: Sweep batch sizes on Q2K and q4km.

**Status**: not started

### I9. GGUF Tensor Reordering

Expert tensors for the same layer are interleaved with attention tensors in GGUF
files. If they were contiguous, sequential readahead would be more effective for
the mmap-wrap path.

**Test**: Analyze GGUF layout, estimate potential readahead improvement. Would
require a custom GGUF repacker tool.

**Status**: not started

---

## RADV Driver Limitations

Confirmed on both Strix Halo (shadow) and Strix Point (local):

| Feature | Status |
|---|---|
| `maxBufferSize` | 2-4 GiB (all shard ranges exceed) |
| `VK_EXT_external_memory_host` import (file-backed mmap) | Always fails on RADV |
| `VK_EXT_external_memory_host` import (anonymous mmap) | Works |
| Compute shader read from pinned host memory | GPU page fault |

---

## Architecture

```
Single backend: moe-flash-cpumoe (CPU_MOE=1, image 6c04589)

llama_params_fit auto-detect:
  ├── Model fits in device memory? → Clear CPU_MOE override
  │   → Vulkan alloc+copy → 20-50 t/s (glm, Q2K)
  │
  └── Model doesn't fit? → Keep CPU_MOE override
      ├── Model ≤ RAM? → Pinned alloc (ggml_vk_host_malloc + copy)
      └── Model > RAM? → mmap-wrap (demand-paged)
                          + Partial prefetch (MADV_WILLNEED, up to MemAvail - 8G)
                          + MADV_RANDOM + MADV_HUGEPAGE → 1.8-6.3 t/s
```

## Completed Phases

- **Phase 0**: Measurement & baseline
- **Phase 1**: Expert file splitter tool
- **Phase 2**: io_uring prefetch prototype
- **Phase 3**: Vulkan integration (host buffer fix, hybrid alloc, prefetch, madvise)
- **Phase 4**: Integration with inference-budget-controller
- **Phase 5**: Auto-detect `--cpu-moe` + single-backend deployment

## CI/CD

- Repo: `github.com/cecil-the-coder/llama-cpp-moe-flash`
- Images: `ghcr.io/cecil-the-coder/llama-cpp-moe-flash:<sha>`
- Workflow: `.github/workflows/build-image.yml` (triggers on `patches/` or `docker/`)
- eh-ops-private: `github.com/themicknugget/eh-ops-private` (Flux-managed)
- Backend: `llamacpp-vulkan-moe-flash-cpumoe` in `backends/kustomization.yaml`
