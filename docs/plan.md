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

### F2. Prediction-Based Expert Prefetch — NO-OP WITH --cpu-moe

Scheduler callback (`ggml_backend_sched_expert_copy_callback`) fires when expert
IDs are copied between backends for MUL_MAT_ID. However, with `--cpu-moe`, all
expert data stays on CPU — no cross-backend copy — **callback never fires**.
Confirmed via I5 instrumentation: 0 callbacks, 0 prefetch calls.

The earlier "12% improvement" on deepseek was measurement noise. The callback
architecture only works when experts are split across GPU/CPU backends (the
`supports_buft` path blocked by RADV GPU page fault).

The blind background prefetch thread is still the effective prefetch mechanism.

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

### I2. NVMe Direct I/O with io_uring — NO BENEFIT

**First attempt** (scheduler callback): SIGSEGV because the callback never fires
with --cpu-moe (discovered in I5). The crash was from other code in the callback.

**Second attempt** (background prefetch thread): Modified the blind prefetch thread
to use io_uring fire-and-forget reads (128KB scratch buffer) instead of posix_fadvise.
Works correctly but shows **no measurable improvement**:

| Path | q4km warm |
|---|---|
| posix_fadvise | 6.70-6.73 t/s |
| io_uring reads | 6.72-6.78 t/s |

Both just hint the kernel to readahead pages. The mechanism (fadvise syscall vs
io_uring read into scratch) doesn't matter when the bottleneck is NVMe bandwidth
and page fault latency, not syscall overhead.

### I3. Prefetch Lookahead Depth — NO BENEFIT

Tested lookahead=3 (prefetch layers N+1, N+2, N+3 from layer N's routing).

| Model | Lookahead=1 | Lookahead=3 | Change |
|---|---|---|---|
| q4km warm | 6.93 | 6.76 | **-2%** (fadvise overhead) |
| deepseek warm | 2.01 | 2.06 | +3% (marginal) |

The extra fadvise syscalls (72 per layer vs 24) cost more than the prefetch
benefit. Expert selection locality across 3 layers isn't high enough to justify
3× the I/O hints. Reverted to lookahead=1.

### I4. AMDVLK vs RADV — NOT VIABLE

AMDVLK was discontinued Sep 2025 (last release v-2025.Q2.1). Benchmarks from
kyuz0/amd-strix-halo-toolboxes show AMDVLK is **8-10× slower on token generation
for large models** (Qwen3-235B: 2.1 vs 18.1 t/s). Only wins prompt processing.
Reports half the shared memory (32 KB vs 64 KB). GPU page fault issue is worse.

RADV remains the correct choice for MoE inference on Strix Halo.

### I5. Expert Frequency Caching — BLOCKED (callback doesn't fire)

Added frequency tracking to prediction callback. Discovered that with `--cpu-moe`,
the scheduler's expert copy callback **never fires** (0 callbacks in 501 pre_graph
calls). All expert data stays on CPU → no cross-backend copy → no callback.

This also invalidates F2 (prediction prefetch) and I3 (lookahead) — both were
no-ops. The "improvements" measured earlier were noise.

**Key finding**: The prediction prefetch architecture requires a different hook
point — not the scheduler's selective copy, but the CPU MUL_MAT_ID dispatch
itself. Need to hook into `ggml_compute_forward_mul_mat_id` where the `used_ids`
bitset is built (line 1510 of ggml-cpu.c).

**Status**: blocked — needs a CPU-side hook, not scheduler-side

### I6. TQ2_KV Testing — WORKS BUT QUALITY UNUSABLE

TQ2_KV type system works correctly. Local debugging confirmed:
- `ggml_type_name(TQ2_KV) = "tq2_kv"` ✓
- `kv_cache_type_from_str("tq2_kv")` matches ✓
- `blck_size=128`, `type_size=34` ✓

**Deployed result** (image `1cb6d44`, env `LLAMA_ARG_CACHE_TYPE_K=tq2_kv`):
- KV cache: **3196 MiB** (tq2_kv) vs 6768 MiB (q4_0) — **2.1× reduction**
- TPS: **20.23 t/s** — no regression
- Output quality: **GARBAGE** — gibberish text, complete attention pattern destruction

The 2.125 bpw quantization is too aggressive for MoE models with HSK=128.
The 4-level symmetric encoding {-1.5d, -0.5d, +0.5d, +1.5d} doesn't preserve
enough precision for the attention score distribution.

Previous debugging confusion was from Docker cache (old binary) and Flux timing
(env var not propagated when pod started).

**Reverted to q4_0.** TQ2_KV needs a higher-precision variant (3-bit or 4-bit
symmetric) to be usable.

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
