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

## Future Directions

### F1. AMX/AVX512 Expert Kernels (highest impact)

**The bottleneck for >RAM models is CPU expert matmul, not I/O.**

Our deepseek runs at 1.8 t/s. KTransformers achieves 28 t/s on the same model by
replacing ggml-cpu matmul with custom AMX (Advanced Matrix Extensions) kernels for
the expert FFN layers. AMX provides 8× throughput over AVX512 for BF16/INT8 matmul.

**Strix Halo has AVX512 but not AMX.** However, AVX512-BF16 + VNNI are available
and could significantly accelerate expert matmul. Fiddler (ICLR 2025) demonstrates
AVX512_BF16 CPU kernels achieving competitive expert throughput.

**Approach**: Profile where time is spent in the `--cpu-moe` path. If expert matmul
dominates (likely), investigate:
- ggml-cpu AVX512 codegen for MoE expert shapes
- Custom BLAS dispatch for expert-sized matrices (~6 MB each)
- Batching multiple expert matmuls to amortize overhead

**Expected impact**: 3-10× improvement on >RAM models (6-18 t/s for deepseek).

### F2. Prediction-Based Expert Prefetch

Current approach: background thread blindly cycles through all experts with
`posix_fadvise(WILLNEED)`. This warms the page cache uniformly but doesn't
prioritize experts that will actually be selected by the router.

**Academic state of the art**:
- **Fate** (2025): cross-layer gate inputs predict next-layer experts with 97.15%
  accuracy. Shallow-favoring cache achieves 99.08% hit rate.
- **PreScope** (2025): learnable layer-aware predictor, async I/O optimizer.
  141% throughput, 74.6% latency improvement.
- **DuoServe-MoE** (2025): MLP predictor for next-layer experts, GPU cache sized
  to exactly K experts.

**Approach for our system**: After routing selects K experts for layer N, use the
gate logits to predict likely experts for layer N+1. Issue targeted `posix_fadvise`
for only those experts' GGUF regions. This replaces blind prefetch with informed
prefetch at near-zero cost (gate logits are already computed).

**Challenge**: Requires reading gate logits back from Vulkan, which is the same
sync stall that made the callback mode 3× slower. Could work if the prediction
runs on CPU from the routing tensor copy that the scheduler already performs.

**Expected impact**: Better page cache utilization for >RAM models, reducing cold
expert faults. Most impactful for deepseek (228 GB, only 55% in cache).

### F3. 1 GB Hugepages

llama.cpp issue #12444 reports **10× speedup** for DeepSeek-R1 using hugetlbfs
with 1 GB pages. Our patch uses `MADV_HUGEPAGE` for transparent 2 MB hugepages,
but 1 GB pages eliminate far more TLB pressure: 228 GB model = 228 entries
(1 GB pages) vs 116,736 entries (2 MB) vs 59,768,832 entries (4 KB).

**Approach**: Pre-allocate 1 GB hugepages at boot (`hugepagesz=1G hugepages=100`),
mmap expert data from hugetlbfs. Requires kernel configuration and may conflict
with other memory users.

**Expected impact**: Significant for >RAM models where TLB misses compound with
page faults. Less impactful for ≤RAM models (already fast).

### F4. RADV Driver Fix

Compute shaders can't read from `VK_EXT_external_memory_host` imported or pinned
host memory on RADV (Mesa 25.2-25.3). If fixed, all expert data could stay in
Vulkan-accessible host buffers and MUL_MAT_ID could run on GPU with 2 graph splits
— potentially 20+ t/s for all model sizes regardless of RAM.

**Status**: Confirmed on both Strix Halo (GFX1151) and Strix Point (GFX1150).
Not yet reported upstream. Would require Mesa/RADV investigation.

### F5. Upstream the Patch

The mmap-wrap + partial prefetch + auto-detect changes are generic enough to
benefit any UMA system (AMD APUs, future Intel with Battlemage iGPU, Apple Silicon
in theory). The patch is clean (456 lines) and non-invasive.

**Blockers**: The patch modifies `llama_params_fit` behavior (auto-clearing
user-set overrides), which may be controversial upstream. The mmap-wrap in
`llama-model.cpp` adds a new code path with `goto` labels. Would need cleanup
and testing on CUDA/Metal backends.

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
