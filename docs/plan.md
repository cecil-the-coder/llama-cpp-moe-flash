# MoE Flash — Implementation Plan

## Current State (Updated 2026-04-03)

**Production image: `4cb7bef`** on b8664 (AMD Strix Halo, 125 GB RAM, Radeon 8060S)

Patches applied: 0001 (core MoE flash + expert cache stable key + sync skip + force-offload guard), 0014 (vec-path byte-range overlap check), 0017 (disable upstream -fit + auto-detect CPU_MOE).

Single-backend architecture with auto-detect `--cpu-moe`:
- Models that fit in GTT (120 GB): auto-detect clears CPU_MOE, full GPU path
- Models exceeding GTT: CPU MoE path with expert GPU cache

| Model | Size | TPS | Loading | Status |
|---|---|---|---|---|
| GLM-4-7-Flash | 17 GB | **~50** | full GPU offload | Coherent |
| Qwen3-235B Q2_K | 80 GB | **20.0** | full GPU offload (no expert cache needed) | Coherent |
| DeepSeek-R1-0528 Q2_K | 228 GB | **4.1** | mmap experts + expert GPU cache + sync skip | Coherent (2.3x baseline) |

**I12 benchmark**: Stock Vulkan 20.7 t/s, moe-flash 20.0 t/s, ik_llama.cpp CPU-only 11.5 t/s (Qwen3).

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
| **KTransformers** | AMX-optimized CPU expert kernels | 28 t/s DeepSeek vs our 4.1 — **7× faster** expert matmul |
| **Fate** | Cross-layer gate prediction (97% accuracy) | Targeted prefetch vs our blind background sweep |
| **HOBBIT** | Mixed-precision: cache-miss experts at lower quant | Reduces I/O for cold experts |
| **flash-moe** | pread() + GCD on Apple Silicon | 4.4 t/s on 397B/48GB MacBook |
| **Fiddler** | Per-expert CPU-vs-GPU profiling | Optimal placement, not all-or-nothing |
| **PreScope** | Async I/O decoupled from compute | 141% throughput improvement |
| **ik_llama.cpp** | FlashMLA, fused MoE FFN, IQ Trellis quants | Better CPU perf, no Vulkan — see I12 |
| **llama.cpp #20757** | Two-tier GPU+RAM expert cache (SLRU) | 14 t/s PoC, seeking C++ implementer — see I10 |

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

## Optimization Research (2026-03-28)

Literature review of MoE inference optimization techniques. Sources: llama.cpp
issues, flash-moe experiments, ik_llama.cpp, academic papers, AMD documentation.
See `docs/findings.md` for full synthesis.

### I10. Two-Tier Expert Cache — TIER 1

**Source**: [llama.cpp #20757](https://github.com/ggml-org/llama.cpp/issues/20757)

Persistent GPU slot buffer with SLRU eviction for MoE experts. Python PoC
achieved **14 tok/s** (from 0.5-1 baseline) on 8 GB VRAM with 98-100% hit rate.

Key design:
- Fixed-address slot buffer on GPU (N slots per layer)
- Segmented LRU eviction: probationary (20%) + protected (80%)
- Frequency-gated admission: only cache expert on second miss
- Hook point: `ggml_backend_sched_compute_splits()` at line 1529

**Why it matters here**: On UMA, "copying" an expert into a GPU slot is a page
table remap, not a data copy. The slot population cost is effectively zero.
This is the single highest-leverage optimization available.

**Result** (images `7ad0e22`, `5ae96d2`): Force-offloaded MUL_MAT_ID to GPU.
GPU compute runs fast (18 t/s for q4km, vs 6.93 CPU) but **output is GARBAGE**.

Root cause: `input_cpy` tensor is allocated at full expert tensor size (5-10 GB
per projection). On RADV, `maxStorageBufferRange` = 4 GiB. The compute shader
descriptor binding can't address beyond 4 GiB → reads uninitialized memory.

Two additional bugs found during investigation:
1. `supports_op` rejects expert tensors > `maxStorageBufferRange` (correctly!)
2. `offload_op` rejects MUL_MAT_ID at bs=1 (`ne[2]=1 < min_batch_size=32`)

**Conclusion**: The full-size copy tensor approach cannot work on RADV.
`VK_EXT_shader_64bit_indexing` is NOT supported on GFX1151 — confirmed via
`vulkaninfo`. The Vulkan backend already has 64-bit indexing support (line 8191
in ggml-vulkan.cpp switches to 64b pipeline when tensor > maxStorageBufferRange),
but the extension is missing from the driver.

The fix requires a **fixed-size GPU slot buffer** (N slots per layer) with expert
ID remapping — the actual "two-tier cache" from #20757 (I10b below).

Bitset cache code retained in patch 0006 (harmless when offload inactive, ready
for slot buffer).

**Status**: BLOCKED — needs I10b (slot buffer) or VK_EXT_shader_64bit_indexing

### I10b. Fixed-Size Slot Buffer for GPU Expert Matmul — TIER 1

**Problem**: Full expert tensor (5-10 GB) exceeds `maxStorageBufferRange` (4 GiB).

**Design**: Pre-allocate a GPU buffer with N slots (e.g., 32 or 64) per MoE
projection, each slot = 1 expert. Total buffer = N × expert_size, fitting within
4 GiB. Before each MUL_MAT_ID dispatch:

1. Read routing IDs (which experts are needed for this token)
2. For each needed expert: check slot table → if miss, copy expert into an
   available slot (LRU eviction). On UMA this is effectively a page remap.
3. Build a `slot_map[expert_id] → slot_idx` mapping
4. Pass slot_map to the MUL_MAT_ID shader
5. Shader indexes `pos_a = slot_map[expert_idx] * stride` instead of `expert_idx * stride`

**Implementation points**:
- **Shader**: Add 1 indirection in `pos_a` calculation (mul_mm.comp line 241).
  Pass slot_map as push constant (fits in 256 bytes for 64 slots) or binding.
- **Scheduler**: In `ggml_backend_sched_compute_splits`, after reading routing
  IDs, populate slots and build the map. Replace full `input_cpy` with smaller
  slot buffer allocated via gallocr.
- **Allocation**: `input_cpy` tensor resized to `[ne0, ne1, N_SLOTS]` instead
  of `[ne0, ne1, n_expert]`. N_SLOTS=32 → ~1.3 GB for Q4_K_M, ~0.9 GB for Q2_K.
  Well within 4 GiB limit.
- **GTT budget**: 32 slots × 3 projections × 94 layers = ~360 GB for Q4_K_M,
  but gallocr reuses buffers across layers, so actual peak = ~4 GB.

**Alternatives considered**:
- `VK_EXT_shader_64bit_indexing`: Not supported on RADV/GFX1151
- Buffer device addresses (BDA): RADV supports BDA but llama.cpp MUL_MAT_ID
  shaders don't use BDA pointers — would need significant shader rewrite
- Multiple descriptor bindings (split tensor): Would need per-expert or
  per-chunk bindings, exceeds descriptor set limits

**First attempt** (images `86825d2`, `cc81e1d`): Implemented shrunk input_cpy
(ne[2]=32), LRU slot assignment, IDS rewrite, force-offload. Build succeeds,
graph split and allocation correct (283 splits, 1142 MiB compute buffer vs
3099 MiB with full tensor). But **SIGSEGV** (exit 139) during first inference.

GDB backtrace revealed crash in `llama_kv_cache::set_input_k_idxs` — NOT in
the slot buffer or MUL_MAT_ID code. The `ne[2]` shrink on the expert weight
tensor propagates through the graph: the MUL_MAT_ID output tensor dimensions
depend on `src[0]->ne[2]`, which corrupts KV cache setup when ne[2]=32
instead of 128.

**Conclusion**: Cannot shrink `ne[2]` on the copy tensor — tensor dimensions
are used throughout the graph for shape inference, not just descriptor binding.
Need a different approach: keep `ne[2]=n_expert` on the tensor but allocate
a smaller buffer underneath, with slot mapping only for data access.

Also found: shared IDS tensor bug — gate/up/down projections share one IDS
tensor, rewriting it for first projection corrupted later projections.
Fixed via `original_ids_cache` that preserves original expert IDs.

**Next approach**: Keep tensor shape intact (`ne[2]=n_expert`), but use a
custom smaller Vulkan buffer allocation (32 slots worth) and map expert IDs
to slot offsets in the copy loop. The Vulkan backend would bind the smaller
buffer but dispatch with the original n_as — the shader early-exits for
experts with zero token count (via `data_expert_count`), so only the 8
active experts (mapped to slots 0-7) need valid data.

**Second attempt** (image `ef1268e`): Keep tensor shape (`ne[2]=n_expert`),
remap active experts to low slots (0-31) within 4 GiB, rewrite IDS.
Still SIGSEGV — GDB shows same crash in `set_input_k_idxs`, called from
`common_speculative_is_compat` → `llama_decode` during model loading.
This is NOT in slot buffer code — it's a pre-existing b8298 bug triggered
by the force-offload graph structure (283 splits vs 190). The
`common_speculative_is_compat` test decode fails when the graph has MoE
expert weight cross-backend splits.

**Root cause of both crashes**: The force-offload changes the graph split
pattern (190→283 splits). `common_speculative_is_compat` does a trial
`llama_decode` during `load_model` that hits an uninitialized KV cache
state. Previous test with 5ae96d2 succeeded because it was tested
interactively (curl after server was ready) — the server survived the
speculative compat check by luck (different memory layout without GDB).

**Speculative compat fix** (image `54ac14c`): Skipped `common_speculative_is_compat`
trial decode that crashed in `set_input_k_idxs` with out-of-bounds slot_info.
Server now loads successfully and processes prompts. But **GPU SIGSEGV** (exit 139)
on first token generation (bs=1).

**Static analysis of crash** (image `54ac14c`):

- `slot_remap_mode = FALSE` for Qwen3-235B Q4_K_M: 128 experts × ~3.4 MB =
  ~435 MB < 4 GiB limit. Normal bitset cache mode runs (copies 8 active experts
  to `input_cpy` at their `id * expert_size` offsets).
- Suballocator offset hypothesis INCORRECT: Vulkan spec says `maxStorageBufferRange`
  limits RANGE only, not `offset + range`. Expert tensor range = 435 MB < 4 GiB, fine.
- The crash occurs specifically during generation (bs=1) → vec path
  (`ggml_vk_use_mul_mat_vec_id` returns TRUE when `src2->ne[1] <= 8`).
- For generation, `ggml_nrows(MUL_MAT_ID output) = top_k = 8 < 32` →
  subsequent ops scheduled to CPU. For prefill (bs=20), `ggml_nrows = 160 >= 32`
  → subsequent ops stay on GPU. Explains why prefill works but generation crashes.
- Actual crash cause unknown from static analysis alone.

Added `[MOE-DBG]` logging before each GPU MUL_MAT_ID split compute and
`[VEC-DBG]` logging at vec path entry (patch 0007). Need runtime log to identify
which specific split/tensor triggers the fault.

**Findings across all I10/I10b attempts**:
1. GPU MUL_MAT_ID compute works (18 t/s when data is valid)
2. Expert tensor range (435 MB) fits within 4 GiB — suballocator offset does NOT
   cause a range violation (Vulkan spec: maxStorageBufferRange limits range only)
3. The `common_speculative_is_compat` trial decode triggers a b8298 KV
   cache bug with changed graph splits (fixed by skipping the check)
4. Changing `ne[2]` on copy tensors breaks graph shape inference (SIGSEGV)
5. `VK_EXT_shader_64bit_indexing` not available on RADV/GFX1151
6. Crash only occurs in vec path (bs=1 generation), not batch path (bs=20 prefill)

**I10b Vec-Path Fix (2026-04-05)**: ROOT CAUSE FOUND AND FIXED.

The original aliasing check (`d_ids.buffer == d_D.buffer`) compared Vulkan
pool handles, not actual memory ranges. Since the suballocator places many
tensors in the same VkBuffer pool, this produced false positives on EVERY
MoE operation, forcing the slower batch path.

Patches applied (image `4c65a2f`):
- **0015**: `ggml_set_input + ggml_set_output` on `selected_experts`
- **0016**: gallocr respects INPUT flag in inplace reuse check
- **0014**: Runtime overlap check using `[offset, offset+size)` byte ranges
- **0017**: Disable upstream `llama_params_fit` (hangs for MoE models)

Result: Vec path works correctly, zero false positives.
**Qwen3-235B Q2_K: 23 t/s** (up from 20.7 baseline, +11%).

**Status**: ✅ COMPLETE — vec path working, no aliasing, coherent output

### I11. Two-Tier Expert GPU Cache — COMPLETE

**Source**: Vulkan spec, local codebase analysis, I10b slot buffer infrastructure

#### Phase 1: DeepSeek Fix (image `7e64a82`)

Root causes found and fixed:
1. Force-offload host buffer guard (patch 0001)
2. gallocr corruption from ggml_set_input/output (patches 0015/0016 removed)
3. Flux YAML duplicate key

#### Phase 2: Expert GPU Cache + Sync Skip (image `74a5930` on b8664)

**Cache mechanism**:
- Changed `expert_cache` key from `input_cpy` (tensor struct, changes each token) to `input->data` (source weight pointer, stable)
- GPU buffer contents persist across tokens (Vulkan reset is no-op, gallocr offsets deterministic)
- Cache warmup: ~5 tokens to ~80% hit rate, ~32 tokens for ~100%
- Memory: zero additional -- uses existing gallocr buffer allocation

**Key finding**: flash-moe `async_prefetch` bypasses the scheduler's expert copy path
entirely (disk -> GPU via io_uring). The expert GPU cache only works on the standard
CPU -> GPU copy path. Disabling flash-moe and using the cache is faster (4.1 vs 2.3 t/s).

**Optimizations applied in 74a5930**:
1. Expert GPU cache (stable cache key: `input->data` instead of `input_cpy`)
2. Sync skip for fully-cached expert tensors (skip IDS read + expert copy + synchronization)
3. b8664 rebase (Vulkan improvements)
4. Auto-detect CPU_MOE (clears for models that fit in GTT)
5. Flash-moe DISABLED for DeepSeek (expert cache outperforms async prefetch)

**Production patch stack** (image `74a5930`):
- 0001: Core MoE flash + expert cache (stable key) + sync skip + force-offload guard
- 0014: Vec-path runtime byte-range overlap check
- 0017: Disable upstream -fit + auto-detect CPU_MOE

**Performance**:
- DeepSeek-R1-0528 Q2_K (228 GB): **4.1 t/s** steady-state (peak 4.7), **2.3x over 1.8 t/s baseline**
- Qwen3-235B Q2_K (80 GB): 20 t/s, unchanged (auto-detect, no expert cache needed)
- GLM-4-7-Flash (17 GB): ~50 t/s, full GPU
- Flash-moe DISABLED for DeepSeek (expert GPU cache outperforms async prefetch)

**Honest assessment of remaining optimizations**:
1. ~~Reduce 284 graph splits~~ -- splits come from CPU<->GPU backend transitions, not expert copying. Can't be reduced without changing the scheduler's hybrid compute model.
2. ~~Combine flash-moe prefetch with expert cache~~ -- mutually exclusive paths; flash-moe bypasses the scheduler's expert copy entirely.
3. Thread count tuning -- minor, already optimal at 16 threads from I1.
4. **The real bottleneck is CPU MoE matmul speed.** AVX-512 on Zen 5 is 7x slower than AMX on Intel for expert matmul. No amount of I/O or copy optimization changes this.

**Status**: COMPLETE

#### Phase 3: Persistent Buffer Pool (current main)

**Mechanism**: Per-projection GPU buffer pool with LRU eviction. Each pool entry holds
one buffer keyed by `input->data` (weight tensor data pointer). Pool lives outside
gallocr so buffers persist across tokens.

**Finding: 0% cache hit rate.** The pool has 9 entries (default `GGML_MOE_POOL_ENTRIES=9`)
but there are 282 unique weight tensors (94 MoE layers x 3 projections: gate, up, down).
Each projection has a different `input->data` key, so within-layer sharing is impossible.
Every projection evicts a different projection's entry.

**Force-offload result**: Qwen3-235B Q4_K_M at **1.4 t/s** — slower than CPU MoE at 6-7 t/s.
The overhead is dominated by expert copy bandwidth: 8 experts x 3 projections x 94 layers
x 3.5 MB = 7.9 GB per token at ~15 GB/s = 527 ms (74% of the 714 ms token time).

#### Phase 3b: Per-Layer Pool Grouping (image `363a6d9`)

**Mechanism**: Group gate/up/down projections into layer-level pool entries. Each
`layer_pool_entry` holds 3 `proj_sub` entries. Same-layer detection via IDS tensor
pointer (gate/up/down share the same `selected_experts` within a layer).

**Result: Still 0% hit rate.** Layer grouping correctly reduces evictions from 282 to 94
per forward pass, but with 15 layer entries and 94 layers, the last 15 layers from one
token get immediately evicted when the next token starts at layer 0.

Tested: Qwen3-235B Q4_K_M at **1.4 t/s** — no improvement over per-projection pool.
Pool stats: `layer_hits=0, layer_misses=30926, evictions=30911, expert_hit_rate=0.0%`.

**Root cause**: Sequential layer execution (0,1,...,93) means the "last N" LRU cache
only preserves layers 79-93. Token 2 starts at layer 0, immediately evicting layer 79.
No layer survives across tokens regardless of pool size < 94.

**Key insight**: Pool-based caching fundamentally cannot help with 94 MoE layers and
sequential execution. Would need P=94 (~141 GB) to cache everything, which exceeds
available RAM. The only viable path for GPU MoE with >GTT models is upstream #20757
(two-tier GPU+RAM cache with proper shader support).

**Status**: COMPLETE — per-layer grouping verified, no performance improvement

### I12. ik_llama.cpp Benchmark — TIER 1

**Source**: [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)

Major llama.cpp fork with CPU-optimized MoE inference:
- **FlashMLA** (PR 273): fastest known CPU-only DeepSeek inference
- **Smart Expert Reduction** (PR 239): reduces computation for rarely-activated experts
- **Fused MoE FFN** (PR 229): batched expert processing
- **Tensor overrides** (PR 232): per-tensor GPU/CPU placement
- **IQ Trellis quantization**: better quality than Q2_K at similar size

**No Vulkan support** — CPU + CUDA only. But with 125 GB RAM and Zen 5,
CPU-only with FlashMLA + fused MoE might match or beat our current 1.37 tok/s
hybrid Vulkan/CPU setup. Worth benchmarking as a baseline.

**Test**: Build ik_llama.cpp with Zen 5 flags (`-march=znver5`), benchmark
DeepSeek-R1-0528 Q2_K CPU-only.

**Status**: not started

### I13. BF16 Expert Weights for CPU Matmul — TIER 2

**Source**: Local codebase analysis (`ggml-cpu/`)

Current Q4_0 dot product (`ggml_vec_dot_q4_0_q8_0` in `quants.c`) only has
AVX2 optimization — no AVX-512 path. But BF16 vec_dot uses `_mm512_dpbf16_ps`
(AVX-512 BF16) which IS optimized on Zen 5.

Even at half-width AVX-512 (256-bit FPU on Strix Halo, 2 cycles per 512-bit op),
BF16 outperforms Q4_0 because `_mm512_dpbf16_ps` does 2× the work per cycle
vs the AVX2 quantized path. The dequantization overhead in Q4_0 is eliminated.

**Trade-off**: BF16 weights are ~2× larger than Q4_0 (~4.5 bpw vs ~4.5 bpw for
Q4_0, but BF16 is 16-bit = no quantization). Expert file size doubles. For
models already near the RAM limit (228 GB on 125 GB), this may not fit.

**Alternative**: Try Q8_0 quant — has the same AVX2-only limitation but higher
accuracy. Or evaluate ik_llama.cpp's IQ4_KSS_R4 which repacks for optimal
AVX-512 access.

**Status**: not started

### I14. io_uring Polish (Registered Buffers + Single Issuer + THP) — TIER 2

**Source**: io_uring man pages, flash-moe experiments, Linux kernel docs

Three low-effort improvements to the existing io_uring background prefetch:

1. **`IORING_REGISTER_BUFFERS`**: Pin staging pool in kernel, skip
   `pin_user_pages()` per read. Eliminates ~752 page-pin operations per token
   (94 layers × 8 experts). Expected 5-15% reduction in read jitter.

2. **`IORING_SETUP_SINGLE_ISSUER`** (6.0+): Single submitter enables kernel
   optimizations. Our prefetch thread is the sole submitter — trivially
   applicable. Expected 2-5% from reduced kernel overhead.

3. **`MADV_HUGEPAGE`** + `MADV_COLLAPSE` on staging pool: 512× TLB pressure
   reduction for expert-sized allocations (2 MB pages vs 4 KB). One `madvise`
   call. Expected 3-8% for GTT access path.

**Avoid**: SQPOLL (wastes a P-core for bursty workload), O_DIRECT (loses 90%+
page cache hit rate), expert prediction heuristics (25-53% accuracy, net negative).

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
Single backend: moe-flash-cpumoe (CPU_MOE=1, image 4cb7bef)

llama_params_fit auto-detect:
  ├── Model fits in device memory? → Clear CPU_MOE override
  │   → Vulkan alloc+copy → 20-50 t/s (glm, Q2K)
  │
  └── Model doesn't fit? → Keep CPU_MOE override
      ├── Model ≤ RAM? → Pinned alloc (ggml_vk_host_malloc + copy)
      └── Model > RAM? → mmap-wrap (demand-paged)
                          + Expert GPU cache (stable key, ~100% hit after 32 tokens)
                          + Sync skip for fully-cached expert tensors
                          + Partial prefetch (MADV_WILLNEED, up to MemAvail - 8G)
                          + MADV_RANDOM + MADV_HUGEPAGE → 4.1 t/s (DeepSeek)
```

## Completed Phases

- **Phase 0**: Measurement & baseline
- **Phase 1**: Expert file splitter tool
- **Phase 2**: io_uring prefetch prototype
- **Phase 3**: Vulkan integration (host buffer fix, hybrid alloc, prefetch, madvise)
- **Phase 4**: Integration with inference-budget-controller
- **Phase 5**: Auto-detect `--cpu-moe` + single-backend deployment

## Lessons Learned (2026-04-03)

### What worked
1. **Auto-detect CPU_MOE**: Single image handles all model sizes. Models <=GTT get full GPU, >GTT get CPU MoE. No manual configuration needed.
2. **Expert GPU cache with stable key**: Using `input->data` (source weight pointer) instead of `input_cpy` (tensor struct) as cache key. GPU buffers persist across tokens because Vulkan reset is a no-op and gallocr offsets are deterministic. Zero extra memory.
3. **mmap-wrap for >RAM models**: Stock `--cpu-moe` mallocs expert tensors and OOMs for 228 GB on 125 GB RAM. Wrapping mmap as CPU buffer enables demand paging from NVMe.
4. **Disabling flash-moe for cached path**: Counter-intuitive but correct. The expert GPU cache only works on the standard CPU->GPU copy path. Flash-moe's async prefetch bypasses this entirely.

### What didn't work
1. **Slot buffer / expert ID remapping (I10b)**: Three separate attempts. Changing `ne[2]` breaks graph shape inference. Keeping `ne[2]` intact but remapping IDS fails because the MUL_MAT_ID shader uses expert IDs for data addressing throughout. Would need deep shader modification that upstream #20757 is better positioned to solve.
2. **Graph split reduction**: Assumed splits came from expert weight copying. Actually come from CPU<->GPU backend transitions for attention vs MoE layers. Fundamental to the hybrid compute model, not an optimization target.
3. **Flash-moe async prefetch for >GTT**: Reads expert weights from disk each token via io_uring instead of using cached GPU buffers. Net slower (2.3 vs 4.1 t/s) because it defeats the expert GPU cache.
4. **Prediction-based expert prefetch (F2, I5)**: Scheduler expert copy callback never fires with `--cpu-moe` because all expert data stays on CPU -- no cross-backend copy to trigger the callback.
5. **io_uring optimizations**: No measurable benefit over posix_fadvise. Both just hint the kernel to readahead. The bottleneck is compute, not I/O syscall overhead.

### Key technical insight
For models exceeding GTT on this hardware, the performance ceiling is set by CPU expert matmul speed (AVX-512 on Zen 5). All I/O, copy, prefetch, and caching optimizations have been exhausted or shown to be irrelevant. The path to higher performance for >GTT models is either better CPU kernels (AMX, fused MoE FFN) or getting expert matmul back on GPU (upstream two-tier cache #20757).

---

## CI/CD

- Repo: `github.com/cecil-the-coder/llama-cpp-moe-flash`
- Images: `ghcr.io/cecil-the-coder/llama-cpp-moe-flash:<sha>`
- Workflow: `.github/workflows/build-image.yml` (triggers on `patches/` or `docker/`)
- eh-ops-private: `github.com/themicknugget/eh-ops-private` (Flux-managed)
- Backend: `llamacpp-vulkan-moe-flash-cpumoe` in `backends/kustomization.yaml`

### How to ship a change

1. **Edit patches or Dockerfile** in this repo (never build Docker images locally)
2. **Commit and push to `main`** — CI triggers on changes to `patches/`, `docker/`,
   or `.github/workflows/build-image.yml`
3. **CI builds and pushes** the image to `ghcr.io/cecil-the-coder/llama-cpp-moe-flash:<short-sha>`
4. **Update the Flux deployment** in eh-ops-private to reference the new image SHA
5. **Test on shadow node** — Flux reconciles, pod restarts with the new image

Do NOT run `docker build` or `docker push` locally. The `docker/build.sh` script
is a legacy convenience script and pushes to the wrong registry.
