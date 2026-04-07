# Next Investigations: Roadmap 2026-Q2

**Status**: COMPLETE -- 5-patch stack on b8664 delivers **7-10 t/s** on Qwen3 Q4_K_M with N_SLOTS=64 (4x baseline). 94.6% expert hit rate with speculative prefetch. GPU compute is now the dominant cost. (2026-04-03)

**Production Image**: `7937441` on b8664

---

## Current State

**Models that fit in GTT (<=120 GB)**: Production-ready. 19-50 t/s, full GPU, no issues.

**Models exceeding GTT (>120 GB) -- COMPLETE**:
- Qwen3-235B Q4_K_M (133 GB): **7-10 t/s** GPU MoE via 64-slot remapping (4x over baseline)
- Expert hit rate: **94.6%** with speculative prefetch (64 slots for 128 experts, K=8)
- ~60 GB persistent GPU buffers, stable on 128 GB node
- DeepSeek-R1-0528 Q2_K (228 GB): ~4 t/s CPU MoE (can't test slot remapping -- 128 GB RAM limit)

**N_SLOTS tuning results:**

| N_SLOTS | Hit Rate | t/s | Memory | Notes |
|---------|----------|-----|--------|-------|
| 32 | 74.9% | 3.5-4.1 | ~35 GB | Default |
| 64 | 94.6% | 7-10 | ~60 GB | **Stable production (128 GB)** |
| 96 | 97.1% | 10.4-11.1 | ~90 GB | Unstable (needs 192+ GB RAM) |
| 128 | -- | OOM | ~123 GB | Exceeds RADV/UMA limits |

**Current state**: At N_SLOTS=64, GPU compute is the dominant cost. The 5.4% miss rate adds modest copy overhead. N_SLOTS=96 delivers +17% but crashes on 128 GB nodes (90 GB pool + mmap exceeds RAM). Further improvement requires faster GPU compute, more RAM, or upstream shader optimizations.

---

## Opportunity Landscape

### Tier 1: High Impact, Low-Medium Effort

#### A: DeepSeek on Slot Remapping
**Goal**: Test slot remapping on DeepSeek-R1-0528 (256 experts, K=8). Currently stuck at ~4 t/s CPU MoE because the model (228 GB) exceeds our 128 GB RAM limit.
**Requires**: Access to a 256 GB node.
**Expected**: 3-5x speedup (4 -> 12-20 t/s) based on Qwen3 results with slot remapping.
**Effort**: Low (once hardware available)

#### C: imatrix Pre-Seeding for Cold Start Elimination
**Goal**: Pre-populate slot cache at startup with the most frequently activated experts from imatrix calibration data. Eliminates the cold-start phase (first ~5 tokens at 1.8 t/s).
**How**: Parse imatrix GGUF file at model load, map expert activation counts to slot assignments, pre-copy hottest experts before first inference.
**Status**: Deferred. The cold-start penalty is only ~3 seconds (5 tokens at 1.8 t/s). Steady-state is 11 t/s. The complexity of imatrix parsing is not justified by 3 seconds of savings.
**Effort**: Medium
**Impact**: Low (only affects first ~5 tokens)

#### D: Graph Split Reduction (282 -> 96 splits) -- DONE
**Goal**: Merge gate/up/down MoE projections within each layer into a single graph split.
**Status**: COMPLETE as patch 0021. Deployed in image `7937441`. Marginal t/s improvement in practice -- GPU compute dominates, not sync overhead.
**Patch**: `patches/0021-merge-moe-splits-within-layer.patch`

### Tier 2: Medium Impact, Medium Effort

#### E: Speculative Expert Prefetch -- DONE
**Goal**: While the GPU is computing layer N, pre-seed layer N+1's slots with predicted experts.
**Status**: COMPLETE as patch 0022. Deployed in image `7937441`. Improves hit rate from 94.4% to 94.6% with N_SLOTS=64. Marginal t/s improvement -- the LRU cache already captures most reuse.
**Patch**: `patches/0022-speculative-expert-prefetch.patch`

#### F: Adaptive N_SLOTS per Layer (Reduce Pool Memory 20-30%) -- INVESTIGATED, SKIP
**Goal**: Instead of a fixed N_SLOTS=96 for all layers, use fewer slots for layers with lower expert diversity.
**How**: Profile expert activation patterns across layers. Layers where fewer unique experts are used can have smaller pools. E.g., early layers might need only 64 slots while later layers need 96.
**Finding**: Expert activation is homogeneous across layers -- all layers use roughly the same number of unique experts. Per-layer tuning yields negligible memory savings. Better to upgrade RAM to enable N_SLOTS=96 globally.
**Effort**: Medium
**Impact**: SKIP -- homogeneous experts make per-layer tuning pointless; upgrade RAM instead

### Tier 3: Research/Exploration

#### G: Expert Routing Prediction (Layer N -> N+1) -- INVESTIGATED, DEFER
**Goal**: Predict which experts will be activated in layer N+1 based on layer N's routing decisions.
**How**: Build a lightweight prediction model (e.g., frequency table or small MLP) from routing statistics. Pre-load predicted experts before they are needed.
**Finding**: At 94.6% hit rate, expert copy is ~3.5ms per miss. Even perfect prediction eliminating all misses saves <0.1 t/s. The bottleneck is GPU compute (~85ms per token), not expert loading. Not worth the complexity.
**Effort**: High
**Impact**: DEFER -- <0.1 t/s gain, GPU-compute-limited

#### H: Rebase Tracking (llama.cpp Upstream)
**Goal**: Stay current with upstream llama.cpp developments, especially MoE-related changes.
**Key items**: #20757 (two-tier expert cache), scheduler improvements, new Vulkan shader optimizations.
**Effort**: Low (monitoring + periodic rebase)
**Impact**: High when upstream lands major MoE improvements

#### I: Patch Surface Reduction
**Goal**: Minimize our patch footprint against upstream to ease rebasing and reduce merge conflicts.
**How**: Factor out self-contained features into separate smaller patches. Upstream compatible changes where possible.
**Effort**: Low-Medium
**Impact**: Low (maintenance quality, not performance)

#### J: Multi-Model Expert Pool Sharing
**Goal**: Share the GPU expert pool across multiple MoE models.
**How**: When multiple models are loaded (e.g., Qwen3 Q2_K and Q4_K_M), share the same GPU buffer pool and evict experts across models.
**Challenge**: Different models have different expert sizes. Would need a slab allocator or size-class pools.
**Effort**: High
**Impact**: Medium (enables running multiple large MoE models on a single GPU)

### Not Pursuing Now

#### B: Upstream Contribution to #20757
**Status**: Deferred. The upstream two-tier cache PR (#20757) has a Python PoC showing 14 t/s but is seeking a C++ implementer. Our slot remapping approach already achieves 10.4-11.1 t/s. Contributing to upstream is valuable but not a priority while we have production workloads to optimize.

---

## Completed Investigations

- **F** - **Adaptive N_SLOTS INVESTIGATED, SKIP**: Homogeneous expert activation across layers -- per-layer tuning yields negligible savings. Better to upgrade RAM.
- **G** - **Routing prediction INVESTIGATED, DEFER**: <0.1 t/s gain. At 94.6% hit rate, GPU compute (~85ms) dominates over expert copy (~3.5ms). Not worth the complexity.
- **I10b** - **Slot remapping COMPLETE**: 7-10 t/s GPU MoE on Qwen3 Q4_K_M (133 GB, >GTT) with N_SLOTS=64. 94.6% hit rate with speculative prefetch. Stable on 128 GB. Configurable via `GGML_MOE_N_SLOTS`.
- **D** - **Split merging COMPLETE** (patch 0021): 282->96 splits. Marginal t/s improvement.
- **E** - **Speculative prefetch COMPLETE** (patch 0022): Pre-seed next layer's slots. Marginal t/s improvement.
- **I11** - Expert GPU cache + sync skip: DeepSeek 4.1 t/s (2.3x baseline)
- **I12** - ik_llama.cpp benchmark: Vulkan 2x faster for in-GTT models
- **I14** - io_uring polish (SINGLE_ISSUER, MADV_HUGEPAGE): no measurable benefit
- **I17** - Prometheus metrics infrastructure
- **I18** - Cache hit tracking fix

---

## Priority Matrix

| Investigation | Impact | Effort | Status | Recommendation |
|---------------|--------|--------|--------|----------------|
| **D: Split reduction** | Marginal | Low-Med | **DONE (patch 0021)** | 282->96 splits, marginal t/s gain |
| **E: Speculative prefetch** | Marginal | Medium | **DONE (patch 0022)** | Pre-seed next layer's slots |
| **A: DeepSeek slot remap** | Very High | Low | Blocked (needs 256 GB) | 3-5x expected once hardware available |
| **C: imatrix pre-seeding** | Low | Medium | Deferred | Only saves ~3s cold start |
| **F: Adaptive N_SLOTS** | Medium | Medium | **INVESTIGATED, SKIP** | Homogeneous experts; upgrade RAM instead |
| **G: Routing prediction** | Low-Med | High | **INVESTIGATED, DEFER** | <0.1 t/s gain, GPU-compute-limited |
| **H: Rebase tracking** | High | Low | Ongoing | Track #20757, upstream releases |
| **I: Patch surface reduction** | Low | Low-Med | Not started | Maintenance quality |
| **J: Multi-model pool** | Medium | High | Not started | Multiple MoE models on one GPU |
| **B: Upstream #20757** | Medium | High | Deferred | Focus on production first |

---

## Diminishing Returns

At 94.6% expert hit rate with N_SLOTS=64, the per-token cost breakdown is:

| Phase | Time | % of Token |
|-------|------|------------|
| **GPU compute** | ~85 ms | ~89% |
| Sync overhead | ~7 ms | ~7% |
| Expert copy (misses) | ~3.5 ms | ~4% |

GPU compute dominates. Optimizations targeting expert copy or miss rate (F, G) yield <0.1 t/s improvement because they only affect the ~4% copy slice. The remaining viable paths to faster inference are:

1. **More RAM (192+ GB)**: Enables N_SLOTS=96 for 10.4-11.1 t/s (+17%)
2. **Upstream #20757**: Proper two-tier GPU cache with SLRU eviction and shader-level expert matmul
3. **Newer llama.cpp**: Upstream Vulkan shader optimizations reduce the 85ms GPU compute
4. **Better hardware**: Intel AMX (28 t/s per KTransformers), faster GPU

---

## Decision Framework

```
Current state: 7-10 t/s Qwen3 Q4_K_M (GPU MoE, 64-slot remapping), 20-50 t/s for <=GTT models.

5-patch stack COMPLETE. N_SLOTS=64 is the stable production config for 128 GB.
GPU compute is now the dominant cost, not expert copy bandwidth.
Split merging (D) and speculative prefetch (E) delivered marginal gains.

Further optimization paths:
    -> A: DeepSeek on slot remap: needs 256 GB node
    -> More RAM (192+ GB): enables N_SLOTS=96 for 10.4-11.1 t/s
    -> H: Upstream #20757 when merged: proper two-tier cache with SLRU
    -> Hardware with Intel AMX: 28 t/s (KTransformers benchmark)
```

---

## Comparison with Other Systems

| System | >GTT t/s | Approach | vs Our 7-10 t/s |
|--------|----------|----------|---------------------|
| **KTransformers** | 28 | Intel AMX CPU kernels | 3-4x faster |
| **llama.cpp #20757 PoC** | 14 | Two-tier GPU cache (Python) | 1.4-2x faster |
| **Our moe-flash (96-slot, unstable)** | 10.4-11.1 | GPU MoE, 96 slots (needs 192+ GB) | +17% (unstable) |
| **Our moe-flash (64-slot, production)** | **7-10** | **GPU MoE, 64 slots, LRU + prefetch** | **baseline** |
| **flash-moe** | 4.4 | Apple SSD + Metal (397B model) | ~2x slower |
| **Our moe-flash (CPU MoE)** | 4.1 | AVX-512 CPU MoE (DeepSeek) | ~2x slower |
| **ik_llama.cpp** | 1.5 | CPU-only, no flash_attn | 5-7x slower |

---

*Last Updated*: 2026-04-03
