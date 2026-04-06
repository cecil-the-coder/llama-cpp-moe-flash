# Next Investigations: Roadmap 2026-Q2

**Status**: COMPLETE -- Slot remapping with N_SLOTS=96 delivers **10.4-11.1 t/s** on Qwen3 Q4_K_M (6x baseline). 97.1% expert hit rate. GPU compute is now the dominant cost. (2026-04-03)

**Production Image**: `17aca27` on b8664

---

## Current State

**Models that fit in GTT (<=120 GB)**: Production-ready. 19-50 t/s, full GPU, no issues.

**Models exceeding GTT (>120 GB) -- COMPLETE**:
- Qwen3-235B Q4_K_M (133 GB): **10.4-11.1 t/s** GPU MoE via 96-slot remapping (6x over 32-slot baseline)
- Expert hit rate: **97.1%** with 96 slots for 128 experts (K=8)
- ~90 GB persistent GPU buffers, configurable via `GGML_MOE_N_SLOTS` env var
- DeepSeek-R1-0528 Q2_K (228 GB): ~4 t/s CPU MoE (can't test slot remapping -- 128 GB RAM limit)

**N_SLOTS tuning results:**

| N_SLOTS | Hit Rate | t/s | Memory | Notes |
|---------|----------|-----|--------|-------|
| 32 | 74.9% | 3.5-4.1 | ~35 GB | Default |
| 64 | 94.4% | 7.5-9.4 | ~60 GB | Good for low-RAM |
| 96 | 97.1% | 10.4-11.1 | ~90 GB | Optimal for 128 GB |
| 128 | -- | OOM | ~123 GB | Exceeds RADV/UMA limits |

**Current state**: At N_SLOTS=96, GPU compute (~90ms/token) is the dominant cost. The 2.9% miss rate adds only ~25ms copy overhead. Further improvement requires faster GPU compute or upstream shader optimizations.

**What we've exhausted** (before slot remapping solved the core problem):
- I/O optimizations (io_uring, posix_fadvise, registered buffers, hugepages) -- no measurable benefit
- Per-projection buffer pool -- 0% hit rate, 282 tensors overwhelm 9 entries
- Per-layer buffer pool grouping -- still 0% hit rate, 94 layers > 15 pool entries
- Flash-moe async prefetch -- slower than cached path for DeepSeek (2.3 vs 4.1 t/s)
- Graph split reduction -- splits are from CPU<->GPU backend transitions, not optimizable

---

## Completed Investigations

- **I10b** - **Slot remapping COMPLETE**: 10.4-11.1 t/s GPU MoE on Qwen3 Q4_K_M (133 GB, >GTT). 97.1% hit rate with 96 slots. N_SLOTS tuning: 32->64->96 (6x speedup). Configurable via `GGML_MOE_N_SLOTS`.
- **I11** - Expert GPU cache + sync skip: DeepSeek 4.1 t/s (2.3x baseline)
- **I12** - ik_llama.cpp benchmark: Vulkan 2x faster for in-GTT models
- **I14** - io_uring polish (SINGLE_ISSUER, MADV_HUGEPAGE): no measurable benefit
- **I17** - Prometheus metrics infrastructure
- **I18** - Cache hit tracking fix

---

## Priority Matrix

| Investigation | Impact | Effort | Status | Recommendation |
|---------------|--------|--------|--------|----------------|
| **Slot remapping (N_SLOTS=96)** | Very High | High | **DONE** | 10.4-11.1 t/s, 97.1% hit rate, 6x baseline |
| **N_SLOTS tuning** | High | Low | **DONE** | 32->64->96 sweep complete. 96 optimal for 128 GB |
| **Cross-layer expert prediction** | Medium | High | Not started | Could reduce remaining 2.9% miss rate |
| **imatrix-based slot pre-seeding** | Low | Medium | Not started | Reduces warmup time, marginal steady-state gain |
| **Upstream rebase tracking** | High | Low | Ongoing | Track #20757, new llama.cpp releases |
| **CPU kernel improvement** | Medium | High | Not started | Less urgent now that GPU MoE works |
| **I13** - BF16 CPU Matmul | Low | Low | Not started | GPU path now preferred over CPU MoE |
| **I7** - Context Scaling | Low | Low | Not started | Not the bottleneck |
| **I8** - Batch Size Tuning | Low | Low | Not started | Minor |

---

## TIER 1: Recommended Next Steps

### 0. Slot Remapping with N_SLOTS Tuning -- DONE (COMPLETE)

**Result**: **10.4-11.1 t/s** on Qwen3 Q4_K_M with 96-slot remapping. 6x speedup from 32-slot baseline. Image `17aca27` on b8664.

**What works:**
- `ne[2]=N_SLOTS` override flows correctly to `n_as=N_SLOTS` in all three shaders (batch, vec, count_experts)
- Persistent pool outside gallocr -- buffers survive across tokens
- LRU slot eviction with bidirectional mapping
- IDS rewrite with deferred write -- prevents overwrite by input copy loop
- Original IDS cache -- prevents gate/up/down cross-contamination
- No shader modifications needed
- Configurable via `GGML_MOE_N_SLOTS` environment variable

**N_SLOTS tuning results:**

| N_SLOTS | Hit Rate | t/s | Memory | Notes |
|---------|----------|-----|--------|-------|
| 32 | 74.9% | 3.5-4.1 | ~35 GB | Default |
| 64 | 94.4% | 7.5-9.4 | ~60 GB | Good for low-RAM |
| 96 | 97.1% | 10.4-11.1 | ~90 GB | Optimal for 128 GB |
| 128 | -- | OOM | ~123 GB | Exceeds RADV/UMA limits |

**Performance at N_SLOTS=96:**
- Expert hit rate: 97.1% (96 slots, 128 experts, K=8)
- 2.9% miss rate -> ~25ms copy overhead per token
- GPU compute: ~90ms per token (dominant cost)
- ~90 GB persistent GPU buffers

### 0a. Further Hit Rate Optimization (DIMINISHING RETURNS)

At 97.1% hit rate, expert copy overhead is only ~25ms/token. GPU compute (~90ms) is now 78% of token time. Further miss rate reduction has limited impact.

**Remaining approaches (low priority):**
1. **Cross-layer expert prediction**: Prefetch likely experts based on routing patterns. Could reduce 2.9% miss rate but impact on t/s is marginal (~25ms savings max).
2. **imatrix-based slot pre-seeding**: Reduces warmup time but steady-state already at 97.1%.

**Effort**: Medium
**Impact**: Low (at most ~10% improvement from eliminating remaining 2.9% miss rate)

### 1. Track Upstream llama.cpp MoE Work

**Goal**: Monitor and rebase when upstream lands MoE improvements.

**Key upstream items**:
- **#20757** (two-tier expert cache): GPU expert matmul with proper shader support for >GTT models. Python PoC showed 14 t/s. Seeking C++ implementer. When merged, this would be the single biggest improvement for DeepSeek-class models (expected 10-15 t/s).
- **Better CPU MoE kernels**: Any upstream improvement to Q2_K/Q4_K matmul with AVX-512 paths.
- **Scheduler improvements**: Reduced graph splits for hybrid CPU/GPU compute.

**Effort**: Low (monitoring + periodic rebase)
**Impact**: High (potentially 3-4x for >GTT models)

### 2. CPU Kernel Improvement

**Goal**: Improve CPU expert matmul speed, which is the dominant bottleneck.

**Options** (in order of feasibility):
1. **I13 - BF16 AVX-512 matmul**: Test if BF16 `_mm512_dpbf16_ps` outperforms Q4_0 AVX2 on Zen 5. Low effort, unclear payoff due to 2x memory increase.
2. **Port ik_llama.cpp fused MoE FFN**: Batched expert processing reduces overhead. High effort (ik_llama.cpp is a major fork), medium payoff.
3. **Wait for hardware with AMX**: Intel AMX gives 7x expert matmul speedup. Requires hardware purchase.

**Effort**: Medium to High
**Impact**: 2-5x for >GTT models (depending on approach)

---

## TIER 2: De-prioritized

### Slot Buffer Shader Modification (CONFIRMED UNNECESSARY)

Slot remapping works without any shader modifications. `ne[2]=32` with slot-remapped IDS
values (0..31) produces correct results in all three shaders (batch, vec, count_experts).
Previous failures were from gallocr corruption, not shader issues. Proven by image `a9e911e`.

### Graph Split Reduction

**Why de-prioritized**: The 284 splits come from CPU<->GPU backend transitions for attention
vs MoE layers, not from expert weight copying. Reducing splits requires changing how the
scheduler assigns backends for hybrid compute -- fundamental architecture change, not an
optimization target.

### Flash-moe Async Prefetch Improvements

**Why de-prioritized**: Flash-moe bypasses the scheduler's expert copy path entirely
(reads from disk via io_uring). This defeats the expert GPU cache, which is the primary
optimization delivering 4.1 t/s. The two approaches are mutually exclusive.

---

## Decision Framework

```
Current state: 10.4-11.1 t/s Qwen3 Q4_K_M (GPU MoE, 96-slot remapping), 19-50 t/s for <=GTT models.

Slot remapping COMPLETE. N_SLOTS=96 is optimal for 128 GB UMA.
GPU compute (~90ms/token) is now the dominant cost, not expert copy bandwidth.

Further optimization paths:
    → Upstream #20757 when merged: proper two-tier cache with SLRU
    → Hardware with Intel AMX: 28 t/s (KTransformers benchmark)
    → Cross-layer prediction: marginal gain (2.9% miss rate already low)
```

---

## Comparison with Other Systems

| System | >GTT t/s | Approach | vs Our 3.5-4.1 t/s |
|--------|----------|----------|---------------------|
| **KTransformers** | 28 | Intel AMX CPU kernels | 2.5x faster |
| **llama.cpp #20757 PoC** | 14 | Two-tier GPU cache (Python) | 1.3x faster |
| **Our moe-flash (96-slot remap)** | **10.4-11.1** | **GPU MoE, 96 slots, LRU eviction** | **baseline** |
| **flash-moe** | 4.4 | Apple SSD + Metal (397B model) | 2.5x slower |
| **Our moe-flash (CPU MoE)** | 4.1 | AVX-512 CPU MoE (DeepSeek) | 2.7x slower |
| **ik_llama.cpp** | 1.5 | CPU-only, no flash_attn | 7x slower |

---

*Last Updated*: 2026-04-03
