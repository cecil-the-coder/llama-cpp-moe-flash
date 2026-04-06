# Next Investigations: Roadmap 2026-Q2

**Status**: MILESTONE -- Slot remapping produces first-ever correct GPU MoE output. Qwen3 Q4_K_M: 3.5-4.1 t/s with 74.9% expert hit rate (32 slots, K=8). Remaining work is hit rate optimization. (2026-04-03)

**Production Image**: `a9e911e` on b8664

---

## Current State

**Models that fit in GTT (<=120 GB)**: Production-ready. 19-50 t/s, full GPU, no issues.

**Models exceeding GTT (>120 GB) -- BREAKTHROUGH**:
- Qwen3-235B Q4_K_M (133 GB): **3.5-4.1 t/s** GPU MoE via slot remapping (was 1.5 t/s always-copy)
- Expert hit rate: 74.9% with 32 slots for 128 experts (K=8)
- DeepSeek-R1-0528 Q2_K (228 GB): ~4 t/s CPU MoE (slot remapping not yet tested)

**Remaining bottleneck**: 25% cache miss rate drives ~560 expert copies per token (~2 GB at ~3.5 MB each). At 15 GB/s UMA bandwidth: ~133ms copy overhead + ~100ms GPU compute = ~233ms/token (~4.3 t/s theoretical, matches measured 3.5-4.1).

**What we've exhausted** (before slot remapping solved the core problem):
- I/O optimizations (io_uring, posix_fadvise, registered buffers, hugepages) -- no measurable benefit
- Per-projection buffer pool -- 0% hit rate, 282 tensors overwhelm 9 entries
- Per-layer buffer pool grouping -- still 0% hit rate, 94 layers > 15 pool entries
- Flash-moe async prefetch -- slower than cached path for DeepSeek (2.3 vs 4.1 t/s)
- Graph split reduction -- splits are from CPU<->GPU backend transitions, not optimizable

---

## Completed Investigations

- **I10b** - **Slot remapping BREAKTHROUGH**: 3.5-4.1 t/s GPU MoE on Qwen3 Q4_K_M (133 GB, >GTT). 74.9% hit rate with 32 slots. First correct GPU MoE output in this project.
- **I11** - Expert GPU cache + sync skip: DeepSeek 4.1 t/s (2.3x baseline)
- **I12** - ik_llama.cpp benchmark: Vulkan 2x faster for in-GTT models
- **I14** - io_uring polish (SINGLE_ISSUER, MADV_HUGEPAGE): no measurable benefit
- **I17** - Prometheus metrics infrastructure
- **I18** - Cache hit tracking fix

---

## Priority Matrix

| Investigation | Impact | Effort | Status | Recommendation |
|---------------|--------|--------|--------|----------------|
| **Slot remapping (N_SLOTS=32)** | Very High | High | **DONE** | 3.5-4.1 t/s, 74.9% hit rate. Remaining: hit rate optimization |
| **Hit rate optimization (more slots)** | High | Medium | Next step | 64/128 slots to reduce 25% miss rate |
| **Cross-layer expert prediction** | High | High | Not started | Prefetch likely experts based on routing patterns |
| **imatrix-based slot pre-seeding** | Medium | Medium | Not started | Pre-load frequent experts from importance matrix |
| **Upstream rebase tracking** | High | Low | Ongoing | Track #20757, new llama.cpp releases |
| **CPU kernel improvement** | Medium | High | Not started | Less urgent now that GPU MoE works |
| **I13** - BF16 CPU Matmul | Low | Low | Not started | GPU path now preferred over CPU MoE |
| **I7** - Context Scaling | Low | Low | Not started | Not the bottleneck |
| **I8** - Batch Size Tuning | Low | Low | Not started | Minor |

---

## TIER 1: Recommended Next Steps

### 0. Slot Remapping with N_SLOTS=32 -- DONE (MILESTONE)

**Result**: First-ever correct GPU MoE output. Qwen3 Q4_K_M: **3.5-4.1 t/s** with 74.9% expert hit rate. Image `a9e911e` on b8664.

**What works:**
- `ne[2]=32` override flows correctly to `n_as=32` in all three shaders (batch, vec, count_experts)
- Persistent pool outside gallocr -- buffers survive across tokens
- LRU slot eviction -- 32 slots with bidirectional mapping
- IDS rewrite with deferred write -- prevents overwrite by input copy loop
- Original IDS cache -- prevents gate/up/down cross-contamination
- No shader modifications needed

**Performance breakdown:**
- Expert hit rate: 74.9% (32 slots, 128 experts, K=8)
- 25% miss rate -> ~560 expert copies/token at ~3.5 MB = ~2 GB
- Copy overhead: ~133ms at 15 GB/s UMA bandwidth
- GPU compute: ~100ms
- Total: ~233ms/token -> ~4.3 t/s theoretical (matches measured 3.5-4.1)

### 0a. Hit Rate Optimization (NEXT STEP)

**Goal**: Reduce the 25% cache miss rate to improve beyond 4.1 t/s.

**Approaches (in order of expected impact):**
1. **More slots (64 or 128)**: Direct hit rate improvement. 64 slots should give ~87% hit rate, 128 slots ~94%. Trade-off: more memory per buffer (64 slots = ~70 GB for P=94).
2. **Cross-layer expert prediction**: Analyze routing patterns to prefetch likely experts before they're needed. Could dramatically reduce cold misses.
3. **imatrix-based slot pre-seeding**: Pre-load frequently-used experts from importance matrix data (from #20757 discussion). Reduces warmup time and improves steady-state hit rate.

**Effort**: Medium
**Impact**: High (expected 6-10 t/s with 90%+ hit rate)

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
Current state: 3.5-4.1 t/s Qwen3 Q4_K_M (GPU MoE, slot remapping), 19-50 t/s for <=GTT models.

Slot remapping WORKS. The question is now: how much faster can we go?

Hit rate optimization (immediate):
    → 64/128 slots: higher hit rate, fewer copies, expected 6-10 t/s
    → Cross-layer prediction: prefetch likely experts
    → imatrix pre-seeding: warm start from importance matrix

Longer term:
    → Upstream #20757 when merged: proper two-tier cache with SLRU
    → Hardware with Intel AMX: 28 t/s (KTransformers benchmark)
```

---

## Comparison with Other Systems

| System | >GTT t/s | Approach | vs Our 3.5-4.1 t/s |
|--------|----------|----------|---------------------|
| **KTransformers** | 28 | Intel AMX CPU kernels | 7x faster |
| **llama.cpp #20757 PoC** | 14 | Two-tier GPU cache (Python) | 3.5x faster |
| **flash-moe** | 4.4 | Apple SSD + Metal (397B model) | ~comparable |
| **Our moe-flash (slot remap)** | 3.5-4.1 | GPU MoE, 32 slots, LRU eviction | baseline |
| **Our moe-flash (CPU MoE)** | 4.1 | AVX-512 CPU MoE (DeepSeek) | comparable |
| **ik_llama.cpp** | 1.5 | CPU-only, no flash_attn | 2.5x slower |

---

*Last Updated*: 2026-04-03
