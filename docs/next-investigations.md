# Next Investigations: Roadmap 2026-Q2

**Status**: All active investigations complete. DeepSeek 4.1 t/s (2.3x baseline), CPU MoE matmul is the bottleneck. (2026-04-03)

**Production Image**: `ghcr.io/cecil-the-coder/llama-cpp-moe-flash:4cb7bef` (b8664)

---

## Current State: Honest Assessment

**Models that fit in GTT (<=120 GB)**: Production-ready. 20-50 t/s, full GPU, no issues.

**Models exceeding GTT (>120 GB)**: 4.1 t/s on DeepSeek-R1-0528 (228 GB). The bottleneck
is CPU expert matmul (AVX-512 on Zen 5). KTransformers achieves 28 t/s with Intel AMX --
7x faster expert matmul. No amount of I/O, copy, or prefetch optimization will close this gap.

**What we've exhausted**:
- I/O optimizations (io_uring, posix_fadvise, registered buffers, hugepages) -- no measurable benefit
- Expert GPU cache + sync skip -- already at ~100% hit rate after 32 tokens
- Flash-moe async prefetch -- slower than cached path for DeepSeek (2.3 vs 4.1 t/s)
- Slot buffer for GPU expert matmul -- shader can't handle remapped IDS, 3 attempts failed
- Graph split reduction -- splits are from CPU<->GPU backend transitions, not optimizable

---

## Completed Investigations

- **I11** - Expert GPU cache + sync skip: DeepSeek 4.1 t/s (2.3x baseline)
- **I12** - ik_llama.cpp benchmark: Vulkan 2x faster for in-GTT models
- **I14** - io_uring polish (SINGLE_ISSUER, MADV_HUGEPAGE): no measurable benefit
- **I10b** - GPU MoE expert matmul: works for in-GTT (auto-detect), blocked for >GTT
- **I17** - Prometheus metrics infrastructure
- **I18** - Cache hit tracking fix

---

## Priority Matrix

| Investigation | Impact | Effort | Status | Recommendation |
|---------------|--------|--------|--------|----------------|
| **Upstream rebase tracking** | High | Low | Ongoing | Track #20757, new llama.cpp releases |
| **CPU kernel improvement** | High | High | Not started | Port ik_llama.cpp fused MoE FFN or wait for AMX |
| **I13** - BF16 CPU Matmul | Medium | Low | Not started | Test if BF16 AVX-512 beats Q4_0 AVX2 |
| **Slot buffer shader mod** | High | Very High | De-prioritized | Wait for upstream #20757 |
| **I7** - Context Scaling | Low | Low | Not started | Not the bottleneck |
| **I8** - Batch Size Tuning | Low | Low | Not started | Minor |

---

## TIER 1: Recommended Next Steps

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

### Slot Buffer Shader Modification

**Why de-prioritized**: Three attempts failed (I10b). The MUL_MAT_ID shader uses expert IDs
for data addressing throughout -- remapping IDS requires deep shader modification that
conflicts with graph shape inference, KV cache setup, and shared IDS tensors across
gate/up/down projections. Upstream #20757 is better positioned to solve this correctly
because it can modify the shader, scheduler, and allocator together.

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
Current state: 4.1 t/s DeepSeek, 20-50 t/s for <=GTT models. All coherent.

The honest question: Is 4.1 t/s on DeepSeek acceptable?

If yes:
    → Monitor upstream, rebase when beneficial improvements land
    → Focus on deploying more <=GTT models (these are production-ready)

If no:
    → Upstream #20757 is the only realistic path to 10+ t/s on this hardware
    → CPU kernel improvements (I13, fused MoE) might reach 8-12 t/s
    → Hardware with Intel AMX would reach 28 t/s (KTransformers benchmark)
```

---

## Comparison with Other Systems

| System | DeepSeek t/s | Approach | vs Our 4.1 t/s |
|--------|-------------|----------|-----------------|
| **KTransformers** | 28 | Intel AMX CPU kernels | 7x faster |
| **llama.cpp #20757 PoC** | 14 | Two-tier GPU cache (Python) | 3.4x faster |
| **ik_llama.cpp** | 1.5 | CPU-only, no flash_attn | 2.7x slower |
| **flash-moe** | 4.4 | Apple SSD + Metal (397B model) | ~comparable |
| **Our moe-flash** | 4.1 | AVX-512 CPU MoE + expert cache | baseline |

---

*Last Updated*: 2026-04-03
