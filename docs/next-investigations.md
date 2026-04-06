# Next Investigations: Roadmap 2026-Q2

**Status**: Pool investigation concluded. Pool caching cannot work with sequential layer execution — needs P=94 (~141 GB) to cache all layers. Slot remapping (N_SLOTS=32) is the only viable path to GPU MoE for >GTT models. Shader analysis confirms feasibility. (2026-04-03)

**Production Image**: latest main on b8664

---

## Current State: Honest Assessment

**Models that fit in GTT (<=120 GB)**: Production-ready. 20-50 t/s, full GPU, no issues.

**Models exceeding GTT (>120 GB)**:
- Qwen3-235B Q4_K_M (133 GB): 1.4 t/s force-offload, ~6-7 t/s CPU MoE
- DeepSeek-R1-0528 Q2_K (228 GB): ~4 t/s CPU MoE (can't test force-offload -- 128 GB RAM too small)

**Persistent buffer pool finding**: 0% cache hit rate. 282 unique weight tensors
(94 layers x 3 projections) with only 9 pool entries. Per-projection keying means
gate/up/down of the SAME layer all have different keys and evict each other.

**What we've exhausted**:
- I/O optimizations (io_uring, posix_fadvise, registered buffers, hugepages) -- no measurable benefit
- Expert GPU cache + sync skip -- already at ~100% hit rate after 32 tokens
- Flash-moe async prefetch -- slower than cached path for DeepSeek (2.3 vs 4.1 t/s)
- Slot buffer for GPU expert matmul -- previous attempts failed from gallocr corruption, NOT shader issues. Shader analysis confirms slot remapping is feasible with ne[2]=32 on pool tensors.
- Graph split reduction -- splits are from CPU<->GPU backend transitions, not optimizable
- Per-projection buffer pool -- 0% hit rate, 282 tensors overwhelm 9 entries
- Per-layer buffer pool grouping -- still 0% hit rate, 94 layers > 15 pool entries, LRU doesn't help with sequential execution

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
| **Slot remapping (N_SLOTS=32)** | Very High | High | Ready to implement | Shader analysis confirms feasibility; ~35 GB for P=94 |
| **Per-layer pool grouping** | None | Medium | Complete | 0% hit rate: 94 layers > 15 entries, sequential execution defeats LRU |
| **Upstream rebase tracking** | High | Low | Ongoing | Track #20757, new llama.cpp releases |
| **CPU kernel improvement** | High | High | Not started | Port ik_llama.cpp fused MoE FFN or wait for AMX |
| **I13** - BF16 CPU Matmul | Medium | Low | Not started | Test if BF16 AVX-512 beats Q4_0 AVX2 |
| **I7** - Context Scaling | Low | Low | Not started | Not the bottleneck |
| **I8** - Batch Size Tuning | Low | Low | Not started | Minor |

---

## TIER 1: Recommended Next Steps

### 0. Slot Remapping with N_SLOTS=32 (READY TO IMPLEMENT)

**Goal**: GPU MoE for >GTT models by remapping expert IDs to a compact 32-slot buffer.

**Why pool caching failed**: 94 layers x 3 projections = 282 entries needing ~141 GB. Sequential execution (layer 0,1,...,93) defeats LRU. Per-layer grouping (15 entries) still 0% hit rate.

**Slot remapping approach**:
1. Allocate persistent pool buffer: 32 slots x expert_size per projection (~376 MB/layer)
2. Copy 8 used experts to LRU slots 0..31 in persistent buffer
3. Override `ne[2]=32` on pool tensor (outside gallocr, no graph corruption)
4. Rewrite IDS tensor: expert_id -> slot_idx
5. Dispatch MUL_MAT_ID with n_as=32

**Shader analysis confirms correctness**:
- **Batch path** (`mul_mm.comp`): `expert_idx = gl_WorkGroupID.z` (0..n_as-1), `pos_a = expert_idx * batch_stride_a`. With n_as=32, only 32 workgroups dispatched. CORRECT.
- **Vec path** (`mul_mat_vec_base.glsl`): `expert_id = data_ids[...]`, `a_offset = expert_id * batch_stride_a`. With slot-remapped IDS (values 0..31), accesses slots. CORRECT.
- **Count experts** (`count_experts.comp`): `expert_id = gl_WorkGroupID.x` (0..n_as-1), counts `data_a[...] == expert_id`. With n_as=32, counts slots. CORRECT.
- **Expert count buffer**: `sizeof(uint32_t) * n_as = 128 bytes`. CORRECT.
- **Output shape**: MUL_MAT_ID output shape from ne01/IDS, NOT ne[2]. No corruption.
- **supports_op**: No ne[2] checks that would reject n_as=32. CONFIRMED.

**Previous failures were NOT shader issues**: The I11 SIGSEGV came from gallocr corruption when `ggml_set_input/output` was applied to graph tensors. The ne[2] override now happens on persistent pool tensors OUTSIDE gallocr.

**Memory budget (P=94, all layers cached)**:
- Per layer: 32 x (3.4 MB gate + 3.4 MB up + 5 MB down) = ~376 MB
- Total: 94 x 376 MB = ~35 GB GPU buffers (UMA = RAM)
- System: 35 GB pool + 10 GB attention + 2 GB KV + 1 GB compute = ~48 GB
- Remaining: ~80 GB for mmap page cache. FEASIBLE on 128 GB.

**Effort**: High (pool tensor management, IDS rewrite, ne[2] override plumbing)
**Impact**: Very High (expected 10-15 t/s, 3x current CPU MoE)

### 0b. Per-Layer Buffer Pool Grouping (COMPLETE — no improvement)

**Result**: 0% hit rate. 94 layers > 15 pool entries, sequential execution defeats LRU.

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

### Slot Buffer Shader Modification (SUPERSEDED by Slot Remapping above)

**Previous assessment was wrong**: The shader does NOT need modification. Detailed analysis
of `mul_mm.comp`, `mul_mat_vec_base.glsl`, and `count_experts.comp` confirms that with
`ne[2]=32` and slot-remapped IDS values (0..31), the existing shaders produce correct results.
The previous failures came from gallocr corruption (ggml_set_input/output on graph tensors),
not from shader incompatibility. Moved to TIER 1 as "Slot Remapping".

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
    → Slot remapping (N_SLOTS=32): shader-verified, ~35 GB feasible, expected 10-15 t/s
    → Upstream #20757 as alternative if slot remapping hits unforeseen issues
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
