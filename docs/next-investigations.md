# Next Investigations: Roadmap 2026-Q2

**Status**: I10b Option B COMPLETE, I14 COMPLETE, I17/I18 COMPLETE (2026-04-03). io_uring polish optimizations implemented.

**Working Image**: `ghcr.io/cecil-the-coder/llama-cpp-moe-flash:f74f3c3`

---

## ✅ Recently Completed

- **I14** - io_uring polish optimizations (SINGLE_ISSUER, MADV_HUGEPAGE)
- **I10b Option B** - Force-offload GPU import (GGUF → staging → GPU cache)
- **I17** - Prometheus metrics infrastructure (HTTP server, 8 Grafana panels)
- **I18** - Cache hit tracking fix (cross-layer expert sharing now counted)

---

## Priority Matrix

| Investigation | Impact | Effort | Status | Recommendation |
|---------------|--------|--------|--------|----------------|
| **I14** - io_uring Polish | Medium | Low | ✅ **COMPLETE** | See [I14-iouring-polish.md](I14-iouring-polish.md) |
| **I10b Option B** - Force-offload | High | Medium | ✅ **COMPLETE** | Ready to test with DeepSeek |
| **I12** - ik_llama.cpp Benchmark | High | Medium | 🚀 **DEPLOYED** | See [I12-ik-llama-benchmark.md](I12-ik-llama-benchmark.md) |
| **I11** - Dynamic Expert Import | High | High | Not started | Future work |
| **I13** - BF16 CPU Matmul | Medium | Low | Not started | Optional |
| **I7** - Context Scaling | Medium | Low | Not started | Needs TQ2 fix |

---

## ⭐ TIER 1: Recommended Next Steps

### 1. I14: io_uring Polish — 10-25% Performance Gain

**Goal**: Optimize existing io_uring background prefetch with minimal code changes.

**Three Improvements**:
```cpp
// 1. IORING_REGISTER_BUFFERS - skip pin_user_pages per read
io_uring_register_buffers(&ring, iovecs, n_slots);
// Eliminates ~752 page-pin ops per token (94 layers × 8 experts)
// Expected: 5-15% reduction in read jitter

// 2. IORING_SETUP_SINGLE_ISSUER - kernel optimization
io_uring_queue_init_params(entries, &ring, &params);
params.flags |= IORING_SETUP_SINGLE_ISSUER;  // (kernel 6.0+)
// Our prefetch thread is sole submitter - trivially applicable
// Expected: 2-5% from reduced kernel overhead

// 3. MADV_HUGEPAGE on staging pool - 512× TLB reduction
madvise(staging_pool, size, MADV_HUGEPAGE | MADV_COLLAPSE);
// One syscall, massive TLB pressure reduction
// Expected: 3-8% for GTT access path
```

**Effort**: ~50 lines of code, 1-2 days
**Risk**: Low (additive improvements)
**Evidence**: flash-moe saw +38% from trusting OS, these are OS-level optimizations

**Files to modify**:
- `src/llama-moe-flash.cpp` - io_uring ring setup

---

### 2. I12: ik_llama.cpp Benchmark — 🚀 DEPLOYED

**Goal**: Establish CPU-only performance baseline for comparison.

**Status**: Image + K8s deployment created. Awaiting CI build + Flux reconciliation.

**Deployment**:
- **Image**: `ghcr.io/cecil-the-coder/ik-llama-cpu:latest` (built via CI)
- **Backend**: `ik-llama-cpu` (CPU-only, no GPU mounts, AVX-512)
- **Model**: `deepseek-r1-0528-ik` (same GGUF files as existing DeepSeek)
- **Details**: See [I12-ik-llama-benchmark.md](I12-ik-llama-benchmark.md)

**Success Metric**:
- If ik_llama.cpp > 2.5 t/s: We should port their CPU kernels
- If ik_llama.cpp < 1.5 t/s: Our hybrid approach is optimal

---

### 3. I11: Dynamic Expert Import — Enable GPU MoE for >GTT Models

**Goal**: Copy active experts to GPU on-demand for models exceeding GTT.

**The Problem**:
- DeepSeek 228 GB > 120 GB GTT limit
- Current: `--cpu-moe` keeps all expert matmul on CPU (1.8 t/s)
- Target: Copy only active experts (K=8 per layer) to GPU → 6-10 t/s

**The Solution**:
```cpp
// Pre-allocate N Vulkan buffer slots (32 slots × expert_size)
// For each token:
//   1. Read routing IDs (which 8 experts needed)
//   2. For each needed expert:
//      - Check slot table (LRU cache)
//      - If miss: io_uring read → pinned staging → import to slot
//      - On UMA: "import" is page-table remap (zero copy!)
//   3. GPU MUL_MAT_ID on slot buffer
```

**Implementation**:
- Reuse slot buffer code from I10b (already in patch 0014-0015)
- Add `LLAMA_FLASH_MOE_FORCE_OFFLOAD=1` flag
- Modify `ggml_backend_sched` to route MUL_MAT_ID to GPU

**Effort**: 3-5 days
**Risk**: Medium (touches scheduler, needs testing)
**Reward**: **5× speedup for DeepSeek** (1.8 → 10 t/s)

**Prerequisites**:
- I14 (staging pool with registered buffers)
- Slot buffer code (I10b - ✅ already done)

---

### 4. I10b Option B: Force-offload for DeepSeek — Quick Win

**Goal**: Activate existing slot buffer code for q4km/deepseek testing.

**Current State**:
- Slot buffer code present in ce76b8d
- Disabled because `--cpu-moe` routes to CPU
- Can be activated per-model for testing

**Test Plan**:
```yaml
# For qwen3-235b-q4km (133 GB, fits in GTT normally)
# Force CPU mode to test slot buffer:
env:
  - name: LLAMA_ARG_CPU_MOE
    value: "1"
  - name: LLAMA_FLASH_MOE_FORCE_OFFLOAD
    value: "1"  # NEW: activate slot buffer GPU path
```

**Expected**:
- Expert matmul runs on GPU via slot buffer
- TPS: 6-7 t/s → 15-18 t/s (if slot buffer works as designed)

**Effort**: 1 day (add flag, test)
**Risk**: Low (opt-in flag, can disable)

---

## TIER 2: Optional/Future

### 5. I13: BF16 Expert Weights for CPU Matmul

**Hypothesis**: BF16 AVX-512 outperforms Q4_0 AVX2 even at 2× size.

**Background**:
- Zen 5 Strix Halo has half-width AVX-512 FPU
- BF16 uses `_mm512_dpbf16_ps` (optimized)
- Q4_0 uses AVX2 only (no AVX-512 path in ggml)

**Test**:
- Convert one expert file to BF16
- Benchmark CPU matmul vs Q4_0

**Effort**: 2-3 days
**Risk**: May not fit in RAM (2× size)

---

### 6. I7: Context Size Scaling with TQ2_KV

**Goal**: Push context to 32K/64K with quantized KV cache.

**Blocker**: TQ2_KV produces garbage output at 2.125 bpw

**Options**:
- Implement TQ3_KV (3-bit symmetric)
- Use Q4_0 KV cache (tested, works)
- Skip: context not our primary bottleneck

---

## Decision Framework

```
If we want 10-25% improvement with low risk:
    → DO I14 (io_uring polish) - 1-2 days

If we want to know if our architecture is optimal:
    → DO I12 (ik_llama.cpp benchmark) - 1-2 days

If we want 5× speedup for DeepSeek (our largest model):
    → DO I11 + I10b Option B - 1 week

If we want all of the above:
    → Parallel: I14 (1 person) + I12 (1 person) + I11 (2 people)
```

---

## Recommended Sequence

### Week 1: Slot Buffer Activation
1. **I10b Option B** - Force-offload flag for q4km testing
2. Validate slot buffer performance vs current 18 t/s
3. If successful: extend to DeepSeek-R1-0528

### Week 2: Optimization
4. **I14** - io_uring polish (50 lines, 10-25% gain)
5. **I12** - ik_llama.cpp benchmark (establish baseline)

### Week 3-4: DeepSeek GPU MoE (if slot buffer works)
6. **I11** - Dynamic expert import refinements
7. Production testing with DeepSeek-R1-0528

---

## Documentation Updates Needed

- [ ] Update README.md with I10b completion
- [ ] Update architecture diagram
- [ ] Add troubleshooting guide (patch 0006 lesson)
- [ ] Create performance tuning guide

---

## Open Questions

1. **Is 18 t/s for q4km the ceiling, or can slot buffer help even ≤GTT models?**
   - Test I10b Option B to find out

2. **What's the actual cost of expert import on UMA?**
   - Page-table remap should be ~zero, but `vkBindBufferMemory` has overhead
   - Need benchmarking on gfx1151

3. **Can we combine I11 streaming with I14 registered buffers?**
   - Yes, and this is the optimal architecture
   - Streaming: experts → staging pool (io_uring)
   - Import: staging → GPU slots (page remap)
   - Compute: GPU matmul on slots

---

*Last Updated*: 2026-03-31  
*Next Review*: After I14 completion (target: 2026-04-07)
