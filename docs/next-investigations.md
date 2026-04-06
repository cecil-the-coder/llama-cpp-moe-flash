# Next Investigations: Roadmap 2026-Q2

**Status**: I11 COMPLETE (expert GPU cache + sync skip), I12 COMPLETE. DeepSeek 4.1 t/s (2.3x baseline). (2026-04-03)

**Production Image**: `ghcr.io/cecil-the-coder/llama-cpp-moe-flash:74a5930` (b8664)

---

## Completed

- **I11** - Expert GPU cache + sync skip: DeepSeek 4.1 t/s (2.3x baseline), stable cache key, sync skip, flash-moe disabled
- **I12** - ik_llama.cpp benchmark: Vulkan 2x faster for in-GTT models
- **I14** - io_uring polish optimizations (SINGLE_ISSUER, MADV_HUGEPAGE)
- **I10b** - GPU MoE expert matmul for in-GTT models (auto-detect)
- **I17** - Prometheus metrics infrastructure
- **I18** - Cache hit tracking fix

---

## Priority Matrix

| Investigation | Impact | Effort | Status | Recommendation |
|---------------|--------|--------|--------|----------------|
| **I11** - Expert GPU Cache + Sync Skip | High | High | **COMPLETE** | 4.1 t/s (2.3x baseline) |
| **I12** - ik_llama.cpp Benchmark | High | Medium | **COMPLETE** | See [I12-ik-llama-benchmark.md](I12-ik-llama-benchmark.md) |
| **I14** - io_uring Polish | Medium | Low | **COMPLETE** | See [I14-iouring-polish.md](I14-iouring-polish.md) |
| **Slot buffer GPU matmul** | High | High | Future | Shader mod needed for >GTT GPU path |
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

### 2. I12: ik_llama.cpp Benchmark — COMPLETE

**Result**: Vulkan hybrid is 2x faster for in-GTT models. DeepSeek: moe-flash 2.05 t/s vs ik_llama 1.5 t/s.

| Model | ik_llama.cpp | Stock Vulkan | moe-flash |
|-------|-------------|-------------|-----------|
| Qwen3-235B Q2_K (80 GB) | 11.5 t/s | 20.7 t/s | 20.0 t/s |
| DeepSeek-R1 Q2_K (228 GB) | 1.5 t/s (no flash_attn) | N/A | 4.1 t/s (expert GPU cache + sync skip) |

**Key finding**: ik_llama.cpp FlashMLA crashes on DeepSeek Q2_K over mmap (NaN logits).
Standard attention path works but isn't faster than our hybrid.

**Details**: See [I12-ik-llama-benchmark.md](I12-ik-llama-benchmark.md)

---

### 3. Remaining Optimizations for DeepSeek (4.1 t/s -> higher)

**Graph split reduction**: Current 284 graph splits could be reduced to ~61 by batching
gate/up/down per layer. Potential 2-3x improvement from reduced dispatch overhead.

**Cold miss prefetch**: Combine flash-moe async prefetch with expert cache for cold misses.
Currently mutually exclusive -- flash-moe bypasses the scheduler's expert copy path entirely.

**Thread count tuning**: DeepSeek CPU_MOE path may benefit from different thread count
than the current default. Zen 5 UMA shares memory bandwidth between CPU and GPU.

**Slot buffer GPU matmul** (future): Shader mod for >GTT GPU expert matmul.
Patches 0015/0016/0019 preserved in repo. Potential further speedup but higher risk.

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
Current state: Expert GPU cache + sync skip delivers 4.1 t/s (2.3x baseline). All models coherent.

If we want further DeepSeek speedup (4.1 → ~8-12 t/s):
    → Graph split reduction (284 → ~61) - highest leverage, 2-3x potential
    → Thread count tuning - low effort
    → Cold miss prefetch (combine flash-moe + cache) - medium effort

If we want incremental improvements:
    → I13 (BF16 CPU matmul) or I7 (context scaling) - low effort

If we want to explore new models:
    → Deploy larger models and test with current stack
```

---

## Recommended Next Steps

1. **Graph split reduction** (highest impact): Batch gate/up/down per layer to reduce
   284 splits to ~61. Target: DeepSeek 4.1 -> ~8-12 t/s.

2. **Thread count tuning** (low effort): Sweep thread counts for DeepSeek CPU_MOE path.

3. **Cold miss prefetch** (medium effort): Combine flash-moe async prefetch with expert
   cache for cold misses during warmup phase.

4. **I13 - BF16 CPU matmul** (low effort): Test if BF16 AVX-512 outperforms
   Q4_0 AVX2 for CPU expert matmul on Zen 5.

---

## Lessons Learned

- Flux YAML validation is critical: a duplicate `value:` key silently blocks reconciliation
- gallocr flag changes (`ggml_set_input/output`) have global side effects -- test all models
- Force-offload guards must check buffer type, not just backend assignment

---

*Last Updated*: 2026-04-03
