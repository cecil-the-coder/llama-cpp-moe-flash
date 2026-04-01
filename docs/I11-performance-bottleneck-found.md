# I11 Performance Bottleneck Analysis

**Date**: 2025-04-01  
**Image**: d23cb29 (I11 debug v4)  
**Status**: ROOT CAUSE IDENTIFIED!

## The Smoking Gun 🎯

### Qwen3 (5 tokens generated)
```
Total data copied to GPU: 22.8 GB
Number of MUL_MAT_ID ops: 831
Cache hit rate: ~5% (mostly complete misses)
```

### DeepSeek (5 tokens generated)
```
Total data copied to GPU: 40.3 GB
Number of MUL_MAT_ID ops: 325
Cache hit rate: ~2% (mostly complete misses)
```

## The Problem

**The expert cache is almost completely ineffective!**

```
[I11-DEBUG-NORM] n_expert=128, used=8, hits=0, misses=8, bytes_per_expert=3538944, total_copy=27648 KB
```

- 8 experts used per token
- 0 cache hits
- 8 cache misses
- 27 MB copied per tensor

With 3 tensors per layer (gate, up, down) × 60 layers = 180 MUL_MAT_ID ops per token

## Root Cause Analysis

### Why Cache Misses?

1. **Per-Token Expert Selection**: Each token uses 8 different experts selected by the gating network
2. **Low Expert Reuse**: Across tokens, the same experts are rarely reused consecutively
3. **Cache Invalidation**: The cache is keyed by `input_cpy` tensor pointer, which changes

### The Math

**Qwen3 per layer per token:**
- 3 tensors (gate, up, down)
- 8 experts each
- ~3.5-5 MB per expert
- Total: 3 × 8 × 4.5 MB = **108 MB per layer per token**

**60 layers:**
- 60 × 108 MB = **~6.5 GB per token**
- For 5 tokens: 5 × 6.5 GB = **32.5 GB** (observed: 22.8 GB with partial caching)

## Why the Performance Regression?

### Before I10b/I11
- Experts were accessed directly from CPU_Mapped memory
- No explicit copying to GPU
- GPU accessed experts via GTT (GPU Translation Table)

### After I10b/I11
- Experts are now **explicitly copied** to GPU for every MUL_MAT_ID
- 22-40 GB of data transfer per request
- The expert copy overhead dominates execution time

## The Cache Implementation Issue

Looking at the code:
```cpp
static std::unordered_map<const ggml_tensor *, std::vector<ggml_bitset_t>> expert_cache;
auto & cached_ids = expert_cache[input_cpy];
```

The cache is keyed by `input_cpy` tensor pointer. Each graph split has different `input_cpy` tensors, so the cache is effectively per-split, not global.

## Evidence

```bash
# Cache hit distribution:
288 used=8, hits=0, misses=8    # Complete miss (96% of ops)
 63 used=2, hits=1, misses=1   # 50% hit rate (rare)
 18 used=9, hits=2, misses=7   # 22% hit rate (very rare)
```

## Conclusion

The I10b/I11 implementation adds **massive data transfer overhead** (~22-40 GB per request) because:

1. Every expert used by every token is copied to GPU
2. The cache is ineffective (hit rate < 5%)
3. The copy happens synchronously before each MUL_MAT_ID

**This is the source of the ~40% performance regression!**

## Possible Solutions

### Option 1: Disable I10b Expert Copy
Revert to CPU_Mapped access without explicit GPU copy. Let GTT handle the memory access.

### Option 2: Fix the Cache
- Use a global cache keyed by expert ID, not tensor pointer
- Pre-load hot experts to GPU
- Keep experts resident on GPU across tokens

### Option 3: Async Expert Loading
- Prefetch next token's experts while computing current token
- Overlap expert copy with computation

### Option 4: Smarter Expert Selection
- Keep top-K most frequently used experts on GPU
- Only copy experts not already resident

## Recommendation

**Immediate**: Revert I10b expert copy, keep only I11 slot buffer (which is dormant anyway)

**Long-term**: Implement proper expert caching that:
1. Keeps frequently-used experts resident on GPU
2. Uses a global LRU cache, not per-split
3. Overlaps expert loading with computation

---

**Status**: Root cause identified - expert copy overhead  
**Impact**: ~40% performance regression  
**Next action**: Decide on fix strategy
