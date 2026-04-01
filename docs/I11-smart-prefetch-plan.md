# I11 Smart Prefetching & io_uring Enhancement Plan

**Date**: 2026-04-01  
**Status**: Planning Phase  
**Goal**: Improve async prefetch efficiency by being smarter about which experts to prefetch

---

## Current Implementation Analysis

### What We Have Now (Working)
```
[I11-ASYNC] Layer 0: 8 experts used, prefetching layer 1 (128 experts)
[I11-ASYNC] Prefetched 128/128 experts for layer 1
```

**Current Approach**:
- Uses `posix_fadvise(fd, offset, len, POSIX_FADV_WILLNEED)` 
- Prefetches ALL 128 experts in layer N+1
- Just "hints" the kernel - actual read happens on first access
- Works but potentially overfetches

### io_uring Infrastructure (Already Present)
The codebase already has `iouring_prefetcher` with:
- io_uring ring setup (SQPOLL fallback to normal mode)
- Staging slots with aligned memory (2MB aligned for O_DIRECT)
- Buffer registration for zero-copy reads
- `iouring_prefetcher_submit()` for manifest-based experts
- `iouring_prefetcher_submit_gguf()` for GGUF-based experts
- `iouring_prefetcher_wait()` for completion

**Problem**: Not currently used in the async_prefetch callback!

---

## Smart Prefetching Strategies

### Strategy 1: Top-k Expert Selection (Immediate Win)
**Idea**: Instead of prefetching ALL 128 experts, only prefetch the ones that were actually selected by the gate.

**How**:
- The callback already receives `used_ids_bitset` - a bitset of which experts were used
- Parse this bitset to get the actual expert IDs
- Only prefetch those experts in the next layer

**Expected Benefit**:
- Current: 128 experts × ~3.4 MB = ~435 MB per layer prefetched
- Smart: 8 experts × ~3.4 MB = ~27 MB per layer prefetched
- **16x reduction in I/O!**

### Strategy 2: io_uring Instead of posix_fadvise
**Idea**: Actually READ the data asynchronously instead of just hinting.

**How**:
- Use `iouring_prefetcher_submit_gguf()` to submit async reads
- Read into staging buffers while GPU computes current layer
- Copy to GPU when expert is needed (zero-copy if possible)

**Expected Benefit**:
- True async I/O without blocking
- Can overlap I/O with computation better
- Better for cold cache scenarios

### Strategy 3: LRU Cache for GPU-Resident Experts
**Idea**: Keep recently used experts in GPU memory rather than reloading.

**How**:
- Maintain LRU cache of expert tensors in GPU memory
- On expert copy: check cache first, only load if miss
- Evict least recently used when cache is full

**Expected Benefit**:
- Avoid redundant I/O for frequently used experts
- Typical MoE patterns: 20-40% of experts handle 80% of traffic

### Strategy 4: Routing Pattern Learning
**Idea**: Learn which experts are typically activated together.

**How**:
- Track co-occurrence matrix: which experts appear together
- If expert A is used, prefetch expert B (which often co-occurs)
- Could use simple heuristics or ML model

**Expected Benefit**:
- Predictive prefetching beyond immediate next layer
- Proactive loading reduces latency spikes

---

## Implementation Phases

### Phase 1: Top-k Selection (Quick Win)
**Time**: ~2-3 hours  
**Impact**: 16x I/O reduction

Changes needed:
1. Modify `expert_copy_callback()` to parse `used_ids_bitset`
2. Extract the actual expert IDs that were used
3. Only prefetch those experts in next layer
4. Add debug logging to verify

```cpp
// Parse used_ids_bitset to get actual expert IDs
std::vector<int> used_experts;
for (int i = 0; i < n_expert; i++) {
    if (ggml_bitset_get(used_ids_bitset, i)) {
        used_experts.push_back(i);
    }
}

// Only prefetch these experts in next layer
for (int expert_id : used_experts) {
    if (expert_id < layer.experts.size()) {
        // Prefetch this specific expert
    }
}
```

### Phase 2: io_uring Integration
**Time**: ~1 day  
**Impact**: True async I/O, better overlap

Changes needed:
1. Connect `iouring_prefetcher` to callback
2. Submit async reads instead of posix_fadvise
3. Don't wait immediately - let reads happen in background
4. Ensure reads complete before expert is needed

Key considerations:
- Need to handle the io_uring completion before next layer uses the data
- May need synchronization point between layers
- Staging buffers need to be managed

### Phase 3: LRU Cache
**Time**: ~2-3 days  
**Impact**: Eliminate redundant I/O

Changes needed:
1. Add GPU memory pool for cached experts
2. Hash map: (layer, expert_id) → GPU buffer
3. LRU eviction when pool is full
4. Cache hit tracking and stats

### Phase 4: Pattern Learning
**Time**: ~1 week  
**Impact**: Predictive prefetching

Changes needed:
1. Co-occurrence matrix tracking
2. Heuristic/ML model for prediction
3. Speculative prefetching beyond next layer
4. A/B testing framework

---

## Expected Performance Impact

### Current (Aggressive Prefetch)
- First request: ~6s (cache warming)
- I/O: 435 MB per layer × 64 layers = ~28 GB per token!
- Subsequent: ~3.8s

### With Top-k Selection
- First request: ~4s (less cache warming needed)
- I/O: 27 MB per layer × 64 layers = ~1.7 GB per token
- **16x less I/O pressure**
- Subsequent: ~3.5s (slightly faster due to less contention)

### With io_uring + Top-k
- Better overlap of I/O and compute
- Could see 10-20% improvement in latency

### With LRU Cache
- Cache hit rate: 20-40% expected
- Further 20-40% reduction in I/O
- Latency improvement: 10-30%

---

## Testing Plan

1. **Baseline**: Current aggressive prefetch (128 experts)
2. **Top-k**: Smart prefetch (8 experts)
3. **Compare**: 
   - Total runtime for 5 requests
   - Page cache hit rates
   - Disk I/O volume (via iostat)
   - Latency distribution

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Top-k selection misses experts | Fallback: if expert not prefetched, load on-demand |
| io_uring completion timing | Add sync point before expert use; measure overhead |
| LRU cache too small | Make configurable; default to 10-20% of experts |
| Complexity increases bugs | Incremental rollout; feature flags per strategy |

---

## Next Steps

1. **Immediate**: Implement Phase 1 (top-k selection) - quick 16x win
2. **Short term**: Test and measure Phase 1 impact
3. **Medium term**: Phase 2 (io_uring) if Phase 1 shows promise
4. **Long term**: Phase 3-4 based on results

---

## Code Structure

```cpp
// Current callback (aggressive)
static void expert_copy_callback(...) {
    // Prefetch ALL experts in next layer
    for (const auto& expert : layer.experts) {
        posix_fadvise(fd, offset, len, POSIX_FADV_WILLNEED);
    }
}

// Smart callback (top-k selection)
static void expert_copy_callback(...) {
    // Parse which experts were actually used
    std::vector<int> used_experts = parse_used_experts(used_ids_bitset, n_expert);
    
    // Only prefetch those in next layer
    for (int expert_id : used_experts) {
        if (expert_id < layer.experts.size()) {
            const auto& expert = layer.experts[expert_id];
            posix_fadvise(fd, offset, expert.total_size, POSIX_FADV_WILLNEED);
        }
    }
}
```

---

**Status**: Ready to implement Phase 1  
**Priority**: HIGH - 16x I/O reduction is significant  
**Estimated Effort**: 2-3 hours for Phase 1
