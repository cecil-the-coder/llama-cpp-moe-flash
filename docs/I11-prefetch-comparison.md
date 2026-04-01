# I11 Prefetching Strategy Comparison

## Visual Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      CURRENT: AGGRESSIVE PREFETCH                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Layer 0 Computing    →    Prefetch Layer 1 (ALL 128 experts)              │
│       ↓                         ↓                                         │
│   [8 experts]              [128 experts × 3.4 MB = 435 MB]                  │
│   used on GPU              prefetched to cache                            │
│       ↓                         ↓                                         │
│  Layer 1 Computing    →    Prefetch Layer 2 (ALL 128 experts)              │
│       ↓                         ↓                                         │
│   [8 experts]              [128 experts × 3.4 MB = 435 MB]                  │
│   used on GPU              prefetched to cache                            │
│                                                                             │
│  TOTAL I/O per token: 435 MB × 64 layers = ~28 GB ❌                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                    SMART: TOP-K PREFETCH (Proposed)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Layer 0 Computing    →    Prefetch Layer 1 (SAME 8 experts)               │
│       ↓                         ↓                                         │
│   [8 experts]              [8 experts × 3.4 MB = 27 MB]                     │
│   used on GPU              prefetched to cache                              │
│       ↓                         ↓                                         │
│  Layer 1 Computing    →    Prefetch Layer 2 (SAME 8 experts)               │
│       ↓                         ↓                                         │
│   [8 experts]              [8 experts × 3.4 MB = 27 MB]                     │
│   used on GPU              prefetched to cache                              │
│                                                                             │
│  TOTAL I/O per token: 27 MB × 64 layers = ~1.7 GB ✅ 16x reduction!       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Insight

The MoE gate selects **8 experts out of 128** (top-k=8). Currently we prefetch ALL 128, but only 8 are used!

**Smart prefetching** = Only prefetch the 8 that will actually be used.

---

## How It Works

### Current Flow
```
1. GPU computes Layer 0 with experts [5, 23, 45, 67, 89, 12, 34, 56]
2. Callback triggers: "Hey, I just used some experts"
3. System: "I don't care which ones, I'll prefetch ALL 128 for Layer 1"
4. Result: 435 MB of I/O, but only 27 MB will be accessed
```

### Smart Flow (Proposed)
```
1. GPU computes Layer 0 with experts [5, 23, 45, 67, 89, 12, 34, 56]
2. Callback triggers: "Hey, I just used experts [5, 23, 45, 67, 89, 12, 34, 56]"
3. System: "Got it! I'll prefetch experts [5, 23, 45, 67, 89, 12, 34, 56] for Layer 1"
4. Result: 27 MB of I/O, all of it will be accessed ✅
```

---

## Why This Works

### MoE Routing Patterns

Most MoE models exhibit **routing locality**:
- Same experts tend to be used for similar inputs
- Certain experts specialize in certain topics/tokens
- Top-k selection is relatively stable within a context

### Example: Qwen3-235B

```
Expert Usage Pattern (typical):
┌────────────────────────────────────────────────────┐
│ Expert 0  ████ (4%)                                 │
│ Expert 1  ████████ (8%)                             │
│ Expert 2  ███ (3%)                                  │
│ ...                                                 │
│ Expert 23 ████████████████ (23%) ← Hot expert      │
│ ...                                                 │
│ Expert 45 ██████████████ (18%) ← Hot expert        │
│ ...                                                 │
│ Expert 127 ██ (2%)                                  │
└────────────────────────────────────────────────────┘

8 experts handle majority of traffic!
```

---

## Implementation: The `used_ids_bitset`

The callback already receives this information!

```cpp
static void expert_copy_callback(
    const struct ggml_tensor * node,
    const uint32_t * used_ids_bitset,  // ← This tells us which experts!
    int n_expert,
    void * user_data) {
    
    // Parse the bitset to get expert IDs
    std::vector<int> used_expert_ids;
    for (int i = 0; i < n_expert; i++) {
        if (ggml_bitset_get(used_ids_bitset, i)) {
            used_expert_ids.push_back(i);
        }
    }
    // used_expert_ids now contains [5, 23, 45, 67, 89, 12, 34, 56]
}
```

---

## Expected Impact

| Metric | Current | Smart | Improvement |
|--------|---------|-------|-------------|
| I/O per layer | 435 MB | 27 MB | **16x less** |
| I/O per token | ~28 GB | ~1.7 GB | **16x less** |
| Cache pollution | High | Low | Better |
| Disk contention | High | Low | Better |
| First request | ~6s | ~4s | **33% faster** |
| Subsequent | ~3.8s | ~3.5s | **8% faster** |

---

## io_uring Enhancement

Beyond smart selection, we can improve HOW we prefetch:

### Current: posix_fadvise (Just a Hint)
```cpp
posix_fadvise(fd, offset, len, POSIX_FADV_WILLNEED);
// Kernel: "OK, I'll read that... eventually... maybe"
```

### io_uring: Actual Async Read
```cpp
// Submit async read
struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
io_uring_prep_read(sqe, fd, buffer, size, offset);
io_uring_submit(&ring);

// Continue computing while read happens in background
// Later: wait for completion
io_uring_wait_cqe(&ring, &cqe);
```

**Benefits**:
- ✅ Actually reads data (not just hints)
- ✅ True async (no blocking)
- ✅ Better overlap of I/O and compute
- ✅ Can batch multiple reads

---

## Combined: Smart + io_uring

```
1. GPU computes Layer 0
2. Parse used_ids_bitset → get 8 expert IDs
3. Submit 8 io_uring reads for Layer 1 experts
4. Continue to Layer 1 (reads happen in parallel!)
5. Before using expert, ensure read completed
6. Result: Maximum overlap, minimum I/O
```

---

## Risk: What If We Miss?

**Q**: What if we prefetch expert 5 for Layer 1, but the gate selects expert 7?

**A**: No problem! The system falls back to on-demand loading:

```cpp
// Smart prefetch (best effort)
for (int expert_id : predicted_experts) {
    posix_fadvise(fd, offset, expert.total_size, POSIX_FADV_WILLNEED);
}

// If prediction misses, kernel loads on-demand (slower but correct)
// No correctness issue, just performance
```

**Expected hit rate**: 60-80% (based on routing stability)

---

## Next Steps

1. ✅ **Document the plan** (this document)
2. 🔄 **Implement Phase 1**: Top-k selection (2-3 hours)
3. ⏳ **Test**: Measure I/O reduction and latency improvement
4. ⏳ **Implement Phase 2**: io_uring integration (1 day)
5. ⏳ **Consider Phase 3-4**: LRU cache and pattern learning

---

**Status**: Ready to implement  
**Priority**: HIGH  
**Impact**: 16x I/O reduction + better performance
