# I11 Implementation Plan

## Overview

Port slot buffer code from `I14-I10b-option-b.patch` to working baseline and activate with `FORCE_OFFLOAD=1`.

## Current State

- ✅ Working baseline: `a15d1b3` (I10b Option B only)
- ✅ Slot buffer code: Exists in `I14-I10b-option-b.patch` 
- ❌ Slot buffer not in working patch yet
- ❌ Not activated by FORCE_OFFLOAD flag

## Implementation Steps

### Phase 1: Extract Slot Buffer Code (1 day)

**From**: `patches/0001-moe-flash-I14-I10b-option-b.patch`

**Files to modify**:
1. `ggml/src/ggml-backend.cpp` - Add slot cache structures and logic
2. `ggml/include/ggml-backend.h` - Add callback declarations

**Code to port**:
```cpp
// In ggml_backend_sched_compute_splits:
// 1. moe_slot_cache structure (lines 113-120)
// 2. expert_cache and slot_caches static variables (lines 109, 120)
// 3. original_ids_cache for shared IDS (lines 123-124)
// 4. Slot remap mode activation (lines 135-141)
// 5. LRU cache management (lines 198-233)
// 6. IDS rewrite logic (lines 252-260)
// 7. Expert copy to slot (lines 216-223)
```

### Phase 2: Integrate with FORCE_OFFLOAD (1 day)

**Current behavior**:
```cpp
if (force_offload) {
    LLAMA_LOG_INFO("...skipping tensor override removal\n", __func__);
    return;  // Just skips auto-detect
}
```

**New behavior**:
```cpp
if (force_offload) {
    LLAMA_LOG_INFO("...activating slot buffer GPU path\n", __func__);
    // Don't return - let auto-detect run but keep overrides
    // Add flag to enable slot buffer mode in backend
    mparams->tensor_buft_overrides = overrides;  // Keep CPU_MOE
    // Set global flag for slot buffer mode
    ggml_set_moe_slot_buffer_mode(true);
}
```

### Phase 3: Vulkan Buffer Allocation (1 day)

**Problem**: Need to pre-allocate GPU slot buffers

**Solution**: In `ggml-vulkan.cpp`, when creating MUL_MAT_ID tensors:
```cpp
// Check if expert tensor > 4 GiB
if (expert_tensor_size > 4*GB) {
    // Allocate slot buffer instead of full tensor
    size_t slot_buffer_size = 32 * expert_size;
    vk_buffer = create_buffer(slot_buffer_size, GPU_MEMORY);
    
    // Store original mmap'd CPU buffer for copy source
    cpu_buffer = original_mmap_buffer;
}
```

### Phase 4: Testing Strategy (1-2 days)

**Test 1**: qwen3-235b-q4km (133 GB) - Model > GTT but not huge
- Expected: 6-7 t/s → 15-18 t/s
- Cache hit rate: >90%

**Test 2**: DeepSeek-R1-0528 (228 GB) - The big one
- Expected: 1.8 t/s → 6-10 t/s
- Memory pressure test

**Test 3**: glm-4-7-flash (17 GB) - Fits in GTT
- Expected: No change (50 t/s)
- Verify no regression

### Phase 5: Performance Validation

**Metrics**:
- TPS improvement >2×
- Cache hit rate >90%
- Copy latency <1ms per expert
- No OOM or corruption

## Technical Risks

### Risk 1: Memory Pressure

32 slots × 94 layers × 5 MB = 15 GB GPU memory
Plus KV cache, activations, etc. = tight fit

**Mitigation**: Dynamic slot count based on free memory

### Risk 2: Graph Split Complexity

MUL_MAT_ID with slot buffer requires:
- CPU routing computation
- GPU slot buffer access
- Potential graph split overhead

**Mitigation**: Profile with `GGML_PERF` flag

### Risk 3: Cache Thrashing

If working set > 32 slots, LRU evictions hurt performance.

**Mitigation**: Monitor hit rate, adjust slot count

## Implementation Order

1. ✅ Port slot buffer structures (ggml-backend.cpp)
2. ✅ Add activation flag
3. ✅ Test with q4km
4. ✅ Measure cache hit rate
5. ⏭️ Optimize if needed
6. ⏭️ Test with DeepSeek

## Code Changes Summary

| File | Lines | Purpose |
|------|-------|---------|
| `ggml-backend.cpp` | +150 | Slot cache, LRU, IDS rewrite |
| `ggml-backend.h` | +20 | Callback declarations |
| `llama.cpp` | +10 | FORCE_OFFLOAD activation |
| `ggml-vulkan.cpp` | +30 | Slot buffer allocation |

**Total**: ~210 lines

## Success Criteria

- [ ] Slot buffer activates when FORCE_OFFLOAD=1
- [ ] GPU MUL_MAT_ID runs successfully
- [ ] TPS improves >2× for >GTT models
- [ ] Cache hit rate >90%
- [ ] No corruption or OOM
- [ ] Coherent output

## Timeline

- **Week 1**: Port slot buffer code, basic testing
- **Week 2**: DeepSeek testing, optimization
- **Week 3**: Documentation, production deployment

---

**Estimated effort**: 1-2 weeks  
**Risk level**: Medium  
**Expected reward**: 3-5× speedup for DeepSeek
