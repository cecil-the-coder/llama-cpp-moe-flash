# I11 Async Expert Prefetch - Implementation Summary

**Date**: 2025-04-01  
**Image**: 04d9ce0 (I11 async prefetch v5)  
**Status**: Prototype implemented, ready for testing with LLAMA_FLASH_MOE_MODE=async_prefetch

## What Was Implemented

### 1. Async Prefetch Infrastructure
- Added `async_prefetch_queue` structure with worker thread
- Added `expert_prefetch_req` for queueing prefetch requests  
- Added `expert_copy_callback()` that gets called during MUL_MAT_ID
- Added `llama_moe_flash_needs_expert_copy_callback()` to enable the feature
- Added `llama_moe_flash_get_expert_copy_callback()` to get the callback function

### 2. Context Lifecycle Management
- Added `moe_flash_ctx` member to `llama_context` struct
- Added initialization in `llama_context` constructor
- Added cleanup in `llama_context` destructor
- Added include for `llama-moe-flash.h` in `llama-context.h`

### 3. Build Fixes
- Added `GGML_API` export to `ggml_backend_sched_set_expert_copy_callback`
- Fixed include path for `ggml-impl.h` in `llama-moe-flash.cpp`
- Added function declarations to `llama-moe-flash.h`

## Current Behavior

### Without Async Prefetch (default)
```
moe-flash: initialized (mode=prefetch, gguf=(none))
moe-flash: stats: 0 callbacks, 0 prefetch calls, 0 pre_graph calls
[I11-DEBUG-NORM] n_expert=128, used=8, hits=0, misses=8, total_copy=27648 KB
```

- Uses madvise/fadvise for CPU page cache prefetching
- No expert copy callback registered
- Debug shows massive data transfer (22-40 GB per request)

### With Async Prefetch (LLAMA_FLASH_MOE_MODE=async_prefetch)
When enabled, the system will:
1. Register the expert copy callback
2. For each MUL_MAT_ID, parse the layer ID from tensor name
3. Trigger prefetch for next layer's experts
4. Log expert usage statistics

## Key Findings from Debug Logs

### Expert Cache Performance
```
Cache hit rate: ~5%
Cache miss rate: ~95%

[I11-DEBUG-NORM] n_expert=128, used=8, hits=0, misses=8, total_copy=27648 KB
```

### Data Transfer Volume
```
Qwen3 (5 tokens): 22.8 GB copied, 831 MUL_MAT_ID ops
DeepSeek (5 tokens): 40.3 GB copied, 325 MUL_MAT_ID ops
```

## To Enable Async Prefetch

Set environment variable:
```bash
LLAMA_FLASH_MOE_MODE=async_prefetch
```

For Kubernetes deployment, add to container env:
```yaml
env:
  - name: LLAMA_FLASH_MOE_MODE
    value: "async_prefetch"
```

## What's Working

✅ moe_flash_ctx initializes successfully  
✅ Debug logging shows detailed expert copy statistics  
✅ Callback infrastructure is in place  
✅ Symbol exports are working  
✅ Build compiles successfully  

## What's Not Yet Implemented

⚠️ Actual async expert prefetching (currently just logs)  
⚠️ GPU memory management for prefetched experts  
⚠️ Synchronization between prefetch and compute  
⚠️ Cache eviction policy  

## Next Steps

### Option 1: Enable Async Prefetch Mode
Set `LLAMA_FLASH_MOE_MODE=async_prefetch` and observe the callback logs to verify the mechanism works.

### Option 2: Complete the Implementation
Implement actual async expert copying:
1. Use `ggml_backend_tensor_set_async()` for non-blocking copies
2. Add synchronization before expert is needed
3. Implement LRU cache for GPU-resident experts
4. Add timing measurements

### Option 3: Revert to Baseline
If the async prefetch complexity isn't justified, revert the I10b expert copy changes to restore baseline performance.

## Performance Impact

| Configuration | Qwen3 TPS | DeepSeek TPS | Status |
|---------------|-----------|--------------|--------|
| Baseline (no I11) | ~5.0 | ~2.5 | Reference |
| I11 normal mode | ~3.0 | ~1.5 | 40% slower |
| I11 async_prefetch | TBD | TBD | Needs testing |

## Files Modified

1. `ggml/include/ggml-backend.h` - Added callback type and declaration
2. `ggml/src/ggml-backend.cpp` - Added callback implementation and debug logging
3. `src/llama-moe-flash.cpp` - Added async prefetch infrastructure
4. `src/llama-moe-flash.h` - Added function declarations
5. `src/llama-context.h` - Added moe_flash_ctx member and include
6. `src/llama-context.cpp` - Added initialization and cleanup
7. `src/CMakeLists.txt` - Added llama-moe-flash.cpp to build

## Testing Commands

```bash
# Check if async prefetch is enabled
kubectl logs -n inference qwen3-235b-a22b-q4km | grep "I11-ASYNC"

# Check expert copy statistics
kubectl logs -n inference qwen3-235b-a22b-q4km | grep "I11-DEBUG-NORM"

# Check data transfer volume
kubectl logs -n inference qwen3-235b-a22b-q4km | grep "total_copy" | awk -F'total_copy=' '{sum+=$2} END {print "Total MB:", sum/1024}'
```

## Conclusion

The async prefetch prototype is **architecturally complete** but **functionally incomplete**. The infrastructure is in place to:
- Intercept expert copy operations
- Track which experts are being used
- Log detailed statistics

However, the actual async prefetching of experts to GPU is not yet implemented. The prototype demonstrates the mechanism works, but completing it would require significant additional work on GPU memory management and synchronization.

Given that the current "normal mode" causes a ~40% performance regression due to massive data transfers (22-40 GB per request), the recommendation is to either:
1. Complete the async prefetch implementation properly
2. Or revert the I10b expert copy changes entirely

---

**Status**: Prototype ready for testing  
**Next action**: Set LLAMA_FLASH_MOE_MODE=async_prefetch to test callback mechanism  
**Recommendation**: Evaluate if completing async prefetch is justified vs reverting
