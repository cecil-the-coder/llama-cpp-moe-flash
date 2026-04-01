# I13 Phase 3: LRU Cache - Implementation Summary

**Date**: 2026-04-01  
**Image**: `40c2926` (LRU cache with hit/miss tracking)  
**Status**: ✅ **CODE COMPLETE** - Infrastructure ready, GPU data storage needs Phase 4

---

## Overview

Implemented LRU (Least Recently Used) cache infrastructure for MoE expert storage. The cache tracks hit/miss rates and provides the foundation for full GPU-based expert caching in Phase 4.

---

## Implementation

### Cache Structure

**`moe_expert_cache` structure:**
```cpp
struct moe_expert_cache {
    size_t max_size;         // Maximum cache size in bytes
    size_t current_size;     // Current cache usage
    size_t entry_size;       // Estimated size per expert
    int max_entries;         // Maximum number of cached experts
    
    std::vector<expert_cache_entry> entries;  // Cache slots
    std::unordered_map<uint64_t, int> lookup; // (layer<<32 | expert) -> index
    
    int64_t access_counter;  // LRU timestamp counter
    int64_t hits;            // Cache hit count
    int64_t misses;          // Cache miss count
    int64_t evictions;       // Number of evictions
};
```

### Helper Functions

**`cache_lookup()`** - Check if expert is in cache
- Returns entry index if found, -1 if not found
- Updates LRU timestamp on hit

**`cache_find_lru()`** - Find entry to evict
- Returns empty slot if available
- Otherwise returns least recently used entry

**`cache_insert()`** - Add expert to cache
- Evicts LRU entry if necessary
- Updates lookup table and metadata

**`cache_key()`** - Generate unique key from (layer, expert_id)

---

## Integration Points

### 1. Cache Initialization
```cpp
// In llama_moe_flash_init()
const char * cache_size_env = getenv("LLAMA_FLASH_MOE_CACHE_SIZE_MB");
size_t cache_size_mb = cache_size_env ? atoi(cache_size_env) : 128;
ctx->cache.max_size = cache_size_mb * 1024 * 1024;
ctx->cache.max_entries = ctx->cache.max_size / (11 * 1024 * 1024);
```

### 2. Cache Check in Callback
```cpp
// In expert_copy_callback()
if (ctx->cache.max_entries > 0) {
    for (int expert_id : used_expert_ids) {
        int cache_idx = cache_lookup(ctx->cache, current_layer, expert_id);
        if (cache_idx >= 0) {
            cache_hits++;
            // TODO: Use cached GPU buffer
        } else {
            cache_misses++;
        }
    }
}
```

### 3. Statistics Output
```cpp
// In llama_moe_flash_free()
if (ctx->cache.max_entries > 0) {
    fprintf(stderr, "moe-flash: cache stats: %lld hits, %lld misses\n",
            ctx->cache.hits, ctx->cache.misses);
    fprintf(stderr, "moe-flash: cache hit rate: %.1f%%\n",
            100.0 * ctx->cache.hits / total);
}
```

---

## Testing Results

### Configuration
- **Cache Size**: 128 MB
- **Max Entries**: 11 experts (~11 MB each for Qwen3 Q4_K_M)
- **Test**: 10 requests, 5 unique + 5 repeated prompts

### Results
```
Test 1: 3448 ms (first request, cold cache)
Test 2: 5623 ms (outlier)
Test 3: 3070 ms
Test 4: 3149 ms
Test 5: 3074 ms
Average: ~3.5s

Repeated requests:
Test 1: 3048 ms
Test 2: 3059 ms
Test 3: 3052 ms
Test 4: 3065 ms
Test 5: 3060 ms
Average: ~3.1s
```

### Cache Hit Rate
```
[I11-CACHE] Layer 37: 0 hits, 2 misses (0.0% hit rate)
[I11-CACHE] Layer 38: 0 hits, 2 misses (0.0% hit rate)
...
```

**Observation**: 0% hit rate because:
1. Cache only tracks metadata (not actual GPU data)
2. Each layer uses 2 different experts
3. 94 layers × 2 experts = 188 experts needed
4. Cache only holds 11 experts
5. No expert survives long enough to be reused

---

## Why 0% Hit Rate?

### Current Limitations
1. **Metadata-only tracking** - Cache stores (layer, expert_id) but not actual weights
2. **Sequential layer access** - Each layer needs different experts
3. **Small cache size** - 128 MB only holds ~11 experts vs 188 needed
4. **No cross-request caching** - Cache resets between requests

### What Would Help
1. **Larger cache** - 1-2 GB could hold hot experts across requests
2. **GPU data storage** - Actually store weights in GPU memory
3. **Cross-request persistence** - Keep cache warm between prompts
4. **Pattern learning** - Predict which experts will be reused

---

## Performance Comparison

| Configuration | Avg Latency | Cache | Status |
|-----------------|-------------|-------|--------|
| Smart prefetch only | ~3.5s | N/A | Baseline |
| LRU cache (metadata) | ~3.5s | 0% hit | Infrastructure ready |
| LRU cache (GPU data) | TBD | TBD | Phase 4 |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_FLASH_MOE_CACHE_SIZE_MB` | 128 | Cache size in megabytes |
| `LLAMA_FLASH_MOE_ENABLED` | 0 | Enable MoE flash optimization |
| `LLAMA_FLASH_MOE_MODE` | prefetch | Operating mode |

---

## Code Structure

```
llama-moe-flash.cpp:
├── expert_cache_entry struct       # Single cache entry
├── moe_expert_cache struct         # Cache container
├── cache_key()                     # Generate lookup key
├── cache_lookup()                  # Find expert in cache
├── cache_find_lru()                # Find eviction candidate
├── cache_insert()                  # Add to cache
├── cache_get_stats()               # Get statistics
│
├── llama_moe_flash_context:        # Context includes:
│   └── moe_expert_cache cache;     # Cache instance
│
├── llama_moe_flash_init():         # Initialize cache from env
├── expert_copy_callback():          # Check cache for each expert
└── llama_moe_flash_free():         # Output cache statistics
```

---

## Next Steps (Phase 4)

To make the cache effective, need to implement:

### 1. GPU Data Storage
```cpp
// Allocate GPU memory for cached experts
void * gpu_buffer = ggml_cuda_malloc(size);
cudaMemcpy(gpu_buffer, host_data, size, cudaMemcpyHostToDevice);
```

### 2. Compute Graph Integration
- Modify `ggml_mul_mat_id` to check cache first
- Use cached weights instead of reading from mmap
- Handle cache misses with fallback to disk

### 3. Cache Persistence
- Keep cache warm between requests
- Pre-populate based on routing patterns
- Async loading via io_uring

### 4. Pattern Learning
- Track expert co-occurrence patterns
- Predict which experts will be needed
- Proactive cache warming

---

## Key Learnings

1. **Infrastructure is ready** - Cache structures and tracking working
2. **GPU storage is hard** - Requires compute graph integration
3. **Size matters** - 128 MB too small, need 1-2 GB for effectiveness
4. **Metadata != performance** - Tracking alone doesn't help
5. **Cross-request needed** - Same-prompt caching not useful enough

---

## Commits

| SHA | Description |
|-----|-------------|
| `40c2926` | I13 Phase 3: LRU cache with hit/miss tracking |

---

## Status

✅ **Phase 3 Complete**: LRU cache infrastructure implemented and tested
- Cache structures working
- Hit/miss tracking functional
- Statistics output at shutdown
- Ready for Phase 4 (GPU data storage)

🎯 **Next**: Phase 4 - GPU-based expert storage with compute graph integration

---

**Image**: `ghcr.io/cecil-the-coder/llama-cpp-moe-flash:40c2926`  
**Configuration**: Set `LLAMA_FLASH_MOE_CACHE_SIZE_MB=128` (or larger for testing)
