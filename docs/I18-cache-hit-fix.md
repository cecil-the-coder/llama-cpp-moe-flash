# I18: Cache Hit Tracking Fix

**Status**: ✅ Complete  
**Date**: 2026-04-03  
**Image**: `ghcr.io/cecil-the-coder/llama-cpp-moe-flash:c791e75`

---

## Problem

The I17 metrics implementation was reporting **0% cache hit rate** even though cross-layer expert sharing was occurring. The logs showed:

```
[I11-CACHE-ANALYSIS] Layer 58: 9 experts, 3 shared with prev layer (33.3%)
[I11-CACHE-ANALYSIS] Layer 93: 8 experts, 3 shared with prev layer (37.5%)
```

But metrics showed:
```
moe_flash_cache_hits_total 0
moe_flash_cache_hit_rate 0.00
```

---

## Root Causes

### 1. Conditional Cache Tracking
The cache tracking code was wrapped in `if (ctx->cache.max_entries > 0)`, which meant:
- In **prefetch mode** (no LRU cache), metrics weren't updated
- Cross-layer sharing was logged but not counted as cache hits

### 2. Missing Variable Declarations
The patch was missing:
- `used_expert_ids` vector declaration
- `next_layer` variable declaration  
- `in_cache` → `in_lru_cache` rename inconsistencies

### 3. Duplicate Metrics Counting
`experts_loaded` was incremented in two places:
- Line 1810: `ctx->metrics.experts_loaded.fetch_add(used_expert_ids.size())`
- Line 1894: Same increment in cache tracking block

---

## Solution

### Fix 1: Unconditional Cache Tracking
Changed from:
```cpp
if (ctx->cache.max_entries > 0) {
    // Track cache hits/misses
}
```

To:
```cpp
if (ctx) {
    // Always track metrics, even without LRU cache
}
```

### Fix 2: Cross-Layer Sharing = Cache Hit
Now counts cross-layer expert reuse as cache hits:
```cpp
bool was_in_prev_layer = std::find(prev_layer_experts.begin(), 
                                   prev_layer_experts.end(), 
                                   expert_id) != prev_layer_experts.end();

bool cache_hit = in_lru_cache || was_in_prev_layer;

if (cache_hit) {
    cache_hits++;
    ctx->metrics.cache_hits.fetch_add(1);
}
```

### Fix 3: Add Missing Declarations
```cpp
// In expert_copy_callback:
std::vector<int> used_expert_ids;
for (int i = 0; i < n_expert; i++) {
    if (ggml_bitset_get(used_ids_bitset, i)) {
        used_expert_ids.push_back(i);
    }
}

int next_layer = current_layer + 1;
```

### Fix 4: Remove Duplicate Counting
Moved `experts_loaded` increment to cache tracking block only:
```cpp
// NOTE: experts_loaded is tracked in the cache section below
```

---

## Results

### Before Fix
```
moe_flash_requests_total         5
moe_flash_cache_hits_total       0
moe_flash_cache_hit_rate         0.00
moe_flash_experts_loaded_total   1293
```

### After Fix
```
moe_flash_requests_total         2
moe_flash_cache_hits_total       225
moe_flash_cache_hit_rate         11.75
moe_flash_experts_loaded_total   1915
```

### Log Output
```
[I17-CACHE] Layer 84: 1 hits, 7 misses (12.5% hit rate), 1 cross-layer
[I17-CACHE] Layer 90: 1 hits, 7 misses (12.5% hit rate), 1 cross-layer
[I17-CACHE] Layer 93: 3 hits, 5 misses (37.5% hit rate), 3 cross-layer
```

---

## Technical Details

### Why Cross-Layer Sharing Matters

In MoE models, consecutive layers often select overlapping experts:
- Layer N selects experts {3, 7, 12, 15, 23}
- Layer N+1 selects experts {7, 12, 19, 31}
- Shared experts: {7, 12} = 40% overlap

This sharing is a form of **temporal locality** - experts used recently are likely to be reused. Counting this as a "cache hit" gives us insight into MoE routing patterns.

### Metric Definitions

| Metric | Definition |
|--------|------------|
| `cache_hits` | Experts in LRU cache OR used in previous layer |
| `cache_misses` | Experts not in cache and not used in previous layer |
| `cache_hit_rate` | `hits / (hits + misses) * 100` |
| `cross_layer_hits` | Experts shared between consecutive layers |

---

## Files Changed

- `src/llama-moe-flash.cpp` - Fixed callback instrumentation
- `patches/0001-moe-flash-working.patch` - Unified patch with fix

---

## Testing

```bash
# Deploy image
kubectl set image -n inference deployment/qwen3-235b-a22b-q4km \
  llama-server=ghcr.io/cecil-the-coder/llama-cpp-moe-flash:c791e75

# Check metrics
curl http://localhost:9090/metrics
```

---

## Related Work

- I17: Prometheus metrics infrastructure
- I16: Dynamic prefetch window
- I11: Async expert prefetch

---

**Author**: Shadow (code-puppy-92fceb)  
**Completed**: 2026-04-03
