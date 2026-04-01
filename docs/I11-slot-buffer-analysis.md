# I11 Slot Buffer Implementation Analysis

## Source

Found in `patches/0001-moe-flash-I14-I10b-option-b.patch` (lines 105-280)

## Key Components

### 1. Slot Cache Data Structure

```cpp
struct moe_slot_cache {
    int n_slots = 0;
    std::vector<int32_t> expert_to_slot;  // [n_expert] → slot or -1
    std::vector<int32_t> slot_to_expert;  // [n_slots] → expert or -1
    std::vector<int64_t> slot_lru;        // [n_slots] → timestamp
    int64_t clock = 0;
};
static std::unordered_map<const ggml_tensor *, moe_slot_cache> slot_caches;
```

### 2. Activation Condition

```cpp
#define GGML_MOE_SLOT_BUFFER_SLOTS 32
const bool slot_remap_mode = (size_t)n_expert * expert_size > (size_t)4 * 1024 * 1024 * 1024;
```

Slot mode activates when expert tensor > 4 GiB (RADV limit).

### 3. LRU Cache Management

```cpp
// Check if expert already in cache
if (sc.expert_to_slot[id] >= 0) {
    sc.slot_lru[sc.expert_to_slot[id]] = sc.clock;
    continue;
}

// Find free slot or evict LRU
int slot = -1;
for (int s = 0; s < n_slots; s++) {
    if (sc.slot_to_expert[s] == -1) { slot = s; break; }
}
if (slot == -1) {
    // Evict LRU
    int64_t min_lru = INT64_MAX;
    for (int s = 0; s < n_slots; s++) {
        if (sc.slot_lru[s] < min_lru) { min_lru = sc.slot_lru[s]; slot = s; }
    }
    int old_expert = sc.slot_to_expert[slot];
    if (old_expert >= 0) sc.expert_to_slot[old_expert] = -1;
}

// Assign new expert to slot
sc.expert_to_slot[id] = slot;
sc.slot_to_expert[slot] = id;
sc.slot_lru[slot] = sc.clock;
```

### 4. Expert Copy to Slot

```cpp
// Copy expert data to LOW slot index (within 4 GiB range)
const size_t src_offset = (size_t)id * expert_size;
const size_t dst_offset = (size_t)slot * expert_size;
const size_t padding = (slot < n_slots - 1) ? std::min<size_t>(expert_size, 512) : 0;

ggml_backend_tensor_set_async(split_backend,
    input_cpy,
    (const uint8_t *)input->data + src_offset, dst_offset,
    expert_size + padding);
```

### 5. IDS Rewrite

```cpp
// Rewrite IDS: replace expert IDs with slot indices
for (size_t i = 0; i < ids.size(); i++) {
    int32_t eid = ids[i];
    GGML_ASSERT(eid >= 0 && eid < n_expert && sc.expert_to_slot[eid] >= 0);
    ids[i] = sc.expert_to_slot[eid];
}

// Write rewritten IDS to GPU
ggml_tensor * ids_gpu = node->src[2];
ggml_backend_tensor_set_async(split_backend,
    ids_gpu, ids.data(), 0, ids.size() * sizeof(int32_t));
```

## Key Insight

The tensor shape stays `ne[2]=n_expert` (graph correctness), but only the first `K*expert_size` bytes of the GPU buffer are accessed. The shader early-exits for experts with 0 token count.

## Cache Coherency

For shared IDS tensors (gate/up/down projections share the same IDS):
```cpp
// Cache original IDS values before rewriting
static std::unordered_map<const ggml_tensor *, std::vector<int32_t>> original_ids_cache;

if (slot_remap_mode) {
    original_ids_cache[ids_tensor] = ids;
}

// When reusing IDS tensor, restore original values
if (slot_remap_mode && ids_tensor == prev_ids_tensor) {
    auto it = original_ids_cache.find(ids_tensor);
    if (it != original_ids_cache.end()) {
        ids = it->second;
    }
}
```

## Integration Points

1. **ggml-backend.cpp**: Slot cache management (already implemented)
2. **llama-moe-flash.cpp**: Prefetch coordination with LRU cache
3. **llama.cpp**: Activate when FORCE_OFFLOAD=1 and model > GTT

## Expected Cache Behavior

With K=8 experts per token and 32 slots:
- Cache size: 32 experts
- Working set: 8 experts
- Cache hit rate: ~90% (assuming temporal locality)
- Miss penalty: 1 expert copy (~5 MB on UMA = page remap)

## Next Steps

1. Port slot buffer code to working patch
2. Add FORCE_OFFLOAD activation logic
3. Test with DeepSeek 228 GB
4. Measure TPS improvement
