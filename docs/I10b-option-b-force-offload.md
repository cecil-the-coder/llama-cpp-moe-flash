# I10b Option B: Force-offload Testing (Slot Buffer GPU Path)

**Status**: 🔄 Infrastructure Added | **Image**: `ghcr.io/cecil-the-coder/llama-cpp-moe-flash:c791e75`

---

## Goal

Test if the slot buffer GPU path works by forcing GPU offload even when `--cpu-moe` is set.

**Why**: For models > 120 GB GTT (like DeepSeek 228 GB), we currently use `--cpu-moe` which routes expert matmul to CPU (1.8 t/s). The slot buffer code is supposed to enable GPU matmul by caching active experts in GPU memory slots.

**Target**: 6-10 t/s for >GTT models (vs current 1.8 t/s)

---

## How It Works

### Current Architecture

```
≤GTT Models (≤120 GB):
  Full GPU offload → 18-20 t/s ✅

>GTT Models (>120 GB):
  --cpu-moe → Expert matmul on CPU → 1.8 t/s
```

### Slot Buffer Concept (I10b Option B)

```
>GTT Models with Slot Buffer:
  --cpu-moe + force_offload=1
  ├── Copy active experts (K=8) to GPU slot buffer
  ├── GPU matmul on slot buffer → Target: 6-10 t/s
  └── LRU eviction for next layer's experts
```

**Key Insight**: Even with 228 GB model, only 8 experts × 3 MB = 24 MB are active per layer!

---

## Implementation Status

### ✅ Completed
- `force_offload` flag in context structure
- Environment variable parsing for `LLAMA_FLASH_MOE_FORCE_OFFLOAD`
- Log message when flag is enabled

### 🔄 Remaining (For Full Test)
- GPU import logic in callback (read expert → GPU buffer)
- Backend integration (route MUL_MAT_ID to use cached experts)
- LRU eviction in cache when slots full

---

## Test Plan

### Phase 1: Validate Flag (Immediate)

```yaml
# Test with qwen3-235b-q4km (133 GB, normally uses full GPU at 18 t/s)
# Force CPU mode + slot buffer to verify infrastructure:

apiVersion: inference.models.eh-ops.io/v1
kind: InferenceModel
metadata:
  name: qwen3-235b-a22b-q4km-test
spec:
  backend: llamacpp-vulkan-moe-flash-cpumoe
  env:
    - name: LLAMA_ARG_CPU_MOE
      value: "1"  # Force CPU mode (disable normal GPU)
    - name: LLAMA_FLASH_MOE_FORCE_OFFLOAD
      value: "1"  # Enable slot buffer GPU path
    - name: LLAMA_FLASH_MOE_ENABLED
      value: "1"
    - name: LLAMA_FLASH_MOE_MODE
      value: "prefetch"
```

**Expected Results**:
- If slot buffer works: 15-18 t/s (same as full GPU)
- If slot buffer broken: 6-7 t/s (CPU matmul)

### Phase 2: DeepSeek Validation (If Phase 1 Passes)

```yaml
# Test with DeepSeek-R1-0528 (228 GB > 120 GB GTT)
# Current: 1.8 t/s (CPU MoE)
# Target: 6-10 t/s (GPU via slot buffer)

apiVersion: inference.models.eh-ops.io/v1
kind: InferenceModel
metadata:
  name: deepseek-r1-test
spec:
  backend: llamacpp-vulkan-moe-flash-cpumoe
  env:
    - name: LLAMA_ARG_CPU_MOE
      value: "1"
    - name: LLAMA_FLASH_MOE_FORCE_OFFLOAD
      value: "1"
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_FLASH_MOE_FORCE_OFFLOAD` | `0` | Enable slot buffer GPU path |
| `LLAMA_ARG_CPU_MOE` | `0` | Force CPU MoE (disable normal GPU) |

**Usage**:
```bash
# For testing slot buffer on a ≤GTT model:
export LLAMA_ARG_CPU_MOE=1
export LLAMA_FLASH_MOE_FORCE_OFFLOAD=1
./llama-server -m model.gguf ...

# Check logs for:
# [I10b] Force GPU offload enabled - slot buffer GPU path active
```

---

## Technical Details

### Slot Buffer Structure

```cpp
struct moe_expert_cache {
    size_t max_size;                          // 128 MB default
    std::vector<expert_cache_entry> entries;  // GPU buffer slots
    std::unordered_map<uint64_t, int> lookup; // (layer<<32 | expert) → slot
};

struct expert_cache_entry {
    int layer_idx, expert_id;
    void* gpu_buffer;                    // GPU memory
    ggml_backend_buffer_t backend_buffer;
    bool valid;
};
```

### Import Flow (When Force Offload Enabled)

```cpp
// In expert_copy_callback:
if (ctx->force_offload) {
    for (int expert_id : used_expert_ids) {
        // Check if expert in GPU cache
        int cache_idx = cache_lookup(ctx->cache, current_layer, expert_id);
        
        if (cache_idx < 0) {
            // Cache miss - import to GPU
            void* expert_data = read_expert_from_gguf(ctx, current_layer, expert_id);
            cache_store_expert(ctx->cache, current_layer, expert_id, 
                               expert_data, expert_size, backend);
        }
        // Now expert is in GPU cache, ready for matmul
    }
}
```

---

## Success Criteria

| Metric | Current (CPU MoE) | Target (Slot Buffer) |
|--------|-------------------|---------------------|
| DeepSeek 228 GB | 1.8 t/s | 6-10 t/s |
| qwen3-235b-q4km | 18 t/s (GPU) | 15-18 t/s (slot buffer) |

---

## Next Steps

1. **Add GPU import logic** to callback (read from GGUF → GPU buffer)
2. **Integrate with backend** to use cached experts for MUL_MAT_ID
3. **Test Phase 1** with qwen3-235b-q4km
4. **If successful**, test Phase 2 with DeepSeek

---

## References

- `next-investigations.md` - Original I10b Option B specification
- `src/llama-moe-flash.cpp` - Implementation (`force_offload` flag)
- `docs/I10b-findings.md` - I10b investigation results

---

**Author**: Shadow (code-puppy-92fceb)  
**Created**: 2026-04-03
