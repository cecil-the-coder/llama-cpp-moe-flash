# I11 Implementation Status

**Date**: 2025-04-01  
**Phase**: Code extraction complete, integration pending

## Completed

### ✅ 1. DeepSeek Baseline Testing
- **Current Performance**: 2.5 TPS (warm) with CPU MoE
- **Target**: 6-10 TPS with GPU MoE (slot buffer)
- **Expected Gain**: 2.4-4×
- **Output Quality**: ✅ Coherent (no corruption)

### ✅ 2. Slot Buffer Code Extraction
Extracted from `patches/0001-moe-flash-I14-I10b-option-b.patch`:

| Component | File | Lines | Status |
|-----------|------|-------|--------|
| Cache structures | `ggml-backend.cpp` | +25 | ✅ Extracted |
| Slot remap logic | `ggml-backend.cpp` | +150 | ✅ Extracted |
| LRU management | `ggml-backend.cpp` | +40 | ✅ Extracted |
| Callback API | `ggml-backend.h` | +20 | ✅ Extracted |

**Patch files created**:
- `patches/i11-slot-buffer-ggml-backend.patch` (247 lines)
- `patches/i11-slot-buffer-ggml-backend-h.patch` (28 lines)
- `patches/0001-moe-flash-I11-slot-buffer.patch` (combined, needs merge)

### ✅ 3. FORCE_OFFLOAD Flag
- Already in working baseline (`a15d1b3`)
- Tested and working
- Ready to activate slot buffer mode

## Remaining Work

### 🔧 1. Manual Merge Required

**Problem**: Both working baseline and slot buffer code modify `ggml-backend.cpp` in overlapping sections.

**Solution**: Manual three-way merge needed:
1. Base: b8298 upstream
2. Ours: working baseline (I10b Option B)
3. Theirs: slot buffer code (I11)

**Merge conflicts expected in**:
- `ggml_backend_sched_compute_splits()` function
- Expert copy logic around line 1500-1600
- Static variable declarations

### 🔧 2. Vulkan Buffer Allocation

Need to verify Vulkan backend allocates slot-sized buffers when `slot_remap_mode` is active:

```cpp
// In ggml-vulkan.cpp tensor creation:
if (tensor is MoE expert weights && slot_remap_mode) {
    // Allocate 32 slots instead of n_expert slots
    size_t slot_buffer_size = 32 * expert_size;
    // ... allocate buffer
}
```

### 🔧 3. Testing & Validation

**Test plan**:
1. Apply merged patch to clean b8298
2. Build Docker image
3. Deploy with FORCE_OFFLOAD=1
4. Test qwen3-235b-q4km (133 GB) first
5. Test DeepSeek 228 GB
6. Measure TPS and cache hit rate

## Technical Details

### Slot Buffer Algorithm

```cpp
// For each MUL_MAT_ID operation:
// 1. Check if expert tensor > 4 GiB (RADV limit)
const bool slot_remap_mode = n_expert * expert_size > 4 GB;

// 2. If slot mode active:
if (slot_remap_mode) {
    // Use 32-slot LRU cache
    // Copy active experts (K=8) to slots 0-7
    // Rewrite IDS tensor: expert_id → slot_id
    // GPU MUL_MAT_ID uses slot buffer
}
```

### Cache Efficiency

With K=8 active experts and 32 slots:
- **Working set**: 8 experts per token
- **Cache capacity**: 32 experts
- **Expected hit rate**: >90% (temporal locality)
- **Miss cost**: 1 expert copy (~5 MB on UMA = page remap)

## Implementation Commands

```bash
# Clone and setup
git clone --depth 1 --branch b8298 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

# Apply working baseline
git apply patches/0001-moe-flash-working.patch

# Apply slot buffer (manual merge needed)
# Edit ggml/src/ggml-backend.cpp to merge slot buffer code

# Build and push
docker build -f docker/Dockerfile.vulkan-moe-flash .

# Deploy with FORCE_OFFLOAD=1
kubectl patch inferencemodel deepseek-r1-0528 -n inference \
  --type merge -p '{"spec":{"env":[{"name":"LLAMA_FLASH_MOE_FORCE_OFFLOAD","value":"1"}]}}'

# Test
curl -s http://10.102.82.101:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-r1-0528","messages":[{"role":"user","content":"Test"}],"max_tokens":20}'
```

## Success Criteria

| Metric | Target | Current |
|--------|--------|---------|
| DeepSeek TPS | 6-10 | 2.5 |
| Cache hit rate | >90% | N/A |
| Output quality | Coherent | ✅ |
| No corruption | No "????" | ✅ |

## Next Steps

1. **Manual merge** of slot buffer code into working baseline
2. **Build** new Docker image
3. **Test** with qwen3-235b-q4km (lower risk)
4. **Test** with DeepSeek 228 GB
5. **Measure** actual TPS improvement

## Risk Assessment

| Risk | Level | Mitigation |
|------|-------|------------|
| Merge conflicts | Medium | Careful manual merge |
| Memory pressure | Medium | Dynamic slot count adjustment |
| Cache thrashing | Low | Monitor hit rate, adjust slots |
| Output corruption | Low | Validate with test suite |

## Estimated Timeline

- **Day 1**: Manual merge, build
- **Day 2**: qwen3 testing, debug
- **Day 3**: DeepSeek testing, optimization
- **Day 4**: Production deployment

---

**Status**: Code extracted, ready for integration  
**Blocker**: Manual merge of overlapping ggml-backend.cpp changes  
**Confidence**: High (code exists, just needs integration)
