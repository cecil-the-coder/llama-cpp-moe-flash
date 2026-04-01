# I11 Investigation: Dynamic Expert Import for GPU MoE

## Goal

Enable GPU MoE for models > GTT (120 GB) by copying active experts to GPU on-demand.

**Current State**: DeepSeek 228 GB runs at 1.8 t/s with CPU MoE
**Target**: 6-10 t/s with GPU MoE via dynamic expert import

## The Problem

For models > GTT:
- Expert tensors exceed 4 GiB `maxStorageBufferRange` per layer
- Can't allocate full expert tensor on GPU (5-10 GB per layer × 94 layers)
- Current `--cpu-moe` keeps experts mmap'd, matmul runs on CPU

## The Solution

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         I11 Dynamic Expert Import                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐         ┌──────────────────┐                  │
│  │  Expert Weights  │         │  GPU Slot Buffer │                  │
│  │  (mmap'd, CPU)   │         │  (32 slots, 4GB) │                  │
│  │  228 GB total    │         │  per layer       │                  │
│  └────────┬─────────┘         └────────┬─────────┘                  │
│           │                            │                              │
│           │  1. Read routing IDs       │                              │
│           │     (which K=8 experts)   │                              │
│           │                            │                              │
│           │  2. Copy active experts ──►│ 3. GPU MUL_MAT_ID            │
│           │     to slots (page remap) │    on slot buffer            │
│           │                            │                              │
│           │  4. LRU cache hits         │                              │
│              skip copies                                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Insight: UMA Zero-Copy

On AMD UMA (Strix Halo), "copying" to GPU is actually **page-table remap**:
- Same physical memory, different virtual address
- No data movement, just permission change
- Cost: ~μs per expert, not ~ms

### Slot Buffer Design

```cpp
// Pre-allocate fixed-size GPU buffer (N slots × expert_size)
// For qwen3-235b: 32 slots × 5 MB = 160 MB per layer
// Total: 94 layers × 160 MB = ~15 GB GPU memory

ggml_tensor * expert_slot_buffer;  // ne[2] = N_SLOTS (32), not n_expert (128)

// Before each MUL_MAT_ID:
// 1. Read routing IDs from CPU (which 8 experts needed)
// 2. For each needed expert:
//    - Check LRU cache (expert_id → slot_id)
//    - If miss: mmap → staging pool → import to slot (page remap on UMA)
//    - On UMA: "import" is page-table remap (zero copy!)
// 3. Rewrite IDS tensor to map expert_id → slot_id
// 4. GPU MUL_MAT_ID uses slot buffer
```

## Implementation Plan

### Phase 1: Slot Buffer Infrastructure

**Files to modify**:
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp` - GPU slot buffer allocation
- `ggml/src/ggml-backend.cpp` - Expert copy scheduling
- `src/llama-moe-flash.cpp` - LRU cache + prefetch coordination

**Key components**:
1. **GPU Slot Pool**: Fixed-size Vulkan buffers per layer
2. **LRU Cache**: Track which experts are in which slots
3. **Copy Scheduler**: Async expert copy before MUL_MAT_ID
4. **IDS Rewrite**: Map expert IDs to slot IDs

### Phase 2: Force-Offload Integration

When `LLAMA_FLASH_MOE_FORCE_OFFLOAD=1`:
- Skip clearing tensor overrides (already done in I10b)
- Route MUL_MAT_ID to GPU with slot buffer
- Activate copy scheduler

### Phase 3: Performance Validation

**Metrics to track**:
- TPS: 1.8 → 6-10 (target)
- Cache hit rate: Should be >90% for K=8, N_SLOTS=32
- Copy latency: <1ms per expert on UMA

## Technical Challenges

### 1. RADV 4 GiB Limit

```
maxStorageBufferRange = 4 GiB (RADV limitation)
Expert tensor per layer = 128 experts × 5 MB = 640 MB (fits!)
```

Individual experts fit, just not all 128 at once. Slot buffer approach solves this.

### 2. Graph Split Complexity

MUL_MAT_ID needs to run on GPU with CPU-provided routing IDs. Requires:
- Split graph at routing computation (CPU)
- Copy IDS to GPU (or use shared buffer)
- MUL_MAT_ID on GPU slot buffer

### 3. Copy Synchronization

Options:
- **Blocking**: Copy before graph, add latency (simple)
- **Async**: Copy while computing previous layer (complex)
- **Predictive**: I10b prefetch copies before needed (optimal)

### 4. Memory Pressure

15 GB for slot buffers + GTT for other tensors = tight on 120 GB.
May need dynamic slot count adjustment.

## Expected Performance

| Model | Current | Target | Improvement |
|-------|---------|--------|-------------|
| DeepSeek 228 GB | 1.8 t/s | 6-10 t/s | **3-5×** |
| qwen3-235b-q4km | 6-7 t/s | 15-18 t/s | **2-3×** |

## Success Criteria

1. ✅ GPU MUL_MAT_ID runs on slot buffer
2. ✅ LRU cache hit rate >90%
3. ✅ No OOM or corruption
4. ✅ Coherent output
5. ✅ TPS improvement >2×

## Related Work

- I10b: Slot buffer code foundation ✅
- I10b Option B: Force-offload flag ✅
- I14: io_uring staging pool optimization
- flash-moe: Proved streaming concept works

## Next Steps

1. Explore existing slot buffer code in patch
2. Design slot buffer allocation strategy
3. Implement GPU buffer pool
4. Integrate with force-offload path
5. Test and benchmark

---

**Status**: Investigation started  
**Target completion**: 1-2 weeks  
**Risk**: Medium (touches scheduler, complex interactions)
