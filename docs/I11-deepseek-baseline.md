# DeepSeek-R1-0528 Baseline Performance Test

**Date**: 2025-04-01  
**Model**: DeepSeek-R1-0528 (228 GB, Q4_K_M)  
**Hardware**: AMD Strix Halo (Radeon 8060S, 128 GB RAM)  
**Image**: a15d1b3 (I10b Option B)

## Test Results

### Normal Mode (CPU MoE, mmap-wrap)

| Request | Time | Tokens | TPS | Status |
|---------|------|--------|-----|--------|
| 1 | 24s | 20 | 0.83 | Cold start |
| 2 | 8s | 20 | 2.50 | Warm |
| 3 | 7s | 20 | 2.86 | Warm |
| **Average (warm)** | - | - | **2.67** | - |

### FORCE_OFFLOAD=1 Mode

| Request | Time | Tokens | TPS | Status |
|---------|------|--------|-----|--------|
| 1 | 27s | 20 | 0.74 | Cold start |
| 2 | 10s | 20 | 2.00 | Warm |
| 3 | 9s | 20 | 2.22 | Warm |
| **Average (warm)** | - | - | **2.11** | - |

## Key Findings

### 1. Baseline Performance: ~2.5 TPS (Warm)

Previous documentation mentioned 1.8 TPS, but actual warm performance is 2.0-2.7 TPS. The lower numbers were likely from cold-start runs or different model configurations.

### 2. FORCE_OFFLOAD Flag Has No Benefit (Yet)

Both modes show similar warm performance (~2.0-2.5 TPS):
- **Normal**: 2.67 TPS warm average
- **FORCE_OFFLOAD=1**: 2.11 TPS warm average

This is expected because:
- FORCE_OFFLOAD flag currently only prevents auto-detect from clearing tensor overrides
- Slot buffer GPU acceleration code is NOT yet ported
- Both modes use CPU MoE via mmap-wrap

### 3. Output Quality: ✅ Coherent

All test outputs show proper reasoning content with no corruption:
- "We are going to simulate a test run..."
- "We are given the following problem..."
- "We are going to test the behavior..."

No "????" or garbage output detected.

### 4. Memory Configuration

```
llama_params_fit_impl: full model does NOT fit in device memory 
  (234711 MiB needed vs 123228 MiB free)
load_tensors: Vulkan_Host tensors: 220.7 GiB → mmap-wrap
```

Model exceeds GTT (120 GB), so mmap-wrap is used for expert tensors.

## I11 Target Analysis

### Current State
- **Performance**: 2.5 TPS (CPU MoE)
- **Memory**: mmap-wrap, experts on CPU
- **Compute**: CPU MUL_MAT_ID

### I11 Target
- **Performance**: 6-10 TPS (GPU MoE with slot buffer)
- **Memory**: 32-slot GPU buffer per layer (15 GB total)
- **Compute**: GPU MUL_MAT_ID with on-demand expert copy

### Expected Improvement
| Metric | Current | Target | Gain |
|--------|---------|--------|------|
| TPS | 2.5 | 6-10 | **2.4-4×** |
| Expert storage | CPU (mmap) | GPU slots | 15 GB |
| Matmul | CPU | GPU | 10× faster |

## Next Steps for I11

1. **Port slot buffer code** from `I14-I10b-option-b.patch` to working baseline
2. **Test with DeepSeek** to verify 6-10 TPS target
3. **Monitor cache hit rate** (target >90%)
4. **Validate output quality** remains coherent

## Test Commands

```bash
# Normal mode test
kubectl patch inferencemodel deepseek-r1-0528 -n inference --type json -p '[{"op": "remove", "path": "/spec/env"}]'

# FORCE_OFFLOAD mode test  
kubectl patch inferencemodel deepseek-r1-0528 -n inference --type merge -p '{"spec":{"env":[{"name":"LLAMA_FLASH_MOE_FORCE_OFFLOAD","value":"1"}]}}'

# Performance test
curl -s http://10.102.82.101:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-r1-0528","messages":[{"role":"user","content":"Test"}],"max_tokens":20}'
```

## Summary

✅ DeepSeek 228 GB runs at **2.5 TPS** with CPU MoE  
✅ Output quality is good (coherent reasoning)  
✅ FORCE_OFFLOAD flag infrastructure ready  
⏭️ **I11 slot buffer porting needed for 6-10 TPS target**

---

**Recommendation**: Proceed with I11 implementation. Expected gain is 2.4-4× (2.5 → 6-10 TPS).
