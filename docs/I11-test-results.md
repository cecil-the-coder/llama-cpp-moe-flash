# I11 Test Results

**Date**: 2025-04-01  
**Image**: 95674c2  
**Status**: Build successful, performance regression observed

## Test Results

### qwen3-235b-q4km (133 GB)

| Request | Time | Tokens | TPS | Notes |
|---------|------|--------|-----|-------|
| 1 | 6s | 13 | 2.2 | Cold |
| 2 | 4s | 13 | 3.2 | Warm |
| 3 | 4s | 13 | 3.2 | Warm |
| 4 | 4s | 12 | 3.0 | Warm |

**Average TPS**: ~3.0 (baseline was ~5.0)

### DeepSeek-R1-0528 (228 GB)

| Request | Time | Tokens | TPS | Notes |
|---------|------|--------|-----|-------|
| 1 | 22s | 14 | 0.6 | Cold |
| 2 | 9s | 16 | 1.8 | Warm |
| 3 | 8s | 13 | 1.6 | Warm |

**Average TPS**: ~1.5 (baseline was ~2.5)

## Key Findings

### 1. No Slot Buffer Activity Detected

Logs show no evidence of slot buffer activation:
```
- No "slot_remap" messages
- No "expert copy" logs
- No I11-specific debug output
```

### 2. Model Configuration

```
DeepSeek:
- n_expert = 256
- n_expert_used = 8
- CPU_Mapped buffer = 46522 MiB (46.5 GB mmap'd experts)
- Model fits in GTT (10.6 GB / 117 GB free)
```

### 3. Code Path Analysis

The slot buffer code is in `ggml_backend_sched_compute_splits()`:
- Trigger condition: `n_expert * expert_size > 4 GB`
- But experts are loaded via `load_tensors()` using mmap
- May not go through scheduler code path

## Possible Issues

### Issue 1: Wrong Code Path
The slot buffer code is in the scheduler, but CPU_Mapped experts may bypass this path.

### Issue 2: Slot Remap Condition Not Met
Even with 256 experts, individual expert size may be < 16 MB (256 * 16 MB = 4 GB).

### Issue 3: Missing Debug Output
Need to add logging to verify:
- Is `slot_remap_mode` being set to true?
- Is the slot buffer code path being executed?
- Are experts being copied to GPU slots?

## Next Steps

1. **Add debug logging** to verify code path execution
2. **Verify expert tensor sizes** in the model
3. **Check if code path is correct** for mmap'd experts
4. **Profile with ggml_perf** to see actual operations

## Hypothesis

The slot buffer code may need to be in a different location - perhaps in `llama-moe-flash.cpp` or in the tensor loading code rather than the scheduler.

The current implementation assumes experts go through `ggml_backend_sched_compute_splits()` but mmap'd experts loaded via `load_tensors()` may use a different code path.

## Recommendation

Investigate the actual code path for MUL_MAT_ID with mmap'd CPU tensors. The slot buffer logic may need to be moved to where the actual expert copying happens.

---

**Status**: Implementation complete but not activating  
**Priority**: High - Need to debug code path  
**Next action**: Add debug logging and verify execution path
