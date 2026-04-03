# ✅ Ready for Testing

**Date**: 2026-04-03  
**Image**: `ghcr.io/cecil-the-coder/llama-cpp-moe-flash:latest`  
**Status**: ✅ I14 + I10b VERIFIED ACTIVE in Kubernetes

---

## 🎉 Verification Complete (2026-04-03)

Both I14 and I10b optimizations have been **verified active** in the Kubernetes cluster:

### I14 io_uring Polish - ✅ ACTIVE
```
[I14] io_uring: enabling SINGLE_ISSUER optimization
[I14] MADV_HUGEPAGE: 16/16 slots enabled (64 MB total)
```

### I10b Force-Offload - ✅ ACTIVE
```
[I10b] Force GPU offload enabled - will use slot buffer GPU path
[I10b] Backend set for GPU offload: Vulkan0
```

### Critical Fix Applied
The backend had `LLAMA_FLASH_MOE_IOURING=0` which disabled the io_uring code path.
Fixed by changing to `LLAMA_FLASH_MOE_IOURING=1` in the cluster configuration.

---

## 🎯 What to Test

### 1. I14: io_uring Polish (10-25% Performance Gain)

Three low-risk optimizations for background prefetch:

| Optimization | Expected Gain | How to Verify |
|--------------|---------------|---------------|
| IORING_SETUP_SINGLE_ISSUER | 2-5% | Log message on startup |
| MADV_HUGEPAGE | 3-8% | `/proc/meminfo` shows increasing AnonHugePages |
| IORING_REGISTER_BUFFERS | 5-15% | Already in place, verify no errors |

**Quick Test**:
```bash
export LLAMA_FLASH_MOE_ENABLED=1
export LLAMA_FLASH_MOE_MODE=prefetch
./llama-server -m model.gguf ...

# Look for in logs:
# [I14] io_uring: enabling SINGLE_ISSUER optimization
# [I14] MADV_HUGEPAGE: 4/4 slots enabled (24 MB total)
```

---

### 2. I10b Option B: Force-Offload (3-5x Speedup for >GTT)

**The Big One**: Test slot buffer GPU path for models exceeding GTT.

**Test Strategy**:
1. **Phase 1** (≤GTT model): Verify infrastructure works
2. **Phase 2** (>GTT model): Measure speedup on DeepSeek

**Phase 1 - Validation Test** (qwen3-235b-q4km, 133 GB ≤ 125 GB GTT):
```bash
# Force CPU mode + slot buffer
export LLAMA_ARG_CPU_MOE=1
export LLAMA_FLASH_MOE_FORCE_OFFLOAD=1
export LLAMA_FLASH_MOE_ENABLED=1
export LLAMA_FLASH_MOE_CACHE_SIZE_MB=128

./llama-server -m qwen3-235b-a22b-q4km.gguf ...

# Expected: 15-18 t/s (same as full GPU)
# If working: slot buffer path is functional
```

**Phase 2 - Speedup Test** (DeepSeek-R1-0528, 228 GB > 120 GB GTT):
```bash
# Same environment variables as Phase 1
export LLAMA_ARG_CPU_MOE=1
export LLAMA_FLASH_MOE_FORCE_OFFLOAD=1
export LLAMA_FLASH_MOE_ENABLED=1
export LLAMA_FLASH_MOE_CACHE_SIZE_MB=128

./llama-server -m DeepSeek-R1-0528-Q2_K.gguf ...

# Baseline (CPU MoE): 1.8 t/s
# Target (I10b GPU):   6-10 t/s  ← 3-5x improvement!
```

---

## 📊 Success Criteria

### I14 io_uring Polish
- [x] Log shows `SINGLE_ISSUER` enabled (kernel 6.0+) ✅
- [x] Log shows `MADV_HUGEPAGE` enabled on all slots ✅
- [ ] `AnonHugePages` increases during inference (pending benchmark)
- [x] No io_uring errors in logs ✅

### I10b Option B (Phase 1 - Validation)
- [x] Log shows `Backend set for GPU offload: Vulkan0` ✅
- [ ] Log shows experts being `Imported to GPU cache` (pending inference)
- [ ] Cache hit rate > 50% after warmup (pending benchmark)
- [ ] Performance ≥ 15 t/s (qwen3-235b-q4km) (pending benchmark)

### I10b Option B (Phase 2 - Speedup)
- [ ] All Phase 1 criteria pass (pending Phase 1 completion)
- [ ] Performance ≥ 6 t/s (DeepSeek 228 GB) (pending benchmark)
- [ ] 3x+ improvement over baseline 1.8 t/s (pending benchmark)
- [ ] No GPU OOM or errors (pending benchmark)

**Status**: Infrastructure verified, awaiting full inference benchmark.

---

## 🚀 Quick Start

### Option A: Local Docker

```bash
# Pull the image
docker pull ghcr.io/cecil-the-coder/llama-cpp-moe-flash:f74f3c3

# Run with I14 + I10b
docker run -it --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  -v /mnt/models:/mnt/models \
  -e LLAMA_FLASH_MOE_ENABLED=1 \
  -e LLAMA_FLASH_MOE_MODE=prefetch \
  -e LLAMA_ARG_CPU_MOE=1 \
  -e LLAMA_FLASH_MOE_FORCE_OFFLOAD=1 \
  ghcr.io/cecil-the-coder/llama-cpp-moe-flash:f74f3c3 \
  ./llama-server -m /mnt/models/qwen3-235b-a22b-q4km.gguf -ngl 99
```

### Option B: Kubernetes

```yaml
apiVersion: inference.models.eh-ops.io/v1
kind: InferenceModel
metadata:
  name: i14-i10b-test
spec:
  backend: llamacpp-vulkan-moe-flash-cpumoe
  env:
    # I14: io_uring optimizations
    - name: LLAMA_FLASH_MOE_ENABLED
      value: "1"
    - name: LLAMA_FLASH_MOE_MODE
      value: "prefetch"
    
    # I10b: Force GPU offload
    - name: LLAMA_ARG_CPU_MOE
      value: "1"
    - name: LLAMA_FLASH_MOE_FORCE_OFFLOAD
      value: "1"
    - name: LLAMA_FLASH_MOE_CACHE_SIZE_MB
      value: "128"
  resources:
    limits:
      memory: "125Gi"
      amd.com/gpu: "1"
```

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| [testing-guide.md](docs/testing-guide.md) | **Start here** - comprehensive testing guide |
| [I14-iouring-polish.md](docs/I14-iouring-polish.md) | Technical details of I14 optimizations |
| [I10b-option-b-force-offload.md](docs/I10b-option-b-force-offload.md) | I10b test plan and architecture |
| [next-investigations.md](docs/next-investigations.md) | Priority matrix and roadmap |

---

## 🎉 If Tests Pass

### I14 Success
- Deploy to production immediately
- Expected: 10-25% performance gain for all MoE models
- Zero risk (additive optimizations)

### I10b Phase 1 Success
- Proceed to Phase 2 (DeepSeek testing)
- Slot buffer GPU path is functional
- Infrastructure is solid

### I10b Phase 2 Success
- **Major achievement**: 3-5x speedup for >GTT models
- DeepSeek 228 GB now runs at 6-10 t/s (was 1.8 t/s)
- Documentation and celebration!

---

## 🐛 If Issues Found

Collect the following:
1. Full startup logs (first 100 lines)
2. Inference logs (showing I14/I10b messages)
3. `cat /proc/meminfo | grep -i huge`
4. `uname -r` (kernel version)
5. Performance metrics (t/s comparison)

File issue with:
- Image tag: `f74f3c3`
- Test case (I14 or I10b Phase 1/2)
- Expected vs actual behavior

---

**Ready to test? Start with the [Testing Guide](docs/testing-guide.md)!**
