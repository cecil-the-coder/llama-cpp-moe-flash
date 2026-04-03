# Testing Guide: I14 + I10b Optimizations

**Image**: `ghcr.io/cecil-the-coder/llama-cpp-moe-flash:f74f3c3`

---

## What to Test

### 1. I14 io_uring Polish (All Models)

Verify the three optimizations are active:
- [ ] IORING_SETUP_SINGLE_ISSUER enabled (kernel 6.0+)
- [ ] MADV_HUGEPAGE enabled on staging pool
- [ ] IORING_REGISTER_BUFFERS working (already in place)

### 2. I10b Option B Force-Offload (>GTT Models)

Test slot buffer GPU path:
- [ ] Backend properly set
- [ ] Experts imported to GPU cache
- [ ] LRU eviction working

---

## Test Commands

### Test 1: I14 Verification (Any MoE Model)

```bash
# Environment setup
export LLAMA_FLASH_MOE_ENABLED=1
export LLAMA_FLASH_MOE_MODE=prefetch

# Run inference
cd /app
./llama-server \
  -m /mnt/models/qwen3-235b-a22b-q4km.gguf \
  --ctx-size 4096 \
  --batch-size 512 \
  -ngl 99 \
  --host 0.0.0.0 \
  --port 8080

# Look for these log messages on startup:
```

**Expected Log Output**:
```
[I14] io_uring: enabling SINGLE_ISSUER optimization
[I14] MADV_HUGEPAGE: 4/4 slots enabled (24 MB total)
moe-flash: io_uring initialized: 4 slots × 6291456 bytes (24 MB total)
```

---

### Test 2: I10b Force-Offload (≤GTT Model First)

Use a model that normally works with full GPU to verify the slot buffer path:

```bash
# Force CPU mode + slot buffer GPU import
export LLAMA_ARG_CPU_MOE=1
export LLAMA_FLASH_MOE_FORCE_OFFLOAD=1
export LLAMA_FLASH_MOE_ENABLED=1
export LLAMA_FLASH_MOE_MODE=prefetch
export LLAMA_FLASH_MOE_CACHE_SIZE_MB=128

./llama-server \
  -m /mnt/models/qwen3-235b-a22b-q4km.gguf \
  --ctx-size 4096 \
  -ngl 99

# Look for these log messages:
```

**Expected Log Output**:
```
[I10b] Force GPU offload enabled - slot buffer GPU path active
[I10b] Backend set for GPU offload: Vulkan0
...
[I10b] Layer 0 Expert 5: Imported to GPU cache (3072000 bytes)
[I10b] Layer 0: 8 imported, 0 cached for GPU offload
[I17-CACHE] Layer 0: 8 hits, 0 misses (100.0% hit rate), 0 cross-layer
```

---

### Test 3: I10b on DeepSeek (>GTT Model)

```bash
# Test with DeepSeek 228 GB (exceeds 120 GB GTT)
export LLAMA_ARG_CPU_MOE=1
export LLAMA_FLASH_MOE_FORCE_OFFLOAD=1
export LLAMA_FLASH_MOE_ENABLED=1
export LLAMA_FLASH_MOE_MODE=prefetch
export LLAMA_FLASH_MOE_CACHE_SIZE_MB=128

./llama-server \
  -m /mnt/models/DeepSeek-R1-0528-Q2_K.gguf \
  --ctx-size 4096 \
  --batch-size 512

# Measure tokens/second
```

**Success Criteria**:
- Tokens/sec: **> 5.0 t/s** (vs baseline 1.8 t/s)
- Cache hit rate: **> 50%** after warmup
- No GPU OOM errors

---

## Validation Checklist

### I14 Validation

| Check | Command/Method | Expected Result |
|-------|----------------|-----------------|
| SINGLE_ISSUER enabled | Check logs for `[I14] io_uring: enabling SINGLE_ISSUER` | Message present (kernel 6.0+) |
| MADV_HUGEPAGE enabled | Check logs for `[I14] MADV_HUGEPAGE: N/M slots enabled` | All slots enabled |
| Huge pages active | `cat /proc/meminfo \| grep AnonHugePages` | Value increasing during inference |
| Buffer registration | Check for no "buffer registration failed" errors | No errors |

### I10b Validation

| Check | Command/Method | Expected Result |
|-------|----------------|-----------------|
| Backend set | Check logs for `[I10b] Backend set for GPU offload: Vulkan0` | Backend name shown |
| Force offload active | Check logs for `force_offload active` | Message on each layer |
| Experts imported | Check logs for `Imported to GPU cache` | Multiple import messages |
| Cache working | Check logs for `I17-CACHE` hit rate | > 50% after warmup |
| Performance gain | Measure t/s vs baseline | 2-5x improvement for >GTT |

---

## Metrics to Collect

### Prometheus Metrics (if enabled)

```bash
# Port-forward to metrics endpoint
kubectl port-forward pod/<llama-pod> 9090:9090

# Query metrics
curl -s http://localhost:9090/metrics | grep moe_flash
```

**Key Metrics**:
- `moe_flash_requests_total` - Total inference requests
- `moe_flash_cache_hits_total` - Cache hit count
- `moe_flash_cache_misses_total` - Cache miss count
- `moe_flash_cache_hit_rate` - Hit rate percentage
- `moe_flash_experts_loaded_total` - Experts loaded from disk
- `moe_flash_io_bytes_loaded_total` - Total I/O bytes
- `moe_flash_io_bytes_saved_total` - I/O saved by cache

### Manual Timing

```bash
# Time a single request
time curl -s http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a hello world program in Python:",
    "n_predict": 128,
    "temperature": 0.7
  }'

# Calculate tokens/second: n_predict / elapsed_time
```

---

## Kubernetes Deployment

### Complete Test Manifest

Save as `test-i14-i10b.yaml`:

```yaml
apiVersion: inference.models.eh-ops.io/v1
kind: InferenceModel
metadata:
  name: i14-i10b-test
  namespace: default
spec:
  # Use the image with I14 + I10b optimizations
  image: ghcr.io/cecil-the-coder/llama-cpp-moe-flash:f74f3c3
  
  backend: llamacpp-vulkan-moe-flash-cpumoe
  
  modelSource:
    modelName: qwen3-235b-a22b-q4km  # Change to DeepSeek-R1-0528-Q2_K for Phase 2
    
  env:
    # I14: io_uring optimizations (automatic when compiled with io_uring)
    - name: LLAMA_FLASH_MOE_ENABLED
      value: "1"
    - name: LLAMA_FLASH_MOE_MODE
      value: "prefetch"
    
    # I10b: Force GPU offload (for testing slot buffer on >GTT models)
    - name: LLAMA_ARG_CPU_MOE
      value: "1"  # Force CPU mode (disable normal GPU)
    - name: LLAMA_FLASH_MOE_FORCE_OFFLOAD
      value: "1"  # Enable slot buffer GPU import
    - name: LLAMA_FLASH_MOE_CACHE_SIZE_MB
      value: "128"
    
    # Prometheus metrics (optional)
    - name: LLAMA_FLASH_MOE_METRICS_PORT
      value: "9090"
    
    # Context size
    - name: LLAMA_ARG_CTX_SIZE
      value: "4096"
    - name: LLAMA_ARG_BATCH_SIZE
      value: "512"
  
  resources:
    limits:
      memory: "125Gi"
      amd.com/gpu: "1"  # Request AMD GPU
  
  # Scale to zero when not in use (optional)
  scaleToZero:
    enabled: false  # Set to true for production, false for testing
```

### Deploy and Test

```bash
# 1. Apply the manifest
kubectl apply -f test-i14-i10b.yaml

# 2. Wait for pod to be ready
kubectl wait --for=condition=ready pod -l app=i14-i10b-test --timeout=300s

# 3. Port-forward to access the API
kubectl port-forward svc/i14-i10b-test 8080:8080 &

# 4. Check logs for I14 startup messages
kubectl logs -l app=i14-i10b-test | grep -E "(I14|I10b|moe-flash)" | head -20

# 5. Send test request to trigger inference
curl -s http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, this is a test. Count from 1 to 10:",
    "n_predict": 128,
    "temperature": 0.7
  }' | jq '.tokens_per_second'

# 6. Watch for I10b messages during inference
kubectl logs -l app=i14-i10b-test -f | grep -E "(I10b|I17-CACHE)"

# 7. Check metrics (if enabled)
kubectl port-forward svc/i14-i10b-test 9090:9090 &
curl -s http://localhost:9090/metrics | grep moe_flash
```

### Expected Results

**I14 Verification** (first 30 seconds):
```
[I14] io_uring: enabling SINGLE_ISSUER optimization
[I14] MADV_HUGEPAGE: 4/4 slots enabled (24 MB total)
moe-flash: io_uring initialized: 4 slots × 6291456 bytes (24 MB total)
```

**I10b Verification** (during first inference):
```
[I10b] Force GPU offload enabled - slot buffer GPU path active
[I10b] Backend set for GPU offload: Vulkan0
[I10b] Layer 0 Expert 5: Imported to GPU cache (3072000 bytes)
[I10b] Layer 0: 8 imported, 0 cached for GPU offload
[I17-CACHE] Layer 0: 8 hits, 0 misses (100.0% hit rate)
```

### Performance Test

```bash
# Run multiple requests and capture t/s
for i in 1 2 3; do
  echo "Run $i:"
  curl -s http://localhost:8080/completion \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Explain quantum computing:", "n_predict": 256}' | \
    jq '.tokens_per_second'
done
```

---

## Troubleshooting

### Issue: "IORING_SETUP_SINGLE_ISSUER not available"

**Cause**: Kernel version < 6.0

**Solution**: Not an error - optimization skipped gracefully

### Issue: "MADV_HUGEPAGE failed"

**Cause**: Transparent huge pages not enabled in kernel

**Check**:
```bash
cat /sys/kernel/mm/transparent_hugepage/enabled
# Should show [always] or [madvise]
```

**Fix**:
```bash
# Enable THP (requires root)
echo madvise > /sys/kernel/mm/transparent_hugepage/enabled
```

### Issue: "No backend available, skipping GPU import"

**Cause**: Backend not set during initialization

**Check**: Verify Vulkan backend is loaded:
```
llama_model_loader: loaded meta data with X key-value pairs...
ggml_vulkan: loaded
```

### Issue: Cache hit rate stays at 0%

**Cause**: Cache not configured or experts not fitting

**Check**:
```bash
# Verify cache size
export LLAMA_FLASH_MOE_CACHE_SIZE_MB=128

# Check logs for cache initialization
```

### Issue: Performance worse than baseline

**Possible Causes**:
1. **Overhead from logging** - Reduce log verbosity
2. **Cache thrashing** - Increase cache size
3. **CPU MoE still active** - Verify `LLAMA_ARG_CPU_MOE=1`

**Debug**:
```bash
# Run with minimal logging
export LLAMA_FLASH_MOE_LOG_ROUTING=0

# Monitor GPU utilization
rocm-smi  # or nvidia-smi
```

---

## Expected Results

### I14 io_uring Polish

| Metric | Before | After I14 | Improvement |
|--------|--------|-----------|-------------|
| TLB misses | Baseline | -50% to -90% | 512× reduction |
| Read latency | Baseline | -5% to -15% | Lower jitter |
| Kernel overhead | Baseline | -2% to -5% | SINGLE_ISSUER |

### I10b Force-Offload

| Model | Baseline (CPU MoE) | Target (I10b) | Improvement |
|-------|-------------------|---------------|-------------|
| qwen3-235b-q4km (≤GTT) | 18 t/s (full GPU) | 15-18 t/s | Slot buffer parity |
| DeepSeek 228 GB (>GTT) | 1.8 t/s | 6-10 t/s | **3-5x speedup** |

---

## Next Steps After Testing

1. **If I14 works**: Deploy to production for 10-25% gain
2. **If I10b works on qwen3**: Test with DeepSeek
3. **If I10b works on DeepSeek**: Document 3-5x speedup achievement
4. **If issues found**: Collect logs and file bug reports

---

**Test Date**: ___________  
**Tester**: ___________  
**Results Summary**: ___________
