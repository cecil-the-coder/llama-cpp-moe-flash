# I12: ik_llama.cpp Benchmark

**Status**: DEPLOYED — awaiting image build + first inference  
**Goal**: Establish CPU-only performance baseline using ik_llama.cpp vs our Vulkan/CPU hybrid  

---

## Why This Matters

ik_llama.cpp is the highest-performing CPU-only llama.cpp fork, with features specifically
optimized for MoE models like DeepSeek:

- **FlashMLA** (PR 273): Multi-head Latent Attention, fastest known CPU DeepSeek inference
- **Fused MoE FFN** (PR 229): Batched expert processing, reduces overhead
- **Smart Expert Reduction** (PR 239): Skip low-weight experts dynamically
- **IQ Trellis quantization**: Better quality than Q2_K at similar size

No Vulkan support — pure CPU with AVX-512 on our Zen 5 hardware. The question:
does their CPU optimization beat our GPU/CPU hybrid for >GTT models?

## Deployment

### Image

Built via CI: `ghcr.io/cecil-the-coder/ik-llama-cpu:<sha>`

- **Dockerfile**: `docker/Dockerfile.ik-llama-cpu`
- **CI workflow**: `.github/workflows/build-ik-image.yml`
- **Source**: https://github.com/ikawrakow/ik_llama.cpp (HEAD)
- **Build flags**: AVX-512F/VBMI/VNNI/BF16, no Vulkan, no CUDA

### Kubernetes

- **Backend**: `ik-llama-cpu` in eh-ops-private `backends/ik-llama-cpu.yaml`
- **Model**: `deepseek-r1-0528-ik` in eh-ops-private `models-crd/deepseek-r1-0528-ik.yaml`
- **Shares storage**: Same `model-cache` PVC + `deepseek-r1-0528` modelDir as existing deployment
- **No GPU mounts**: Pure CPU, no `/dev/dri` or `/dev/kfd`

### Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Threads | 16 | Optimal for Zen 5 (same as I1 finding) |
| Context | 8192 | Match existing DeepSeek config |
| Flash Attention | ON | ik_llama.cpp FlashMLA should auto-enable for DeepSeek |
| KV Cache | q4_0 | Match existing config |
| Batch | 4096/1024 | Match existing config |

## Test Plan

### Benchmark Protocol

Run identical prompts against both backends and compare:

```bash
# Baseline: existing Vulkan/CPU hybrid
curl -s http://deepseek-r1-0528.inference.svc:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-r1-0528","messages":[{"role":"user","content":"Explain the Riemann hypothesis in detail."}],"max_tokens":512}' \
  | jq '.usage'

# I12: ik_llama.cpp CPU-only
curl -s http://deepseek-r1-0528-ik.inference.svc:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-r1-0528-ik","messages":[{"role":"user","content":"Explain the Riemann hypothesis in detail."}],"max_tokens":512}' \
  | jq '.usage'
```

### Metrics to Collect

1. **Generation t/s** (primary metric) — from `/metrics` endpoint or response timing
2. **Prompt processing t/s** — first token latency
3. **Memory usage** — `kubectl top pod`
4. **Cold start time** — time from pod start to `/health` ready
5. **Output quality** — verify coherent output (not garbage)

### Thread Sweep (if baseline looks promising)

```bash
# Test 12, 16, 24 threads (same as I1)
# ik_llama.cpp may have different optimal thread count
# since it's CPU-only (no GPU competing for memory bandwidth)
```

## Success Criteria

| ik_llama.cpp t/s | Verdict | Action |
|------------------|---------|--------|
| > 2.5 t/s | **Significant win** | Consider porting CPU kernels or switching to ik_llama.cpp for >GTT models |
| 1.5 - 2.5 t/s | **Comparable** | Keep current hybrid; ik_llama.cpp not worth the switch |
| < 1.5 t/s | **Our hybrid wins** | Current architecture is optimal for this hardware |

**Current baseline**: DeepSeek-R1-0528 Q2_K on Vulkan/CPU hybrid = **1.37-1.8 t/s**

## What Happens If ik_llama.cpp Wins

If ik_llama.cpp significantly outperforms our hybrid (>2.5 t/s):

1. **Short term**: Use ik_llama.cpp backend for DeepSeek (>GTT models) in production
2. **Medium term**: Investigate porting FlashMLA to our Vulkan pipeline
3. **Long term**: Hybrid approach — ik_llama.cpp CPU kernels + our Vulkan attention

## Shipping Sequence

1. Push Dockerfile + CI to llama-cpp-moe-flash `main` (triggers image build)
2. Push backend + model CRD to eh-ops-private `main` (Flux deploys)
3. Wait for image build + Flux reconciliation
4. Scale up `deepseek-r1-0528-ik` and run benchmarks
5. Record results in `docs/measurements.md`
