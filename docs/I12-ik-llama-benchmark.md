# I12: ik_llama.cpp Benchmark

**Status**: COMPLETE (2026-04-04)  
**Verdict**: Our Vulkan hybrid wins. ik_llama.cpp is ~2x slower on ≤GTT models and comparable on >GTT.  

---

## Results

### Qwen3-235B-A22B Q2_K (80 GB — fits in RAM/GTT)

| Backend | Image | Gen t/s (warm) | Prompt t/s (warm) | Notes |
|---------|-------|---------------|-------------------|-------|
| **Stock Vulkan** | `kyuz0/vulkan-radv` (b8298) | **20.72-20.74** | 20.44-20.49 | Unpatched llama.cpp, 12 threads |
| **moe-flash** | `cecil/llama-cpp-moe-flash` | **20.21-20.77** | ~20 | Our patched build, 16 threads |
| **ik_llama.cpp** | `cecil/ik-llama-cpu` | **10.47-11.46** | 10.74-15.97 | CPU-only, AVX-512, no Vulkan |

**Finding**: For models that fit in GTT, Vulkan GPU matmul is ~2x faster than ik_llama.cpp's
optimized CPU kernels (IQK + fused MoE FFN). Stock Vulkan and our moe-flash patch perform
identically — the patch overhead is zero for ≤GTT models.

### DeepSeek-R1-0528 Q2_K (228 GB — exceeds GTT)

| Backend | Gen t/s (warm) | Config | Notes |
|---------|---------------|--------|-------|
| **moe-flash** (CPU MoE) | **1.37-1.8** | Vulkan attn + CPU expert, q4_0 KV, flash_attn | Production config |
| **ik_llama.cpp** | **1.48-1.62** | CPU-only, f16 KV, flash_attn OFF | FlashMLA disabled (crash) |

**Finding**: Comparable performance. ik_llama.cpp couldn't use FlashMLA (NaN logits crash),
and the standard attention path with f16 KV cache performs similarly to our Vulkan hybrid.

---

## Key Findings

### 1. Vulkan GPU is 2x faster than CPU for ≤GTT models
The Radeon 8060S iGPU (40 CUs, RDNA 3.5) delivers 20 t/s on Qwen3-235B Q2_K, while
ik_llama.cpp's optimized CPU kernels (AVX-512 + IQK + fused MoE FFN) reach only 11 t/s.
The GPU wins despite being an integrated part.

### 2. Stock Vulkan = moe-flash for ≤GTT models
Our patch adds zero overhead when models fit in GTT. The auto-detect `--cpu-moe` logic
correctly clears the override, and full GPU offload runs at stock speed.

### 3. FlashMLA crashes on DeepSeek Q2_K over mmap
ik_llama.cpp's FlashMLA (`llm_prepare_mla: need to compute 61 wkv_b tensors`) produces
NaN logits on DeepSeek-R1-0528 Q2_K when the model is mmap'd from disk (228 GB on
125 GB RAM). The standard attention path works correctly. This is likely a bug in the
IQK FA kernels with MLA + Q2_K + demand-paged memory.

### 4. Binary missing VBMI/VNNI/BF16
The ik_llama.cpp binary was built with `-march=x86-64-v4` which only provides base
AVX-512. Zen 5 supports VBMI/VNNI/BF16 but these weren't compiled in. A rebuild with
`-march=znver4` or `-march=znver5` might improve CPU performance by 10-20%.

### 5. ik_llama.cpp has `--cpu-moe` and `--n-cpu-moe`
These flags exist in the fork for fine-grained MoE placement. Could be useful for
future hybrid approaches.

---

## Verdict

| Model size | Winner | Margin |
|-----------|--------|--------|
| ≤ GTT (120 GB) | **Vulkan (stock or moe-flash)** | 2x faster (20 vs 11 t/s) |
| > GTT (228 GB) | **Tie** | Both ~1.5-1.8 t/s |

**Recommendation**: Keep our current Vulkan hybrid architecture. ik_llama.cpp doesn't
justify a switch for any model size on this hardware. The theoretical FlashMLA advantage
couldn't be realized due to the crash.

**Future opportunity**: If ik_llama.cpp fixes the FlashMLA crash for >RAM models, re-test.
FlashMLA could potentially push DeepSeek CPU-only to 3-5 t/s, which would make it
competitive with our I10b slot buffer target.

---

## Deployment Details

### Image
- `ghcr.io/cecil-the-coder/ik-llama-cpu:111ae6e` (`:latest`)
- Built from `docker/Dockerfile.ik-llama-cpu` via `.github/workflows/build-ik-image.yml`
- Source: https://github.com/ikawrakow/ik_llama.cpp commit `d557d6c`
- Flags: `-march=x86-64-v4` (base AVX-512, CPU-only)

### Kubernetes (eh-ops-private)
- **Backend**: `ik-llama-cpu` — CPU-only, no GPU mounts, 16 threads
- **Models tested**:
  - `qwen3-235b-a22b-ik` — 80 GB, shares `qwen3-235b-a22b` modelDir
  - `deepseek-r1-0528-ik` — 228 GB, flash_attn=0, f16 KV, --cpu-moe
  - `qwen3-235b-a22b-stock` — stock `llamacpp-vulkan-moe` backend for comparison

### Raw Server Timings

**Qwen3-235B ik_llama.cpp** (4 runs):
```
eval time =    4171.61 ms /    32 tokens ( 130.36 ms/tok,  7.67 t/s)  [cold]
eval time =   12222.98 ms /   128 tokens (  95.49 ms/tok, 10.47 t/s)
eval time =   11621.83 ms /   128 tokens (  90.80 ms/tok, 11.01 t/s)
eval time =   11170.37 ms /   128 tokens (  87.27 ms/tok, 11.46 t/s)
```

**Qwen3-235B Stock Vulkan** (4 runs):
```
eval time =    1563.81 ms /    32 tokens (  48.87 ms/tok, 20.46 t/s)  [cold]
eval time =    6387.80 ms /   128 tokens (  49.90 ms/tok, 20.04 t/s)
eval time =    6176.32 ms /   128 tokens (  48.25 ms/tok, 20.72 t/s)
eval time =    6172.44 ms /   128 tokens (  48.22 ms/tok, 20.74 t/s)
```

**DeepSeek ik_llama.cpp** (3 runs, flash_attn=0, f16 KV):
```
eval time =   41405.14 ms /    64 tokens ( 646.96 ms/tok,  1.55 t/s)  [cold]
eval time =   79041.39 ms /   128 tokens ( 617.51 ms/tok,  1.62 t/s)
eval time =   82808.10 ms /   128 tokens ( 646.94 ms/tok,  1.55 t/s)
eval time =   86432.73 ms /   128 tokens ( 675.26 ms/tok,  1.48 t/s)
```
