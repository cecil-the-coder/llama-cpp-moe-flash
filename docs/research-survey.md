# MoE Inference on Constrained Hardware: Research Survey (2024-2026)

**Target platform**: AMD Radeon 8060S iGPU (RDNA 3.5), 120 GB GTT + 125 GB RAM, llama.cpp Vulkan
**Model**: DeepSeek-R1-0528 (228 GB Q2_K, 256 experts/layer, top-8 routing)
**Current performance**: 1.37 tok/s with `--cpu-moe`

---

## 1. Frameworks and Projects

### 1.1 llama.cpp (upstream)

**Status**: The baseline you are already using. `--cpu-moe` keeps expert tensors on CPU RAM while running dense/attention layers on GPU. Each decode step copies the top-K selected experts CPU->GPU via `ggml_backend_sched_compute_splits()`.

**Critical RFC -- Issue #20757**: "Two-tier GPU+RAM expert cache for MoE offload"
- https://github.com/ggml-org/llama.cpp/issues/20757
- Opened March 19, 2026 by e1n00r
- Python PoC achieves **14 tok/s steady state** (vs 0.5-1 tok/s baseline) on 8 GB GPU
- Proposes persistent GPU slot buffer with pluggable eviction policies (SLRU recommended)
- Frequency-gated admission filter: only admit experts on second miss
- Key code locations: `ggml/src/ggml-backend.cpp:1445-1564`, `common/arg.cpp:2284,2291`, `src/llama.cpp:481-509`
- Author is seeking a C++ implementer; no code has been merged yet
- **Relevance**: Directly applicable to your setup. If implemented, could dramatically improve your tok/s

**Issue #19825**: Managed SSD offloading for MoE on macOS
- https://github.com/ggml-org/llama.cpp/issues/19825
- Request for Metal-optimized SSD-to-GPU pipeline to prevent kernel panics
- Users report ik_llama.cpp fork works well for large MoE models

**Existing flags**:
- `--cpu-moe` / `--n-cpu-moe N` -- route MoE expert tensors to CPU (already used)
- `LLAMA_ARG_CPU_MOE` / `LLAMA_ARG_N_CPU_MOE` -- environment variable equivalents

### 1.2 ik_llama.cpp

**URL**: https://github.com/ikawrakow/ik_llama.cpp

Major llama.cpp fork by ikawrakow with significantly better CPU and hybrid GPU/CPU performance. This is currently the most actively developed alternative for constrained MoE inference.

**Key MoE-specific features**:
- **Smart Expert Reduction (PR 239)**: Dynamically reduces expert computation based on routing weights
- **Fused FFN ops (PR 229)**: Fuses MoE FFN operations for faster inference
- **Better MoE on CUDA (PR 283)**: Optimized CUDA kernels for MoE
- **Better MoE on Metal (PR 307)**: Optimized Metal kernels for MoE on Apple Silicon
- **Better TG for MoE on CUDA (PR 248)**: Improved token generation for MoE models

**FlashMLA-3**: Fastest known CPU-only inference for DeepSeek models
- CPU version (PR 273): Optimized MLA attention on CPU
- CUDA version (PR 247): GPU-accelerated MLA
- Critical for DeepSeek-V3/R1 which uses Multi-head Latent Attention

**Tensor overrides (PR 232)**: Hybrid GPU/CPU inference with per-tensor placement control -- allows keeping some experts on GPU while others are on CPU

**Quantization advances**:
- IQ1_KT, IQ2_KT, IQ3_KT, IQ4_KT (Trellis-based quants for better quality at low bitrates)
- IQ4_KSS, IQ2_KS (size-optimized variants)
- Q8_KV for 8-bit KV cache (PR 208)
- Hadamard transforms for K-cache (PR 1033)

**Critical limitation**: Only CPU (AVX2/ARM_NEON) and CUDA backends are fully functional. **No Vulkan support**. This makes it unsuitable for your AMD iGPU setup unless you switch to pure CPU inference (which could actually be faster than your current hybrid setup -- worth benchmarking).

**Supported models**: DeepSeek-V3/R1, Qwen3 MoE, GLM-4, Command-A, LLaMA-4, Gemma3, and many more.

### 1.3 Flash-MoE

**URL**: https://github.com/danveloper/flash-moe

A pure C/Metal inference engine that runs **Qwen3.5-397B-A17B** (397B params, 512 experts/layer) on a MacBook Pro with 48 GB RAM at **4.4+ tok/s**. The entire 209 GB model streams from SSD.

**Key techniques directly relevant to your problem**:

1. **SSD Expert Streaming**: Expert weights read from NVMe on demand via parallel `pread()` with GCD dispatch groups. Only the K=4 active experts per layer are loaded (~6.75 MB each). The OS page cache manages caching -- no custom cache needed ("Trust the OS" principle). Inspired by Apple's "LLM in a Flash" paper.

2. **FMA-Optimized Dequant Kernel**: Rearranges 4-bit dequant+matmul from `(nibble * scale + bias) * x` to `fma(nibble, scale*x, bias*x)`. The GPU FMA unit does dequant+multiply in one instruction. 12% speedup.

3. **Deferred GPU Expert Compute**: Expert forward pass submitted without waiting. GPU executes while CPU prepares next layer.

4. **Trust the OS**: Every custom caching approach they tested (Metal LRU, malloc cache, LZ4 compressed cache) was slower than letting the OS page cache handle it. Page cache achieved ~71% hit rate naturally.

**Their discarded approaches (58 experiments)**:
- LZ4 expert compression: -13% (decompress overhead > cache savings)
- F_RDADVISE prefetch: net 0% (SSD DMA slows GPU via memory controller arbitration)
- Temporal expert prediction: -18% (25% hit rate)
- MTP speculative decoding: break-even (MoE I/O scales per-token unlike dense)
- mmap expert files: -5x (per-page fault overhead on cold data)

**Pipeline per layer (4.28ms average at 4-bit)**:
```
CMD3(prev) -> CMD1: attention projections + delta-net  [1.22ms GPU]
           -> CPU: flush results                       [0.01ms CPU]
           -> CMD2: o_proj + norm + routing + shared    [0.55ms GPU]
           -> CPU: softmax + topK routing               [0.003ms]
           -> I/O: parallel pread K=4 experts           [2.41ms SSD]
           -> CMD3: expert forward + combine + norm     [0.04ms encode, DEFERRED]
```

**Relevance**: The "Trust the OS" finding and the SSD streaming pipeline architecture are directly applicable. The FMA kernel optimization concept could be ported to Vulkan compute shaders.

### 1.4 vLLM

**URL**: https://github.com/vllm-project/vllm

Production-grade inference server with extensive MoE support:

- **Expert Parallelism (EP)**: Distributes experts across multiple GPUs. Each GPU holds a subset of experts; tokens are all-to-all dispatched. Optimized for DeepSeek-V3/R1.
- **Fused MoE kernels**: `fused_moe` module with CUDA/Triton kernels. Supports FP8, BF16, FP16, INT4 quantization. The `moe_wna16` method supports INT4/INT8 weight-only quantization for MoE.
- **EPLB (Expert Parallelism Load Balancer)**: Rebalances expert placement across GPUs based on load.
- **Offloading connector**: `vllm/adapter/offloading_connector.py` supports expert offloading between GPU and CPU. Uses LRU and ARC eviction policies for KV cache offloading.
- **KV cache offloading**: `kv_offload` with LRU and ARC policies for managing KV cache between GPU and host memory.

**Limitation**: CUDA-only (NVIDIA). No Vulkan or ROCm support for your iGPU. Architecture designed for multi-GPU datacenter inference, not single constrained device.

### 1.5 SGLang

**URL**: https://github.com/sgl-project/sglang

Fast inference engine with MoE optimizations:
- Expert Parallelism support
- DeepSeek-V3/R1 day-0 support
- AMD MI300X optimizations (ROCm-based)
- Prefill-Decode (PD) disaggregation: separate prefill and decode servers, each optimized for their workload

**Limitation**: ROCm-only for AMD, requires datacenter GPUs (MI300X). Not applicable to iGPU/Vulkan.

### 1.6 DeepSpeed

**URL**: https://github.com/microsoft/DeepSpeed

Microsoft's distributed inference/training framework:
- **ZeRO-Inference**: Offloads model parameters to CPU/NVMe, claims 20x faster inference than naive offloading. Supports weight quantization (FP6, INT4) and KV cache offloading.
- **ZeroQuant**: Model compression for inference, supports MoE models.
- **FP6-centric serving**: Research on 6-bit quantization that balances quality and speed.

**Limitation**: Designed for multi-GPU datacenter setups. CUDA-focused. Concepts (ZeRO offloading, weight quantization strategies) are transferable but code is not.

### 1.7 Petals

**URL**: https://github.com/bigscience-workshop/petals

BitTorrent-style distributed inference:
- Splits model across multiple machines over the internet
- Supports Mixtral 8x22B and Llama 3.1 405B
- Each peer runs a subset of layers (or experts for MoE)
- Could theoretically distribute experts across multiple machines

**Relevance**: Conceptually interesting for MoE -- each peer could hold a subset of experts. Latency over internet makes it impractical for interactive use, but the architecture pattern of distributed expert serving is relevant.

### 1.8 AITER (AMD AI Tensor Engine)

**URL**: https://github.com/ROCm/aiter

AMD's centralized AI operator repository:
- FusedMoE kernels (including mixed-precision A4W4 via FlyDSL)
- MLA (Multi-head Latent Attention) for DeepSeek models
- KV Cache operations
- Quantization: BF16/FP16 -> FP8/INT4
- Kernels from Triton, CK (Composable Kernels), and assembly backends

**Limitation**: Targets ROCm GPUs (MI300X, MI250X, etc.). Does NOT support Vulkan or iGPU. Not directly usable for your Radeon 8060S, but the kernel techniques (fused MoE, MLA) represent the state of the art for AMD hardware and could inform Vulkan shader development.

---

## 2. Expert Quantization and Compression

### 2.1 Quantization Formats

**FP8 (E4M3/E5M2)**: 8-bit floating point, supported by vLLM, SGLang, AITER. Best quality at 8-bit but requires hardware FP8 support. AMD RDNA 3.5 iGPU does not have native FP8 matrix units (that's RDNA 4 / CDNA).

**INT4 Weight-Only (W4A16)**: 4-bit weights, 16-bit activations. The standard for deployment quantization. vLLM's `moe_wna16` implements this for MoE experts specifically.

**MXFP4 / NVFP4**: Microscaling FP4 formats (OCP standard). Supported in newer vLLM versions. Block-scaled 4-bit with better quality than naive INT4.

**IQ (Integer Quantization) family from ik_llama.cpp**:
- IQ1_KT, IQ2_KT, IQ3_KT, IQ4_KT: Trellis-based quantization using trellis-coded modulation for better quality at low bitrates
- IQ4_KSS, IQ2_KS: Size-optimized variants with K-quant superblocks
- These consistently outperform standard Q2_K/Q3_K at equivalent sizes

**TQ2_KV**: Your project's 2-bit KV cache quantization (found to degrade quality significantly in testing)

**Q8_KV**: ik_llama.cpp's 8-bit KV cache quantization (PR 208) -- better quality trade-off

### 2.2 MoE-Specific Quantization Strategies

**Asymmetric expert quantization**: Since MoE routing is skewed, frequently-used experts could use higher precision while rarely-used experts use lower precision. No mainstream framework implements this yet, but it's discussed in academic literature.

**Expert pruning/merging**: "Smart Expert Reduction" (ik_llama.cpp PR 239) dynamically reduces computation based on routing weights. Conceptually related to expert pruning where rarely-activated experts are removed or merged.

**Shared-expert optimization**: DeepSeek-V3/R1 has a shared expert active for every token. This can always be kept in GPU memory since it's always needed, unlike routed experts.

### 2.3 Quantization Recommendations for Your Setup

With 228 GB Q2_K on 120 GB GTT + 125 GB RAM:
- Consider **IQ2_KT or IQ2_KS** from ik_llama.cpp for better quality at similar size
- **Q8_KV** for KV cache instead of TQ2_KV for acceptable memory savings without quality loss
- If you can tolerate slightly larger files, **IQ3_KT** at ~3 bits would dramatically improve quality while still fitting in RAM
- The shared expert should always be on GPU; only routed experts need the cache

---

## 3. Memory Reduction Techniques

### 3.1 KV Cache Optimization

**KV Cache Quantization**:
- Q8_KV (ik_llama.cpp): 8-bit KV cache, good quality/speed trade-off
- TQ2_KV (your project): 2-bit, poor quality in practice
- FP8 KV cache (vLLM/AITER): Best 8-bit option on supported hardware

**MLA (Multi-head Latent Attention)**: DeepSeek-V3/R1 uses MLA which compresses KV cache to a latent representation. FlashMLA (ik_llama.cpp PR 273, 247) optimizes this for CPU and CUDA. The latent KV is much smaller than standard KV, effectively reducing memory requirements.

**KV Cache Offloading**: vLLM supports offloading KV cache to CPU with LRU/ARC policies. For long contexts, this can save significant GPU memory.

### 3.2 Expert Memory Management

**Two-Tier Expert Cache (llama.cpp #20757)**:
The most directly relevant proposal:
```
Tier 1: GPU VRAM  -- persistent slot buffer, N fixed-address expert slots, SLRU eviction
Tier 2: CPU RAM   -- pinned memory, backing store for all local experts
Tier 3: SSD/mmap  -- full weight tensor, demand-paged by OS
```
PoC results: 0.5-1 tok/s baseline -> 14 tok/s steady state (on 8 GB GPU)

**"Trust the OS" (Flash-MoE)**:
Counter-intuitively, letting the OS page cache manage expert data outperformed all custom caching approaches. The OS page cache achieved 71% hit rate naturally on their workload. Custom Metal LRU, malloc cache, and LZ4 compression all performed worse.

**Pinned Memory for Tier 2**: Using `cudaMallocHost`/`mlock` for CPU-side expert data enables faster async copies to GPU. On UMA (your setup), this is less critical since GPU and CPU share physical memory, but the allocation strategy still matters for avoiding page faults during transfers.

### 3.3 Memory Layout Optimizations

**Expert file clustering**: Flash-MoE tried clustering expert weights by access pattern on disk -- found 0% benefit because NVMe ignores scatter at 7 MB granularity. This suggests SSD-level optimization is not worth pursuing.

**mmap with madvise**: Using `POSIX_MADV_WILLNEED` after each decode step for experts just used (they're likely needed next token). `POSIX_MADV_DONTNEED` to release cold expert pages. The WILLNEED pattern exists in llama.cpp at `src/llama-mmap.cpp:436` for model load.

**Transparent Huge Pages (THP)**: For large expert allocations, THP reduces TLB pressure. Linux kernel supports `madvise(MADV_HUGEPAGE)` for explicit huge page backing. Relevant for CPU-side expert matmul where memory bandwidth is the bottleneck.

---

## 4. Novel Architectures for MoE Inference

### 4.1 Apple "LLM in a Flash" (arXiv:2312.11514)

**Paper**: "LLM in a flash: Efficient Large Language Model Inference with Limited Memory"
**Authors**: Keivan Alizadeh et al. (Apple), ACL 2024
**URL**: https://arxiv.org/abs/2312.11514

Two key techniques for running models larger than DRAM:
1. **Windowing**: Strategically reduces data transfer by reusing previously activated neurons. Exploits sparsity in activation patterns.
2. **Row-column bundling**: Increases data chunk sizes read from flash memory, aligning with flash's sequential read strengths.

Results: Run models up to 2x DRAM size, 4-5x faster inference on CPU, 20-25x faster on GPU compared to naive loading.

This paper directly inspired Flash-MoE and the broader "stream from SSD" approach to oversized model inference.

### 4.2 Block-Sparse MoE Operations (MegaBlocks)

**Paper**: "MegaBlocks: Efficient Sparse Training with Mixture of Experts" (arXiv:2211.15841)
**URL**: https://arxiv.org/abs/2211.15841

Reformulates MoE computation as block-sparse matrix operations. Instead of processing each expert independently, all expert computations are batched into a single large sparse matmul. This is primarily a training optimization but the concept applies to inference:

- **Dropless MoE**: No token dropping -- all tokens are processed by their assigned experts
- **Block-sparse GPU kernels**: Single kernel launch instead of N separate expert matmuls
- **Efficient for many-small-experts**: When experts are small (as in DeepSeek's 256 experts), the overhead of N separate kernels is significant. Batching into one sparse operation reduces launch overhead.

**Relevance to your setup**: DeepSeek-R1 has 256 experts per layer, each relatively small (~224 MB per expert at Q2_K). The MegaBlocks approach of batching expert computations into a single sparse operation could reduce kernel launch overhead significantly, especially on Vulkan where dispatch overhead is non-trivial.

### 4.3 Prefill-Decode Disaggregation

**Implemented in**: SGLang, vLLM (experimental)

Separates the inference pipeline into two specialized servers:
- **Prefill server**: Optimized for processing the initial prompt (compute-bound, processes all tokens)
- **Decode server**: Optimized for token-by-token generation (memory-bound, processes one token at a time)

For MoE, the decode server only needs to load K experts per token (vs all experts for prefill), making expert caching far more effective. The prefill server can use a different strategy (stream all experts, or process on CPU).

**Relevance**: For single-device inference, the concept translates to "don't try to optimize prefill and decode the same way." During prefill, accept slower throughput and stream experts. During decode, use the aggressive expert cache. This is essentially what the SLRU proposal in llama.cpp #20757 does with its frequency-gated admission.

### 4.4 Speculative Decoding for MoE

**Approaches**: EAGLE, MTP (Multi-Token Prediction), Medusa

Speculative decoding generates multiple candidate tokens with a small draft model, then verifies them with the large model in parallel. For dense models this gives 2-3x speedup.

**Challenge for MoE**: Each token may route to different experts, so speculation scales I/O cost per candidate token. Flash-MoE tested MTP and found it only broke even -- not worth the complexity.

**Potential bright spot**: If expert caching is effective (high hit rate), the additional I/O cost of speculative tokens is near-zero for cache hits. This makes speculative decoding more attractive as the cache improves, but it's a second-order optimization.

---

## 5. Community Developments and Practical Findings

### 5.1 Active llama.cpp MoE Issues

- **#20757**: Two-tier expert cache RFC (detailed above) -- most relevant to your work
- **#19825**: SSD offloading for MoE on macOS -- community demand for SSD streaming
- **#20140**: `--cpu-moe` KV cache corruption bug -- indicates the feature is actively used
- **#19683**: Degenerate output with MoE on CUDA -- routing bugs in specific model configurations
- **#19672**: 512-expert MoE assertion failure -- scale limits in current implementation
- **#20848**: IQ3 dequant shaders use 64 VGPRs on AMD, limiting occupancy -- directly relevant to Vulkan performance on your hardware

### 5.2 ik_llama.cpp Community

Active fork with rapid development. The community around ik_llama.cpp is focused on:
- Pushing CPU inference performance (especially for DeepSeek models with FlashMLA)
- Better quantization quality at low bitrates
- Hybrid GPU/CPU inference for constrained setups
- Metal optimizations for Apple Silicon (not relevant to you, but the techniques translate)

### 5.3 Key Lessons from Flash-MoE (58 Experiments)

| Approach | Result | Lesson |
|----------|--------|--------|
| Trust OS page cache | +38% vs custom cache | Don't fight the OS |
| FMA dequant kernel | +12% | Fused ops matter |
| LZ4 expert compression | -13% | Decompress overhead dominates |
| Temporal expert prediction | -18% | Expert access is hard to predict |
| MTP speculative decoding | break-even | MoE I/O per-token cost kills it |
| mmap expert files | -5x | Per-page fault overhead on cold data |
| GPU private buffer compression | -20% | Blit cost exceeds matvec savings |

### 5.4 AMD-Specific Vulkan Considerations

**RDNA 3.5 (your 8060S iGPU)**:
- UMA architecture: GPU and CPU share physical memory (no PCIe bottleneck)
- GTT (GART) memory is system RAM mapped as VRAM
- Vulkan compute shaders are the path for GPU acceleration
- No ROCm support -- Vulkan is the only GPU compute API available
- IQ3 shaders have VGPR pressure issues on AMD RADV (#20848) -- 40% occupancy loss
- The `VK_EXT_external_memory_host` extension allows importing host memory as Vulkan buffer, potentially useful for zero-copy expert access on UMA

**Memory bandwidth on UMA**:
- On your hardware, GTT memory IS system RAM -- there is no copy needed between CPU and GPU
- The current `--cpu-moe` approach copies expert data CPU->GPU unnecessarily on UMA
- A zero-copy approach using Vulkan's host-visible memory could eliminate expert copy overhead entirely
- This is a unique advantage of UMA that none of the CUDA/ROCm frameworks can exploit

---

## 6. Actionable Recommendations for Your Setup

### Immediate (no code changes)

1. **Benchmark ik_llama.cpp pure CPU inference** against your current hybrid Vulkan/CPU setup. With 125 GB RAM and Zen 5 AVX-512, CPU-only with FlashMLA might match or beat 1.37 tok/s, especially with their optimized MoE kernels. This is the lowest-risk path to potentially better performance.

2. **Try IQ2_KT or IQ2_KS quantization** from ik_llama.cpp if you switch to that fork. Better quality than Q2_K at similar or smaller size.

3. **Increase thread count** -- your earlier testing found 16 threads was the real win for CPU-side MoE matmul. Ensure you're using optimal thread counts.

### Short-term (code modifications to llama.cpp)

4. **Implement zero-copy expert access on UMA**: On your AMD APU, GTT memory is system RAM. Instead of copying expert data CPU->GPU, use `VK_EXT_external_memory_host` to import the host-mapped expert data directly as a Vulkan buffer. This eliminates the copy overhead that `--cpu-moe` currently incurs. The data is already in physical RAM; Vulkan just needs to know about it.

5. **Add `POSIX_MADV_WILLNEED` hints after decode**: After each decode step, advise the kernel that the just-used experts will be needed again. This is a ~10 line patch to `llama-mmap.cpp`.

6. **Allocate expert buffers with `MADV_HUGEPAGE`**: Reduces TLB pressure for the large expert memory regions during CPU-side matmul.

### Medium-term (significant development)

7. **Implement the two-tier expert cache from #20757**: This is the highest-impact change. The RFC author has proven 14x speedup. On UMA, the "copy" from Tier 2 to Tier 1 would be a simple pointer remap (no actual data movement), making the cache effectively free to populate. You'd only need:
   - A persistent GPU slot buffer for hot experts
   - A mapping from expert_id to slot_idx
   - SLRU eviction policy
   - The slot index remapping in `ggml_backend_sched_compute_splits()`

8. **Explore MegaBlocks-style batched sparse matmul**: Instead of launching 8 separate expert matmul kernels per layer (top-8 routing), batch them into a single sparse operation. Reduces Vulkan dispatch overhead, which is significant for small expert sizes.

### Long-term (architectural)

9. **Dedicated Vulkan compute shaders for MoE**: Port the fused MoE kernels from vLLM/AITER to Vulkan compute. Specifically:
   - Fused dequant+matmul (like Flash-MoE's FMA kernel)
   - Fused router + topK + expert dispatch
   - Fused expert combination (weighted sum of expert outputs)

10. **Expert pre-computation on CPU**: While GPU handles attention, CPU could pre-compute the next layer's routing and begin loading/pinning expert data. This overlaps CPU expert matmul with GPU attention computation.

---

## Source Links

- llama.cpp Issue #20757 -- Two-tier expert cache: https://github.com/ggml-org/llama.cpp/issues/20757
- llama.cpp Issue #19825 -- SSD offloading: https://github.com/ggml-org/llama.cpp/issues/19825
- llama.cpp Issue #20848 -- IQ3 VGPR pressure on AMD: https://github.com/ggml-org/llama.cpp/issues/20848
- ik_llama.cpp fork: https://github.com/ikawrakow/ik_llama.cpp
- ik_llama.cpp Smart Expert Reduction (PR 239): https://github.com/ikawrakow/ik_llama.cpp/pull/239
- ik_llama.cpp Fused FFN (PR 229): https://github.com/ikawrakow/ik_llama.cpp/pull/229
- ik_llama.cpp FlashMLA CPU (PR 273): https://github.com/ikawrakow/ik_llama.cpp/pull/273
- ik_llama.cpp FlashMLA CUDA (PR 247): https://github.com/ikawrakow/ik_llama.cpp/pull/247
- ik_llama.cpp Tensor overrides (PR 232): https://github.com/ikawrakow/ik_llama.cpp/pull/232
- ik_llama.cpp Q8_KV (PR 208): https://github.com/ikawrakow/ik_llama.cpp/pull/208
- Flash-MoE: https://github.com/danveloper/flash-moe
- "LLM in a Flash" paper: https://arxiv.org/abs/2312.11514
- MegaBlocks paper: https://arxiv.org/abs/2211.15841
- AITER (AMD): https://github.com/ROCm/aiter
- vLLM: https://github.com/vllm-project/vllm
- SGLang: https://github.com/sgl-project/sglang
- DeepSpeed: https://github.com/microsoft/DeepSpeed
- Petals: https://github.com/bigscience-workshop/petals
- VK_EXT_external_memory_host spec: https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_external_memory_host.html
