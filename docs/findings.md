# Key Findings: flash-moe & What Transfers to Linux/io_uring

## Source

Analyzed `danveloper/flash-moe` (cloned to `/tmp/flash-moe`):
- `CLAUDE.md` — project overview and results table
- `docs/io-and-gpu-exploration.md` — 58-experiment breakdown
- `docs/plan-async-pread-pipeline.md` — pipeline design attempts

## flash-moe Results Summary

Runs Qwen3.5-397B-A17B (209 GB at 4-bit) on a 48 GB MacBook Pro M3 Max at **4.4 tok/s**.

Key specs: 60 layers, 512 experts, K=4 active per token per layer. Expert files ~6.75 MB each
at 4-bit. Pipeline averages 4.28 ms/layer:

```
CMD3(prev) → CMD1: attention projections + delta-net  [1.22ms GPU]
           → CPU: flush results                        [0.01ms CPU]
           → CMD2: o_proj + norm + routing + shared    [0.55ms GPU]
           → CPU: softmax + topK routing               [0.003ms]
           → I/O: parallel pread K=4 experts           [2.41ms SSD]
           → CMD3: expert forward + combine + norm     [0.04ms encode, DEFERRED]
```

I/O is 56% of total wall time. GPU utilization was 2.4% — fully I/O bound.

## What Worked (Kept)

| Technique | Impact | Notes |
|---|---|---|
| **Trust the OS page cache** | +38% | Deleting the 9.8 GB Metal cache was the single biggest win |
| Parallel `pread()` (4 threads) | 9.2x over sequential | Superlinear due to NVMe command queue depth |
| 2 MB-aligned DMA buffers | +5% | Free with `posix_memalign`; 3.6x better DMA throughput |
| FMA dequant kernel | +12% GPU | Only matters if GPU compute is the bottleneck |
| Deferred CMD3 (async GPU) | Pipeline | Submit expert compute, continue to next layer's attn |
| BLAS for linear attention | +64% attn | Only applies to GatedDeltaNet layers (not standard MHA) |

## What Failed (58 experiments — key lessons)

| Technique | Result | Why |
|---|---|---|
| `mmap` + `memcpy` | **5x slower** | 240 page faults per 3.9 MB expert vs 1 `pread` call |
| Custom Metal LRU cache (9.8 GB) | **−38%** | Wired memory shrinks OS page cache; compressor thrash |
| `F_RDADVISE` prefetch (immediate) | −8% | NVMe command contention — double-issues reads |
| `F_RDADVISE` with lead-time prediction | −4% | 65–80% of predictions wrong; wastes bandwidth |
| Expert routing prediction | 25–53% accuracy | Not worth the overhead |
| LZ4 compression | −13% | Decompress overhead > cache savings |
| `F_NOCACHE` flag | +3% (2-bit only) | Avoids thrash only when working set >> page cache |
| Speculative early routing | −38% | Cache pollution + overhead |
| Spin-poll GPU wait | −23% | CPU thermal throttle competes with GPU |

### The Custom Cache Lesson (Critical)

Every cache approach made things worse once it exceeded ~500 entries. Root cause:

> Metal buffer allocations are **wired memory** (pinned). A 9.8 GB cache wired 9.8 GB of
> physical RAM, leaving only ~25 GB for the OS page cache instead of ~35 GB. The OS page
> cache (macOS CLOCK-Pro) achieves 71% hit rate naturally and has zero lookup overhead
> (it's the MMU itself). A custom cache got 55% hit rate with hash table overhead.

PostgreSQL analogy: `shared_buffers` at 25% of RAM, not 60%. Same principle.

### The mmap Lesson (Critical for Linux)

`mmap` is wrong for cold expert data. On both macOS and Linux, a 4–7 MB expert read via
`mmap` triggers ~240 individual page faults (each 16–64 KB page faulted separately). A
single `pread()` call issues one contiguous NVMe DMA command for the whole region.
Measured: `mmap` = 0.12 GB/s vs `pread` = 5.5 GB/s for cold data. **5x difference.**

> Note: llama.cpp's default `--mmap` is fine for weights already in the page cache. The
> problem is cold access. For the streaming path, use `pread` (or `io_uring`) directly.

## What Transfers to Linux/io_uring

### Direct transfers (same physics, same principle)

| Lesson | Linux equivalent |
|---|---|
| Use `pread` not `mmap` for cold data | `io_uring_prep_read` (same as pread, async) |
| 2 MB-aligned buffers | `posix_memalign(buf, 1<<21, size)` |
| Parallel reads (4 threads) | io_uring SQ depth ≥ K; submit all K SQEs at once |
| Trust the OS page cache | Don't `malloc` a custom expert cache |
| No prediction / speculative loads | Same accuracy limits on any architecture |
| Don't use `F_NOCACHE` equivalent | Linux: `O_DIRECT` — avoid unless working set >> RAM |

### Linux advantages over macOS

1. **True I/O + GPU overlap is possible.** Apple Silicon's memory controller serializes NVMe
   DMA and GPU compute (they share the bus). AMD hardware has a separate NVMe controller.
   io_uring reads genuinely run concurrently with Vulkan attention compute.

2. **io_uring SQPOLL mode** submits SQEs from a kernel thread with zero syscall overhead.
   Flash-moe's GCD dispatch requires `dispatch_group_enter/leave` overhead per expert.

3. **Larger page cache.** With 125 GB RAM (vs 48 GB), warm-cache hit rate will be much
   higher for repeated expert access patterns. The 71% hit rate flash-moe measured was on
   a 48 GB machine; this hardware could reach 90%+.

### Linux disadvantages vs macOS

1. **Slower NVMe.** Apple's "Apple Fabric" SSD hits 17.5 GB/s. NVMe Gen4 is ~7 GB/s.
   This is a 2.5x disadvantage in the cold-read path.

2. **Linux page size is 4 KB** (macOS ARM64 uses 16 KB). More page faults per expert file
   on cold access. Not relevant if using `pread`/io_uring directly.

3. **Vulkan vs Metal.** The flash-moe Metal shaders are hand-tuned for Apple's GPU tile
   architecture. Vulkan GLSL shaders for AMD RDNA3/4 don't have the same FMA fusion
   guarantees, though ROCm/Vulkan on RDNA4 (gfx1151) is quite capable.

## Unified Memory Implication (This Hardware)

On Strix Halo, system RAM = GTT = Vulkan "VRAM". This means:

- io_uring reads into a `posix_memalign` buffer land in system RAM
- That same buffer is accessible to the GPU via GTT without a copy
- The "upload to Vulkan" step is **free** (just a pointer, no DMA copy)

This is actually better than flash-moe's scenario, where expert data had to be DMA'd from
the page cache into a Metal buffer. On this hardware, the io_uring destination buffer IS
the GPU buffer.

## Model-Specific Notes

**Qwen3-235B-A22B** (deployed as `qwen3-235b-a22b`):
- 94 layers, 128 experts, K=8 active
- Q2_K quantization: ~5.9 MB per expert
- Total I/O per token: ~4.3 GB (cold), ~148 ms at warm cache

**Qwen3.5-REAP-212B-A17B** (deployed as `qwen35-reap-212b-a17b`):
- Architecture TBD (custom REAP model); estimated K=8
- IQ4_XS quantization: ~12.8 MB per expert
- Total I/O per token: ~6.4 GB (cold)

Flash streaming is only useful when model > 120 GB GTT limit or when multiple large
models need to coexist in the budget simultaneously.

---

## MoE Inference Optimization Landscape (2026-03-28)

Broad research survey of MoE inference optimization techniques, focused on
what's applicable to our setup (AMD Strix Halo UMA, Vulkan, 125 GB RAM).

### Key Finding: UMA is a Unique Advantage

On our hardware, GTT memory IS system RAM. CPU and GPU share the same physical
DRAM. The 0.45x penalty from `--cpu-moe` is NOT a bandwidth problem — it's a
compute throughput problem (12 CPU cores vs 40 GPU CUs for matmul). Both have
~120-180 GB/s memory bandwidth from the same DDR5X controller.

This means:
- "Copying" experts to GPU is page table remap, not data movement
- `buffer_from_host_ptr` zero-copy works for models ≤ GTT
- The path to GPU expert matmul for oversized models is through dynamic
  expert import, not faster copies

### CPU-Side: Zen 5 Strix Halo Characteristics

Critical discovery: the Ryzen AI MAX 385+ has a **half-width AVX-512 FPU**
(256-bit data path, 2 cycles per 512-bit operation). This is fundamentally
different from desktop Zen 5 (full 512-bit).

| Feature | Status | Impact |
|---|---|---|
| AVX-512 BF16 | Yes (half-width) | `_mm512_dpbf16_ps` — best CPU matmul path |
| AVX-512 VNNI | Yes (half-width) | INT8 dot product, already enabled |
| AVX-512 FP16 | No | — |
| AVX2 (256-bit) | Full throughput | Competitive with half-width AVX-512 |

The Q4_0 dot product path (`ggml_vec_dot_q4_0_q8_0`) has **no AVX-512
optimization** — AVX2 only. BF16 vec_dot IS AVX-512 BF16 optimized. BF16
experts would be faster on CPU despite larger size (see I13).

### Vulkan-Side: RDNA 3.5 on RADV

- **Cooperative matrix**: Enabled on RADV for all AMD archs. Already active
  in our build for all matmul and flash attention shaders.
- **Single queue**: iGPU exposes 1 queue family with 1 queue (compute+graphics).
  No async compute/transfer overlap possible. `single_queue = true`.
- **Architecture detection gap**: Code detects gfx1151 as `AMD_RDNA3` (not 3.5).
  Gets RDNA3-tuned parameters. May be suboptimal.
- **IQ3 VGPR pressure** (#20848): IQ3 dequant uses 64 VGPRs on AMD, 40%
  occupancy loss. Prefer Q4_K_M or Q8_0 for expert matmul on AMD.
- **maxStorageBufferRange = 4 GiB** (RADV): Blocks large expert tensor GPU access.
  MUL_MAT_ID src0 for Qwen3-235B Q4_K_M ≈ 10 GiB > 4 GiB limit. Required I10b
  fix to route through correct Vulkan buffer write path (see below).

### I10b: GPU MoE Matmul Crash Fix (2026-03-30)

**Problem**: `set_input_k_idxs` crashes with SIGSEGV when MoE matmul offloads to GPU.

Root cause chain:
1. `ggml_backend_sched_reserve` calls `gallocr_reserve_n` with zero-initialized
   `node_backend_ids` (swap bug: A/B arrays swapped after `sched_split_graph`)
2. Zero backend_id = Vulkan (first backend) → `k_idxs` tensor allocated in
   `Vulkan_Host` buffer instead of CPU
3. `Vulkan_Host` passes `ggml_backend_buffer_is_host()` (buft claims host = true)
   but `ggml_backend_buffer_get_base()` returns `vk_ptr_base = 0x1000` (sentinel)
4. `tensor->data = 0x1000 + gpu_offset` — writing to this address causes SIGSEGV

**Diagnostic** (patch 0007): `[K_IDXS] buft=Vulkan_Host base=0x1000 data=0x803000`
confirmed the root cause.

**Fix series** (patches 0008-0012): Multiple crash types, all same root cause.

**Type 1 — set_input writes** (patches 0008-0011): Replace direct `dst->data` write
with `ggml_backend_tensor_set()`. For `Vulkan_Host`, dispatches to `ggml_vk_buffer_write()`
which computes `vk_offset = tensor->data - 0x1000 = gpu_offset`. Covers:
- 0008: `set_input_k_idxs`
- 0009: `set_input_v_idxs`
- 0010: `set_input_kq_mask`, `set_input_k_shift`, `set_input_pos_bucket`
- 0011: 10 `set_input` functions in `llama-graph.cpp`

**Type 2 — CPU compute kernels** (patch 0012): CPU kernels (e.g. `ggml_compute_forward_get_rows`)
dereference `tensor->data` for READ/WRITE operations. Cannot use `ggml_backend_tensor_set`.
Fix: keep a `static std::unordered_map<void*, void*> s_vk_host_cpu_ptrs` mapping buffer
context → real CPU pointer from `ggml_vk_host_malloc`. Override `buffer->iface.get_base`
per-buffer to return the real CPU ptr. Works because `ggml_backend_buffer` stores
`iface` by value (mutable after `ggml_backend_buffer_init`). All Vulkan buffer ops that
compute `vk_offset = tensor->data - get_base()` continue to work correctly.

The `ggml_vk_buffer_write` offset calculation is important:
- Before 0012: `vk_offset = tensor->data - 0x1000`
- After 0012: `vk_offset = tensor->data - cpu_ptr = offset` (same result, different base)

Note: `ggml_backend_vk_buffer_set_tensor` already uses `ggml_backend_buffer_get_base(buffer)`
in the offset calculation, so changing `get_base()` fixes the path automatically.

Useful RADV env vars for debugging:
- `RADV_DEBUG=vm,syncshaders` — VA gap + shader sync for GPU fault debugging
- `ACO_DEBUG=force-waitdeps` — force wait dependencies on gfx1151
- `GGML_VK_DISABLE_COOPMAT=1` — A/B test cooperative matrix contribution

### io_uring: What to Avoid

| Technique | Verdict | Why |
|---|---|---|
| O_DIRECT | **AVOID** | Loses 90%+ page cache hit rate |
| SQPOLL | **AVOID** | Wastes a P-core for bursty workload |
| Expert prediction | **AVOID** | 25-53% accuracy, net negative (flash-moe confirmed) |
| mmap for cold experts | **AVOID** | 5× slower than pread (240 page faults per expert) |
| LZ4 compression | **AVOID** | Decompress overhead > cache savings (-13%) |

What works (see I14 for implementation):
- `IORING_REGISTER_BUFFERS` — skip pin_user_pages per read
- `IORING_SETUP_SINGLE_ISSUER` — kernel optimization for single submitter
- `MADV_HUGEPAGE` on staging pool — 512× TLB pressure reduction
- Trust OS page cache — +38% in flash-moe (71% natural hit rate)

### Related Projects

**llama.cpp #20757** (Two-tier expert cache):
- RFC with Python PoC: 0.5-1 tok/s → 14 tok/s on 8 GB VRAM
- SLRU eviction with frequency-gated admission
- Author seeking C++ implementer
- Most directly applicable optimization for our setup

**ik_llama.cpp** (ikawrakow fork):
- FlashMLA: fastest known CPU DeepSeek inference
- Fused MoE FFN, Smart Expert Reduction
- IQ Trellis quantization (IQ1_KT through IQ4_KT)
- No Vulkan support — CPU + CUDA only
- Worth benchmarking as CPU-only baseline

**Flash-MoE** (danveloper):
- 4.4 tok/s on 397B/48GB MacBook
- Key insight: "trust the OS" — custom caches all worse than page cache
- F_RDADVISE failure was Apple-specific (shared memory controller)
- Our hardware advantage: separate NVMe and GPU memory paths

**"LLM in a Flash"** (Apple, ACL 2024):
- Windowing + row-column bundling for flash-resident models
- Run models up to 2× DRAM size, 20-25× faster than naive
- Foundation for flash-moe and this project's approach
