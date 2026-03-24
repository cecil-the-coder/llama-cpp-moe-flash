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
