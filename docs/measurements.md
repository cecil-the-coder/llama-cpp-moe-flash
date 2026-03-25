# Phase 0 Measurements

Measured 2026-03-23 on the `shadow` node (AMD Strix Halo, Radeon 8060S RDNA4).

Model: **glm-4-7-flash** (GLM-4.7-Flash Q4_K_M, 17.2 GiB, deepseek2 arch, 64 experts, top-4)

---

## P0.5 — llama.cpp Version

| Field | Value |
|---|---|
| Build | **b8298** |
| Commit | `f90bd1dd` (tag `b8298`) |
| Compiler | GCC 15.2.1 |
| OS | Fedora 43 (container) |
| GPU | Radeon 8060S Graphics (RADV GFX1151, RDNA4) |
| Backend | Vulkan (ggml-vulkan), UMA system |

```
$ kubectl exec -n inference glm-4-7-flash -- llama-server --version
version: 8298 (f90bd1dd)
built with GNU 15.2.1 for Linux x86_64
```

---

## P0.4 — Routing Tensor Backend Assignment

**Result: ARGSORT (routing) runs on Vulkan, NOT CPU.**

The Vulkan backend in b8298 fully implements `GGML_OP_ARGSORT` (confirmed in
`ggml-vulkan.cpp` at lines 459, 9797, 12930, 15318, 16214). The source code in
`llama-graph.cpp:1302-1304` shows:

```cpp
ggml_tensor * selected_experts = ggml_argsort_top_k(ctx0, selection_probs, n_expert_used);
cb(selected_experts->src[0], "ffn_moe_argsort", il);
cb(selected_experts, "ffn_moe_topk", il);
```

Comments in the Vulkan source confirm the tensor assignment:
```
node #980 (   ARGSORT):   ffn_moe_argsort-15 (   0K) [Vulka         ]
node #439 (   ARGSORT):   ffn_moe_argsort-10 ( 256K) [Vulka         ]
```

Corroborating evidence from the scheduler:
- `graph splits = 2` (only 2 splits between backends)
- `sched copies = 1` (one tensor copy, likely the output logits)
- Vulkan compute buffer = 334.51 MiB, Vulkan_Host = 12.02 MiB

### Impact on Phase 2 Architecture

This is the "medium likelihood, high impact" risk from the risk register — **confirmed**.

The eval callback approach (`cb_eval`) intercepts tensors after computation. Since
`ffn_moe_topk` lives on Vulkan, reading the routing decision on CPU requires a
**synchronous readback** from GPU memory. On a UMA system (Strix Halo), this readback
should be fast (GTT→CPU is just a cache-line fetch, no PCIe transfer), but it still
requires a `vkMapMemory` or `ggml_backend_tensor_get` call that may stall the Vulkan
command queue.

**Action needed**: Add task P0.6 to design the readback approach before Phase 2.

---

## P0.2 — Time to First Token (Cold Start, Scale-from-Zero)

**Total TTFT: 6.50s** (from curl request to first token response)

Breakdown:
| Phase | Time |
|---|---|
| Controller routing + pod scheduling | ~2.5s |
| Model load (from page cache) | ~3.0s |
| Prompt processing (8 tokens) | 0.33s |
| First token generation | <0.001s |
| **Total** | **6.50s** |

```
$ time curl -s http://10.111.150.85:9000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"glm-4-7-flash","messages":[{"role":"user","content":"Say hello."}],"max_tokens":1}'

# Response timings:
#   prompt_n: 8, prompt_ms: 328.114 (41.0 ms/tok, 24.4 t/s)
#   predicted_n: 1
# curl TIME_TOTAL: 6.500856s
```

Pod timing (from Kubernetes conditions):
- PodScheduled: 22:51:52
- ContainerStarted: 22:51:53
- Ready: 22:51:56

Note: The model was likely still in page cache from a previous run. A truly cold start
(empty page cache) with 17.2 GiB from NVMe at ~5-7 GB/s would add ~2-3s to the
model load phase, bringing total TTFT to ~8-9s.

---

## P0.3 — Warm Generation Speed (TPS)

**Sustained generation: ~59 tokens/sec** (consistent across two runs)

### Run 1: 200 tokens
```
Prompt: 22 tokens, 88.5 t/s (11.3 ms/tok)
Generation: 200 tokens, 59.0 t/s (16.9 ms/tok)
Total gen time: 3389 ms
```

### Run 2: 500 tokens
```
Prompt: 14 tokens, 121.5 t/s (8.2 ms/tok)
Generation: 500 tokens, 58.8 t/s (17.0 ms/tok)
Total gen time: 8501 ms
```

At 59 t/s with top-4 routing across 64 experts, the generation window per token is
~17ms. The io_uring prefetch system needs to complete expert reads within this window
to be useful.

For glm-4-7-flash experts:
- Expert size: `expert_feed_forward_length = 1536`, quantized at Q4_K ≈ ~1.5 KB per expert per layer
- 4 experts × ~1.5 KB = ~6 KB per layer per token (very small)
- This is a small MoE model — the real test will be qwen3-235b-a22b

---

## P0.1 — Page Fault and NVMe I/O Profile

### Warm Inference (200 tokens)

| Metric | Before | After | Delta |
|---|---|---|---|
| Minor faults | 138,339 | 140,319 | **1,980** |
| Major faults | 1 | 1 | **0** |
| IO read bytes | 0 | 0 | **0** |
| NVMe sectors read | — | — | **0** |

**Zero major faults and zero NVMe I/O during warm generation.** The entire model
is served from the page cache.

### Memory Layout

The GGUF file is **memory-mapped** (mmap):
```
7f6a25530000-7f6a2ff59000 r--s  /models/glm-4-7-flash/zai-org_GLM-4.7-Flash-Q4_K_M.gguf
```

Process memory:
- VmRSS: 461 MB (only ~2.6% of the 17.2 GB model)
- Page cache: 28.7 GB (holds entire model + OS data)
- Free memory: 76.8 GB

### System Context (vmstat during inference)
```
procs ---memory--- --swap-- ---io--- --cpu--
 r  b  free    cache     bi  bo   us sy id
 2  0  76.8G   28.7G     0   0    1  0  99
```

### Implications for Flash MoE Design

For glm-4-7-flash (17.2 GB), the model fits entirely in page cache with room to spare.
Flash MoE streaming adds no value here — there are zero major faults.

**However**, for the target model qwen3-235b-a22b (80 GB Q2_K), the situation is different:
- 80 GB model + ~30 GB KV cache + OS = ~110 GB minimum
- System has ~105 GB total RAM (76.8 free + 28.7 cache)
- The model will not fit entirely in page cache alongside KV cache
- Major faults during generation are expected → flash MoE streaming becomes valuable

The measurement should be repeated on qwen3-235b-a22b to confirm page fault behavior
under memory pressure. This is where the io_uring prefetch system will show its value.

---

## P2.4 — io_uring Expert Reads

Tested with io_uring staging on glm-4-7-flash (CPU mode, IPC_LOCK capability).

### io_uring Configuration
- 16 staging slots × 6,291,456 bytes = 96 MB staging pool
- SQPOLL failed (needs CAP_SYS_NICE), normal mode used
- Ring queue depth: 32 SQEs
- Buffer registration: successful

### Results (7 tokens, 46 MoE layers)

| Metric | Value |
|---|---|
| Callbacks | 322 |
| io_uring submits | 322 batches |
| io_uring waits | 322 completions |
| Expert reads | 8,248 |
| Total bytes read | 15.9 GB |
| Avg submit time | **0.8 µs** |
| Avg wait time | **1,598 µs** (1.6 ms) |
| Throughput | 49.4 MB/batch, ~9.9 GB/s from page cache |

### Analysis

The 1.6ms average wait includes reading ~49 MB per batch from page cache. For cold
cache (NVMe reads at ~7 GB/s sequential), estimated wait would be ~7ms (or ~2ms with
4-deep io_uring parallelism). Both are well within the 17ms per-token window.

The submit overhead (0.8 µs) is negligible — the io_uring ring submission adds
essentially zero latency to the inference path.

---

## P4 — Full Stack Testing on qwen3-235b-a22b

Model: **qwen3-235b-a22b** Q2_K (80 GB, 2 GGUF shards, 94 layers, 92 MoE, 128 experts, top-8)

### Baseline (no flash, llamacpp-vulkan-moe)

```
Prompt: 12 tokens @ 18.2 t/s (54.9 ms/tok)
Generation: 200 tokens @ 20.9 t/s (47.9 ms/tok)
TTFT (cold start): 82.8s
```

### Flash with eval callback (llamacpp-vulkan-moe-flash, io_uring warmcache)

```
Prompt: 12 tokens @ 13.4 t/s (74.6 ms/tok)
Generation: 200 tokens @ 3.0 t/s (330.1 ms/tok)
io_uring: 55 MoE layers loaded (shard 1 only), 128 experts/layer
```

### Eval Callback Overhead Analysis

| Mode | Generation t/s | ms/token | Slowdown |
|---|---|---|---|
| Baseline (no callback) | 20.9 | 47.9 | 1.0x |
| glm-4-7-flash w/ callback | 25.3 (was 59) | 39.5 | 2.3x |
| **qwen3-235b w/ callback** | **3.0** (was 20.9) | **330.1** | **7.0x** |

**Root cause**: The `ggml_backend_sched` eval callback path (ggml-backend.cpp:1585-1622)
serializes graph execution. For each node where callback returns `ask=true`:
1. Compute a subgraph up to that node
2. `ggml_backend_synchronize()` — wait for GPU
3. Call observe callback with data
4. Resume next subgraph

With 92 MoE layers, each containing routing + expert matmul nodes, this creates ~184
GPU sync points per token instead of running the entire graph as one submission.

**Conclusion**: The `cb_eval` approach is too expensive for production use on large
models. Need a mechanism that doesn't serialize the graph.

### Mode Comparison (qwen3-235b-a22b, 200 tokens, image 1790a42)

| Mode | Prompt t/s | Gen t/s | ms/tok | vs Baseline | Notes |
|---|---|---|---|---|---|
| Baseline (no flash) | 18.2 | **20.9** | 47.9 | 1.00x | No eval callback |
| **Prefetch** (bg thread) | 18.3 | **21.0** | 47.5 | **1.00x** | Zero overhead, default mode |
| Fadvise (pre-graph all) | 17.8 | 14.0 | 71.6 | 0.67x | 21K fadvise syscalls/graph |
| Callback (eval cb) | 13.4 | 3.0 | 330.1 | 0.14x | Serializes graph execution |

**Winner: Prefetch mode** — background thread running `posix_fadvise(WILLNEED)` on
expert GGUF regions independently of inference. Zero measured overhead.

The fadvise-all mode is too chatty (55 layers × 128 experts × 3 parts = 21K syscalls
per graph compute). The callback mode serializes GPU execution via
`ggml_backend_synchronize` between every node batch.

---

## DeepSeek-R1-0528 671B Q2_K (228 GB, 5 shards)

### Multi-shard loading
```
detected 5 GGUF shards
loaded GGUF shard: 12 MoE layers, 256 experts/layer
loaded GGUF shard: 12 MoE layers, 256 experts/layer
loaded GGUF shard: 12 MoE layers, 256 experts/layer
loaded GGUF shard: 13 MoE layers, 256 experts/layer
loaded GGUF shard: 12 MoE layers, 256 experts/layer
total after merging 5 shards: 58 MoE layers
io_uring staging: 128 MB (16 slots × 8 MB), max expert size=6.3 MB
```

### CPU-only test (--n-gpu-layers 0)
Model loaded successfully despite being 1.82x RAM (228 GB / 125 GB).
```
Prompt: 10 tokens @ 0.88 t/s
Generation: 50 tokens @ 1.31 t/s (766 ms/tok)
```
The prefetch thread was active, warming page cache for expert regions.

### Vulkan test (--n-gpu-layers all, no cpu-moe) — OOM

**Node crashed.** Two root causes identified:

1. **Vulkan `buffer_from_host_ptr` hardcoded to false** (ggml-vulkan.cpp:14948).
   Despite having `VK_EXT_external_memory_host` support, the backend forces a full
   memcpy of all model weights into new Vulkan buffers. 228 GB mmap + 228 GB copy = OOM.

2. **mmap prefetch pages in entire model** (llama-model.cpp:7492).
   `init_mappings(true)` calls `posix_madvise(MADV_WILLNEED)` on all 228 GB, forcing
   the kernel to read the entire model from NVMe into page cache — exceeds 125 GB RAM.

3. **GTT limit**: Even with zero-copy `buffer_from_host_ptr`, the 120 GB GTT pool
   can't hold 228 GB of imported buffers. The RADV driver accounts imported host
   memory against GTT.

### Working configuration: --cpu-moe + --no-warmup + prefetch disabled

Three patches that together enable 228 GB on 125 GB RAM:

| Fix | What it does |
|---|---|
| **Disable mmap prefetch** for models > RAM | Skips `MADV_WILLNEED` when model > MemAvailable. Pages fault on demand instead of all at once. |
| **`--cpu-moe`** (built-in llama.cpp) | Expert tensors (`ffn_*_exps`) stay on CPU mmap. Only 7.4 GB non-expert weights go to Vulkan. |
| **`--no-warmup`** | Skips the initial forward pass. Without this, warmup pages in too many expert pages and OOMs. |

```
Memory layout:
  Vulkan0 model buffer:   7,409 MB  (non-expert weights)
  CPU_Mapped buffers:   228,597 MB  (expert weights, mmap'd, not prefetched)
  Vulkan0 KV cache:        154 MB  (MLA compressed, tiny)
  Vulkan0 compute:       3,099 MB
  Vulkan_Host compute:      44 MB
  Total Vulkan/GTT:     10,706 MB  (8.5% of 120 GB GTT)
```

### Results (Vulkan + cpu-moe + no-warmup, image f7bbf5f)

```
Prompt: 10 tokens @ 1.13 t/s
Generation: 50 tokens @ 1.37 t/s (728 ms/tok)
```

Compared to CPU-only (1.31 t/s), the Vulkan path is marginally faster (1.37 t/s)
because attention and routing run on GPU while expert matmul runs on CPU.

**Limitation**: Expert matmul runs on CPU (~9 GB/s bandwidth to expert data from page
cache) instead of Vulkan GPU (~200 GB/s GTT bandwidth on UMA). This is because
`--cpu-moe` puts expert tensors on `ggml_backend_cpu_buffer_type`, causing the
scheduler to route `MUL_MAT_ID` to the CPU backend.

**Next step**: Get expert matmul on Vulkan GPU. On UMA, CPU memory is physically
GPU-accessible. Need to either register CPU expert buffers as Vulkan host memory,
or implement the staging buffer approach from the original design.

---

## Summary

### All benchmark results

| Model | Size | Config | Prompt t/s | Gen t/s | Notes |
|---|---|---|---|---|---|
| glm-4-7-flash | 17 GB | Baseline (no flash) | 88.5 | **59.0** | Fits in RAM, no benefit |
| glm-4-7-flash | 17 GB | Prefetch thread | 88.5 | **59.0** | Zero overhead confirmed |
| glm-4-7-flash | 17 GB | Eval callback | — | **25.3** | 2.3x slower (serialized) |
| qwen3-235b-a22b | 80 GB | Baseline (no flash) | 18.2 | **20.9** | Fits in RAM |
| qwen3-235b-a22b | 80 GB | buffer_from_host_ptr (full Vulkan) | 16.3 | **20.4** | Zero-copy mmap import |
| qwen3-235b-a22b | 80 GB | Prefetch thread | 18.3 | **21.0** | Zero overhead |
| qwen3-235b-a22b | 80 GB | Eval callback | 13.4 | **3.0** | 7x slower |
| qwen3-235b-a22b | 80 GB | Fadvise (pre-graph) | 17.8 | **14.0** | 21K syscalls overhead |
| qwen3-235b-a22b | 80 GB | cpu-moe (plain CPU) | 16.5 | **9.5** | Expert matmul on CPU |
| qwen3-235b-a22b | 80 GB | cpu-moe (Vulkan_Host, 190 splits) | 18.6 | **9.0** | GPU expert matmul, split overhead |
| qwen3-235b-a22b | 80 GB | cpu-moe (Vulkan_Host, supports_buft) | — | **SIGSEGV** | 1 split but crashes in non-MoE ops |
| DeepSeek-R1-0528 | 228 GB | CPU-only | 0.88 | **1.31** | Pure CPU, mmap |
| DeepSeek-R1-0528 | 228 GB | cpu-moe + no-warmup | 1.13 | **1.37** | Vulkan attn, CPU experts |

### Key findings

1. **Models ≤ GTT (120 GB):** `buffer_from_host_ptr` zero-copy import works perfectly.
   20.4 t/s = baseline. No code changes needed beyond the Vulkan backend fix.

2. **Prefetch thread mode** is zero overhead for models ≤ RAM. Background `posix_fadvise`
   keeps expert pages warm.

3. **Eval callback** serializes GPU execution (7x slower on large models). Not viable.

4. **Models > GTT** require: `--cpu-moe` + `--no-warmup` + disabled mmap prefetch.
   Expert matmul runs on CPU at ~9.5 t/s (vs 20.9 t/s baseline). Works for any model size.

5. **GPU expert matmul with --cpu-moe (the open problem):**
   - Vulkan_Host buffer type makes experts GPU-accessible (registered in pinned_memory)
   - With 190 graph splits: 9.0 t/s (sync overhead negates GPU advantage)
   - With supports_buft to eliminate splits: SIGSEGV in non-MoE ops
   - Next: targeted scheduler change to route ONLY MUL_MAT_ID to Vulkan

6. **Vulkan_Host buffer context fix:** Changed `ggml_backend_vk_host_buffer_type_alloc_buffer`
   to create proper `ggml_backend_vk_buffer_context` (with dev_buffer) instead of CPU context.
   This is correct for MUL_MAT_ID dispatch but causes SIGSEGV when combined with
   global `supports_buft` (non-MoE ops crash).

4. **buffer_from_host_ptr** fix (alignment + conditional enable) allows zero-copy mmap
   import on UMA for models ≤ GTT (120 GB). **Validated on qwen3-235b (80 GB): 20.4 t/s
   matching baseline 20.9 t/s.** Zero additional memory allocation.

5. **Models > GTT** (228 GB on 120 GB GTT) can't import full model via buffer_from_host_ptr.
   `--cpu-moe` keeps experts on CPU mmap (no GTT consumption). Expert matmul runs on CPU.
   GPU expert matmul requires either scheduler modification or staging buffer approach.

6. **GPU expert matmul — MUL_MAT_ID offload (image `5486e15`, deployed)**:
   Added unconditional GPU offload for MUL_MAT_ID in `backend_id_from_cur`.
   Expert weights stay on CPU mmap. Scheduler creates graph splits with selective
   expert copy (only used experts copied per split). Results:

   | Model | Size vs GTT | Splits (bs=1) | TPS | Bottleneck |
   |---|---|---|---|---|
   | qwen3-235b Q2_K | 80 GB (under) | 2 | 20.4 | None (buffer_from_host_ptr) |
   | qwen3-235b Q4_K_M | 133 GB (over) | 190 | 4.71 | Split sync overhead |
   | deepseek-r1-0528 Q2_K | 228 GB (over) | 118 | 1.48 | I/O + split overhead |

   Split count = 2 × n_MoE_layers (one per MUL_MAT_ID weight tensor change).
   qwen3-235b: 92 layers × 2 = 184 + bookends = 190.
   deepseek-r1: 58 layers × 2 = 116 + bookends = 118.

7. **supports_buft approaches — all failed (OOM/SIGSEGV)**:
   - `supports_buft` for Vulkan_Host (f62620a, c265012): SIGSEGV
   - `supports_buft` for all host buffers via `buft_is_host` (b806384): OOM on node
   Root cause: supports_buft merges ALL ops into one Vulkan split. The gallocr
   allocates compute buffer for ALL weight copies simultaneously. For 130+ GB of
   expert data this exceeds available memory. supports_buft is per-buffer-type,
   not per-op — fundamentally unsuitable for incremental expert loading.

7. **GPU memory on UMA** (Strix Halo):
   - Single memory heap: 125 GB (= physical RAM)
   - Single type: DEVICE_LOCAL + HOST_VISIBLE + HOST_COHERENT + HOST_CACHED
   - GTT limit: 120 GB (kernel param `amdgpu.gttsize=122880`)
   - VRAM carve-out: 512 MB
   - Vulkan budget: 125 GB, but kernel limits to 120 GB GTT
   - `VK_EXT_external_memory_host` supported: yes (import host ptrs as Vulkan buffers)
   - `minImportedHostPointerAlignment`: 4096 bytes
