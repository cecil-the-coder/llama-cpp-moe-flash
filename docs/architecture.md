# llama.cpp MoE Architecture: Internals & Hook Points

## How MoE Inference Works in llama.cpp

### Graph construction (build time, once per context config)

In `src/llama-graph.cpp:build_moe_ffn()` (~line 1159), for each MoE layer:

```
selected_experts = ggml_argsort_top_k(probs, n_expert_used)
    ↓ [CPU tensor: int32[K, n_tokens] — the routing decision]

gate_up = ggml_mul_mat_id(gate_up_exps, cur, selected_experts)
    ↑ src0 = ALL expert weights: float[n_embd, n_ff*2, n_expert]  ← contiguous
    ↑ src2 = selected_experts (the ids)
```

`ggml_mul_mat_id` (GGML_OP_MUL_MAT_ID) is the key operation. It takes:
- `src0`: a single tensor containing **all** expert weights stacked on the 3rd dimension
- `src1`: the input activations
- `src2`: the routing indices (which K of N experts to use)

The Vulkan/CPU kernel iterates through selected expert indices and performs matmul only for
the K active rows of `src0`, skipping the others. But `src0->data` must point to a buffer
containing ALL expert weights.

### Tensor layout in memory

For a model like Qwen3-235B (128 experts, n_embd=4096, n_ff=1536, Q2_K):

```
src0->data → [expert_0_gate_up | expert_1_gate_up | ... | expert_127_gate_up]
              ←————————— 128 × 5.9 MB = 754 MB per layer ————————————————→
```

All 94 layers × 754 MB = ~69 GB of expert weight data, all contiguous in one Vulkan buffer.

### Execution pipeline (Vulkan path)

In `ggml/src/ggml-vulkan/ggml-vulkan.cpp`:

```
ggml_vk_mul_mat_id()
  → ggml_vk_mul_mat_id_q_f16()       (for quantized types)
      → ggml_vk_count_experts_and_clear() (count tokens per expert)
      → ggml_vk_matmul_id()            (dispatch the batched matmul shader)
```

The Vulkan shader receives a single VkBuffer for all expert weights, plus a push constant
`n_as` (= n_expert) and the expert index buffer. It uses the index to stride into the
correct slice of the weight buffer.

### The eval callback hook (already exists)

In `ggml/src/ggml-backend.cpp:ggml_backend_sched_compute_splits()`:

```c
// When callback_eval is registered:
for each node in split:
    bool need = callback_eval(t, /*ask=*/true, user_data);   // BEFORE compute
    ggml_backend_graph_compute_async(backend, &subgraph);
    ggml_backend_synchronize(backend);
    callback_eval(t, /*ask=*/false, user_data);              // AFTER compute
```

The callback signature:
```c
typedef bool (*ggml_backend_sched_eval_callback)(
    struct ggml_tensor * t,
    bool ask,        // true = "do you need this node's output?", false = "node just finished"
    void * user_data
);
```

Registered via `ggml_backend_sched_set_eval_callback()`, which is already called in
`src/llama-context.cpp:1197` with `cparams.cb_eval`.

**This is the primary hook point.** No new API needed.

### The gap window

In a MoE layer, the DAG order is:

```
[argsort_top_k]        → callback(t, ask=false): routing done, expert IDs known here
[attention Q/K/V proj] → ~1.2 ms GPU time
[attention compute]    → ~0.5 ms GPU time
[MUL_MAT_ID gate_up]   → callback(t, ask=true): expert data must be ready HERE
```

The ~1.7 ms window between routing and the first expert matmul is the overlap window for
io_uring reads. This matches flash-moe's 2.41 ms pread window on 6.75 MB experts at
17.5 GB/s SSD. On this hardware with 7 GB/s NVMe and 5.9 MB experts (Q2_K):

```
Expected I/O time = K × expert_size / bandwidth
  = 8 × 5.9 MB / 7 GB/s × (1 / parallelism_factor)
  ≈ 8 × 5.9 / 7000 × 1000 ms ÷ 4 (parallel reads)
  ≈ 1.7 ms
```

Near-perfect overlap with the attention compute window if io_uring parallel reads are used.

### Where tensor names are set (for callback identification)

In `build_moe_ffn()`:
```c
cb(selected_experts, "ffn_moe_topk", il);   // ← argsort result, named by layer index
```

The `cb` callback sets tensor names like `"ffn_moe_topk.5"` for layer 5. The eval callback
can match on this name to identify exactly when routing is complete.

## Model Weight Loading

In `src/llama-model-loader.cpp`, controlled by `use_mmap` (default: true on Linux):

```c
impl(struct llama_file * file, size_t prefetch, bool numa) {
    // Linux path:
    posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);  // hint: sequential access
    if (prefetch) flags |= MAP_POPULATE;              // optionally pre-fault
    addr = mmap(NULL, file->size(), PROT_READ, MAP_SHARED, fd, 0);
    if (prefetch > 0)
        posix_madvise(addr, prefetch, POSIX_MADV_WILLNEED);
}
```

With mmap, weights are lazy-loaded on first access. The Vulkan backend uploads mmap'd
weight data to GTT-backed Vulkan buffers at model load time
(`ggml_backend_tensor_set` → copies from mmap region to Vulkan buffer).

**After upload, the mmap is no longer needed for inference** — the Vulkan buffer in GTT
is the live copy. The mmap region stays in the page cache as a cache of the on-disk data.

## The Streaming Architecture Problem

The fundamental constraint: `src0->data` in `GGML_OP_MUL_MAT_ID` is a pointer to a
**contiguous** Vulkan buffer holding all experts. To stream individual experts:

**Option A — Full expert tensor stays in Vulkan buffer (current)**
- Pros: no changes to Vulkan shaders
- Cons: entire model must fit in GTT; no streaming benefit

**Option B — Sparse Vulkan buffer with expert slots**
- Allocate a small buffer: K slots × max_expert_size × 2 (double-buffer)
- After routing: io_uring-read K active experts into the slots
- Modify Vulkan shader: accept an indirection table (expert_id → slot_index)
- Cons: requires shader changes; expert data no longer at fixed stride

**Option C — CPU-side matmul for expert layers only**
- Keep expert weights in system RAM (not uploaded to Vulkan at all)
- After routing: use CPU GEMV for the K expert matmuls (AVX-512 available on Zen 5)
- Vulkan handles attention; CPU handles expert FFN
- Pros: no Vulkan changes needed; naturally uses page cache
- Cons: slower per-expert matmul (CPU vs GPU)

**Option D — posix_fadvise + mmap (minimal change)**
- Keep current architecture (full Vulkan buffer)
- After routing, call `posix_fadvise(WILLNEED)` on the next layer's expert regions
- OS async prefetch fills page cache in background
- Does NOT help on cold misses (fadvise is advisory, may be ignored)
- flash-moe found `F_RDADVISE` (equivalent) to be neutral or harmful on Apple

## Relevant Source Files

| File | Purpose | Lines |
|---|---|---|
| `src/llama-graph.cpp` | `build_moe_ffn()` — MoE graph construction | 2735 |
| `src/llama-context.cpp` | `cb_eval` registration, graph compute call | ~1400 |
| `src/llama-mmap.cpp` | `llama_mmap` — model weight loading via mmap | 752 |
| `src/llama-model-loader.cpp` | Weight loading, `use_mmap` flag | 1655 |
| `ggml/src/ggml-backend.cpp` | `ggml_backend_sched_compute_splits()` — eval callback | ~2100 |
| `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | `ggml_vk_mul_mat_id()` — Vulkan MoE dispatch | ~15000 |
| `ggml/src/ggml-cpu/repack.cpp` | `forward_mul_mat_id()` — CPU MoE fallback | ~5000 |
| `src/models/qwen3moe.cpp` | Qwen3 MoE model graph builder | 131 |

## Target GPU: AMD Radeon 8060S (gfx1151)

- Architecture: RDNA 3.5 (GC_11_5_0, same IP as gfx1150 but with different feature set)
- 40 Compute Units, 80 SIMDs
- PCI Device ID: `0x1586`
- Vulkan driver: RADV (Mesa, `vulkan-radv` image tag)
- Memory: GTT-backed (system RAM), no discrete VRAM
- GPU and NVMe share the same memory controller bus — but AMD's memory controller
  (unlike Apple's) uses separate agents, so NVMe DMA and GPU compute **can** overlap

When writing or modifying Vulkan shaders, target `gfx1151` / RDNA 3.5 capabilities:
- Wave32 and Wave64 support
- 128 KB LDS per CU
- FP16 matrix units (WMMA instructions via Vulkan `VK_KHR_shader_float16_int8`)
- No hardware ray tracing needed here

## llama.cpp Commit Used for Analysis

From `/tmp/llama.cpp` (shallow clone, main branch, March 2026):
```bash
git -C /tmp/llama.cpp log --oneline -1
```
