# io_uring Expert Prefetcher: Design

## Overview

Add async NVMe expert weight streaming to llama.cpp's MoE execution path using Linux
`io_uring`. Expert weights are stored as per-expert files and streamed into double-buffered
staging areas during the attention compute window, hiding I/O latency behind GPU work.

## Target: Option B (Sparse Vulkan Buffer with Indirection)

Chosen over Option C (CPU matmul) because:
- Vulkan matmul is ~10x faster than CPU GEMV for the expert projection sizes
- gfx1151 (Strix Halo) has strong compute; wasting it on CPU fallback is suboptimal
- The indirection table change in the Vulkan shader is localized (one GLSL file)

## Component Breakdown

### 1. Offline: Expert File Splitter (`tools/split_experts.py`)

Reads a GGUF file containing MoE expert tensors and writes each expert's weights to a
separate binary file with 2 MB alignment padding.

Input:  `model.gguf` (contains `blk.N.ffn_gate_exps`, `blk.N.ffn_up_exps`, `blk.N.ffn_down_exps`)
Output: `experts/L{layer:02d}_E{expert:04d}.bin` — packed gate+up+down, 2 MB aligned
        `experts/manifest.json` — maps (layer, expert_id) → {file, offset, size_bytes}

Key detail: store gate_proj + up_proj together (merged), then down_proj separately,
matching llama.cpp's `gate_up_exps` merged tensor layout.

File layout per expert:
```
[2MB-aligned header: 64 bytes metadata]
[gate_up weights: n_embd × n_ff × 2 × bpw/8, padded to 4096]
[down weights:    n_ff × n_embd × bpw/8, padded to 4096]
```

### 2. Runtime: Expert Staging Pool

A fixed-size staging pool with K×2 slots (double-buffered). On unified memory (GTT):
each slot is a `posix_memalign(1<<21)` allocation that is also registered as a
Vulkan external memory buffer.

```c
#define MAX_K_EXPERTS  16
#define N_BUFS         2   // double buffer

typedef struct {
    int     expert_id;          // which expert is currently in this slot
    int     layer;
    void *  data;               // 2MB-aligned host pointer
    size_t  size;               // bytes
    VkBuffer vk_buf;            // Vulkan view of same memory (GTT = zero copy)
    VkDeviceMemory vk_mem;
} expert_slot;

typedef struct {
    expert_slot slots[N_BUFS][MAX_K_EXPERTS];
    int         active_set;     // 0 or 1
} expert_staging_pool;
```

### 3. io_uring Submission Layer (`src/expert-prefetcher.cpp`)

```c
typedef struct {
    struct io_uring         ring;
    expert_staging_pool *   pool;
    const expert_manifest * manifest;   // (layer, expert_id) → file + offset
    int                     n_pending;
    int                     pending_layer;
    int32_t                 pending_ids[MAX_K_EXPERTS];
} expert_prefetcher;

// Called immediately after argsort_top_k completes (eval callback, ask=false)
void expert_prefetcher_submit(
        expert_prefetcher * pf,
        int layer,
        const int32_t * expert_ids,   // K selected expert indices
        int k) {

    int next_set = pf->pool->active_set ^ 1;

    for (int i = 0; i < k; i++) {
        const expert_file_entry * e = manifest_lookup(pf->manifest, layer, expert_ids[i]);
        expert_slot * slot = &pf->pool->slots[next_set][i];

        struct io_uring_sqe * sqe = io_uring_get_sqe(&pf->ring);
        io_uring_prep_read(sqe,
            e->fd,                  // file descriptor (kept open)
            slot->data,             // 2MB-aligned destination
            e->size,
            e->offset);
        io_uring_sqe_set_data64(sqe, i);  // tag with slot index
        slot->expert_id = expert_ids[i];
        slot->layer = layer;
    }
    io_uring_submit(&pf->ring);

    pf->n_pending = k;
    pf->pending_layer = layer;
    memcpy(pf->pending_ids, expert_ids, k * sizeof(int32_t));
}

// Called just before MUL_MAT_ID executes (eval callback, ask=true)
// Returns slot mapping: slot_index[expert_id] for the shader indirection table
void expert_prefetcher_wait(
        expert_prefetcher * pf,
        uint32_t * slot_map_out) {   // slot_map_out[expert_id] = slot_index or UINT32_MAX

    struct io_uring_cqe * cqe;
    for (int i = 0; i < pf->n_pending; i++) {
        io_uring_wait_cqe(&pf->ring, &cqe);
        GGML_ASSERT(cqe->res > 0);  // check read succeeded
        io_uring_cqe_seen(&pf->ring, cqe);
    }

    // flip active set
    pf->pool->active_set ^= 1;

    // build slot map for Vulkan shader
    memset(slot_map_out, 0xff, MAX_EXPERTS * sizeof(uint32_t));
    for (int i = 0; i < pf->n_pending; i++) {
        slot_map_out[pf->pending_ids[i]] = i;
    }
}
```

### 4. Eval Callback Hook (add to `src/llama-context.cpp`)

```c
struct flash_moe_ctx {
    expert_prefetcher *  prefetcher;
    int                  n_expert_used;
    bool                 enabled;
};

static bool flash_moe_eval_cb(ggml_tensor * t, bool ask, void * user_data) {
    auto * fctx = (flash_moe_ctx *) user_data;
    if (!fctx->enabled) return true;

    if (!ask) {
        // Node finished computing: check if this is a routing result
        if (ggml_op(t) == GGML_OP_ARGSORT && strstr(t->name, "ffn_moe_topk")) {
            int layer = parse_layer_from_name(t->name);  // "ffn_moe_topk.5" → 5
            // selected_experts tensor is on CPU — read directly
            const int32_t * ids = (const int32_t *) t->data;
            expert_prefetcher_submit(fctx->prefetcher, layer, ids, fctx->n_expert_used);
        }
    } else {
        // About to compute: if it's an expert matmul, wait for I/O
        if (t->op == GGML_OP_MUL_MAT_ID && strstr(t->name, "ffn_moe_gate_up")) {
            uint32_t slot_map[MAX_EXPERTS];
            expert_prefetcher_wait(fctx->prefetcher, slot_map);
            // patch the Vulkan push constants with the slot map
            // (requires passing slot_map to ggml_vk_mul_mat_id somehow)
        }
    }
    return true;
}
```

### 5. Vulkan Shader Modification (the hard part)

Current shader accesses expert weights with a fixed stride:
```glsl
// Current: expert weight at fixed offset in one big buffer
uint expert_row_start = expert_id * nb02;
```

Needs to become an indirect lookup via a staging table:
```glsl
// New: expert slot indirection
layout(binding = 5) readonly buffer ExpertSlotMap {
    uint slot_index[];  // slot_index[expert_id] = which staging slot
};
layout(binding = 6) readonly buffer StagingData {
    // K × expert_size staging area, indexed by slot
    float staging[];
};
uint slot = slot_index[expert_id];
uint expert_row_start = slot * staging_stride;
```

This requires:
- New GLSL shader variant in `ggml/src/ggml-vulkan/vulkan-shaders/`
- New pipeline in `ggml-vulkan.cpp` for `MUL_MAT_ID_STREAMING`
- New push constants struct `vk_mat_mat_id_streaming_push_constants`
- New buffer bindings in the descriptor set layout

**Alternatively** (simpler but slightly slower): use `vkCmdCopyBuffer` to copy each
expert slot into its correct position in the full expert tensor before dispatch. This
avoids shader changes but adds a copy step. On unified memory, this copy is a
GTT→GTT memcpy — fast but not zero-copy.

## Sequencing: Per-Token Timeline

```
Token T:

Layer N-1:
  ├─ [GPU: attention CMD1]                          ~1.2ms
  │    ├─ io_uring submit: K experts for layer N    ~0ms (async)
  │    └─ NVMe reads in flight...                   concurrent
  ├─ [GPU: attention CMD2 + routing]                ~0.55ms
  └─ [io_uring wait CQEs]                           ~0ms (already done)

Layer N:
  ├─ [GPU: MUL_MAT_ID with staging bufs]            ~0.04ms
  └─ [GPU: combine + norm]                          deferred

... repeat for each layer
```

The io_uring reads for layer N are submitted during layer N-1's attention compute.
By the time the wait is called, the reads have had 1.75ms to complete.

Expected I/O completion time for K=8 experts, 5.9 MB each at 7 GB/s NVMe:
```
8 × 5.9 MB / 7 GB/s = 6.7 ms  (sequential)
8 × 5.9 MB / 7 GB/s / 4 (parallel) = 1.7 ms  (4-deep io_uring queue)
```

With 4+ concurrent SQEs submitted to io_uring, the completion time should match the
1.75ms attention window. This is tight but achievable, especially with warm page cache
(warm NVMe reads: 30 GB/s → 0.4 ms).

## io_uring Configuration

```c
struct io_uring_params params = {0};
params.flags = IORING_SETUP_SQPOLL;        // kernel-side submission thread
params.sq_thread_idle = 2000;              // park after 2ms of idle
io_uring_queue_init_params(64, &ring, &params);

// Register all staging buffers for zero-copy fixed buffers
// (avoids page table walk on each read)
io_uring_register_buffers(&ring, iov_array, K * N_BUFS);
```

Use `io_uring_prep_read_fixed` for reads into registered buffers — eliminates per-read
`mmap`/page-walk overhead.

## Memory Layout

```
GTT (system RAM, GPU-accessible):
├── Non-expert weights (attention, norms, embeddings): ~5-10 GB
├── Expert staging pool: K × 2 × max_expert_size
│   = 8 × 2 × 13 MB (IQ4_XS) = ~208 MB                        ← tiny
└── KV cache: depends on context size

NVMe (model cache PVC):
└── Expert files: N_layers × N_experts × expert_size
    = 94 × 128 × 5.9 MB (Qwen3-235B Q2_K) = ~71 GB
```

The staging pool is tiny (~200 MB) regardless of model size — only K active experts per
layer are ever in memory simultaneously.

## Open Questions

1. **Routing tensor location**: Is `selected_experts` (the argsort output) always on the
   CPU backend, or can it be on Vulkan? If on Vulkan, we need `vkGetBufferDeviceAddress`
   or a sync read-back before we can issue io_uring. Check `llama-context.cpp` scheduler
   assignment for the argsort node.

2. **Layer pipelining**: The design above does one-layer lookahead (submit reads for layer N
   during layer N-1). Could extend to two-layer lookahead if needed, but the 1.75ms window
   should be sufficient.

3. **First layer**: Layer 0 has no previous layer to hide behind. Need a synchronous read
   (or submit layer 0's io_uring before the graph starts).

4. **Shared expert**: Some MoE architectures (Qwen3, DeepSeek) have a shared expert that
   always runs. It should be kept in the staging pool permanently (never evicted).

5. **vkCmdCopyBuffer vs shader indirection**: Start with vkCmdCopyBuffer (simpler) and
   measure overhead before committing to shader changes.
