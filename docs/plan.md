# Implementation Plan & Task Tracking

## Status Legend
- `[ ]` — not started
- `[~]` — in progress
- `[x]` — done
- `[!]` — blocked / needs decision

---

## Phase 0: Measurement & Baseline

Establish ground truth before writing any code.

### Tasks

- [x] **P0.1** — Profile expert page-fault overhead during generation
  ```bash
  # Trigger a model load (e.g., qwen3-235b-a22b)
  # In a pod on shadow, during active inference:
  perf stat -e page-faults,minor-faults,major-faults \
    -p $(pgrep llama-server) -- sleep 5

  # Also check NVMe throughput during generation:
  iostat -x 1 nvme0n1
  ```
  **Goal**: Quantify how many major faults (cold reads) occur per token after cold start
  vs warm (steady-state). If major faults are near zero at steady-state, the page cache
  is already doing its job and flash streaming adds little value.

- [x] **P0.2** — Measure time-to-first-token (TTFT) after scale-from-zero
  ```bash
  # Use the inference-budget-controller proxy endpoint
  time curl -s http://<svc>:9000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen3-235b-a22b","messages":[{"role":"user","content":"hi"}],"max_tokens":1}'
  ```
  **Goal**: Understand whether TTFT is dominated by model load I/O or container startup.

- [x] **P0.3** — Measure per-token latency (TPS) on glm-4-7-flash (warm)
  ```bash
  # Run a ~200 token generation and measure tokens/second
  ```
  **Goal**: Establish baseline TPS to compare against after any changes.

- [x] **P0.4** — Check if argsort (routing) tensor lives on CPU or Vulkan backend
  **Result**: Routing tensor (`ffn_moe_topk`, `ffn_moe_argsort`) is on **Vulkan**.
  Vulkan fully implements `GGML_OP_ARGSORT`. Confirmed via source analysis of
  `ggml-vulkan.cpp` and `llama-graph.cpp` at commit `f90bd1dd`.
  **Blocks**: Phase 2 — eval callback needs sync readback from Vulkan (see P0.6).

- [x] **P0.5** — Check llama.cpp commit in running container
  **Result**: b8298 (`f90bd1dd`), GCC 15.2.1, Fedora 43, Vulkan on Radeon 8060S (GFX1151).

- [x] **P0.6** — Design Vulkan readback approach for routing tensor
  Since P0.4 confirmed the routing tensor is on Vulkan, design the readback mechanism
  for the eval callback before Phase 2 begins. Options:
  1. `ggml_backend_tensor_get()` — synchronous readback, simplest approach
  2. `vkMapMemory` on the staging buffer — lower-level, may avoid scheduler overhead
  3. UMA pointer aliasing — on Strix Halo (UMA), GTT memory is CPU-accessible; may be
     possible to read the Vulkan buffer directly without an explicit copy
  **Goal**: Choose the readback approach with lowest latency and least disruption to the
  Vulkan command queue. Must complete within the ~17ms per-token window.

---

## Phase 1: Expert File Splitter Tool

Create the offline tool that reformats GGUF expert tensors into per-expert files.

### Tasks

- [x] **P1.1** — Understand GGUF tensor layout for MoE models
  **Results:**
  - Expert index IS the slowest-changing (last) dimension → contiguous per-expert slices
  - GLM-4.7-Flash / Qwen3: **separate** `gate_exps`, `up_exps`, `down_exps` (not merged)
  - Some models use merged `gate_up_exps`; splitter handles both patterns
  - Tensor shapes: `[n_embd, n_ff, n_expert]` for gate/up, `[n_ff, n_embd, n_expert]` for down

- [x] **P1.2** — Write `tools/split_experts.py`
  Implemented with: mmap-based extraction, both merged and separate gate/up support,
  per-expert combined files `L{NNN}_E{NNNN}.bin`, 2MB alignment padding, `manifest.json`.
  Uses only Python stdlib (no gguf package needed). Includes `--info` and `--verify` modes.

- [x] **P1.3** — Validate round-trip: split + reconstruct matches original tensor data
  `--verify` flag confirmed byte-identical match for all 2944 expert files on glm-4-7-flash.

- [x] **P1.4** — Test on glm-4-7-flash (small MoE model)
  46 MoE layers × 64 experts = 2944 files, 17.25 GB total. Split took ~30s.
  Per expert: gate (1.69MB Q4_K) + up (1.69MB Q4_K) + down (2.46MB Q6_K) = 5.84MB raw, 6MB padded.

- [x] **P1.5** — Estimate total disk usage post-split
  | Model | Layers | Experts | Per-Expert | Padded | Total Split | Original | Overhead |
  |---|---|---|---|---|---|---|---|
  | glm-4-7-flash | 46 | 64 | 5.84 MB | 6 MB | 17.25 GB | 17.2 GB | ~0% |
  | qwen3-235b-a22b Q2_K | 92 | 128 | 6.50 MB | 8 MB | 91.75 GB | 80 GB | ~15% |
  PVC: 3.6 TB total, 2.3 TB free — no space issues.
  **Note**: qwen3-235b-a22b GGUF is split across 2 shards; splitter needs multi-shard
  support (layer 54 spans both shards). TODO for P1.2 enhancement.

---

## Phase 2: io_uring Prefetch Prototype (posix_fadvise first)

Before implementing full io_uring, test whether the simpler `posix_fadvise(WILLNEED)`
approach works on Linux (flash-moe found it neutral on macOS; Linux behavior may differ).

### Tasks

- [x] **P2.1** — Fork llama.cpp at b8298
  Cloned to `/workspace/llama-cpp-moe-flash/llama.cpp`, branch `moe-flash` from tag `b8298`.

- [x] **P2.2** — Add eval callback that logs routing decisions
  Implemented `src/llama-moe-flash.cpp` with eval callback that intercepts `ffn_moe_topk`
  tensors. On UMA, `ggml_backend_tensor_get()` is a zero-overhead memcpy (confirmed in
  P0.6). Tested on cluster: 322 callbacks across 46 MoE layers, all expert IDs readable.
  Controlled by `LLAMA_FLASH_MOE_ENABLED=1` and `LLAMA_FLASH_MOE_LOG_ROUTING=1`.

- [x] **P2.3** — Add `posix_fadvise(WILLNEED)` in the callback
  Implemented in same module. After reading routing for layer N, issues
  `posix_fadvise(WILLNEED)` for next layer N+1's selected expert files.
  Tested on cluster: 7560 fadvise calls across one inference pass.
  **Decision gate result**: On glm-4-7-flash (17 GB, fits in page cache), fadvise has
  no measurable TPS impact — expected since there are zero major faults (P0.1).
  The real test needs a model under memory pressure (qwen3-235b-a22b).
  **Proceeding to P2.4** — io_uring reads are needed for the target scenario where
  experts are NOT in page cache.

- [x] **P2.4** — Add real io_uring reads (replace posix_fadvise)
  Implemented in `llama-moe-flash.cpp`, gated behind `-DGGML_IOURING=ON`.
  - 16 staging slots × 6 MB = 96 MB aligned staging pool
  - SQPOLL attempted first, falls back to normal mode
  - Buffer registration for zero-copy fixed reads
  - **Requires `IPC_LOCK` capability** (added to eh-ops-private)
  - Tested results (glm-4-7-flash, 7 tokens, CPU mode):
    | Metric | Value |
    |---|---|
    | Submit batches | 322 |
    | Total read | 15.9 GB |
    | Avg submit | 0.8 µs |
    | Avg wait | 1.6 ms (page cache) |
  - 1.6ms wait is well within the 17ms/token window

- [ ] **P2.5** — Verify correctness: outputs match baseline (same logits/tokens)
  Deferred — io_uring reads are into separate staging buffers, not used for
  inference yet (Phase 3 wires them into Vulkan). Current inference output
  is unmodified and correct by construction.

---

## Phase 3: Vulkan Integration

### Architecture Evolution

**Phase 3a (completed): Page cache warming** — works when model fits in RAM.
io_uring reads warm the page cache so mmap'd tensor access doesn't page-fault.
Zero Vulkan changes, zero inference overhead with prefetch thread.

**Phase 3b (in progress): Expert tensor offloading** — needed when model > RAM.
DeepSeek-R1-0528 (228 GB) OOM'd because llama.cpp allocates ALL tensors (including
expert weights) as Vulkan buffers at load time. On UMA, Vulkan buffers consume real
RAM, so 228 GB of Vulkan allocations on 125 GB RAM = OOM.

The fix: **keep expert tensors on CPU (mmap-only), don't allocate them in Vulkan**.
Only non-expert weights (~15 GB), KV cache (~5 GB), and staging buffers (~128 MB)
go into Vulkan memory. Expert data is streamed from mmap/disk on demand.

This requires modifying how llama.cpp assigns tensors to backends — either via the
existing `--override-tensor` mechanism or by patching the model loader to force
expert tensors to CPU.

### Tasks

- [x] **P3.1** — Page cache warming via GGUF offset reads
  Implemented `load_gguf_source()` + multi-shard support + prefetch thread.
  Works for models that fit in RAM. Zero overhead (21 t/s = baseline on qwen3-235b).

- [x] **P3.2** — Zero-overhead prefetch mode
  Background thread with `posix_fadvise(WILLNEED)` — no eval callback needed.
  Three modes: prefetch (default), fadvise, callback.

- [x] **P3.3** — Force expert tensors to CPU backend
  llama.cpp b8298 has `--cpu-moe` / `LLAMA_ARG_CPU_MOE=1`. Expert tensors stay on CPU
  mmap, only 7.4 GB non-expert weights go to Vulkan. MUL_MAT_ID runs on CPU.

- [x] **P3.4** — Disable mmap prefetch for models > RAM
  Added check: if model size > MemAvailable, skip `posix_madvise(MADV_WILLNEED)`.
  Without this, the prefetch pages in all 228 GB → OOM.

- [x] **P3.5** — Enable `buffer_from_host_ptr` with GTT-aware fallback
  Set `buffer_from_host_ptr = device->external_memory_host` in Vulkan backend caps.
  Added size check: skip import if it would exceed GTT (with 8 GB headroom).
  Added fallback in model loader: if import fails, fall back to alloc+copy.

- [x] **P3.6** — Benchmark DeepSeek-R1-0528 (228 GB)
  **Working at 1.37 t/s** with --cpu-moe + --no-warmup + disabled prefetch.
  Vulkan uses 10.7 GB (8.5% of GTT). Expert matmul on CPU.

- [x] **P3.7** — Validate buffer_from_host_ptr on qwen3-235b (80 GB < 120 GB GTT)
  **Validated: 20.4 t/s** — matches 20.9 t/s baseline. Zero-copy mmap import works
  perfectly for models ≤ GTT. Model loads with zero additional memory allocation.
  The alignment fix (round up to 4K) was needed for the import to succeed.

- [~] **P3.8** — Get expert matmul on Vulkan GPU for models > GTT

  **Current state (deployed as image `3153412`):**
  mmap-wrap + Vulkan_Host fixes + CPU matmul. No GPU offload for MoE.
  Over-GTT models load reliably on cold boot via demand-paged mmap.

  | Model | Size | Image | Splits (bs=1) | TPS | Notes |
  |---|---|---|---|---|---|
  | qwen3-235b Q2_K | 80 GB | 3153412 | 190 | 9.68 | Matches 998a216 baseline |
  | qwen3-235b Q4_K_M | 133 GB | 3153412 | 190 | 4.52→6.4 | Cold→warm, prev OOM'd |
  | deepseek-r1-0528 | 228 GB | 3153412 | 118 | 1.69→2.84 | Cold→warm, **2× prev** |
  | qwen3-235b Q2_K | 80 GB | 998a216 | 190 | 9.72 | Original baseline |
  | qwen3-235b Q4_K_M | 133 GB | 998a216 | 190 | 4.71 | Only warm cache, OOM cold |
  | deepseek-r1-0528 | 228 GB | 998a216 | 118 | 1.37 | Only warm cache, OOM cold |

  **Root cause of ALL previous SIGSEGVs (fixed in `d101d54`, carried to `3153412`):**

  Patch 0001's c265012 change created `ggml_backend_vk_buffer_context` for
  Vulkan_Host with virtual pointers (vk_ptr_base=0x1000 + offset). The
  scheduler's selective expert copy (ggml-backend.cpp:1547) dereferences
  `input->data` as a host pointer → SIGSEGV at 0x1000+offset.

  Fix: revert Vulkan_Host `alloc_buffer` to `ggml_backend_cpu_buffer_from_ptr`
  — real host pointers, `is_host=true`, `host_get` finds vk_buffer in
  `pinned_memory` (registered by `ggml_vk_host_malloc`).

  This also explains why `998a216` (no MUL_MAT_ID offload) worked but `5486e15`
  (with offload) crashed: without offload, MUL_MAT_ID stays on CPU → no
  selective expert copy → virtual pointers never dereferenced.

  Also fixed: Vulkan_Host alignment changed from `minMemoryMapAlignment` to
  `minStorageBufferOffsetAlignment` — required when expert tensors are used
  directly in Vulkan compute shaders (P3.8c bypass path).

  **Q2_K regression (20.4 → 9.55 t/s):**

  `LLAMA_ARG_CPU_MOE=1` is baked in the Dockerfile. For models ≤ GTT (like
  Q2_K 80 GB), this forces expert weights to CPU even though buffer_from_host_ptr
  could import them directly into Vulkan. The MUL_MAT_ID offload then creates
  190 splits to copy experts back to GPU. Without `--cpu-moe`, Q2_K had 94
  splits and 20.4 t/s.

  **What was attempted and ruled out:**

  1. **Full buffer_from_host_ptr import (228 GB):** exceeds 120 GB GTT.
  2. **Vulkan_Host for experts (a888d75):** 190 splits → 9.0 t/s. OOM > RAM.
  3. **supports_buft (c265012, b806384):** gallocr allocates ALL copies → OOM.
  4. **Pipeline parallelism (55060be):** n_copies=2 doubles all tensors → OOM.

  **P3.8c implementation (in `d101d54`, needs split bypass debug):**

  - [x] Scheduler: skip `need_new_split` for MUL_MAT_ID + host buffer +
        `buffer_from_host_ptr` capability (ggml-backend.cpp:1205-1217)
  - [x] Scheduler: skip copy creation under same conditions (line 1284-1293)
  - [x] Vulkan: dynamic import in tensor_subbuffer + mul_mat_id (ggml-vulkan.cpp)
  - [x] ggml.c: 4096-byte alignment for CPU buffers (page-aligned for import)
  - [x] Vulkan_Host: alignment fix (minStorageBufferOffsetAlignment)
  - [x] Vulkan_Host: revert to CPU buffer interface (SIGSEGV root cause fix)
  - [x] Offload fix: handle `src_backend_id == -1` for Vulkan_Host weights
        (backend_from_buffer returns -1 because no backend claims Vulkan_Host buft)
  - [x] Offload fix: handle `src_backend_id == -1` for Vulkan_Host weights
  - [~] **Split bypass reduces splits to 2 but SIGSEGV on inference**:
        With `kv_unified=true` (2 splits), bypass is fully active — expert
        weights stay on Vulkan_Host, no copies. But SIGSEGV on first request.

        Alignment ruled out (vkMapMemory page-aligned, gallocr correct).
        Coopmat disabled doesn't help. Bounds check added (image `2ab7da0`).

        The SIGSEGV is a GPU fault during command buffer execution. The
        vk_buffer from `host_get` is valid, offset+size within bounds,
        alignment correct — but the GPU crashes reading the pinned buffer.
        The SAME data works when memcpy'd to a Vulkan compute buffer
        (94-split selective copy path).

        **Two separate issues discovered:**

        **Issue 1: RADV GPU page fault with Vulkan_Host pinned memory**
        Pinned memory from `ggml_vk_host_malloc` cannot be read by compute
        shaders as storage buffer input on RADV/AMD UMA. Signal handler
        didn't catch SIGSEGV → GPU fault from driver. Re-importing via
        VK_EXT_external_memory_host and changing memory type (eDeviceLocal)
        doesn't help. Bypass restricted to plain CPU buffers only (not
        Vulkan_Host) via `buft_is_host + buft_device != split_device` check.

        **Issue 2: Over-GTT models can't load fresh (malloc OOM)**
        `--cpu-moe` fallback_alloc mallocs ALL expert data (~120 GB q4km,
        ~210 GB deepseek). On 125 GB RAM → OOM. Earlier 4.71/1.48 t/s
        results were from warm-cache sessions only. Image `d101d54` also
        fails on fresh boot. The mmap-wrap approach prevents this OOM
        (node survived with `b52361a`) but causes glibc heap corruption
        (`malloc(): unaligned tcache chunk detected`).

        **mmap-wrap heap corruption** (blocks over-GTT testing):
        `ggml_backend_cpu_buffer_from_ptr` wrapping mmap'd expert data.
        Corruption persists even after:
        - Reverting 4096 alignment (back to 64)
        - Aligning `first` down to TENSOR_ALIGNMENT
        - Removing mmap-wrap entirely (still crashes on later images
          due to other patch changes — but d101d54 without mmap-wrap loads
          q4km on warm cache)

        Root cause likely: `get_base` pads pointer to TENSOR_ALIGNMENT,
        shifting buffer base. Tensors placed by gallocr extend past the
        mmap boundary. Or `load_all_data` writes to incorrect addresses
        relative to the mmap-wrapped buffer.

        **RESOLVED in `2e60aff`**: Clean minimal patch with mmap-wrap + all
        fixes. The heap corruption was from earlier builds that had extra
        P3.8c bypass + debug code. The clean patch works:
        - q4km 133GB loads cold on 125GB RAM (1.35 t/s, warms to 1.42 t/s)
        - Node survives (no OOM)
        - 284 graph splits (selective expert copy, working path)

        **Key mmap-wrap changes**:
        - `ggml_backend_cpu_buffer_from_ptr` wraps mmap directly (demand-paged)
        - `buffer_from_host_ptr` failure falls to `goto mmap_wrap` for host buffers
          instead of `goto fallback_alloc` (which malloc'd all data → OOM)
        - Align `first` down to TENSOR_ALIGNMENT (32) to prevent get_base padding

        **RADV limitation confirmed**: Direct GPU access to host/pinned/imported
        memory via compute shaders doesn't work. P3.8c bypass (2 splits) is
        architecturally correct but blocked by this driver limitation.

  **Future improvements (prioritized):**

  - [x] **P3.9** — Per-shard buffer_from_host_ptr + page alignment fix
    Two issues found and fixed:

    **Issue 1**: All-or-nothing shard import. Fixed: keep successful imports,
    mmap-wrap only failed shards (in `8a37b8d`).

    **Issue 2 (ROOT CAUSE)**: `buffer_from_host_ptr` was silently failing
    for ALL shards because the pointer `addr + first` was NOT page-aligned.
    `VK_EXT_external_memory_host` requires 4096-byte alignment. The mmap base
    (`addr`) is page-aligned, but `first` (offset of first tensor in shard)
    is NOT. The alignment check at `ggml_vk_buffer_from_host_ptr` line 15586
    returned `{}` silently — no error message.

    Fixed in `663fee9`: align `first` DOWN to 4096 before passing to
    `buffer_from_host_ptr`. BUT: revealed the REAL limit — `maxBufferSize`
    is only **4 GiB** on RADV/Strix Halo. No single shard fits.

    **Issue 3 (ACTUAL LIMIT)**: `VkPhysicalDeviceMaintenance4Properties::maxBufferSize`
    = 4 GiB on RADV. All shard mapping ranges exceed this. Regular Vulkan buffers
    work via suballocation (multiple smaller buffers). `buffer_from_host_ptr` tries
    to create ONE buffer for the entire shard → exceeds 4 GiB → fails.

    Attempted fix: chunked import (3 GiB chunks, images `bb3ea90`/`71e225a`).
    **Also fails**: `ErrorOutOfDeviceMemory` at offset 0. RADV cannot import
    mmap'd virtual pages via `VK_EXT_external_memory_host` at all — the issue
    is not buffer size but the memory source. The extension only works for
    `ggml_vk_host_malloc` (pinned) pages, not mmap'd file pages.

    **Conclusion**: `buffer_from_host_ptr` cannot work for mmap'd model data
    on RADV. The current mmap-wrap + CPU matmul is the best achievable path.

    **RADV limitations summary**:
    - `maxBufferSize` = 4 GiB (all shard ranges exceed this)
    - `VK_EXT_external_memory_host` import fails for mmap'd pages (ErrorOutOfDeviceMemory)
    - Direct pinned memory read by compute shaders → GPU page fault (SIGSEGV)
    - These are driver-level limitations, not fixable from our side

  - [x] **P3.10** — Verify io_uring prefetch for mmap-wrapped buffers
    **CONFIRMED WORKING**: The prefetch thread uses `posix_fadvise(WILLNEED)`
    on the GGUF file descriptor at expert tensor offsets. Since mmap-wrap maps
    the SAME file, `fadvise` warms the SAME page cache pages. The prefetch
    thread activates ("94 layers, 128 experts/layer") and contributes to the
    4.52→6.4 t/s warmup on q4km. No changes needed.

  - [ ] **P3.11** — RADV bug report for host memory limitations
    File upstream RADV bug covering three issues:
    1. `VK_EXT_external_memory_host` fails for mmap'd pages (ErrorOutOfDeviceMemory)
    2. Compute shaders can't read from pinned host memory (GPU page fault)
    3. `maxBufferSize` = 4 GiB (limits single-buffer imports)
    Build with `VK_LAYER_KHRONOS_validation` for exact error details.
    If any of these are fixed, significant performance gains are possible.

  - [ ] **P3.12** — Disable --cpu-moe for under-GTT models
    On RADV, `buffer_from_host_ptr` fails for mmap'd data regardless of size,
    so this optimization is blocked until RADV fixes the import limitation.
    If RADV is fixed: when all shards import successfully, skip --cpu-moe
    to avoid the 190-split CPU matmul overhead → 20+ t/s for all models.

---

## Phase 4: Integration with inference-budget-controller

### Status: Mostly complete

- [x] **P4.1** — Container image with CI/CD
  Image: `ghcr.io/cecil-the-coder/llama-cpp-moe-flash:<sha>`
  GitHub Actions workflow builds on push to `patches/` or `docker/`.
  Dockerfile: multi-stage from `kyuz0/amd-strix-halo-toolboxes:vulkan-radv`.
  Baked env vars: `LLAMA_ARG_CPU_MOE=1`, `LLAMA_FLASH_MOE_ENABLED=1`, `LLAMA_FLASH_MOE_IOURING=1`.

- [x] **P4.2** — InferenceBackend CRD: `llamacpp-vulkan-moe-flash`
  Created in `eh-ops-private/kubernetes/infrastructure/inference/backends/`.
  Includes IPC_LOCK capability for io_uring. Image tag updated via git.

- [x] **P4.3** — Deployed models
  - qwen3-235b-a22b Q2_K (80 GB): on `llamacpp-vulkan-moe-flash` backend
  - qwen3-235b-a22b-q4km (133 GB): downloaded, not yet tested stable
  - deepseek-r1-0528 Q2_K (228 GB): downloaded, works with manual pod + --no-warmup
  - glm-4-7-flash (17 GB): on flash backend for testing

- [x] **P4.4** — Budget tracking: set memory request to ~30Gi for flash models
  (actual mmap'd model doesn't count against budget; only KV + compute + staging)

- [ ] **P4.5** — Fix Flux reconciliation for Helm-based models
  The `models/` directory has HelmRelease YAMLs but HelmReleases aren't being
  rendered. Models are actually managed via `models-crd/` raw InferenceModel CRDs.
  Need to update `models-crd/` files for backend changes, not `models/`.

- [ ] **P4.6** — Add `--no-warmup` to backend args
  The controller constructs CLI args from `_helpers.tpl`. Need to add `--no-warmup`
  for the flash backend to prevent OOM during warmup on large models.
  Currently only works via manual pod or env var.

---

## Decisions Needed

| # | Decision | Options | Status |
|---|---|---|---|
| D1 | Start with vkCmdCopyBuffer or shader indirection? | Copy (simpler) vs Indirect (faster) | Prefer copy first; measure |
| D2 | Per-expert files or offset-within-GGUF? | Files simpler; offsets avoid doubling disk | Files for now; offsets later |
| D3 | GGML_IOURING as build flag or runtime flag? | Build flag (no liburing dep by default) | Build flag `-DGGML_IOURING=ON` |
| D4 | Target model for initial testing? | Small MoE (glm-4-7-flash) first | Small model first |
| D5 | Merge upstream or maintain fork? | Upstream if accepted; fork otherwise | Fork initially |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Argsort tensor on Vulkan (not CPU) | **Confirmed** | High | P0.6: design readback; UMA should make this cheap |
| io_uring read latency > attention window | Medium | High | P0 measurement; fallback: larger io_uring depth |
| Vulkan buffer aliasing issues on GTT | Low | High | Test with small model first |
| Expert split doubles PVC disk usage | High | Medium | Check PVC size; use offset-in-GGUF approach if needed |
| llama.cpp upstream changes break patch | Medium | Low | Pin to a commit; rebase periodically |

---

## Reference: Useful Commands

```bash
# Check llama.cpp version in running pod
kubectl exec -n inference deploy/controller-manager -- env | grep -i llama

# Check PVC usage on shadow
kubectl exec -n inference <any-pod> -- df -h /models

# Monitor NVMe I/O during inference
# (from a privileged pod or on the node directly)
iostat -x 1 /dev/nvme0n1

# Check GTT usage
# from a pod with /dev/dri access:
cat /sys/kernel/debug/dri/0/amdgpu_gtt_mm 2>/dev/null || \
  cat /sys/class/drm/card0/device/mem_info_gtt_used 2>/dev/null

# Run flash-moe reference (macOS only, for reference):
# cd /tmp/flash-moe/metal_infer && make && ./infer --timing --tokens 20 --prompt "Hello"
```
