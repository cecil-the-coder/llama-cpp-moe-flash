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

  **Current state (stable, deployed as image `998a216`):**
  `--cpu-moe` with plain CPU buffers. Expert matmul on CPU. 9.5 t/s (qwen3-235b Q2_K).
  Models > GTT work with `--no-warmup` + disabled mmap prefetch.

  **What was attempted and why it failed:**

  1. **Full buffer_from_host_ptr import (228 GB):** Kernel OOM — importing 228 GB
     via VK_EXT_external_memory_host exceeds what the kernel allows on 125 GB RAM.

  2. **Vulkan_Host buffer type for experts (image `a888d75`):**
     Changed model loader to keep experts on Vulkan_Host instead of CPU. Expert
     data allocated via `ggml_vk_host_malloc` + alloc+copy from mmap. Expert
     MUL_MAT_ID routes to Vulkan BUT with 190 graph splits → 9.0 t/s (slower
     than baseline due to sync overhead).

  3. **supports_buft for Vulkan_Host (image `c265012`):**
     Made `ggml_backend_vk_device_supports_buft` return true for Vulkan_Host.
     Eliminated graph splits (190→1). BUT caused SIGSEGV during warmup.

     Root cause of SIGSEGV: Vulkan_Host `alloc_buffer` originally created a
     **CPU buffer context** (`ggml_backend_cpu_buffer_from_ptr` at line 13414).
     We fixed this to create a proper `ggml_backend_vk_buffer_context` with
     `dev_buffer` from the pinned memory. However, with supports_buft=true,
     ALL ops (not just MUL_MAT_ID) run in one Vulkan split. The crash happens
     BEFORE MUL_MAT_ID — in attention or early ops that process the Vulkan_Host
     output buffer or intermediate tensors.

     Key finding: `vk_tensor_offset` uses `tensor->data - vk_ptr_base` (where
     vk_ptr_base = 0x1000). ALL Vulkan buffer tensors get virtual pointers at
     0x1000+offset. `ggml_vk_host_get` searches pinned_memory by ACTUAL host
     pointer — so it never finds these virtual pointers. The fallback to
     `buf_ctx->dev_buffer` + `vk_tensor_offset` should work, but something
     in non-MoE ops fails.

  **Implemented approach: MUL_MAT_ID offload in scheduler (patch 0002)**

  Key insight: the existing `op_offload` mechanism in `backend_id_from_cur`
  already routes CPU weights to GPU, but gated by `offload_op()` which
  requires `batch_size >= 32`. For MUL_MAT_ID during single-token generation,
  `ne[2] = 1` (batch dimension), so it never triggers.

  The fix: after the batch-size-gated offload loop, add unconditional GPU
  offload specifically for `GGML_OP_MUL_MAT_ID`. Expert weights stay on plain
  CPU buffer (from `--cpu-moe`), no Vulkan_Host needed. The scheduler's
  selective expert copy (lines 1480-1564) copies only used experts from
  CPU → Vulkan compute buffer.

  Also reverted `supports_buft` for Vulkan_Host (SIGSEGV source).

  Changes:
  - `ggml/src/ggml-backend.cpp:837-847`: MUL_MAT_ID offload (8 lines)
  - `ggml/src/ggml-vulkan/ggml-vulkan.cpp:15517-15520`: removed supports_buft for Vulkan_Host
  - Patch: `patches/0002-moe-gpu-expert-offload.patch`

  **Status: needs deployment testing on cluster**
  Expected: scheduler creates graph splits at MoE boundaries (~2-4 per layer),
  selective expert copy handles data transfer, MUL_MAT_ID runs on Vulkan GPU.

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
