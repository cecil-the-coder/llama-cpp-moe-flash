---
name: Vec path garbage output debugging
description: Active investigation into why GPU MoE vec path (generation, bs=1) produces garbage output while batch path (prefill) works
type: project
---

# GPU MoE Vec Path Garbage Output — Active Debug

**Status**: ONGOING. Root cause NOT yet found.

## Situation

Image `1146005` (all 12 patches applied) runs Qwen3-235B Q4_K_M with `LLAMA_ARG_CPU_MOE=1` and `LLAMA_FLASH_MOE_ENABLED=1`. Patches 0008–0012 fixed crashes (exit 139 → running). But:

- **Prefill (batch path)**: produces CORRECT output ✓
- **Generation (vec path, bs=1)**: produces all `?????` garbage ✗
- Speed is 12.74 t/s (GPU IS working) vs 6.93 baseline (CPU)

## Key Facts (don't re-derive these)

- Expert tensor size: Qwen3-235B Q4_K gate/up = **~432 MiB** (128 experts × 3.38 MB each), Q6_K down = ~630 MiB. Well under 4 GiB.
- `slot_remap_mode = FALSE` for this model — the slot buffer code (plan foamy-orbiting-forest.md) activates at >4 GiB, which is NOT triggered here.
- `ggml_vk_use_mul_mat_vec_id` returns TRUE for bs=1 (generation). Uses `ggml_vk_mul_mat_vec_id_q_f16`.
- 64b indexing path NOT triggered (src0 < 4 GiB).
- Expert copy code path: normal bitset mode, copies expert `id` from `input->data + id*expert_size` to `input_cpy` at same offset.

## Test Clone

All 12 patches are applied at `/tmp/llama-patch-test/`. Key files:
- `/tmp/llama-patch-test/ggml/src/ggml-backend.cpp` — expert copy logic (lines 1490–1725)
- `/tmp/llama-patch-test/ggml/src/ggml-vulkan/ggml-vulkan.cpp` — Vulkan dispatch (vec path: lines 8387–8630)
- `/tmp/llama-patch-test/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_base.glsl` — vec shader base (expert_id→a_offset)
- `/tmp/llama-patch-test/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q4_k.comp` — Q4_K vec shader
- `/tmp/llama-patch-test/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q6_k.comp` — Q6_K vec shader

Patches live at `/workspace/llama-cpp-moe-flash/patches/0001-*.patch` through `0012-*.patch`.

## Runtime Log (from deployed pod)

```
[MOE-DBG] split=259 GPU MUL_MAT_ID node=ffn_moe_gate-86 src0=Vulkan0#blk.86.ffn_gate_exps.weight#0(buf=0x435d8810 ne=4096x1536x128) src2=ffn_moe_topk-86(buf=0x435d8740 ne=8x1)
[VEC-DBG] mul_mat_vec_id src0=...#0(buf=0x435d8810 nbytes=452984832) src1=ffn_norm-86(buf=0x435d8740) ids=ffn_moe_topk-86(buf=0x435d8740 nei0=8 nei1=1) dst=ffn_moe_gate-86(buf=0x435d8740)
[VEC-DBG] d_Qx.buf=0x435d86f0 d_ids.buf=0x40337080 d_D.buf=0x40337080
```

- `input_cpy` (the GPU expert copy buffer) is in `buf=0x435d8810`
- `d_Qx.buf=0x435d86f0` is the underlying VkBuffer handle for `input_cpy`
- IDS, src1, dst all in the main GPU compute buffer `0x435d8740`

## What Has Been Verified as Correct

1. Expert tensor size calculation — 432 MiB, not 10 GiB (earlier hypothesis was wrong)
2. `vk_tensor_offset` after patch 0012 — returns correct byte offset for both Vulkan and Vulkan_Host buffers
3. Expert copy byte math — `src_offset = id * expert_size`, `dst_offset = id * expert_size`, both correct
4. Shader `a_offset` calculation — `expert_id * (batch_stride_a / QUANT_K) = expert_id * 24576` blocks, which equals `expert_id * expert_size / sizeof(block_q4_K)` ✓
5. UMA buffer path — `ggml_vk_buffer_write_2d_async` finds source in pinned memory, issues `vkCmdCopyBuffer` at correct src/dst offsets
6. `d_Qx` subbuffer — input_cpy is a GPU device buffer (NOT pinned), so subbuffer correctly returns `{input_cpy_dev_buffer, 0, full_size}`

## Summary of Findings

### Issue
Qwen3-235B-A22B generates garbage output (`?????????`, Chinese characters like `手手手手手`, or random text) when using the Vulkan backend with MoE flash enabled.

### Root Cause
Buffer aliasing in the Vulkan backend's gallocr (buffer allocator). The gallocr assigns the same GPU buffer to multiple tensors that have non-overlapping lifetimes. For MoE operations:

1. `ffn_moe_topk-XX` generates expert indices in buffer X
2. `ffn_moe_gate-XX`, `ffn_moe_up-XX` write to the SAME buffer X (aliasing)
3. This corrupts the expert indices before `MUL_MAT_ID` can read them
4. Both vec path and batch path fail because they read corrupted indices

### Debug Evidence
```
[VEC-DBG] d_Qx.buf=0x20481140 d_ids.buf=0x20481140 d_D.buf=0x20481140
```
All buffers (input, ids, destination) point to the same address.

### Fixes Attempted

1. **Staging buffer approach**: Copy expert IDs to a staging buffer before corruption
   - Status: FAILED - Still garbage output
   - Issue: Host-visible staging buffer may not work correctly for compute shaders

2. **Skip vec path for MoE**: Force batch path by modifying `ggml_vk_use_mul_mat_vec_id()`
   - Status: PARTIAL - Vec path is skipped, but batch path has same aliasing issue
   - Issue: The corruption happens in gallocr before either path executes

3. **Disable MoE flash**: Test without MoE flash
   - Status: FAILED - Still garbage
   - Issue: The problem is in the base Vulkan backend, not MoE flash

### The Real Problem
The gallocr buffer aliasing is the root cause. The MoE expert indices tensor (`topk_ids`) has a longer lifetime than the gallocr expects, causing it to be aliased with other intermediate tensors.

### Potential Solutions

1. **Modify gallocr**: Mark `topk_ids` tensor as non-aliasable
   - Pros: Fixes root cause
   - Cons: Requires changes to llama.cpp core gallocr logic

2. **Add explicit copy**: Copy expert IDs to a dedicated buffer in the compute graph
   - Pros: Workaround without modifying gallocr
   - Cons: Performance overhead, complex graph modification

3. **Use CPU backend for MoE**: Fall back to CPU for MUL_MAT_ID operations
   - Pros: Avoids Vulkan aliasing
   - Cons: Performance impact

### Current Status
ISSUE NOT RESOLVED - requires further investigation or upstream llama.cpp fix

## Next Steps

1. **Follow the command buffer submission chain**: Find where `cpy_ctx` (from `ggml_backend_vk_set_tensor_async`) is submitted relative to the compute context. Look at `ggml_vk_synchronize` (lines 13677–13730) and how `cpy_ctx` vs compute context queue submission is ordered.

2. **Check if transfer→compute ordering is guaranteed**: On RADV/gfx1151 with a single queue, submissions to the same queue are ordered. But if separate queues (transfer vs compute) are used, there must be semaphores. Check `ctx->device->async_use_transfer_queue` — if TRUE, transfers go to a separate transfer queue and need cross-queue synchronization.

3. **Check fusion_flags**: Determine if `MUL_MAT_ID + MUL` is being fused (SCALE0 flag). If yes, check what `data_fuse0` is pointing to and whether its indexing by `expert_i0` (0–7) is correct.

4. **Alternative**: Add more debug logging — print `expert_id` values read from IDS, the actual bytes written to `input_cpy`, and the shader dispatch parameters. Push a new image via CI.

## Deploying Changes

```bash
cd /workspace/llama-cpp-moe-flash
git push  # triggers CI, builds Docker image

# Watch CI until completion (blocks until done):

gh run watch --repo cecil-the-coder/llama-cpp-moe-flash $(gh run list --repo cecil-the-coder/llama-cpp-moe-flash --limit 1 --json databaseId -q '.[0].databaseId')

# Alternative: check status without blocking:
gh run list --repo cecil-the-coder/llama-cpp-moe-flash --limit 1

# Get new tag from CI (short SHA):
gh run view --repo cecil-the-coder/llama-cpp-moe-flash <run_id> --json headSha -q '.headSha[:7]'

# Update deployment YAML and push:
# /workspace/eh-ops-private/kubernetes/infrastructure/inference/backends/llamacpp-vulkan-moe-flash-cpumoe.yaml
# Set tag: "<new_image_tag>"
# Push to eh-ops-private to deploy
```

**Tip:** Use `gh run watch` instead of polling `gh run list` - it blocks until the workflow completes and shows live progress.

## Patch 0013: Debug Logging Added

Created patch `0013-vec-path-debug-logging.patch` with enhanced logging:

1. **EXPERT-COPY logs** (backend.cpp): Shows which experts are copied, src/dst offsets, sizes
2. **VEC-DBG logs** (ggml-vulkan.cpp): Shows fusion_flags, n_fused, d_F0 buffer pointer

These logs will help verify:
- Whether experts are being copied to correct offsets
- Whether fusion is active (SCALE0 flag) and what the fusion buffer is
- Whether buffer pointers are valid

**Why:** Garbage output = systematic bug, not random. The batch path works and uses the same expert copy logic. The vec path differs only in the shader and dispatch. Synchronization between transfer and compute is the strongest remaining hypothesis.
