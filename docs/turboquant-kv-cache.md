# TQ2_KV Fused 2-bit KV Cache Quantization

## Project Description and Motivation

Large mixture-of-experts (MoE) models such as Qwen3-235B and DeepSeek-R1 require
enormous KV caches at long context lengths. With a head size of 128 (HSK=128) and
FP16 storage, 8K context already consumes ~3.0 GB of KV cache memory; at 32K context
this grows to ~12.3 GB. On a system with limited VRAM (e.g., Radeon 8060S with 16 GB
unified memory shared with system RAM), this leaves little headroom for model weights
and expert activations.

TQ2_KV introduces a new 2.125 bits-per-weight (bpw) quantization type specifically
for KV cache tensors. It delivers a 7.5x memory reduction at 8K context (3.0 GB →
0.4 GB) and approximately the same ratio at longer contexts. The quantization error is
bounded by a symmetry-exploiting 2-bit encoding that maps to {-1.5d, -0.5d, +0.5d,
+1.5d}, which empirically achieves NMSE < 0.05 for attention scores at HSK=128.

The current system runs at ~6 tokens/second with `--cpu-moe`, where the CPU expert
matmul is the throughput bottleneck. At that rate, KV bandwidth is not the limiting
factor, so the immediate TPS impact is negligible (< 0.1%). The primary value is
enabling longer context at affordable memory cost, and future-proofing the pipeline
for when the RADV GPU page fault fix lands and experts move back to the GPU — at that
point, attention throughput will scale with context length, and KV bandwidth reduction
will meaningfully improve performance.

---

## Architecture Decision Summary

### Why TQ2_KV?

- **Type slot**: `GGML_TYPE_TQ2_KV = 41` — the next clean slot after `NVFP4 = 40`.
  No tombstone collision with any existing or reserved type.
- **Block layout**: `struct block_tq2_kv { float16_t d; uint8_t qs[32]; }` — exactly
  34 bytes per block of 128 elements, yielding 2.125 bpw.
- **Encoding**: 2-bit symmetric. Quantized values q ∈ {0,1,2,3} map to
  {-1.5d, -0.5d, +0.5d, +1.5d}, where d = max_abs / 1.5.
- **NaN guard**: encoder replaces NaN/Inf with 0.0 before computing max_abs, matching
  the pattern used by other GGML quantization types.
- **Scope**: HSK=128 only (the head dimension used by Qwen3-235B and DeepSeek-R1).
  Other head dimensions fall back to F16 unchanged.

### Why coopmat1 (not coopmat2)?

`VK_NV_cooperative_matrix2` is an NVIDIA-vendor extension. It is not present on RDNA4
(RADV driver). The decoder is therefore implemented on the `VK_KHR_cooperative_matrix`
(coopmat1) path, gated by `device->coopmat1_fa_support`. This is the same path used
by existing quantized KV types (e.g., Q8_0) in the flash attention shader.

The decoder lives in `flash_attn_base.glsl` as a new `dequantize4()` overload under
`#if defined(DATA_A_TQ2_KV)`, using LDS staging (shared memory) to unpack 2-bit
values into the cooperative matrix tiles before the attention computation.

### Why SET_ROWS (not ggml_vk_cpy)?

`ggml_vk_cpy` is not the KV write operation. The actual KV write dispatched by the
llama.cpp graph builder is `GGML_OP_SET_ROWS`. The encoder is therefore a new
`#if defined(DATA_A_TQ2_KV)` block inside `copy_to_quant.comp`, using the existing
SET_ROWS dispatch framework. No new `.comp` shader file is required.

---

## Memory Impact Table

| Context | F16 KV Cache | TQ2_KV Cache | Reduction |
|---------|-------------|--------------|-----------|
| 8K      | 3.0 GB      | 0.4 GB       | 7.5x      |
| 32K     | 12.3 GB     | 1.6 GB       | 7.7x      |

Figures are for Qwen3-235B / DeepSeek-R1 at HSK=128 with typical MoE layer counts.

---

## 13-Item Implementation Checklist

Items must be implemented in the order listed; later items depend on earlier ones.
The critical path is: 1 → 2 → 5 → 9 (everything unblocks after item 9 compiles).

- [x] Item 1: `ggml/include/ggml.h` — add `GGML_TYPE_TQ2_KV = 41`, increment `GGML_TYPE_COUNT` to 42. (~3 lines, no deps)
- [x] Item 2: `ggml/src/ggml-common.h` — add `QK_TQ2_KV=128`, `block_tq2_kv` struct with `static_assert` that `sizeof(block_tq2_kv) == 34`. (~4 lines, deps: 1)
- [x] Item 3: `ggml/src/ggml.c` — add type_traits entry (`blck_size=128`, `type_size=34`) plus reference CPU encode/decode functions with NaN guard. (~50 lines, deps: 1, 2)
- [x] Item 4: `common/arg.cpp` — add `GGML_TYPE_TQ2_KV` to `kv_cache_types[]` array so `-ctk tq2_kv` is accepted as a CLI argument. (~1 line, deps: 1)
- [x] Item 5: `ggml-vulkan/vulkan-shaders/types.glsl` — add `QUANT_K_TQ2_KV=128`, the block struct, and a packed16 alias under `DATA_A_TQ2_KV` guard. (~12 lines, no C deps)
- [x] Item 6: `ggml-vulkan/vulkan-shaders/flash_attn_base.glsl` — add `dequantize4()` overload for `DATA_A_TQ2_KV` with `BLOCK_SIZE=128`, `BLOCK_BYTE_SIZE=34`, LDS staging. (~20 lines, deps: 5)
- [x] Item 7: `ggml-vulkan/vulkan-shaders/copy_to_quant.comp` — add `DATA_A_TQ2_KV` quantize block: per-warp max_abs reduction, NaN guard, `d=max_abs/1.5`, 2-bit pack into `qs[]`. (~30 lines, deps: 5)
- [x] Item 8: `ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp` — register tq2_kv in the coopmat1 FA shader generation loop; add tq2_kv to the SET_ROWS type loop. (~15 lines, deps: 5, 6, 7)
- [x] Item 9: `ggml-vulkan/ggml-vulkan.cpp` — `CREATE_FA` for TQ2_KV (coopmat1 only), SET_ROWS pipeline registration, `supports_op` additions (FA gated on `coopmat1_fa_support`, SET_ROWS accepted, CPU FA rejected). (~20 lines, deps: 1, 6, 8)
- [x] Item 10: `ggml-vulkan/CMakeLists.txt` — add tq2_kv to the explicit shader list if the build system requires it (may be a no-op if shaders are globbed). (~2–5 lines, deps: 8)
- [ ] Item 11: `src/llama-context.cpp` — **no code change needed**. The existing HSK divisibility check handles TQ2_KV correctly via `ggml_blck_size(128)`. (0 lines)
- [ ] Item 12: `src/llama-graph.cpp` — **no code change needed**. SET_ROWS dispatch is handled at the backend level, not in the graph builder. (0 lines)
- [x] Item 13: `tests/test-backend-ops.cpp` — add `GGML_TYPE_TQ2_KV` to the `flash_attn_ext` `type_KV` loop for HSK=128; set NMSE threshold to 0.05. (~5 lines, deps: 1, 6, 9)

---

## Open Questions

The following questions must be resolved during implementation, not deferred to a
follow-up PR.

**Q1: Non-power-of-2 stride in GLSL**
Does GLSL handle `block_tq2_kv[ib]` array indexing at a 34-byte stride correctly?
GLSL struct array indexing may silently pad members to power-of-2 alignment, which
would make `ib * 34` compute the wrong byte offset. Resolution: measure actual struct
size in a test shader via `sizeof` equivalent or manual offset arithmetic, and add
2 bytes of padding (making the block 36 bytes) if needed.

**Q2: SET_ROWS source type**
Is the source tensor for the KV write F32 or F16? Check the existing Q8_0 `set_rows`
path in `copy_to_quant.comp` to confirm which source format the encoder must handle.
If the graph passes F32, the encoder reads `float`; if F16, it reads `float16_t` (or
`uint16_t` unpacked via `unpackHalf2x16`). The NaN guard must be applied after the
format conversion.

**Q3: LDS size with BLOCK_SIZE=128**
Verify that the LDS (shared memory) allocation for the coopmat1 dequantize4() path
with BLOCK_SIZE=128 fits within the device's 64 KB LDS limit. The staging buffer for
128 elements at FP16 is 256 bytes per subgroup; with 4 subgroups per workgroup this
is 1 KB, well within limits. Confirm this against the actual shader workgroup
configuration before enabling.

**Q4: Multi-stream i03 stride**
When `n_stream > 1`, each stream works on a different layer shard. Verify that the
i03 (layer/batch) stride in the flash attention shader is computed correctly when
`n_embd_k_gqa` is a multiple of 128. Specifically, check that the byte offset
`i03 * stride_a03` uses the quantized byte size (34 bytes/block × blocks) rather
than the element count, to avoid off-by-stride bugs in multi-stream inference.

---

## Success Criteria

The implementation is considered complete and correct when all of the following hold:

1. `tests/test-backend-ops` flash_attn_ext with `type_KV=TQ2_KV`, HSK=128:
   NMSE < 0.05 on RDNA4 (Radeon 8060S).
2. Qwen3-235B at 8K context: KV cache size reported in llama.cpp startup log drops
   from ~3 GB to ~0.4 GB.
3. No GPU faults, no hangs, no validation layer errors during a full 8K-context
   generation run.
4. No regression on existing F16 and Q8_0 flash attention tests.
5. `cmake -DGGML_VULKAN=ON` build completes cleanly with no new warnings related to
   TQ2_KV shaders or C++ code.

---

## Notes for Implementor

### Q1 Investigation: GLSL 34-byte stride

Start by checking how the Q8_0 block (18 bytes: 2-byte scale + 16-byte qs) is indexed
in `flash_attn_base.glsl`. If it uses raw byte-offset arithmetic (`layout(buffer_reference,
scalar)` or manual pointer math) rather than struct array indexing, TQ2_KV can follow
the same pattern without padding. If the existing code uses GLSL struct arrays, add
`uint8_t _pad[2]` to `block_tq2_kv` to reach 36 bytes and adjust the BLOCK_BYTE_SIZE
constant accordingly. The NMSE test (item 13) will catch any stride error as a large
numerical error.

### Q2 Investigation: SET_ROWS source type

Search `copy_to_quant.comp` for the Q8_0 or Q4_0 quantize block and read what type
is declared for the source (input) buffer. Also check `ggml-vulkan.cpp` for the
pipeline creation and push-constant layout for SET_ROWS to confirm whether the
source tensor's element type is declared as F32 or F16. If both types appear in
different code paths, TQ2_KV should handle both (or at minimum match what the
graph builder actually sends for KV cache writes in llama-graph.cpp).

### Q3 Investigation: LDS sizing

In `flash_attn_base.glsl`, find the `shared` array declarations for the coopmat1
path and add up the bytes consumed per workgroup. With 128-element blocks at FP16,
each 128-element tile is 256 bytes. The coopmat1 path typically stages one tile per
subgroup per k-iteration; at 4 subgroups/WG this is 1 KB, far below the 64 KB limit.
However, if the shader also stages Q and scale tiles simultaneously, sum all `shared`
arrays and confirm the total. Document the measured LDS usage in a comment above
the `dequantize4()` implementation.

### Q4 Investigation: Multi-stream stride

In `ggml-vulkan.cpp`, find where `op_params` or push constants are built for the
flash attention dispatch. Look for how `stride_a03` (or equivalent layer stride) is
computed — whether it is `ne03 * element_size` or `nb03` (pre-computed byte stride
from the tensor metadata). The ggml tensor `nb` fields store byte strides, not
element counts, so using `nb[3]` directly is correct regardless of type. Verify that
the SET_ROWS dispatch (item 9) also uses byte strides from `nb` rather than
re-computing from element counts, to avoid the multi-stream stride bug.
