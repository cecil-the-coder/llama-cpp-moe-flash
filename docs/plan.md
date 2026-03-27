# MoE Flash — Implementation Plan

## Current State

**Production image: `836d36a`** (deployed via Flux on shadow node)

| Model | Size | Path | Cold TPS | Warm TPS |
|---|---|---|---|---|
| glm-4-7-flash | 17 GB | pinned alloc | ~29 | ~30 |
| qwen3-235b Q2_K | 80 GB | pinned alloc | 9.30 | ~10 |
| qwen3-235b Q4_K_M | 133 GB | mmap-wrap | 6.46 | 6.64 |
| deepseek-r1-0528 | 228 GB | mmap-wrap | ~2-3 (est) | ~3-4 (est) |

**Original baseline (`998a216`)**: Q2_K 20.4 t/s (no --cpu-moe, experts on Vulkan via pinned memory)

**Key features in patch 0002:**
- Vulkan_Host CPU buffer interface (SIGSEGV fix)
- Hybrid pinned/mmap allocation (fast for ≤ RAM, safe for > RAM)
- Partial page cache prefetch (posix_madvise WILLNEED up to MemAvail - 8 GiB)
- madvise tuning (MADV_RANDOM + MADV_HUGEPAGE for mmap'd expert data)
- Per-shard buffer_from_host_ptr fallback with page alignment
- host_malloc nullptr guard (GTT exhausted fallback)

---

## Q2_K Performance Gap: 9.30 vs 20.4 t/s

The `998a216` baseline achieved 20.4 t/s because:
- No `--cpu-moe` → expert weights stayed on Vulkan_Host (pinned memory)
- Pinned memory was allocated via `ggml_vk_host_malloc` (alloc+copy from mmap)
- MUL_MAT_ID ran on CPU but expert data was already in host-accessible Vulkan buffers
- Only 94 graph splits

Current `836d36a` gets 9.30 t/s because:
- `--cpu-moe` baked in Dockerfile → experts forced to CPU/Vulkan_Host
- `buffer_from_host_ptr` fails (maxBufferSize = 4 GiB, shard ranges ~47 GiB)
- Hybrid detects model fits in RAM → uses pinned alloc (same as 998a216)
- But 190 splits (vs 94) — the MUL_MAT_ID offload code creates extra splits

**Root cause of the gap**: `buffer_from_host_ptr` failure forces fallback to pinned alloc,
which works but creates a different buffer type (Vulkan_Host pinned vs Vulkan device).
The scheduler generates more splits for Vulkan_Host buffers.

**Root cause found**: `998a216` had `supports_buft=true` for Vulkan_Host → 94 splits.
Our `836d36a` has `supports_buft=false` → 190 splits. Re-enabling it (image `c2f139f`)
causes GPU page fault (SIGSEGV exit 139).

The issue: `998a216` used `ggml_backend_vk_buffer_context` (virtual pointers) for
Vulkan_Host. The dispatch used `buf_ctx->dev_buffer` which worked. Our fix changed to
CPU buffer interface (real host pointers) for the selective copy path. The dispatch
now uses `host_get` → finds the pinned vk_buffer → but the GPU faults reading it.

**Both paths resolve to the same physical memory** but through different Vulkan objects.
The `dev_buffer` path works; the `host_get` path doesn't. This is the same RADV
limitation as P3.8c.

**Attempted**: dual-interface buffer (image `09540b1`). Used `vk_buffer_context`
(virtual pointers) for dispatch + `ggml_backend_tensor_get` for selective copy.
Achieved 1 graph split but GPU page fault persists — RADV can't read from
`ggml_vk_host_malloc` pinned memory in compute shaders regardless of access path
(`dev_buffer` or `host_get`).

`998a216` likely worked at 20.4 t/s because `buffer_from_host_ptr` imported the
non-expert shard as a regular Vulkan buffer (not pinned), so the GPU never read
from pinned memory. With our code, `buffer_from_host_ptr` fails (maxBufferSize 4G)
→ falls to pinned alloc → GPU fault.

**Key discovery**: `998a216` also gets ~8.6 t/s with 190 splits on current GGUF files.
The original 20.4 t/s was from OLDER GGUF files where `buffer_from_host_ptr` succeeded
(shard mapping ranges were under 4 GiB maxBufferSize). The re-downloaded files have
47 GiB ranges → import always fails → 190 splits.

**Path to 20+ t/s**: make `buffer_from_host_ptr` succeed. Options:
1. **Per-tensor import** — split the 47 GiB mapping range into individual tensor
   imports (each typically a few hundred MB, well under 4 GiB maxBufferSize).
   Requires model loader changes to import tensors individually instead of as
   one buffer per shard. Most tractable option.
2. Use GGUF files with smaller shard ranges (re-quantize with different split)
3. GGML_VK_FORCE_MAX_BUFFER_SIZE — tested, driver returns ErrorOutOfDeviceMemory
   for 46 GiB even with override. The 4 GiB limit is a real driver constraint.
4. Fix chunked import GPU page fault — RADV driver bug, not in our control

---

## RADV Driver Limitations

Confirmed on both Strix Halo (shadow) and Strix Point (local):

| Feature | Status |
|---|---|
| `maxBufferSize` | 2-4 GiB (all shard ranges exceed) |
| `VK_EXT_external_memory_host` import (mmap) | Buffer creation OK, compute shader read → GPU page fault |
| `VK_EXT_external_memory_host` import (pinned) | Works on local (kernel 6.19), fails on shadow (kernel 6.18) |
| Compute shader read from pinned host memory | GPU page fault on both machines |

The GPU page fault when reading from imported/pinned memory in compute shaders
blocks all GPU expert matmul paths. Expert matmul stays on CPU (`--cpu-moe`).

**Driver comparison:**
- Shadow: Mesa 25.3.6, kernel 6.18.15-talos, Strix Halo (GFX1151)
- Local: Mesa 25.2.8, kernel 6.19.8-cachyos, Strix Point (890M)

---

## Remaining Tasks

### To test
- [ ] **deepseek-r1-0528** on `836d36a` — verify cold boot + partial prefetch
- [ ] **glm-4-7-flash** on `836d36a` — sanity check

### To investigate
- [ ] **Q2K 190 vs 94 splits** — understand why `836d36a` has more splits than `998a216`.
  Removing the MUL_MAT_ID offload code (dead code since --cpu-moe routes to CPU anyway)
  might reduce splits back to 94 → possible TPS improvement.
- [ ] **998a216 buffer_from_host_ptr success** — `998a216` showed `buffer_from_host_ptr`
  working for some shards (20.4 t/s). Why did it succeed where `836d36a` fails?
  The `998a216` patch had different Vulkan_Host alloc code (vk_buffer_context with
  virtual pointers). Maybe the import path was different.

### Cleanup
- [ ] **Strip debug/dead code from patch** — maxBufferSize logging, alignment warnings,
  chunked import, P3.8c bypass remnants, selective callback code. Target ~80 lines.
- [ ] **Update measurements.md** — consolidate final benchmark numbers

### Blocked on RADV
- [ ] **GPU expert matmul** — P3.8c bypass (2 splits) is architecturally correct.
  Blocked by GPU page fault on imported/pinned memory. If RADV fixes this:
  all models → 20+ t/s regardless of size.
- [ ] **Auto-disable --cpu-moe** — when buffer_from_host_ptr succeeds for all shards,
  skip --cpu-moe → full Vulkan speed. Blocked by same RADV limitation.

---

## Architecture

```
Model file (GGUF, mmap'd)
    │
    ├── Model ≤ RAM? ──→ Pinned alloc (ggml_vk_host_malloc + copy) → 9.7 t/s
    │                     Expert matmul on CPU (--cpu-moe)
    │
    └── Model > RAM? ──→ mmap-wrap (ggml_backend_cpu_buffer_from_ptr)
                          + Partial prefetch (MADV_WILLNEED, up to MemAvail - 8G)
                          + MADV_RANDOM + MADV_HUGEPAGE
                          Expert matmul on CPU (--cpu-moe) → 6.5 t/s
```

## Completed Phases

- **Phase 0**: Measurement & baseline (P0.1-P0.6)
- **Phase 1**: Expert file splitter tool (P1.1-P1.5)
- **Phase 2**: io_uring prefetch prototype (P2.1-P2.4)
- **Phase 3**: Vulkan integration
  - P3.1-P3.7: Page cache warming, buffer_from_host_ptr, --cpu-moe
  - P3.8: GPU expert matmul (blocked by RADV)
  - P3.9: Per-shard import + page alignment fix
  - P3.10: Selective prefetch (reverted — 3× slower from Vulkan sync stall)
  - P3.13: Hybrid pinned/mmap allocation
  - P3.14: Partial page cache prefetch
  - P3.15: madvise tuning (MADV_RANDOM + MADV_HUGEPAGE)
- **Phase 4**: Integration with inference-budget-controller (P4.1-P4.4)

## CI/CD

- Repo: `github.com/cecil-the-coder/llama-cpp-moe-flash`
- Images: `ghcr.io/cecil-the-coder/llama-cpp-moe-flash:<sha>`
- Workflow: `.github/workflows/build-image.yml` (triggers on `patches/` or `docker/`)
- eh-ops-private: `github.com/themicknugget/eh-ops-private` (Flux-managed)
- Backend: `llamacpp-vulkan-moe-flash` in `backends/kustomization.yaml`
