# MoE Flash — Implementation Plan

## Current State

**Production image: `3fbe06f`** (deployed on shadow node)

**Two-backend architecture** (controller passes `spec.env` correctly):
- `moe-flash` backend: `CPU_MOE=0` → models that fit in GPU memory
- `moe-flash-cpumoe` backend: `CPU_MOE=1` → models that exceed GPU/RAM

| Model | Size | Backend | Path | TPS (confirmed) |
|---|---|---|---|---|
| glm-4-7-flash | 17 GB | moe-flash | Vulkan alloc, 2 splits | **50.57** |
| qwen3-235b Q2_K | 80 GB | moe-flash | Vulkan alloc, 2 splits | **20.77** |
| qwen3-235b Q4_K_M | 133 GB | moe-flash-cpumoe | mmap-wrap, partial prefetch | 2.96 cold → 6.0-6.3 warm |
| deepseek-r1-0528 | 228 GB | moe-flash-cpumoe | mmap-wrap, partial prefetch | pending test |

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

**ROOT CAUSE FOUND**: `VK_EXT_external_memory_host` on RADV only works for
`MAP_PRIVATE|MAP_ANONYMOUS` mmap, NOT file-backed mmap (`MAP_SHARED` or
`MAP_PRIVATE` with a file descriptor). Tested locally: 64 MB anonymous → SUCCESS,
64 MB file-backed → FAILED (ErrorOutOfDeviceMemory). This means `buffer_from_host_ptr`
can NEVER import GGUF mmap'd data on RADV.

The original 20.4 t/s was likely from a run WITHOUT `--cpu-moe` where the model
loaded entirely via `ggml_vk_create_buffer_device` (regular Vulkan alloc+copy from
mmap, suballocated). The `buffer_from_host_ptr` path was never the source of 20 t/s.

**Actual path to faster performance**:
1. Remove `--cpu-moe` for models ≤ GTT — lets all data go to Vulkan device memory
   via regular alloc+copy. No mmap import needed. BUT: this malloc+copies ~80 GB
   which risks OOM on 125 GB RAM (needs hybrid check).
2. Investigate why `998a216` recently gets 8.6 t/s (same as `836d36a`) when it
   originally got 20.4 — may be controller arg differences or GGUF version changes.

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

### Confirmed (image `3fbe06f`)
- [x] **Q2K** on moe-flash (CPU_MOE=0) — 20.77 t/s, 2 splits
- [x] **q4km** on moe-flash-cpumoe (CPU_MOE=1) — 2.96 cold → 6.0-6.3 warm

### Confirmed (image `3fbe06f`)
- [x] **glm-4-7-flash** on moe-flash — 50.57 t/s, 2 splits

### To test
- [ ] **deepseek-r1-0528** on moe-flash-cpumoe — verify cold boot + partial prefetch

### Cleanup
- [ ] **Strip debug/dead code from patch** — target minimal diff
- [ ] **Update measurements.md** — consolidate final benchmark numbers
- [ ] **Auto-detect --cpu-moe** — safety net for misassigned models (nice-to-have)

### Blocked on RADV
- [ ] **GPU expert matmul** — compute shaders can't read from pinned/imported host
  memory on RADV. If driver fixes this: all models → 20+ t/s regardless of size.

---

## Architecture

```
Two backends, same image (3fbe06f):

moe-flash (CPU_MOE=0):
  Model → Vulkan alloc+copy → 20+ t/s
  Used for: models ≤ GTT (Q2K 80G, glm 17G)

moe-flash-cpumoe (CPU_MOE=1):
  Model file (GGUF, mmap'd)
    ├── Model ≤ RAM? → Pinned alloc (ggml_vk_host_malloc + copy)
    └── Model > RAM? → mmap-wrap (demand-paged)
                        + Partial prefetch (MADV_WILLNEED, up to MemAvail - 8G)
                        + MADV_RANDOM + MADV_HUGEPAGE
  Used for: models > GTT (q4km 133G, deepseek 228G)
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
