# Patches

Patches against llama.cpp main branch for io_uring expert streaming.

## Status

Patch 0001 is complete and tested. See `docs/plan.md` for full status.

## Patches

| File | Description | Phase | Status |
|---|---|---|---|
| `0001-moe-flash-complete.patch` | Eval callback + fadvise + io_uring staging + GGUF warmcache | P2–P3 | Done |
| `0003-vulkan-expert-copy-staging.patch` | vkCmdCopyBuffer staging → Vulkan buffer | P3.1 | Planned |
| `0004-vulkan-shader-indirection.patch` | Shader indirection table (skip copy) | P3.3 | If needed |

## Applying

```bash
cd /workspace/llama-cpp-moe-flash/llama.cpp  # cloned in P2.1
git am ../patches/0001-*.patch
```

## Base commit

```
Tag:    b8298
Commit: f90bd1dd (llama : whitespace cleanup (#20422))
Date:   2026-03-23 (confirmed in running container)
```

To checkout:
```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout b8298
```
