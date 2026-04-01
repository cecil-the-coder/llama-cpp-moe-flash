# Patches

Patches against llama.cpp `b8298` for MoE flash expert streaming + Vulkan UMA offload + TQ2_KV.

## Main Patches

| File | Description |
|---|---|
| `0001-moe-flash-working.patch` | **CURRENT** - Consolidated patch with I11 async prefetch fully functional |
| `0001-moe-flash-complete-consolidated.patch` | Pre-I10b consolidated patch (stable, production-ready) |
| `0001-moe-flash-I11-*.patch` | Individual I11 async prefetch development patches |
| `0002-moe-expert-offload.patch` | Vulkan_Host CPU buffer interface (not vk_buffer_context), supports_buft removal |
| `0003-tq2-kv.patch` | TQ2_KV 2-bit KV cache type: type system, encode/decode with NaN guard, GLSL shaders, Vulkan backend, tests |

## I11 Async Prefetch Patches (Development History)

The async prefetch feature was developed incrementally. Key patches:

| Patch | Purpose |
|---|---|
| `0001-moe-flash-I11-async-prefetch.patch` | Initial callback infrastructure |
| `0001-moe-flash-I11-fix-parse-layer.patch` | Fixed layer ID parsing for `ffn_moe_gate-N` names |
| `0001-moe-flash-I11-fix-gguf-loading.patch` | Fixed GGUF loading outside io_uring block |
| `0001-moe-flash-I11-actual-prefetch.patch` | Added actual `posix_fadvise` prefetch logic |
| `0001-moe-flash-working.patch` | **FINAL** - Complete working implementation (symlink to latest) |

## Applying

```bash
git clone --depth 1 --branch b8298 https://github.com/ggml-org/llama.cpp
cd llama.cpp
git apply ../patches/0001-moe-flash-working.patch
```

## Base commit

```
Tag:    b8298
Commit: f90bd1dd (llama : whitespace cleanup (#20422))
Date:   2026-03-23
```
