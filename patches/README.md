# Patches

Patches against llama.cpp `b8298` for MoE flash expert streaming + Vulkan UMA offload + TQ2_KV.

## Patches

| File | Description |
|---|---|
| `0001-moe-flash-complete.patch` | Core moe-flash feature: eval callback, fadvise, io_uring staging, GGUF warmcache, auto-detect --cpu-moe |
| `0002-moe-expert-offload.patch` | Vulkan_Host CPU buffer interface (not vk_buffer_context), supports_buft removal |
| `0003-tq2-kv.patch` | TQ2_KV 2-bit KV cache type: type system, encode/decode with NaN guard, GLSL shaders, Vulkan backend, tests |

## Applying

```bash
git clone --depth 1 --branch b8298 https://github.com/ggml-org/llama.cpp
cd llama.cpp
git apply ../patches/0001-moe-flash-complete.patch
git apply ../patches/0002-moe-expert-offload.patch
git apply ../patches/0003-tq2-kv.patch
```

## Base commit

```
Tag:    b8298
Commit: f90bd1dd (llama : whitespace cleanup (#20422))
Date:   2026-03-23
```
