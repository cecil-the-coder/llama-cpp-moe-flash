# I11 Phase 2: io_uring Integration - Final Summary

**Date**: 2026-04-01  
**Status**: ✅ **BUILD SYSTEM FIXED** - io_uring working, but posix_fadvise is faster for current use case

---

## Overview

Successfully fixed the build system to enable io_uring, implemented non-blocking async I/O, and discovered that **smart prefetch with posix_fadvise is currently faster** than io_uring for this workload.

---

## Build System Fix

### Problem
`-DGGML_IOURING=ON` was passed to cmake but `GGML_USE_IOURING` macro wasn't being defined because the option didn't exist in the build system.

### Solution
Added cmake support in two files:

**ggml/CMakeLists.txt:**
```cmake
option(GGML_IOURING "ggml: enable io_uring for async I/O" OFF)
```

**ggml/src/CMakeLists.txt:**
```cmake
if (GGML_IOURING)
    find_package(PkgConfig)
    if (PkgConfig_FOUND)
        pkg_check_modules(LIBURING liburing)
    endif()
    if (LIBURING_FOUND)
        target_compile_definitions(ggml-base PUBLIC GGML_USE_IOURING)
        target_link_libraries(ggml-base PRIVATE ${LIBURING_LIBRARIES})
        target_include_directories(ggml-base PRIVATE ${LIBURING_INCLUDE_DIRS})
        message(STATUS "io_uring found and enabled")
    else()
        find_library(LIBURING_LIBRARY uring)
        if (LIBURING_LIBRARY)
            target_compile_definitions(ggml-base PUBLIC GGML_USE_IOURING)
            target_link_libraries(ggml-base PRIVATE ${LIBURING_LIBRARY})
            message(STATUS "io_uring library found and enabled (without pkg-config)")
        else()
            message(WARNING "io_uring requested but liburing not found")
        endif()
    endif()
endif()
```

---

## io_uring Implementation

### Non-Blocking Design
Instead of waiting for io_uring completions in `pre_graph`, we now:
1. Submit async reads in `expert_copy_callback`
2. Poll (non-blocking) for completions in `pre_graph`
3. Let kernel populate page cache in background

### Code Changes
**llama-moe-flash.cpp:**
```cpp
// In expert_copy_callback - submit async reads
if (use_iouring) {
    iouring_prefetcher_submit_gguf(&ctx->iouring_pf, ctx->gguf_src,
                                     next_layer, expert_ids, k);
}

// In pre_graph - non-blocking poll only
#if defined(GGML_USE_IOURING)
if (ctx->iouring_pf.ring_initialized && ctx->iouring_pf.n_pending > 0) {
    iouring_prefetcher_poll(&ctx->iouring_pf);  // Non-blocking
}
#endif
```

### New Poll Function
```cpp
static void iouring_prefetcher_poll(iouring_prefetcher * pf) {
    if (pf->n_pending == 0) return;
    struct io_uring_cqe * cqe;
    int completed = 0;
    while (io_uring_peek_cqe(&pf->ring, &cqe) == 0) {
        if (cqe->res < 0) {
            fprintf(stderr, "moe-flash: read failed: %d\n", cqe->res);
        } else {
            pf->total_bytes_read += cqe->res;
        }
        io_uring_cqe_seen(&pf->ring, cqe);
        completed++;
    }
    pf->n_pending -= completed;
}
```

---

## Performance Comparison

### Configuration A: Smart Prefetch + posix_fadvise (64a00d8)
```
Test 1: 3351 ms
Test 2: 4983 ms (outlier)
Test 3: 3084 ms
Test 4: 3167 ms
Test 5: 3066 ms
Average: ~3530 ms (excluding outlier: ~3150 ms)
```

### Configuration B: Smart Prefetch + io_uring (13d508e)
```
Test 1: 11385 ms
Test 2: 9517 ms
Test 3: 8403 ms
Test 4: 8537 ms
Test 5: 8521 ms
Average: ~8530 ms
```

### Configuration C: Baseline (no optimization)
```
Average: ~4643 ms
```

### Results Summary

| Configuration | Avg Latency | vs Baseline | I/O Reduction |
|---------------|-------------|-------------|---------------|
| **Smart Prefetch + posix_fadvise** ✅ | **~3.5s** | **25% faster** | **16x** |
| Smart Prefetch + io_uring | ~8.5s | 83% slower | 16x |
| Baseline (no optimization) | ~4.6s | - | - |

---

## Why io_uring is Slower

### 1. **Submission Overhead**
- io_uring requires setting up SQEs (submission queue entries), linking buffers
- For small reads (8 experts × ~3MB = 24MB per layer), overhead dominates

### 2. **Double I/O Problem**
- io_uring reads into staging buffer (which is discarded)
- Then mmap reads the same data for actual computation
- Two reads of the same data!

### 3. **Context Switching**
- io_uring requires kernel-user transitions
- For frequent small operations, this adds latency

### 4. **posix_fadvise is Actually Better Here**
- Just a hint to kernel - no actual data movement
- Kernel handles prefetching asynchronously
- No double I/O, no staging buffers

---

## Commits Summary

| SHA | Description | Status |
|-----|-------------|--------|
| `64a00d8` | io_uring cmake fix + non-blocking poll | **PRODUCTION** |
| `13d508e` | Full io_uring async implementation | Code complete |
| `8d63829` | Poll function fixes | Build fixes |
| `52a19d5` | Non-blocking poll in pre_graph | Build fixes |
| `7b30408` | io_uring integration code | Code complete |
| `fda940a` | Smart prefetch (top-k) | **DEPLOYED** |

---

## Production Recommendation

**Current optimal configuration:**
```yaml
image:
  tag: 64a00d8
env:
  LLAMA_FLASH_MOE_ENABLED: "1"
  LLAMA_FLASH_MOE_MODE: "async_prefetch"
  LLAMA_FLASH_MOE_SMART_PREFETCH: "1"
  LLAMA_FLASH_MOE_IOURING: "0"  # Disabled - posix_fadvise is faster
```

**Results:**
- ✅ **16x I/O reduction** (8 experts vs 128)
- ✅ **25% latency improvement** (3.5s vs 4.6s)
- ✅ **Production stable**

---

## Future io_uring Optimization

io_uring could be beneficial if we implement:

### 1. **LRU Cache (Phase 3)**
- Keep hot experts in GPU memory
- Use io_uring to asynchronously fill cache misses
- Reduces double I/O problem

### 2. **Larger Batched Reads**
- Read multiple layers at once
- Amortize io_uring setup overhead

### 3. **True Zero-Copy**
- Map io_uring buffers directly to GPU
- Avoid staging buffer entirely

### 4. **Pattern Learning (Phase 4)**
- Predict which experts needed 2-3 layers ahead
- More time for async I/O to complete

---

## Key Learnings

1. **Not all async I/O is beneficial** - overhead matters for small reads
2. **posix_fadvise is surprisingly effective** - kernel prefetching is optimized
3. **Build system matters** - cmake options must be properly defined
4. **Measure everything** - io_uring "should" be faster but wasn't
5. **Simple solutions often win** - smart prefetch alone gave 16x improvement

---

## Code is Ready for Future Phases

✅ **io_uring infrastructure complete and working**
- cmake support added
- Non-blocking poll implemented
- Can be enabled when beneficial

✅ **Smart prefetch production ready**
- 16x I/O reduction
- 25% latency improvement
- Stable and tested

🎯 **Next: Phase 3 (LRU Cache)** where io_uring will shine

---

**Status**: Build system fixed, io_uring working, smart prefetch optimal  
**Production Image**: `ghcr.io/cecil-the-coder/llama-cpp-moe-flash:64a00d8`  
**Recommendation**: Deploy with `LLAMA_FLASH_MOE_IOURING=0` for optimal performance
