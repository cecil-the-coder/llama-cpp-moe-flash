# Research Phase: Complete

**Status**: RESEARCH COMPLETE -- 8-patch stack on b8664 delivers **7-10 t/s** on Qwen3 Q4_K_M with N_SLOTS=64 (4x baseline). All viable software optimizations explored. GPU compute (85ms/token) is the hardware ceiling. (2026-04-03)

**Production Image**: `7937441` on b8664

---

## Final State

Expert copy: **3.5ms/token** (optimized from 530ms -- 150x reduction). GPU compute: **85ms/token** (hardware-limited, dominant). 7-10 t/s is near the ceiling for Radeon 8060S UMA.

### Research findings (2026-04-03)

| Investigation | Result |
|---------------|--------|
| KHR_coopmat | Already active in RADV. No free performance. |
| Least-stale eviction (0023) | Works but equivalent to LRU at full pool size (all 94 layers fit) |
| AMDVLK driver | Not available in base image. Benchmarks show 8-10x slower anyway. |
| APEX requant | Skipped -- risky for marginal gain |
| MoEpic / KTransformers / PuzzleMoE | Too complex for marginal gains on UMA |

### Production config

- **N_SLOTS=64** on 128 GB UMA: 7-10 t/s, 94.6% hit rate, ~60 GB pool, stable
- **N_SLOTS=96** needs 192+ GB RAM for 10.4-11.1 t/s (+17%)
- GPU compute dominates -- optimizing expert copy or miss rate yields <0.1 t/s

---

## Remaining Hardware-Dependent Paths

All software optimizations have been exhausted. These require hardware or upstream changes:

| Path | Expected Gain | Requirement |
|------|---------------|-------------|
| **More RAM (192+ GB)** | N_SLOTS=96, 10.4-11.1 t/s (+17%) | RAM upgrade |
| **DeepSeek slot remap** | 3-5x on DeepSeek (4 -> 12-20 t/s) | 256 GB node |
| **Upstream #20757** | 14 t/s PoC with proper SLRU | Upstream merge + rebase |
| **Newer llama.cpp** | Reduce 85ms GPU compute | Upstream Vulkan shader work |
| **Intel AMX hardware** | 28 t/s (KTransformers) | Different hardware platform |

## Maintenance-Only Items

| Item | Notes |
|------|-------|
| Rebase tracking (H) | Monitor #20757 and upstream releases |
| Patch surface reduction (I) | Factor patches for easier rebasing |
| Upstream contribution (B) | When bandwidth allows |

---

## Completed Investigations (Full List)

- **KHR_coopmat**: Already active. No free performance.
- **Least-stale eviction (0023)**: Equivalent to LRU at full pool size.
- **AMDVLK**: Not available; benchmarks show 8-10x slower anyway.
- **APEX requant**: Skipped (risky).
- **MoEpic / KTransformers / PuzzleMoE**: Too complex for marginal UMA gains.
- **D** - Split merging COMPLETE (patch 0021): 282->96 splits. Marginal t/s.
- **E** - Speculative prefetch COMPLETE (patch 0022): Marginal t/s.
- **F** - Adaptive N_SLOTS: SKIP. Homogeneous experts; upgrade RAM instead.
- **G** - Routing prediction: DEFER. <0.1 t/s gain, GPU-compute-limited.
- **I10b** - Slot remapping COMPLETE: 7-10 t/s with N_SLOTS=64. 94.6% hit rate.
- **I11** - Expert GPU cache + sync skip: DeepSeek 4.1 t/s (2.3x baseline).
- **I12** - ik_llama.cpp benchmark: Vulkan 2x faster for in-GTT models.
- **I14** - io_uring polish: no measurable benefit.
- **I17** - Prometheus metrics infrastructure.
- **I18** - Cache hit tracking fix.

---

## Per-Token Cost Breakdown (N_SLOTS=64)

| Phase | Time | % of Token |
|-------|------|------------|
| **GPU compute** | ~85 ms | ~89% |
| Sync overhead | ~7 ms | ~7% |
| Expert copy (misses) | ~3.5 ms | ~4% |

GPU compute dominates. Expert copy was reduced 150x (530ms -> 3.5ms). Further software optimization yields <0.1 t/s.

---

## Comparison with Other Systems

| System | >GTT t/s | Approach | vs Our 7-10 t/s |
|--------|----------|----------|---------------------|
| **KTransformers** | 28 | Intel AMX CPU kernels | 3-4x faster |
| **llama.cpp #20757 PoC** | 14 | Two-tier GPU cache (Python) | 1.4-2x faster |
| **Our moe-flash (96-slot, unstable)** | 10.4-11.1 | GPU MoE, 96 slots (needs 192+ GB) | +17% (unstable) |
| **Our moe-flash (64-slot, production)** | **7-10** | **GPU MoE, 64 slots, LRU + prefetch** | **baseline** |
| **flash-moe** | 4.4 | Apple SSD + Metal (397B model) | ~2x slower |
| **Our moe-flash (CPU MoE)** | 4.1 | AVX-512 CPU MoE (DeepSeek) | ~2x slower |
| **ik_llama.cpp** | 1.5 | CPU-only, no flash_attn | 5-7x slower |

---

*Last Updated*: 2026-04-03
