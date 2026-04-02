# I15: Prometheus Metrics Export

**Status**: ✅ Implementation Complete  
**Date**: 2026-04-01  
**Component**: Metrics Collection & Observability

---

## Overview

Added lightweight performance metrics collection to MoE Flash with optional Prometheus export. This provides visibility into:
- Cache effectiveness (hit/miss rates)
- I/O savings vs baseline
- Request latency distributions
- Expert loading performance

---

## Metrics Collected

### Counters

| Metric | Type | Description |
|--------|------|-------------|
| `moe_flash_requests_total` | Counter | Total inference requests processed |
| `moe_flash_layers_total` | Counter | Total MoE layers processed |
| `moe_flash_tokens_total` | Counter | Total tokens generated |
| `moe_flash_experts_loaded_total` | Counter | Experts loaded from disk |
| `moe_flash_experts_prefetched_total` | Counter | Experts prefetched proactively |
| `moe_flash_experts_from_cache_total` | Counter | Experts served from cache |
| `moe_flash_cache_hits_total` | Counter | Cache hits |
| `moe_flash_cache_misses_total` | Counter | Cache misses |
| `moe_flash_bytes_loaded_total` | Counter | Bytes actually read from disk |
| `moe_flash_bytes_requested_total` | Counter | Bytes that would be read without optimization |

### Gauges

| Metric | Type | Description |
|--------|------|-------------|
| `moe_flash_active_experts_cached` | Gauge | Current number of experts in cache |
| `moe_flash_cache_size_mb` | Gauge | Cache size limit in MB |
| `moe_flash_cache_hit_rate` | Gauge | Cache hit rate as percentage |
| `moe_flash_io_savings_percent` | Gauge | I/O savings vs loading full model |

### Histograms

| Metric | Type | Description | Buckets |
|--------|------|-------------|---------|
| `moe_flash_request_duration_ms` | Histogram | End-to-end request latency | 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000 |
| `moe_flash_expert_load_duration_ms` | Histogram | Time to load expert from disk | 1, 5, 10, 25, 50, 100, 250, 500, 1000 |
| `moe_flash_graph_compute_duration_ms` | Histogram | Graph compute time | 100, 250, 500, 1000, 2500, 5000, 10000 |

---

## Usage

### Simple Metrics (Always Available)

Simple metrics are always collected with zero overhead. They can be accessed via:

```cpp
// Get metrics snapshot
moe_flash_metrics_c metrics;
moe_flash_metrics_get(ctx->metrics, &metrics);

// Print to stderr
moe_flash_metrics_print(ctx->metrics);
```

Example output:
```
╔════════════════════════════════════════════════════════════════╗
║           MoE Flash Metrics Report                             ║
╠════════════════════════════════════════════════════════════════╣
║ Requests:                          42                          ║
║ Experts Loaded:                   336                          ║
║ Cache Hit Rate:                  12.5%                          ║
║ I/O Savings:                     84.0%                          ║
║ Bytes Loaded:                    3212 MB                       ║
║ Bytes Requested:                20045 MB                       ║
╚════════════════════════════════════════════════════════════════╝
```

### Prometheus Export (Optional)

To enable Prometheus metrics export:

```bash
# Build with prometheus support
cmake -DLLAMA_USE_PROMETHEUS=ON -DPROMETHEUS_CPP_DIR=/path/to/prometheus-cpp ..
make

# Run with metrics endpoint on port 9090
./llama-server --moe-flash-metrics-port 9090
```

Metrics will be available at:
```
http://localhost:9090/metrics
```

---

## Grafana Dashboard

A pre-built Grafana dashboard is provided at:
```
grafana/moe-flash-dashboard.json
```

### Dashboard Panels

1. **Request Rate** - Requests per second
2. **Cache Hit Rate** - Gauge showing hit percentage (target: >70% green)
3. **I/O Savings** - Percentage reduction in I/O vs baseline
4. **Active Experts Cached** - Current cache utilization
5. **Request Latency** - p50/p95/p99 latency over time
6. **Expert Load Time** - Distribution of expert load times
7. **Cache Hits vs Misses** - Comparative rates
8. **I/O Throughput** - Loaded vs requested bytes
9. **Experts per Request** - Average experts loaded per inference
10. **Cache Utilization** - Percentage of cache in use

### Key Metrics to Watch

| Alert | Condition | Severity |
|-------|-----------|----------|
| High Cache Miss Rate | `hit_rate < 30%` for 5m | Warning |
| Low I/O Savings | `savings < 50%` for 5m | Critical |
| High Latency | `p95 > 5s` for 5m | Warning |
| Cache Full | `utilization > 95%` | Info |

---

## Prometheus Queries

### Essential Queries

```promql
# Cache hit rate over time
100 * rate(moe_flash_cache_hits_total[5m]) / 
  (rate(moe_flash_cache_hits_total[5m]) + rate(moe_flash_cache_misses_total[5m]))

# I/O savings percentage
100 * (1 - rate(moe_flash_bytes_loaded_total[5m]) / 
  rate(moe_flash_bytes_requested_total[5m]))

# Request latency percentiles
histogram_quantile(0.95, 
  sum(rate(moe_flash_request_duration_ms_bucket[5m])) by (le))

# Experts per request
rate(moe_flash_experts_loaded_total[5m]) / 
  rate(moe_flash_requests_total[5m])
```

### Alerting Rules

```yaml
groups:
  - name: moe-flash
    rules:
      - alert: MoEFlashHighCacheMisses
        expr: moe_flash_cache_hit_rate < 30
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "MoE Flash cache miss rate is high"
          
      - alert: MoEFlashLowIOSavings
        expr: moe_flash_io_savings_percent < 50
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "MoE Flash I/O optimization not working"
```

---

## Implementation Details

### Code Structure

```
src/
├── llama-moe-flash-metrics.h      # Metrics definitions
├── llama-moe-flash-metrics.cpp    # C interface implementation
└── llama-moe-flash.cpp            # Main file with instrumentation

grafana/
└── moe-flash-dashboard.json       # Pre-built dashboard
```

### Instrumentation Points

Metrics are collected at these key points:

1. **Request start** - `llama_moe_flash_pre_graph()`
   - Increment `requests_total`
   - Start request timer

2. **Expert loading** - `expert_copy_callback()`
   - Increment `experts_loaded_total`
   - Record bytes loaded
   - Time the operation
   - Check/update cache stats

3. **Cache operations**
   - Hit: Increment `cache_hits_total`
   - Miss: Increment `cache_misses_total`, load from disk

4. **Request end** - Destructor or explicit call
   - Stop request timer
   - Update latency histogram

### Zero-Overhead Design

- Metrics use `std::atomic` for thread-safety without locks
- Collection is branch-predictor friendly (always same path)
- Disabled counters compile to no-ops when not accessed
- Prometheus export is completely optional (compile-time)

---

## Testing

### Verify Metrics Collection

```bash
# Run with debug output
LLAMA_FLASH_MOE_ENABLED=1 ./llama-server -m model.gguf 2>&1 | grep -i metrics

# Check metrics endpoint (if Prometheus enabled)
curl -s http://localhost:9090/metrics | grep moe_flash
```

### Expected Output

```
# HELP moe_flash_requests_total Total inference requests
# TYPE moe_flash_requests_total counter
moe_flash_requests_total 42

# HELP moe_flash_cache_hit_rate Cache hit rate percentage
# TYPE moe_flash_cache_hit_rate gauge
moe_flash_cache_hit_rate 12.5

# HELP moe_flash_io_savings_percent I/O savings vs baseline
# TYPE moe_flash_io_savings_percent gauge
moe_flash_io_savings_percent 84.0
```

---

## Future Enhancements

### Planned
- [ ] Per-model metrics labels
- [ ] Expert-level detail (which experts are hot)
- [ ] Pattern analysis metrics
- [ ] GPU memory pressure correlation

### Under Consideration
- Pushgateway support for ephemeral instances
- OpenTelemetry integration
- Custom metric exporters (StatsD, CloudWatch)

---

## References

- [Prometheus Best Practices](https://prometheus.io/docs/practices/naming/)
- [Grafana Dashboard Guidelines](https://grafana.com/docs/grafana/latest/dashboards/build-dashboards/best-practices/)
- Original issue: I15 - Add observability to MoE Flash

---

**Author**: Shadow (code-puppy-92fceb)  
**Related**: I11, I13, I14
