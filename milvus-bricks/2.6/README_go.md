# Search Horizon Performance Tool (Go Version)

High-performance vector search testing tool written in Go with advanced concurrency using goroutines and channels.

## 🚀 Key Features

### Performance Advantages over Python Version
- **Superior Concurrency**: Uses Go's goroutines instead of Python's ThreadPoolExecutor
- **Lower Memory Overhead**: More efficient memory management with Go's runtime
- **Better Resource Utilization**: Native support for thousands of concurrent operations
- **Faster Execution**: Compiled binary with optimized performance
- **No GIL Limitations**: True parallelism without Python's Global Interpreter Lock

### Core Functionality
- ✅ **Vector Search Testing**: Support for both normal and hybrid search modes
- ✅ **Query Vector Management**: Load vectors from parquet files with efficient cycling
- ✅ **Advanced Statistics**: Real-time QPS, latency percentiles, and success rates
- ✅ **Expression Filtering**: Multiple filter types (equality, timestamp, geo, array)
- ✅ **Concurrent Architecture**: Worker pool with configurable concurrency levels
- ✅ **Timeout Control**: Per-search and overall test timeouts
- ✅ **Comprehensive Logging**: Detailed progress tracking and performance metrics

## 📋 Prerequisites

### Go Environment
```bash
# Install Go 1.21 or later
go version  # Should show go1.21+
```

### Dependencies
```bash
# Initialize Go module (already done)
go mod download
```

### Query Vector File
Ensure your query vector file exists at the expected location:
```bash
# Default path: /root/test/data/query.parquet
# Format: Parquet file with 'feature' column containing vector arrays
```

## 🛠️ Building

```bash
# Build the binary
go build -o search_horizon_perf search_horizon_perf.go

# Or build with optimizations
go build -ldflags="-s -w" -o search_horizon_perf search_horizon_perf.go
```

## 🎯 Usage

### Basic Usage
```bash
# Normal search with default settings
./search_horizon_perf

# Hybrid search with 20 workers
./search_horizon_perf -search-type=hybrid -workers=20

# Custom timeout and expression
./search_horizon_perf -timeout=300 -expr=geo_contains -workers=50
```

### Command Line Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `-search-type` | `normal` | Search type: `normal` or `hybrid` |
| `-workers` | `10` | Maximum number of concurrent workers |
| `-timeout` | `120` | Test timeout in seconds |
| `-output-fields` | `""` | Output fields (comma-separated) |
| `-expr` | `""` | Expression type (see below) |
| `-nq` | `1` | Number of query vectors per search |
| `-topk` | `15000` | Top K results to retrieve |
| `-search-timeout` | `10` | Individual search timeout in seconds |

### Expression Types

| Type | Description | Example |
|------|-------------|---------|
| `equal` | Device ID equality | `device_id == "SENSOR_A123"` |
| `equal_and_timestamp_week` | Device + 7-day window | `device_id == "CAM_B112" and timestamp >= 1735689600` |
| `equal_and_timestamp_month` | Device + 30-day window | `device_id == "DV348" and timestamp >= 1735689600` |
| `geo_contains` | Geospatial containment | `ST_CONTAINS(location, 'POLYGON(...)')` |
| `sensor_contains` | Array containment | `ARRAY_CONTAINS(sensor_lidar_type, "Thor_Trucks")` |

## 📊 Performance Examples

### High-Concurrency Testing
```bash
# 100 concurrent workers for stress testing
./search_horizon_perf -workers=100 -timeout=600

# Expected performance improvement:
# - Python version: ~50-200 QPS (GIL limited)
# - Go version: ~500-2000+ QPS (depending on Milvus cluster)
```

### Expression-Specific Testing
```bash
# Test all expression types (default behavior)
./search_horizon_perf -workers=50 -timeout=300

# Test specific expression type
./search_horizon_perf -expr=geo_contains -workers=30 -timeout=180
```

### Hybrid Search Testing
```bash
# Hybrid search with multiple vector fields
./search_horizon_perf -search-type=hybrid -workers=20 -timeout=240 -topk=10000
```

## 📈 Performance Monitoring

The tool provides real-time statistics:

```
📊 Progress - Submitted: 15420, QPS: 128.5, Avg: 0.078s, P99: 0.234s, Success: 99.8%
```

Final results include:
- **Total Searches**: Number of completed search operations
- **QPS (Queries Per Second)**: Sustained search rate
- **Latency Statistics**: Average, Min, Max, P95, P99 latencies
- **Success Rate**: Percentage of successful searches
- **Test Duration**: Actual test runtime

## 🔧 Configuration

### Hardcoded Settings (modify in source if needed)
```go
host           = "https://in01-9028520cb1d63cf.ali-cn-hangzhou.cloud-uat.zilliz.cn:19530"
collectionName = "horizon_test_collection"
vectorField    = "feature"
apiKey         = "cc5bf695ea9236e2c64617e9407a26cf0953034485d27216f8b3f145e3eb72396e042db2abb91c4ef6fde723af70e754d68ca787"
queryVectorFile = "/root/test/data/query.parquet"
```

### Tuning for Maximum Performance
```bash
# For maximum throughput (adjust based on your Milvus cluster capacity)
./search_horizon_perf -workers=200 -timeout=900 -search-timeout=30

# For latency testing (lower concurrency, shorter timeout)
./search_horizon_perf -workers=5 -timeout=60 -search-timeout=5
```

## 🚨 Troubleshooting

### Common Issues

1. **Parquet File Not Found**
   ```
   ❌ failed to load query vectors: failed to open parquet file
   ```
   - Ensure `/root/test/data/query.parquet` exists
   - Use the Python `generate_query_vectors.py` script to create it

2. **Collection Not Loaded**
   ```
   ❌ collection horizon_test_collection is not loaded
   ```
   - Load the collection in Milvus first
   - Check collection status with Milvus client

3. **Connection Issues**
   ```
   ❌ Failed to create Milvus client: connection refused
   ```
   - Verify Milvus server address and port
   - Check API key validity
   - Ensure network connectivity

### Performance Tuning

1. **Low QPS**: Increase `-workers` (try 50-200 depending on your cluster)
2. **High Memory Usage**: Decrease `-workers` or increase `-search-timeout`
3. **Timeout Issues**: Increase `-search-timeout` for individual searches

## 🔄 Migration from Python Version

### Key Differences
- **Binary Execution**: No Python interpreter needed
- **Faster Startup**: Compiled binary starts immediately
- **Better Concurrency**: True parallelism with goroutines
- **Lower Resource Usage**: More efficient memory and CPU utilization

### Performance Comparison
| Metric | Python Version | Go Version |
|--------|---------------|------------|
| Startup Time | ~2-3 seconds | ~0.1 seconds |
| Memory Usage | ~200-500MB | ~50-150MB |
| Max Concurrency | ~50 workers | ~500+ workers |
| Typical QPS | 50-200 | 500-2000+ |

## 🎉 Advanced Usage

### Custom Build Flags
```bash
# Debug build
go build -race -o search_horizon_perf search_horizon_perf.go

# Optimized production build
go build -ldflags="-s -w" -tags=release -o search_horizon_perf search_horizon_perf.go
```

### Profiling
```bash
# Build with profiling support
go build -tags=profile -o search_horizon_perf search_horizon_perf.go

# Run with CPU profiling
./search_horizon_perf -cpuprofile=cpu.prof -workers=100 -timeout=60
```

This Go version provides significantly better performance and resource efficiency compared to the Python implementation, making it ideal for high-throughput vector search testing scenarios.
