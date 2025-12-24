# Turbopuffer Query Benchmark Tool

A high-performance Go-based benchmarking tool for testing Turbopuffer vector query performance with optimized concurrency.

## Features

- **High-Performance Concurrency**: Lock-free atomic operations for maximum throughput
- **Memory-Optimized**: Pre-loads and converts all vectors to float32 in memory
- **Configurable concurrency** (serial or parallel queries)
- **JSON-based query vector input** with one-time loading
- **Pre-generated namespace pool** for efficient round-robin selection
- **Configurable test duration**
- **Comprehensive performance statistics** (QPS, latency percentiles)
- **Periodic progress logging** during test execution

## Prerequisites

- Go 1.16 or higher
- Turbopuffer API key and namespace

## Installation

```bash
cd /Users/yanliang/fork/vectordb-testbricks/tpuffer

# Initialize go module if not already done
go mod init tpuffer-benchmark

# Install Turbopuffer Go SDK
go get github.com/turbopuffer/turbopuffer-go

# Download dependencies
go mod tidy
```

## Usage

```bash
go run query.go \
  -json <path-to-json-file> \
  -key <your-api-key> \
  -region <turbopuffer-region> \
  -user-id-start <start-id> \
  -user-id-end <end-id> \
  -concurrency <number-of-concurrent-queries> \
  -duration <duration-in-seconds> \
  -topk <number-of-results>
```

### Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `-json` | Yes | - | Path to JSON file containing query vectors |
| `-key` | No* | - | Your Turbopuffer API key (*or set `TURBOPUFFER_API_KEY` env var) |
| `-region` | No | `us-east-1` | Turbopuffer region (e.g., `us-east-1`, `gcp-us-central1`) |
| `-user-id-start` | No | 1 | Start of user ID range for namespaces |
| `-user-id-end` | No | 1 | End of user ID range for namespaces |
| `-concurrency` | No | 1 | Number of concurrent queries (1 for serial mode) |
| `-duration` | No | 10 | Duration to run queries in seconds |
| `-topk` | No | 10 | Number of results to return per query |
| `-log-interval` | No | 10 | Statistics logging interval in seconds |

### API Key Configuration

You can provide the API key in two ways:
1. Using the `-key` flag
2. Setting the `TURBOPUFFER_API_KEY` environment variable

```bash
# Method 1: Using flag
go run query.go -json vectors.json -key tpuf_xxxxxxxxxxxx ...

# Method 2: Using environment variable
export TURBOPUFFER_API_KEY=tpuf_xxxxxxxxxxxx
go run query.go -json vectors.json ...
```

### Namespace Selection

The tool queries namespaces based on user ID ranges. For each query, a random user ID is selected from the range `[user-id-start, user-id-end]`, and the namespace is formatted as `id_<userid>`. 

For example, if `-user-id-start=100` and `-user-id-end=200`, the tool will randomly query namespaces like `id_100`, `id_150`, `id_200`, etc.

### JSON File Format

The JSON file should contain an array of vectors (arrays of float numbers):

```json
[
  [0.004807, 0.002700, 0.020795, ...],
  [0.015234, -0.003456, 0.012345, ...],
  ...
]
```

See `query_vectors_100.json` for an example.

## Examples

### Serial Query (Concurrency = 1) - Single Namespace

```bash
go run query.go \
  -json query_vectors_100.json \
  -key tpuf_xxxxxxxxxxxx \
  -region us-east-1 \
  -user-id-start 1000 \
  -user-id-end 1000 \
  -concurrency 1 \
  -duration 30
```

### Serial Query - Multiple Namespaces

```bash
go run query.go \
  -json query_vectors_100.json \
  -key tpuf_xxxxxxxxxxxx \
  -region gcp-us-central1 \
  -user-id-start 1000 \
  -user-id-end 1010 \
  -concurrency 1 \
  -duration 30
```

### Parallel Queries (Concurrency = 10) - Multiple Namespaces

```bash
# Using API key flag
go run query.go \
  -json query_vectors_100.json \
  -key tpuf_xxxxxxxxxxxx \
  -region us-east-1 \
  -user-id-start 1000 \
  -user-id-end 2000 \
  -concurrency 10 \
  -duration 60

# Or using environment variable
export TURBOPUFFER_API_KEY=tpuf_xxxxxxxxxxxx
go run query.go \
  -json query_vectors_100.json \
  -region us-east-1 \
  -user-id-start 1000 \
  -user-id-end 2000 \
  -concurrency 10 \
  -duration 60
```

## Output

The tool will output statistics including:

- **Total Queries**: Total number of queries executed
- **Total Duration**: Total time the test ran
- **QPS**: Queries per second
- **Average Latency**: Mean query latency
- **P95 Latency**: 95th percentile latency
- **P99 Latency**: 99th percentile latency
- **Max Latency**: Maximum query latency
- **Min Latency**: Minimum query latency

Example output:

```
========== Query Statistics ==========
Total Queries:     1234
Total Duration:    30.001s
QPS:               41.13
Average Latency:   24.32ms
P95 Latency:       45.67ms
P99 Latency:       78.90ms
Max Latency:       123.45ms
Min Latency:       12.34ms
======================================
```

## Performance Optimizations

This tool is optimized for high-concurrency scenarios based on the design patterns from `search_horizon_perf_3.go`:

### 1. **Lock-Free Vector Access**
- Uses `atomic.AddInt64()` for lock-free counter increment
- `GetVectorLockFree()` method avoids mutex contention
- Significantly improves throughput in high-concurrency scenarios

### 2. **Memory Pre-Allocation**
- Loads entire JSON file into memory at startup (one-time cost)
- Pre-converts all vectors from float64 to float32 before queries start
- Pre-generates all namespace strings in a pool
- Pre-allocates latency slice with 100K capacity

### 3. **Efficient Namespace Selection**
- Round-robin selection through pre-generated namespace pool
- No random number generation overhead during queries
- Uses atomic operations for thread-safe access

### 4. **Optimized Query Loop**
- Minimal allocation in hot path
- Direct slice access without copying
- No locks in query execution path

### 5. **Periodic Progress Logging**
- Non-blocking statistics logging every N seconds
- Doesn't impact query performance

These optimizations trade memory for performance, making the tool suitable for sustained high-QPS testing.

This tool uses the official Turbopuffer Go SDK:

- **SDK**: `github.com/turbopuffer/turbopuffer-go`
- **Client creation**: 
  ```go
  client := turbopuffer.NewClient(
      option.WithAPIKey(apiKey),
      option.WithRegion(region),
  )
  ```
- **Namespace format**: `id_{userid}` where userid is selected from `[user-id-start, user-id-end]` in round-robin fashion
- **Query method**:
  ```go
  ns := client.Namespace(namespace)
  result, err := ns.Query(
      ctx,
      turbopuffer.NamespaceQueryParams{
          RankBy: turbopuffer.NewRankByVector("vector", vectorFloat32),
          TopK:   turbopuffer.Int(int64(topK)),
      },
  )
  ```

For more details, see:
- [Turbopuffer Go SDK GitHub](https://github.com/turbopuffer/turbopuffer-go)
- [Turbopuffer Vector Search Guide](https://turbopuffer.com/docs/vector)
- [Turbopuffer Query Documentation](https://turbopuffer.com/docs/query)
- [Turbopuffer Regions](https://turbopuffer.com/docs/regions)

## Notes

- Uses the official Turbopuffer Go client SDK (not HTTP client)
- **High-performance design**: Lock-free atomic operations for concurrent access
- **Memory optimization**: All vectors pre-loaded and converted to float32
- **Efficient namespace selection**: Round-robin through pre-generated pool (not random)
- When concurrency is set to 1, queries run serially (one after another)
- When concurrency > 1, multiple workers run queries in parallel with no lock contention
- Vectors and namespaces are distributed round-robin among all workers
- Query errors are logged but don't stop the benchmark
- Namespaces are in format `id_<userid>`, e.g., `id_1000`, `id_1001`, `id_1002` (sequential)
- API key can be provided via flag or environment variable
- Periodic progress logging shows real-time QPS during test execution

## Troubleshooting

**Error: "query failed with status 401"**
- Check your API key is correct

**Error: "query failed with status 404"**
- Verify your namespaces exist (id_xxx format)
- Check that the user ID range corresponds to existing namespaces

**Error: "Failed to parse JSON file"**
- Ensure your JSON file format matches the expected structure (array of float arrays)

**Error: "user-id-start must be <= user-id-end"**
- Make sure the start ID is not greater than the end ID

**High latency values**
- Check your network connection
- Consider the cold vs hot data performance characteristics of Turbopuffer
- Try increasing concurrency to maximize throughput
- Verify that namespaces in the user ID range exist and have data

