package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/turbopuffer/turbopuffer-go"
	"github.com/turbopuffer/turbopuffer-go/option"
)

// VectorPool manages query vectors with lock-free concurrent access
// Pre-loads all vectors into memory for maximum performance
type VectorPool struct {
	vectors    [][]float32 // Pre-converted to float32 for SDK
	currentIdx int64       // Atomic counter for lock-free access
}

// NewVectorPool creates a new VectorPool and loads vectors from JSON file
func NewVectorPool(filePath string) (*VectorPool, error) {
	log.Printf("📖 Loading vectors from %s", filePath)

	// Read entire file at once for best performance
	data, err := io.ReadAll(mustOpen(filePath))
	if err != nil {
		return nil, fmt.Errorf("failed to read JSON file: %v", err)
	}

	// Parse JSON - expecting array of float64 vectors
	var vectors64 [][]float64
	if err := json.Unmarshal(data, &vectors64); err != nil {
		return nil, fmt.Errorf("failed to parse JSON: %v", err)
	}

	if len(vectors64) == 0 {
		return nil, fmt.Errorf("no vectors found in JSON file")
	}

	// Pre-convert all vectors to float32 (required by turbopuffer SDK)
	// Note: Go json.Unmarshal always parses JSON numbers as float64,
	// so we must convert to float32 for the SDK even if the original
	// precision in JSON doesn't require float64
	vectors32 := make([][]float32, len(vectors64))
	for i, vec64 := range vectors64 {
		vec32 := make([]float32, len(vec64))
		for j, v := range vec64 {
			vec32[j] = float32(v)
		}
		vectors32[i] = vec32
	}

	log.Printf("✅ Loaded %d vectors (dimension: %d) and converted to float32",
		len(vectors32), len(vectors32[0]))

	return &VectorPool{
		vectors:    vectors32,
		currentIdx: 0,
	}, nil
}

// GetVectorLockFree returns next vector using atomic operations (lock-free)
// This provides best performance for high-concurrency scenarios
func (vp *VectorPool) GetVectorLockFree() []float32 {
	if len(vp.vectors) == 0 {
		panic("Vector pool is empty")
	}

	poolSize := int64(len(vp.vectors))
	idx := atomic.AddInt64(&vp.currentIdx, 1) % poolSize
	return vp.vectors[idx]
}

// GetSize returns the number of vectors in the pool
func (vp *VectorPool) GetSize() int {
	return len(vp.vectors)
}

// mustOpen opens a file or panics
func mustOpen(filePath string) *os.File {
	file, err := os.Open(filePath)
	if err != nil {
		panic(fmt.Sprintf("failed to open file: %v", err))
	}
	return file
}

// Statistics collects query performance metrics with optimized lock usage
type Statistics struct {
	latencies          []time.Duration
	namespaceLatencies []time.Duration // Latencies for namespace creation
	first10            []time.Duration // Store first 10 query latencies
	mu                 sync.Mutex
	totalQueries       int64 // Atomic counter
	totalFailures      int64 // Atomic counter for failed queries
	startTime          time.Time
	endTime            time.Time
	firstQueryTime     time.Time // Time of first successful query
	lastQueryTime      time.Time // Time of last successful query
	stopped            int32     // Atomic flag: 1 when test duration expired
	lastLoggedCount    int       // Track last logged latency count for interval stats
}

func NewStatistics() *Statistics {
	return &Statistics{
		latencies:          make([]time.Duration, 0, 100000), // Pre-allocate for efficiency
		namespaceLatencies: make([]time.Duration, 0, 100000), // Pre-allocate for namespace latencies
		first10:            make([]time.Duration, 0, 10),     // Pre-allocate for first 10
		startTime:          time.Now(),
	}
}

func (s *Statistics) recordLatency(latency time.Duration) {
	s.mu.Lock()
	now := time.Now()
	s.latencies = append(s.latencies, latency)
	// Record first 10 queries
	if len(s.first10) < 10 {
		s.first10 = append(s.first10, latency)
	}
	// Record first and last query time for accurate QPS calculation
	if s.firstQueryTime.IsZero() {
		s.firstQueryTime = now
	}
	s.lastQueryTime = now
	s.mu.Unlock()
}

func (s *Statistics) recordLatencyWithStartTime(latency time.Duration, queryStartTime time.Time) {
	s.mu.Lock()
	// Only record if:
	// 1. Query was started before test stopped
	// 2. Query completed before test stopped (queryStartTime + latency <= endTime)
	// Add a small safety margin to avoid edge cases
	queryEndTime := queryStartTime.Add(latency)

	// Stricter check: query must complete at least 1ms before endTime
	if s.endTime.IsZero() || (queryStartTime.Before(s.endTime) && queryEndTime.Before(s.endTime)) {
		s.latencies = append(s.latencies, latency)
		// Record first 10 queries
		if len(s.first10) < 10 {
			s.first10 = append(s.first10, latency)
		}
		// Record first and last query time for accurate QPS calculation
		if s.firstQueryTime.IsZero() {
			s.firstQueryTime = queryStartTime
		}
		s.lastQueryTime = queryStartTime
	}
	s.mu.Unlock()
}

func (s *Statistics) incrementQueries() {
	atomic.AddInt64(&s.totalQueries, 1)
}

func (s *Statistics) incrementFailures() {
	atomic.AddInt64(&s.totalFailures, 1)
}

func (s *Statistics) markStopped() {
	atomic.StoreInt32(&s.stopped, 1)
}

func (s *Statistics) isStopped() bool {
	return atomic.LoadInt32(&s.stopped) == 1
}

func (s *Statistics) recordNamespaceLatency(latency time.Duration) {
	s.mu.Lock()
	s.namespaceLatencies = append(s.namespaceLatencies, latency)
	s.mu.Unlock()
}

func (s *Statistics) printStats() {
	s.mu.Lock()
	defer s.mu.Unlock()

	totalQueries := atomic.LoadInt64(&s.totalQueries)
	totalFailures := atomic.LoadInt64(&s.totalFailures)
	totalDuration := s.endTime.Sub(s.startTime)

	if totalQueries == 0 {
		log.Println("No queries executed.")
		return
	}

	if len(s.latencies) == 0 {
		log.Println("No latency data recorded.")
		return
	}

	// Sort latencies for percentile calculations
	sort.Slice(s.latencies, func(i, j int) bool {
		return s.latencies[i] < s.latencies[j]
	})

	// Calculate statistics
	var totalLatency time.Duration
	var maxLatency time.Duration
	minLatency := s.latencies[0]

	for _, lat := range s.latencies {
		totalLatency += lat
		if lat > maxLatency {
			maxLatency = lat
		}
		if lat < minLatency {
			minLatency = lat
		}
	}

	avgLatency := totalLatency / time.Duration(len(s.latencies))

	// Calculate success rate
	successRate := float64(totalQueries-totalFailures) / float64(totalQueries) * 100

	// Calculate percentiles
	p95Index := int(math.Ceil(float64(len(s.latencies))*0.95)) - 1
	p99Index := int(math.Ceil(float64(len(s.latencies))*0.99)) - 1

	if p95Index >= len(s.latencies) {
		p95Index = len(s.latencies) - 1
	}
	if p99Index >= len(s.latencies) {
		p99Index = len(s.latencies) - 1
	}

	p95Latency := s.latencies[p95Index]
	p99Latency := s.latencies[p99Index]

	// Calculate QPS based on actual query execution time (wall clock QPS)
	wallClockQPS := float64(totalQueries) / totalDuration.Seconds()

	// Calculate pure QPS based on first and last successful query
	var pureQPS float64
	if !s.firstQueryTime.IsZero() && !s.lastQueryTime.IsZero() {
		actualQueryDuration := s.lastQueryTime.Sub(s.firstQueryTime).Seconds()
		if actualQueryDuration > 0 {
			pureQPS = float64(len(s.latencies)-1) / actualQueryDuration
		}
	}

	// Helper function to format duration to 3 decimal places in milliseconds
	formatLatency := func(d time.Duration) string {
		ms := float64(d.Nanoseconds()) / 1e6
		return fmt.Sprintf("%.3fms", ms)
	}

	// Print results in a compact, easy-to-copy format
	log.Printf("✅ Query test completed!")

	// Build QPS string conditionally
	qpsStr := fmt.Sprintf("   QPS: %.2f", wallClockQPS)
	if pureQPS > 0 {
		qpsStr = fmt.Sprintf("   QPS (wall clock): %.2f\n   QPS (pure): %.2f", wallClockQPS, pureQPS)
	}

	log.Printf("📊 Final Results:\n   Total Queries: %d\n   Total Failures: %d\n   %s\n   Average Latency: %s\n   Min Latency: %s\n   Max Latency: %s\n   P95 Latency: %s\n   P99 Latency: %s\n   Success Rate: %.2f%%\n   Test Duration: %v",
		totalQueries,
		totalFailures,
		qpsStr,
		formatLatency(avgLatency),
		formatLatency(minLatency),
		formatLatency(maxLatency),
		formatLatency(p95Latency),
		formatLatency(p99Latency),
		successRate,
		totalDuration)

	// Print first 10 queries latency
	if len(s.first10) > 0 {
		var first10Lines string
		for i, lat := range s.first10 {
			first10Lines += fmt.Sprintf("   Query %2d: %s\n", i+1, formatLatency(lat))
		}
		// Remove the last "\n" to avoid extra empty line
		if len(first10Lines) > 0 {
			first10Lines = first10Lines[:len(first10Lines)-1]
		}
		log.Printf("\n🔥 First 10 Queries Latency:\n%s", first10Lines)
	}

	// Print namespace latency statistics
	if len(s.namespaceLatencies) > 0 {
		sort.Slice(s.namespaceLatencies, func(i, j int) bool {
			return s.namespaceLatencies[i] < s.namespaceLatencies[j]
		})

		var totalNsLatency time.Duration
		for _, lat := range s.namespaceLatencies {
			totalNsLatency += lat
		}
		avgNsLatency := totalNsLatency / time.Duration(len(s.namespaceLatencies))
		minNsLatency := s.namespaceLatencies[0]
		maxNsLatency := s.namespaceLatencies[len(s.namespaceLatencies)-1]

		p50NsIndex := int(float64(len(s.namespaceLatencies)) * 0.50)
		p95NsIndex := int(float64(len(s.namespaceLatencies)) * 0.95)
		p99NsIndex := int(float64(len(s.namespaceLatencies)) * 0.99)

		if p50NsIndex >= len(s.namespaceLatencies) {
			p50NsIndex = len(s.namespaceLatencies) - 1
		}
		if p95NsIndex >= len(s.namespaceLatencies) {
			p95NsIndex = len(s.namespaceLatencies) - 1
		}
		if p99NsIndex >= len(s.namespaceLatencies) {
			p99NsIndex = len(s.namespaceLatencies) - 1
		}

		p50NsLatency := s.namespaceLatencies[p50NsIndex]
		p95NsLatency := s.namespaceLatencies[p95NsIndex]
		p99NsLatency := s.namespaceLatencies[p99NsIndex]

		log.Printf("\n📦 Namespace Creation Latency:\n   Count: %d\n   Avg: %s\n   Min: %s\n   Max: %s\n   P50: %s\n   P95: %s\n   P99: %s",
			len(s.namespaceLatencies),
			formatLatency(avgNsLatency),
			formatLatency(minNsLatency),
			formatLatency(maxNsLatency),
			formatLatency(p50NsLatency),
			formatLatency(p95NsLatency),
			formatLatency(p99NsLatency))
	}
}

// NamespacePool manages namespace selection with pre-generated list
type NamespacePool struct {
	namespaces []string
}

// NewNamespacePool creates a namespace pool with pre-generated namespaces
func NewNamespacePool(userIDStart, userIDEnd int) *NamespacePool {
	count := userIDEnd - userIDStart + 1
	namespaces := make([]string, count)

	for i := 0; i < count; i++ {
		namespaces[i] = fmt.Sprintf("id_%d", userIDStart+i)
	}

	if count == 1 {
		log.Printf("✅ Single namespace mode: id_%d (all workers will query this namespace concurrently)", userIDStart)
	} else {
		log.Printf("✅ Pre-generated %d namespaces (id_%d to id_%d)",
			count, userIDStart, userIDEnd)
		log.Printf("   Namespace selection: Random")
	}

	return &NamespacePool{
		namespaces: namespaces,
	}
}

// GetNamespaceLockFree returns a random namespace (thread-safe in Go 1.20+)
func (np *NamespacePool) GetNamespaceLockFree() string {
	poolSize := len(np.namespaces)
	if poolSize == 1 {
		return np.namespaces[0]
	}
	idx := rand.Intn(poolSize)
	return np.namespaces[idx]
}

// performQuery executes a single query (excludes namespace initialization time)
func performQuery(ctx context.Context, ns turbopuffer.Namespace,
	vector []float32, topK int) error {

	_, err := ns.Query(
		ctx,
		turbopuffer.NamespaceQueryParams{
			RankBy: turbopuffer.NewRankByVector("vector", vector),
			TopK:   turbopuffer.Int(int64(topK)),
		},
	)

	return err
}

// worker performs queries continuously until stopped
func worker(ctx context.Context, client turbopuffer.Client,
	vectorPool *VectorPool, namespacePool *NamespacePool,
	topK int, stop <-chan struct{}, stats *Statistics, wg *sync.WaitGroup) {

	defer wg.Done()

	for {
		select {
		case <-stop:
			return
		default:
			// Check if test duration has expired before starting a new query
			if stats.isStopped() {
				return
			}

			// Lock-free access to get vector and namespace
			vector := vectorPool.GetVectorLockFree()
			namespaceStr := namespacePool.GetNamespaceLockFree()

			// Measure namespace creation time
			nsStart := time.Now()
			ns := client.Namespace(namespaceStr)
			nsLatency := time.Since(nsStart)

			// Record namespace latency if test is still running
			if !stats.isStopped() {
				stats.recordNamespaceLatency(nsLatency)
			}

			// Record start time before the actual query
			queryStartTime := time.Now()
			err := performQuery(ctx, ns, vector, topK)
			latency := time.Since(queryStartTime)

			// Only count queries that were started before duration expired
			// Check again with the query start time
			if !stats.isStopped() {
				stats.incrementQueries()

				if err != nil {
					stats.incrementFailures()
					log.Printf("Query error (namespace: %s): %v", namespaceStr, err)
				} else {
					// Record latency only if query was initiated before test stopped
					stats.recordLatencyWithStartTime(latency, queryStartTime)
				}
			}
		}
	}
}

// periodicLogger logs statistics periodically
func periodicLogger(stats *Statistics, interval time.Duration, stop <-chan struct{}) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-stop:
			return
		case <-ticker.C:
			totalQueries := atomic.LoadInt64(&stats.totalQueries)
			totalFailures := atomic.LoadInt64(&stats.totalFailures)
			duration := time.Since(stats.startTime).Seconds()
			qps := float64(totalQueries) / math.Max(duration, 0.001)
			successRate := float64(totalQueries-totalFailures) / float64(math.Max(float64(totalQueries), 1)) * 100

			// Get interval statistics
			stats.mu.Lock()
			currentCount := len(stats.latencies)
			var intervalLatencies []time.Duration
			if currentCount > stats.lastLoggedCount {
				// Copy interval latencies for calculations
				intervalLatencies = make([]time.Duration, currentCount-stats.lastLoggedCount)
				copy(intervalLatencies, stats.latencies[stats.lastLoggedCount:currentCount])
			}
			stats.lastLoggedCount = currentCount
			stats.mu.Unlock()

			// Calculate interval statistics
			var intervalMinMs, intervalAvgMs, intervalP50Ms, intervalP90Ms, intervalP99Ms, intervalP999Ms float64
			var top3Slowest []float64
			if len(intervalLatencies) > 0 {
				// Sort for percentile calculations
				sort.Slice(intervalLatencies, func(i, j int) bool {
					return intervalLatencies[i] < intervalLatencies[j]
				})

				// Min
				intervalMin := intervalLatencies[0]

				// Top 3 slowest (last 3 elements after sorting)
				numSlowest := 3
				if len(intervalLatencies) < numSlowest {
					numSlowest = len(intervalLatencies)
				}
				top3Slowest = make([]float64, numSlowest)
				for i := 0; i < numSlowest; i++ {
					idx := len(intervalLatencies) - numSlowest + i
					top3Slowest[i] = float64(intervalLatencies[idx].Nanoseconds()) / 1e6
				}

				// Average
				var totalLatency time.Duration
				for _, lat := range intervalLatencies {
					totalLatency += lat
				}
				intervalAvg := totalLatency / time.Duration(len(intervalLatencies))

				// Percentiles
				p50Index := int(float64(len(intervalLatencies)) * 0.50)
				p90Index := int(float64(len(intervalLatencies)) * 0.90)
				p99Index := int(float64(len(intervalLatencies)) * 0.99)
				p999Index := int(float64(len(intervalLatencies)) * 0.999)

				if p50Index >= len(intervalLatencies) {
					p50Index = len(intervalLatencies) - 1
				}
				if p90Index >= len(intervalLatencies) {
					p90Index = len(intervalLatencies) - 1
				}
				if p99Index >= len(intervalLatencies) {
					p99Index = len(intervalLatencies) - 1
				}
				if p999Index >= len(intervalLatencies) {
					p999Index = len(intervalLatencies) - 1
				}

				intervalP50 := intervalLatencies[p50Index]
				intervalP90 := intervalLatencies[p90Index]
				intervalP99 := intervalLatencies[p99Index]
				intervalP999 := intervalLatencies[p999Index]

				// Convert to milliseconds
				intervalMinMs = float64(intervalMin.Nanoseconds()) / 1e6
				intervalAvgMs = float64(intervalAvg.Nanoseconds()) / 1e6
				intervalP50Ms = float64(intervalP50.Nanoseconds()) / 1e6
				intervalP90Ms = float64(intervalP90.Nanoseconds()) / 1e6
				intervalP99Ms = float64(intervalP99.Nanoseconds()) / 1e6
				intervalP999Ms = float64(intervalP999.Nanoseconds()) / 1e6
			}

			// Format top 3 slowest
			var top3Str string
			if len(top3Slowest) > 0 {
				top3Str = "["
				for i, v := range top3Slowest {
					if i > 0 {
						top3Str += ", "
					}
					top3Str += fmt.Sprintf("%.3f", v)
				}
				top3Str += "]ms"
			} else {
				top3Str = "[0.000]ms"
			}

			log.Printf("📊 Progress - Total: %d, Failures: %d, QPS: %.2f, Success: %.2f%%, Duration: %.1fs\n   Interval: Avg=%.3fms, Min=%.3fms, Top3Slowest=%s, P50=%.3fms, P90=%.3fms, P99=%.3fms, P99.9=%.3fms",
				totalQueries, totalFailures, qps, successRate, duration,
				intervalAvgMs, intervalMinMs, top3Str, intervalP50Ms, intervalP90Ms, intervalP99Ms, intervalP999Ms)
		}
	}
}

func main() {
	// Command line flags
	concurrency := flag.Int("concurrency", 1, "Number of concurrent queries (1 for serial)")
	jsonFile := flag.String("json", "data/query.json", "Path to JSON file containing query vectors")
	duration := flag.Int("duration", 10, "Duration to run queries in seconds")
	apiKey := flag.String("key", "", "Turbopuffer API key (or set TURBOPUFFER_API_KEY env var)")
	region := flag.String("region", "aws-us-west-2", "Turbopuffer region (e.g., aws-us-west-2)")
	topK := flag.Int("topk", 10, "Number of results to return (top_k)")
	userIDStart := flag.Int("user-id-start", 1, "Start of user ID range for namespaces (id_xxx)")
	userIDEnd := flag.Int("user-id-end", 1, "End of user ID range for namespaces (id_xxx)")
	logInterval := flag.Int("log-interval", 30, "Statistics logging interval in seconds")

	flag.Parse()

	// Set up logging
	log.SetFlags(log.LstdFlags | log.Lshortfile)

	// Validate required parameters
	if *jsonFile == "" {
		log.Fatal("Error: -json flag is required")
	}

	// Get API key from flag or environment variable
	apiKeyValue := *apiKey
	if apiKeyValue == "" {
		apiKeyValue = os.Getenv("TURBOPUFFER_API_KEY")
	}
	if apiKeyValue == "" {
		log.Fatal("Error: -key flag or TURBOPUFFER_API_KEY environment variable is required")
	}

	if *userIDStart > *userIDEnd {
		log.Fatal("Error: user-id-start must be <= user-id-end")
	}

	// Load vectors into memory (one-time operation)
	vectorPool, err := NewVectorPool(*jsonFile)
	if err != nil {
		log.Fatalf("Failed to load vectors: %v", err)
	}

	// Pre-generate all namespaces
	namespacePool := NewNamespacePool(*userIDStart, *userIDEnd)

	log.Printf("Concurrency: %d", *concurrency)
	log.Printf("Duration: %d seconds", *duration)
	log.Printf("Region: %s", *region)
	log.Printf("TopK: %d", *topK)

	// Create Turbopuffer client
	client := turbopuffer.NewClient(
		option.WithAPIKey(apiKeyValue),
		option.WithRegion(*region),
	)

	// Create context
	ctx := context.Background()

	// Initialize statistics with pre-allocated capacity
	stats := NewStatistics()

	// Create stop channel and wait group
	stop := make(chan struct{})
	var wg sync.WaitGroup

	// Start workers
	log.Println("\nStarting query workers...")
	if *concurrency == 1 {
		log.Println("Running in SERIAL mode")
	} else {
		log.Printf("Running with %d concurrent workers", *concurrency)
	}

	for i := 0; i < *concurrency; i++ {
		wg.Add(1)
		go worker(ctx, client, vectorPool, namespacePool, *topK, stop, stats, &wg)
	}

	// Start periodic logger
	logStop := make(chan struct{})
	go periodicLogger(stats, time.Duration(*logInterval)*time.Second, logStop)

	// Run for specified duration
	log.Printf("\nRunning queries for %d seconds...\n", *duration)
	time.Sleep(time.Duration(*duration) * time.Second)

	// Mark test as stopped - queries after this won't be counted
	stats.markStopped()
	stats.endTime = time.Now()

	// Stop workers and logger
	log.Printf("Duration completed, stopping workers...")
	close(stop)
	close(logStop)

	// Give workers a short grace period to finish current queries
	// Don't wait indefinitely for stuck queries
	gracePeriod := 3 * time.Second
	log.Printf("Waiting %v for workers to complete current queries...", gracePeriod)

	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		log.Printf("All workers completed gracefully")
	case <-time.After(gracePeriod):
		log.Printf("Grace period expired, some workers may still be running (will be abandoned)")
	}

	// Print final statistics
	stats.printStats()
}
