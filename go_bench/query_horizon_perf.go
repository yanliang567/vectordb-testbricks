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
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
)

// expandPath expands ~ to user home directory
func expandPath(path string) (string, error) {
	if strings.HasPrefix(path, "~/") {
		homeDir, err := os.UserHomeDir()
		if err != nil {
			return "", fmt.Errorf("failed to get home directory: %v", err)
		}
		return filepath.Join(homeDir, path[2:]), nil
	}
	return path, nil
}

// QueryStats contains performance statistics
type QueryStats struct {
	totalQueries   int64
	totalFailures  int64
	totalLatencyMS int64
	totalResults   int64
	minLatencyMS   int64
	maxLatencyMS   int64
	latencies      []int64
	startTime      time.Time
	mu             sync.RWMutex
}

// NewQueryStats creates a new QueryStats instance
func NewQueryStats() *QueryStats {
	return &QueryStats{
		startTime:    time.Now(),
		minLatencyMS: math.MaxInt64,
		maxLatencyMS: 0,
		latencies:    make([]int64, 0, 10000),
	}
}

// RecordQuery records a query result
func (s *QueryStats) RecordQuery(latency time.Duration, resultCount int, success bool) {
	s.mu.Lock()
	defer s.mu.Unlock()

	latencyMS := latency.Nanoseconds() / 1e6
	atomic.AddInt64(&s.totalQueries, 1)
	atomic.AddInt64(&s.totalLatencyMS, latencyMS)
	atomic.AddInt64(&s.totalResults, int64(resultCount))

	if !success {
		atomic.AddInt64(&s.totalFailures, 1)
	}

	if latencyMS < s.minLatencyMS {
		s.minLatencyMS = latencyMS
	}
	if latencyMS > s.maxLatencyMS {
		s.maxLatencyMS = latencyMS
	}

	// Keep only recent samples for percentile calculation
	if len(s.latencies) >= 10000 {
		s.latencies = s.latencies[1000:]
	}
	s.latencies = append(s.latencies, latencyMS)
}

// Reset clears all statistics and restarts timing
func (s *QueryStats) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.totalQueries = 0
	s.totalFailures = 0
	s.totalLatencyMS = 0
	s.totalResults = 0
	s.minLatencyMS = math.MaxInt64
	s.maxLatencyMS = 0
	s.latencies = make([]int64, 0, 10000)
	s.startTime = time.Now()
}

// GetStats returns current statistics
func (s *QueryStats) GetStats() map[string]interface{} {
	s.mu.RLock()
	defer s.mu.RUnlock()

	duration := time.Since(s.startTime).Seconds()
	totalQueries := float64(atomic.LoadInt64(&s.totalQueries))
	totalFailures := float64(atomic.LoadInt64(&s.totalFailures))
	totalLatencyMS := float64(atomic.LoadInt64(&s.totalLatencyMS))
	totalResults := atomic.LoadInt64(&s.totalResults)

	var qps, avgLatency, successRate float64
	if duration > 0 {
		qps = totalQueries / duration
	}
	if totalQueries > 0 {
		avgLatency = totalLatencyMS / totalQueries
		successRate = ((totalQueries - totalFailures) / totalQueries) * 100
	}

	// Calculate percentiles
	latenciesCopy := make([]int64, len(s.latencies))
	copy(latenciesCopy, s.latencies)
	sort.Slice(latenciesCopy, func(i, j int) bool {
		return latenciesCopy[i] < latenciesCopy[j]
	})

	var p95, p99 float64
	if len(latenciesCopy) > 0 {
		p95Idx := int(float64(len(latenciesCopy)) * 0.95)
		p99Idx := int(float64(len(latenciesCopy)) * 0.99)
		if p95Idx >= len(latenciesCopy) {
			p95Idx = len(latenciesCopy) - 1
		}
		if p99Idx >= len(latenciesCopy) {
			p99Idx = len(latenciesCopy) - 1
		}
		p95 = float64(latenciesCopy[p95Idx])
		p99 = float64(latenciesCopy[p99Idx])
	}

	var avgResults float64
	if totalQueries > 0 {
		avgResults = float64(totalResults) / totalQueries
	}

	return map[string]interface{}{
		"qps":            qps,
		"avg_latency":    avgLatency,
		"min_latency":    float64(s.minLatencyMS),
		"max_latency":    float64(s.maxLatencyMS),
		"p95_latency":    p95,
		"p99_latency":    p99,
		"success_rate":   successRate,
		"total_queries":  totalQueries,
		"total_failures": totalFailures,
		"total_results":  float64(totalResults),
		"avg_results":    avgResults,
		"duration":       duration,
	}
}

// PreGeneratedQuery represents a query with pre-generated expression
type PreGeneratedQuery struct {
	QueryID          int                    `json:"query_id"`
	OriginalQuery    map[string]interface{} `json:"original_query"`
	MilvusExpression string                 `json:"milvus_expression"`
}

// ExpressionFileData represents the structure of expression JSON files
type ExpressionFileData struct {
	SourceFile   string              `json:"source_file"`
	TotalQueries int                 `json:"total_queries"`
	Queries      []PreGeneratedQuery `json:"queries"`
}

// ExpressionGenerator generates random filter expressions
type ExpressionGenerator struct {
	preGeneratedExprs []string // Pre-loaded expressions from JSON files
	mu                sync.RWMutex
}

// NewExpressionGenerator creates a new expression generator
func NewExpressionGenerator() *ExpressionGenerator {
	return &ExpressionGenerator{
		preGeneratedExprs: make([]string, 0),
	}
}

// LoadPreGeneratedExpressions loads pre-generated expressions from JSON files
func (eg *ExpressionGenerator) LoadPreGeneratedExpressions(expressionFiles []string) error {
	eg.mu.Lock()
	defer eg.mu.Unlock()

	var allExpressions []string

	for _, filePath := range expressionFiles {
		// Expand ~ path to full path
		expandedPath, err := expandPath(filePath)
		if err != nil {
			return fmt.Errorf("failed to expand file path %s: %v", filePath, err)
		}

		// Check if file exists
		if _, err := os.Stat(expandedPath); os.IsNotExist(err) {
			return fmt.Errorf("❌ Expression file does not exist: %s", expandedPath)
		}

		log.Printf("📖 Loading pre-generated expressions from %s", expandedPath)

		file, err := os.Open(expandedPath)
		if err != nil {
			return fmt.Errorf("failed to open expression file: %v", err)
		}
		defer file.Close()

		// Read and parse JSON
		data, err := io.ReadAll(file)
		if err != nil {
			return fmt.Errorf("failed to read expression file: %v", err)
		}

		var exprData ExpressionFileData
		if err := json.Unmarshal(data, &exprData); err != nil {
			return fmt.Errorf("failed to parse expression JSON: %v", err)
		}

		if len(exprData.Queries) == 0 {
			log.Printf("⚠️ Warning: No queries found in %s", filePath)
			continue
		}

		// Extract expressions
		for _, query := range exprData.Queries {
			if query.MilvusExpression != "" {
				allExpressions = append(allExpressions, query.MilvusExpression)
			}
		}

		log.Printf("✅ Loaded %d expressions from %s (source: %s)",
			len(exprData.Queries), filepath.Base(expandedPath), exprData.SourceFile)
	}

	if len(allExpressions) == 0 {
		return fmt.Errorf("❌ No expressions loaded from any file")
	}

	eg.preGeneratedExprs = allExpressions
	log.Printf("✅ Total pre-generated expressions loaded: %d", len(eg.preGeneratedExprs))

	return nil
}

// GenerateExpression generates a random filter expression
func (eg *ExpressionGenerator) GenerateExpression() string {
	eg.mu.RLock()
	defer eg.mu.RUnlock()

	if len(eg.preGeneratedExprs) == 0 {
		log.Printf("⚠️ Warning: No pre-generated expressions loaded")
		return ""
	}

	// Randomly select an expression from the loaded expressions
	idx := rand.Intn(len(eg.preGeneratedExprs))
	return eg.preGeneratedExprs[idx]
}

// QueryTask represents a single query task
type QueryTask struct {
	Limit        int
	OutputFields []string
	Timeout      time.Duration
	ResultRatio  float64 // Ratio to check if result count is sufficient (e.g., 0.9 means 90% of limit)
}

// QueryResult represents the result of a query task
type QueryResult struct {
	Success     bool
	Latency     time.Duration
	ResultCount int
	Error       string
}

// QueryWorker performs query operations
type QueryWorker struct {
	client     *milvusclient.Client
	collection string
	exprGen    *ExpressionGenerator
	stats      *QueryStats
}

// NewQueryWorker creates a new query worker
func NewQueryWorker(milvusClient *milvusclient.Client, collectionName string,
	exprGen *ExpressionGenerator, stats *QueryStats) *QueryWorker {
	return &QueryWorker{
		client:     milvusClient,
		collection: collectionName,
		exprGen:    exprGen,
		stats:      stats,
	}
}

// PerformQuery executes a query task
func (qw *QueryWorker) PerformQuery(ctx context.Context, task *QueryTask) *QueryResult {
	startTime := time.Now()

	queryCtx, cancel := context.WithTimeout(ctx, task.Timeout)
	defer cancel()

	result := &QueryResult{}

	// Generate expression
	expr := qw.exprGen.GenerateExpression()
	if expr == "" {
		result.Success = false
		result.Error = "empty expression"
		result.Latency = time.Since(startTime)
		return result
	}

	// Execute query using v2.6 milvusclient API
	queryOption := milvusclient.NewQueryOption(qw.collection).
		WithFilter(expr).
		WithOutputFields(task.OutputFields...).
		WithLimit(task.Limit)

	queryResult, err := qw.client.Query(queryCtx, queryOption)
	result.Latency = time.Since(startTime)

	if err != nil {
		result.Success = false
		result.Error = fmt.Sprintf("query failed: %v", err)
		// Only log error details occasionally to avoid flooding logs
		return result
	}

	result.Success = true
	result.ResultCount = queryResult.ResultCount

	// Log warning if result count is less than expected
	if task.ResultRatio > 0 && result.ResultCount < int(float64(task.Limit)*task.ResultRatio) {
		log.Printf("⚠️ Warning: Query returned %d results, less than requested limit=%d * %.2f = %.0f (filter: %s)",
			result.ResultCount, task.Limit, task.ResultRatio, float64(task.Limit)*task.ResultRatio, expr)
	}

	return result
}

// QueryHorizonPerf is the main query performance tester
type QueryHorizonPerf struct {
	client     *milvusclient.Client
	collection string
	stats      *QueryStats
	exprGen    *ExpressionGenerator
}

// NewQueryHorizonPerf creates a new query performance tester
func NewQueryHorizonPerf(uri, token, collectionName string, expressionFiles []string) (*QueryHorizonPerf, error) {
	ctx := context.Background()

	// Create Milvus client using v2.6 milvusclient API
	clientConfig := &milvusclient.ClientConfig{
		Address: uri,
	}

	if token != "" {
		clientConfig.APIKey = token
	}

	milvusClient, err := milvusclient.New(ctx, clientConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create Milvus client: %v", err)
	}

	// Verify collection exists and is loaded
	has, err := milvusClient.HasCollection(ctx, milvusclient.NewHasCollectionOption(collectionName))
	if err != nil {
		return nil, fmt.Errorf("failed to check collection '%s' existence: %v", collectionName, err)
	}
	if !has {
		return nil, fmt.Errorf("collection '%s' does not exist", collectionName)
	}

	// Check if collection is loaded
	loadState, err := milvusClient.GetLoadState(ctx, milvusclient.NewGetLoadStateOption(collectionName))
	if err != nil {
		return nil, fmt.Errorf("failed to get load state: %v", err)
	}

	if loadState.State != entity.LoadStateLoaded {
		return nil, fmt.Errorf("collection %s is not loaded (current state: %v)", collectionName, loadState.State)
	}

	// Create expression generator
	exprGen := NewExpressionGenerator()

	// Load pre-generated expressions
	if len(expressionFiles) > 0 {
		if err := exprGen.LoadPreGeneratedExpressions(expressionFiles); err != nil {
			return nil, fmt.Errorf("failed to load pre-generated expressions: %v", err)
		}
	}

	return &QueryHorizonPerf{
		client:     milvusClient,
		collection: collectionName,
		stats:      NewQueryStats(),
		exprGen:    exprGen,
	}, nil
}

// RunQueryTest runs the query performance test
func (qhp *QueryHorizonPerf) RunQueryTest(ctx context.Context, limit, maxWorkers int,
	timeout time.Duration, outputFields []string, queryTimeout time.Duration, resultRatio float64) error {

	// Reset statistics for this individual test
	qhp.stats.Reset()

	log.Printf("🚀 Starting query performance test...")
	log.Printf("   Collection: %s", qhp.collection)
	log.Printf("   Max Workers: %d", maxWorkers)
	log.Printf("   Test Timeout: %v", timeout)
	log.Printf("   Limit: %d", limit)
	log.Printf("   Output Fields: %v", outputFields)

	// Create worker pool
	workers := make([]*QueryWorker, maxWorkers)
	for i := 0; i < maxWorkers; i++ {
		workers[i] = NewQueryWorker(qhp.client, qhp.collection, qhp.exprGen, qhp.stats)
	}

	// Create channels for task distribution
	taskChan := make(chan *QueryTask, maxWorkers*2)
	resultChan := make(chan *QueryResult, maxWorkers*2)

	// Start worker goroutines
	var wg sync.WaitGroup
	testCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	for i := 0; i < maxWorkers; i++ {
		wg.Add(1)
		go func(worker *QueryWorker) {
			defer wg.Done()
			for {
				select {
				case <-testCtx.Done():
					return
				case task := <-taskChan:
					if task == nil {
						return
					}
					result := worker.PerformQuery(testCtx, task)
					select {
					case resultChan <- result:
					case <-testCtx.Done():
						return
					}
				}
			}
		}(workers[i])
	}

	// Result collector goroutine
	errorCount := int64(0)
	go func() {
		for result := range resultChan {
			qhp.stats.RecordQuery(result.Latency, result.ResultCount, result.Success)
			if !result.Success {
				// Only log first 10 errors and then sample every 100th error
				errNum := atomic.AddInt64(&errorCount, 1)
				if errNum <= 10 || errNum%100 == 0 {
					log.Printf("❌ Query failed (#%d): %s", errNum, result.Error)
				}
			}
		}
	}()

	// Task generator
	taskCount := int64(0)
	lastLogTime := time.Now()
	lastLoggedTasks := int64(0)
	logInterval := int64(maxWorkers * 100)
	if logInterval > 1000 {
		logInterval = 1000
	}

	taskGenWg := sync.WaitGroup{}
	taskGenWg.Add(1)

	go func() {
		defer taskGenWg.Done()
		defer close(taskChan)

		for {
			select {
			case <-testCtx.Done():
				return
			default:
				// Generate task
				task := &QueryTask{
					Limit:        limit,
					OutputFields: outputFields,
					Timeout:      queryTimeout,
					ResultRatio:  resultRatio,
				}

				select {
				case taskChan <- task:
					current := atomic.AddInt64(&taskCount, 1)

					// Log progress periodically
					if current-lastLoggedTasks >= logInterval || time.Since(lastLogTime) > 10*time.Second {
						stats := qhp.stats.GetStats()
						log.Printf("📊 Progress - Submitted: %d, QPS: %.1f, Avg: %.1f ms, P99: %.1f ms, Success: %.1f%%",
							current, stats["qps"], stats["avg_latency"].(float64),
							stats["p99_latency"].(float64), stats["success_rate"])
						lastLoggedTasks = current
						lastLogTime = time.Now()
					}

				case <-testCtx.Done():
					return
				}
			}
		}
	}()

	// Wait for test completion
	<-testCtx.Done()
	log.Printf("⏰ Test timeout reached, stopping task generation...")

	taskGenWg.Wait()
	log.Printf("📝 Task generation stopped, waiting for workers to finish...")

	// Wait for workers to complete
	wg.Wait()
	close(resultChan)

	log.Printf("✅ Query test completed!")

	// Print final statistics
	finalStats := qhp.stats.GetStats()
	log.Printf("📊 Final Results:")
	fmt.Printf("RESULTS | Total_Queries=%d | Total_Failures=%d | Total_Results=%d | QPS=%.1f | Avg_Results=%.1f | Avg_Latency=%.1fms | Min_Latency=%.1fms | Max_Latency=%.1fms | P95_Latency=%.1fms | P99_Latency=%.1fms | Success_Rate=%.2f%% | Duration=%.2fs\n",
		int64(finalStats["total_queries"].(float64)),
		int64(finalStats["total_failures"].(float64)),
		int64(finalStats["total_results"].(float64)),
		finalStats["qps"],
		finalStats["avg_results"],
		finalStats["avg_latency"].(float64),
		finalStats["min_latency"].(float64),
		finalStats["max_latency"].(float64),
		finalStats["p95_latency"].(float64),
		finalStats["p99_latency"].(float64),
		finalStats["success_rate"],
		finalStats["duration"])

	return nil
}

// Close closes the Milvus client connection
func (qhp *QueryHorizonPerf) Close() error {
	ctx := context.Background()
	return qhp.client.Close(ctx)
}

func main() {
	// Command line arguments
	var (
		fileWorkers     = flag.String("file-workers", "all:1", "File and worker pairs: 'file1.json:10,file2.json:20' or 'all:10' for all files")
		testTimeout     = flag.Int("timeout", 300, "Test timeout in seconds")
		queryTimeoutSec = flag.Int("query-timeout", 30, "Individual query timeout in seconds")
		exprDir         = flag.String("expr-dir", "/root/horizon/horizonPoc/data/query_expressions", "Directory containing expression JSON files")
		filePattern     = flag.String("file-pattern", "*_exprs.json", "File pattern to match when using 'all' (e.g., '*_pic.json', '*.json')")
		limit           = flag.Int("limit", 15000, "Limit for query results")
		sleepSec        = flag.Int("sleep-sec", 120, "Sleep seconds between tests")
		outputFieldsStr = flag.String("output-fields", "timestamp,device_id,expert_collected,sensor_lidar_type,gcj02_lon,gcj02_lat", "Comma-separated output fields")
		resultRatio     = flag.Float64("result-ratio", 1.0, "Ratio to check if result count is sufficient (e.g., 0.9 means warn if results < 90% of limit)")

		// Hardcoded values
		host           = "https://in01-3e1a7693c28817d.ali-cn-hangzhou.cloud-uat.zilliz.cn:19530"
		collectionName = "horizon_test_collection"
		apiKey         = "cc5bf695ea9236e2c64617e9407a26cf0953034485d27216f8b3f145e3eb72396e042db2abb91c4ef6fde723af70e754d68ca787"
	)

	flag.Parse()

	log.SetFlags(log.LstdFlags | log.Lshortfile)

	// Parse output fields
	var outputFields []string
	if *outputFieldsStr != "" {
		outputFields = strings.Split(*outputFieldsStr, ",")
		for i := range outputFields {
			outputFields[i] = strings.TrimSpace(outputFields[i])
		}
	}

	// Check if file-workers is provided
	if *fileWorkers == "" {
		log.Fatalf("❌ --file-workers parameter is required. Example: 'all:10' or 'file1.json:10,file2.json:20'")
	}

	// Define test configuration structure
	type TestConfig struct {
		ExpressionFiles []string
		Workers         int
		TestName        string
	}

	// Parse file-workers parameter
	var testConfigs []TestConfig

	exprDirPath, err := expandPath(*exprDir)
	if err != nil {
		log.Fatalf("❌ Failed to expand expression directory path: %v", err)
	}

	configs := strings.Split(*fileWorkers, ",")
	for _, config := range configs {
		config = strings.TrimSpace(config)
		parts := strings.Split(config, ":")
		if len(parts) != 2 {
			log.Fatalf("❌ Invalid file-workers format: %s. Expected format: 'file:workers' or 'all:workers'", config)
		}

		fileName := strings.TrimSpace(parts[0])
		workersStr := strings.TrimSpace(parts[1])

		workers, err := strconv.Atoi(workersStr)
		if err != nil {
			log.Fatalf("❌ Invalid worker count in %s: %s", config, workersStr)
		}

		if fileName == "all" {
			// Expand "all" to create a separate test for each expression file matching the pattern
			entries, err := os.ReadDir(exprDirPath)
			if err != nil {
				log.Fatalf("❌ Failed to read expression directory: %v", err)
			}

			log.Printf("🔍 Searching for files matching pattern '%s' in %s", *filePattern, exprDirPath)

			var allFiles []string
			for _, entry := range entries {
				if entry.IsDir() {
					continue
				}

				// Use filepath.Match to match the pattern
				matched, err := filepath.Match(*filePattern, entry.Name())
				if err != nil {
					log.Fatalf("❌ Invalid file pattern '%s': %v", *filePattern, err)
				}

				if matched {
					fullPath := filepath.Join(exprDirPath, entry.Name())
					allFiles = append(allFiles, fullPath)
					log.Printf("   ✓ Matched: %s", entry.Name())
				}
			}

			if len(allFiles) == 0 {
				log.Fatalf("❌ No expression files matching pattern '%s' found in %s", *filePattern, exprDirPath)
			}

			log.Printf("✅ Found %d matching file(s)", len(allFiles))

			// Create a separate test config for each file
			for _, filePath := range allFiles {
				baseName := filepath.Base(filePath)
				// Remove common suffixes
				baseName = strings.TrimSuffix(baseName, "_exprs.json")
				baseName = strings.TrimSuffix(baseName, ".json")
				testName := fmt.Sprintf("%s_%d_workers", baseName, workers)

				testConfigs = append(testConfigs, TestConfig{
					ExpressionFiles: []string{filePath},
					Workers:         workers,
					TestName:        testName,
				})
			}
		} else {
			// Load specific file
			var fullPath string
			if filepath.IsAbs(fileName) {
				fullPath = fileName
			} else {
				fullPath = filepath.Join(exprDirPath, fileName)
			}

			// Check if file exists
			if _, err := os.Stat(fullPath); os.IsNotExist(err) {
				log.Fatalf("❌ Expression file does not exist: %s", fullPath)
			}

			// Extract base name without _expressions.json suffix for cleaner test name
			baseName := filepath.Base(fileName)
			baseName = strings.TrimSuffix(baseName, "_exprs.json")
			baseName = strings.TrimSuffix(baseName, ".json")
			testName := fmt.Sprintf("%s_%d_workers", baseName, workers)

			testConfigs = append(testConfigs, TestConfig{
				ExpressionFiles: []string{fullPath},
				Workers:         workers,
				TestName:        testName,
			})
		}
	}

	// Log the test configurations
	log.Printf("📋 Test configurations (%d tests):", len(testConfigs))
	for i, config := range testConfigs {
		fileCount := len(config.ExpressionFiles)
		if fileCount == 1 {
			log.Printf("   %d. %s: %s with %d workers",
				i+1, config.TestName, filepath.Base(config.ExpressionFiles[0]), config.Workers)
		} else {
			log.Printf("   %d. %s: %d files with %d workers",
				i+1, config.TestName, fileCount, config.Workers)
		}
	}

	// Run tests sequentially for each configuration
	for i, config := range testConfigs {
		log.Printf("🎯 Running test %d/%d: %s", i+1, len(testConfigs), config.TestName)

		// Log expression files being loaded
		if len(config.ExpressionFiles) == 1 {
			log.Printf("📖 Loading expression file: %s", filepath.Base(config.ExpressionFiles[0]))
		} else {
			log.Printf("📖 Loading %d expression files:", len(config.ExpressionFiles))
			for _, f := range config.ExpressionFiles {
				log.Printf("   - %s", filepath.Base(f))
			}
		}

		// Create query performance tester for this specific test
		qhp, err := NewQueryHorizonPerf(host, apiKey, collectionName, config.ExpressionFiles)
		if err != nil {
			log.Fatalf("❌ Failed to create query performance tester: %v", err)
		}

		ctx := context.Background()
		err = qhp.RunQueryTest(ctx, *limit, config.Workers,
			time.Duration(*testTimeout)*time.Second, outputFields,
			time.Duration(*queryTimeoutSec)*time.Second, *resultRatio)

		if err != nil {
			log.Printf("❌ Test %d/%d failed: %s: %v", i+1, len(testConfigs), config.TestName, err)
		} else {
			log.Printf("✅ Test %d/%d completed: %s", i+1, len(testConfigs), config.TestName)
		}

		// Close the client
		qhp.Close()

		// Brief pause between tests (skip for last test)
		if i < len(testConfigs)-1 {
			log.Printf("⏸️  Pausing %d seconds before next test...", *sleepSec)
			time.Sleep(time.Duration(*sleepSec) * time.Second)
		}
	}

	log.Printf("🎉 All tests completed!")
}
