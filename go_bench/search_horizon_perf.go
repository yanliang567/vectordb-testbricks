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

// SearchStats contains performance statistics
type SearchStats struct {
	totalSearches  int64
	totalFailures  int64
	totalLatencyMS int64
	minLatencyMS   int64
	maxLatencyMS   int64
	latencies      []int64
	startTime      time.Time
	mu             sync.RWMutex
}

// NewSearchStats creates a new SearchStats instance
func NewSearchStats() *SearchStats {
	return &SearchStats{
		startTime:    time.Now(),
		minLatencyMS: math.MaxInt64,
		maxLatencyMS: 0,
		latencies:    make([]int64, 0, 10000),
	}
}

// RecordSearch records a search result
func (s *SearchStats) RecordSearch(latency time.Duration, success bool) {
	s.mu.Lock()
	defer s.mu.Unlock()

	latencyMS := latency.Nanoseconds() / 1e6
	atomic.AddInt64(&s.totalSearches, 1)
	atomic.AddInt64(&s.totalLatencyMS, latencyMS)

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
func (s *SearchStats) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.totalSearches = 0
	s.totalFailures = 0
	s.totalLatencyMS = 0
	s.minLatencyMS = math.MaxInt64
	s.maxLatencyMS = 0
	s.latencies = make([]int64, 0, 10000)
	s.startTime = time.Now()
}

// GetStats returns current statistics
func (s *SearchStats) GetStats() map[string]interface{} {
	s.mu.RLock()
	defer s.mu.RUnlock()

	duration := time.Since(s.startTime).Seconds()
	qps := float64(s.totalSearches) / math.Max(duration, 0.001)

	var avgLatency, p95Latency, p99Latency float64
	successRate := 100.0

	if s.totalSearches > 0 {
		avgLatency = float64(s.totalLatencyMS) / float64(s.totalSearches)
		successRate = float64(s.totalSearches-s.totalFailures) / float64(s.totalSearches) * 100
	}

	if len(s.latencies) > 0 {
		// Calculate percentiles
		sortedLatencies := make([]int64, len(s.latencies))
		copy(sortedLatencies, s.latencies)

		// Efficient sort using sort package
		sort.Slice(sortedLatencies, func(i, j int) bool {
			return sortedLatencies[i] < sortedLatencies[j]
		})

		p95Index := int(float64(len(sortedLatencies)) * 0.95)
		p99Index := int(float64(len(sortedLatencies)) * 0.99)

		if p95Index >= len(sortedLatencies) {
			p95Index = len(sortedLatencies) - 1
		}
		if p99Index >= len(sortedLatencies) {
			p99Index = len(sortedLatencies) - 1
		}

		p95Latency = float64(sortedLatencies[p95Index])
		p99Latency = float64(sortedLatencies[p99Index])
	}

	return map[string]interface{}{
		"total_searches": s.totalSearches,
		"total_failures": s.totalFailures,
		"qps":            qps,
		"avg_latency":    avgLatency,
		"min_latency":    float64(s.minLatencyMS),
		"max_latency":    float64(s.maxLatencyMS),
		"p95_latency":    p95Latency,
		"p99_latency":    p99Latency,
		"success_rate":   successRate,
		"duration":       duration,
	}
}

// QueryVectorPool manages query vectors from parquet file
type QueryVectorPool struct {
	vectors    [][]float32
	currentIdx int64
	mu         sync.RWMutex
}

// NewQueryVectorPool creates a new QueryVectorPool
func NewQueryVectorPool(filePath string) (*QueryVectorPool, error) {
	pool := &QueryVectorPool{
		vectors:    make([][]float32, 0),
		currentIdx: 0,
	}

	if err := pool.LoadVectorsFromFile(filePath); err != nil {
		return nil, err
	}

	return pool, nil
}

// LoadVectorsFromFile loads vectors from file (supports CSV and text formats)
func (qvp *QueryVectorPool) LoadVectorsFromFile(filePath string) error {
	log.Printf("📖 Loading query vectors from %s", filePath)

	// Expand ~ path to full path
	expandedPath, err := expandPath(filePath)
	if err != nil {
		return fmt.Errorf("failed to expand file path %s: %v", filePath, err)
	}

	// Check if file exists
	if _, err := os.Stat(expandedPath); os.IsNotExist(err) {
		return fmt.Errorf("❌ Query vector file does not exist: %s", expandedPath)
	} else if err != nil {
		return fmt.Errorf("❌ Failed to check file status: %s, error: %v", expandedPath, err)
	}

	log.Printf("✅ Found query vector file: %s", expandedPath)

	// Only support JSON format
	ext := strings.ToLower(filepath.Ext(expandedPath))
	if ext != ".json" {
		return fmt.Errorf("❌ Only JSON format is supported. Please use .json file extension")
	}

	qvp.mu.Lock()
	defer qvp.mu.Unlock()

	return qvp.loadFromJSON(expandedPath)
}

// loadFromJSON reads vectors from JSON file (format: [][]float64)
func (qvp *QueryVectorPool) loadFromJSON(filePath string) error {
	file, err := os.Open(filePath)
	if err != nil {
		return fmt.Errorf("failed to open JSON file: %v", err)
	}
	defer file.Close()

	// Read entire file once
	data, err := io.ReadAll(file)
	if err != nil {
		return fmt.Errorf("failed to read JSON file: %v", err)
	}

	// Parse JSON as [][]float64
	var vectors [][]float64
	if err := json.Unmarshal(data, &vectors); err != nil {
		return fmt.Errorf("failed to parse JSON: %v", err)
	}

	if len(vectors) == 0 {
		return fmt.Errorf("❌ No vectors found in JSON file")
	}

	// Pre-allocate slice with known capacity to avoid resizing
	qvp.vectors = make([][]float32, 0, len(vectors))

	// Convert float64 to float32 efficiently
	for _, vector64 := range vectors {
		vector32 := make([]float32, len(vector64))
		for i, val := range vector64 {
			vector32[i] = float32(val)
		}
		qvp.vectors = append(qvp.vectors, vector32)
	}

	log.Printf("✅ Loaded %d vectors from JSON file (dimension: %d)", len(qvp.vectors), len(qvp.vectors[0]))
	return nil
}

// GetVectors returns nq vectors starting from current index
// Returns references to original vectors to avoid memory copying
func (qvp *QueryVectorPool) GetVectors(nq int) [][]float32 {
	qvp.mu.Lock()
	defer qvp.mu.Unlock()

	if len(qvp.vectors) == 0 {
		log.Printf("❌ ERROR: No vectors loaded from JSON file!")
		panic("Vector pool is empty - ensure JSON file was loaded correctly")
	}

	result := make([][]float32, nq)
	poolSize := int64(len(qvp.vectors))

	for i := 0; i < nq; i++ {
		idx := (qvp.currentIdx + int64(i)) % poolSize
		// Return reference to original vector instead of copying
		// This is safe since vectors are read-only in search operations
		result[i] = qvp.vectors[idx]
	}

	qvp.currentIdx = (qvp.currentIdx + int64(nq)) % poolSize
	return result
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

// GenerateExpression generates a random filter expression from pre-loaded expressions
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

// SearchTask represents a single search task
type SearchTask struct {
	SearchType     string
	VectorField    string
	NQ             int
	TopK           int
	OutputFields   []string
	Timeout        time.Duration
	ExpressionType string
	ResultRatio    float64 // Ratio to check if result count is sufficient (e.g., 0.9 means 90% of topK)
}

// SearchResult represents the result of a search task
type SearchResult struct {
	Success     bool
	Latency     time.Duration
	ResultCount int
	SearchType  string
	Error       string
}

// SearchWorker performs search operations
type SearchWorker struct {
	client     *milvusclient.Client
	collection string
	vectorPool *QueryVectorPool
	exprGen    *ExpressionGenerator
	stats      *SearchStats
}

// NewSearchWorker creates a new search worker
func NewSearchWorker(milvusClient *milvusclient.Client, collectionName string,
	vectorPool *QueryVectorPool, exprGen *ExpressionGenerator, stats *SearchStats) *SearchWorker {
	return &SearchWorker{
		client:     milvusClient,
		collection: collectionName,
		vectorPool: vectorPool,
		exprGen:    exprGen,
		stats:      stats,
	}
}

// PerformSearch executes a search task
func (sw *SearchWorker) PerformSearch(ctx context.Context, task *SearchTask) *SearchResult {
	startTime := time.Now()

	searchCtx, cancel := context.WithTimeout(ctx, task.Timeout)
	defer cancel()

	result := &SearchResult{
		SearchType: task.SearchType,
	}

	if task.SearchType == "hybrid" {
		// Get fresh query vectors for each search to ensure diversity
		// get nq + 1 vectors from the vector pool for hybrid search
		queryVectors := sw.vectorPool.GetVectors(task.NQ + 1)

		// Convert [][]float32 to []entity.Vector
		vectors := make([]entity.Vector, len(queryVectors))
		for i, vector := range queryVectors {
			vectors[i] = entity.FloatVector(vector)
		}

		// Hybrid search using v2.6 milvusclient API
		// Create ANN requests with independent expressions from GenerateExpression
		expr1 := sw.exprGen.GenerateExpression()
		// expr2 := sw.exprGen.GenerateExpression()

		annReq1 := milvusclient.NewAnnRequest(task.VectorField, task.TopK, vectors[0])
		annReq2 := milvusclient.NewAnnRequest(task.VectorField, task.TopK, vectors[1])

		// Set the same expression for each ANN request
		if expr1 != "" {
			annReq1 = annReq1.WithFilter(expr1)
			annReq2 = annReq2.WithFilter(expr1)
		}

		// Use WeightedRanker instead of RRFReranker
		searchOpt := milvusclient.NewHybridSearchOption(
			sw.collection,
			task.TopK,
			annReq1,
			annReq2,
		).WithReranker(milvusclient.NewWeightedReranker([]float64{0.6, 0.4}))

		// Add output fields if specified
		if len(task.OutputFields) > 0 {
			searchOpt = searchOpt.WithOutputFields(task.OutputFields...)
		}

		searchResults, err := sw.client.HybridSearch(searchCtx, searchOpt)
		if err != nil {
			result.Error = err.Error()
		} else {
			result.Success = true
			if len(searchResults) > 0 {
				// Get the length from the IDs column
				if idsCol := searchResults[0].IDs; idsCol != nil {
					result.ResultCount = idsCol.Len()
					// Check if results are insufficient
					if task.ResultRatio > 0 && result.ResultCount < int(float64(task.TopK)*task.ResultRatio) {
						log.Printf("⚠️ Warning: Hybrid search returned %d results, less than requested topK=%d * %.2f = %.0f (expr1: %s, expr2: %s)",
							result.ResultCount, task.TopK, task.ResultRatio, float64(task.TopK)*task.ResultRatio, expr1, expr1)
					}
				}
			}
		}
	} else {
		// get nq vectors from the vector pool for normal search
		queryVectors := sw.vectorPool.GetVectors(task.NQ)

		// Convert [][]float32 to []entity.Vector
		vectors := make([]entity.Vector, len(queryVectors))
		for i, vector := range queryVectors {
			vectors[i] = entity.FloatVector(vector)
		}
		// Normal search using v2.6 milvusclient API
		searchOpt := milvusclient.NewSearchOption(sw.collection, task.TopK, vectors).WithANNSField(task.VectorField)

		// Generate fresh filter expression for each search (same as hybrid search)
		filter := ""
		if task.ExpressionType != "" {
			filter = sw.exprGen.GenerateExpression()
		}
		if filter != "" {
			searchOpt = searchOpt.WithFilter(filter)
		}

		// Add output fields if specified
		if len(task.OutputFields) > 0 {
			searchOpt = searchOpt.WithOutputFields(task.OutputFields...)
		}

		searchResults, err := sw.client.Search(searchCtx, searchOpt)
		if err != nil {
			result.Error = err.Error()
		} else {
			result.Success = true
			if len(searchResults) > 0 {
				// Get the length from the IDs column
				if idsCol := searchResults[0].IDs; idsCol != nil {
					result.ResultCount = idsCol.Len()
					// Check if results are insufficient
					if task.ResultRatio > 0 && result.ResultCount < int(float64(task.TopK)*task.ResultRatio) {
						log.Printf("⚠️ Warning: Normal search returned %d results, less than requested topK=%d * %.2f = %.0f (filter: %s)",
							result.ResultCount, task.TopK, task.ResultRatio, float64(task.TopK)*task.ResultRatio, filter)
					}
				}
			}
		}
	}

	result.Latency = time.Since(startTime)
	return result
}

// SearchHorizonPerf main search performance testing struct
type SearchHorizonPerf struct {
	client     *milvusclient.Client
	collection string
	vectorPool *QueryVectorPool
	stats      *SearchStats
	exprGen    *ExpressionGenerator
}

// NewSearchHorizonPerf creates a new search performance tester
func NewSearchHorizonPerf(uri, token, collectionName, queryVectorFile string, expressionFiles []string) (*SearchHorizonPerf, error) {
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

	// Load query vectors
	vectorPool, err := NewQueryVectorPool(queryVectorFile)
	if err != nil {
		return nil, fmt.Errorf("failed to load query vectors: %v", err)
	}

	// Verify collection exists and is loaded
	// Using v2.6 milvusclient API with NewHasCollectionOption
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

	// Load pre-generated expressions if files provided
	if len(expressionFiles) > 0 {
		if err := exprGen.LoadPreGeneratedExpressions(expressionFiles); err != nil {
			return nil, fmt.Errorf("failed to load pre-generated expressions: %v", err)
		}
	}

	return &SearchHorizonPerf{
		client:     milvusClient,
		collection: collectionName,
		vectorPool: vectorPool,
		stats:      NewSearchStats(),
		exprGen:    exprGen,
	}, nil
}

// RunSearchTest runs the search performance test
func (shp *SearchHorizonPerf) RunSearchTest(ctx context.Context, searchType, vectorField string,
	nq, topK, maxWorkers int, timeout time.Duration, outputFields []string,
	exprType string, searchTimeout time.Duration, resultRatio float64) error {

	// Reset statistics for this individual test
	shp.stats.Reset()

	log.Printf("🚀 Starting search performance test...")
	log.Printf("   Collection: %s", shp.collection)
	log.Printf("   Search Type: %s", searchType)
	log.Printf("   Vector Field: %s", vectorField)
	log.Printf("   Max Workers: %d", maxWorkers)
	log.Printf("   Test Timeout: %v", timeout)
	log.Printf("   nq: %d, topk: %d", nq, topK)
	log.Printf("   Expression Type: %s", exprType)

	// Create worker pool
	workers := make([]*SearchWorker, maxWorkers)
	for i := 0; i < maxWorkers; i++ {
		workers[i] = NewSearchWorker(shp.client, shp.collection, shp.vectorPool, shp.exprGen, shp.stats)
	}

	// Create channels for task distribution
	taskChan := make(chan *SearchTask, maxWorkers*2)
	resultChan := make(chan *SearchResult, maxWorkers*2)

	// Start worker goroutines
	var wg sync.WaitGroup
	testCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	for i := 0; i < maxWorkers; i++ {
		wg.Add(1)
		go func(worker *SearchWorker) {
			defer wg.Done()
			for {
				select {
				case <-testCtx.Done():
					return
				case task := <-taskChan:
					if task == nil {
						return
					}
					result := worker.PerformSearch(testCtx, task)
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
	go func() {
		for result := range resultChan {
			shp.stats.RecordSearch(result.Latency, result.Success)
			if !result.Success {
				log.Printf("❌ Search failed: %s", result.Error)
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
				// Note: query vectors and filter expressions will be generated fresh during each search execution
				task := &SearchTask{
					SearchType:     searchType,
					VectorField:    vectorField,
					NQ:             nq,
					TopK:           topK,
					OutputFields:   outputFields,
					Timeout:        searchTimeout,
					ExpressionType: exprType,
					ResultRatio:    resultRatio,
				}

				select {
				case taskChan <- task:
					current := atomic.AddInt64(&taskCount, 1)

					// Log progress periodically
					if current-lastLoggedTasks >= logInterval || time.Since(lastLogTime) > 10*time.Second {
						stats := shp.stats.GetStats()
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

	// Final statistics
	finalStats := shp.stats.GetStats()
	log.Printf("✅ Search test completed!")
	log.Printf("📊 Final Results:\n   Total Searches: %d\n   Total Failures: %d\n   QPS: %.1f\n   Average Latency: %.1f ms\n   Min Latency: %.1f ms\n   Max Latency: %.1f ms\n   P95 Latency: %.1f ms\n   P99 Latency: %.1f ms\n   Success Rate: %.2f%%\n   Test Duration: %.2f seconds",
		finalStats["total_searches"],
		finalStats["total_failures"],
		finalStats["qps"],
		finalStats["avg_latency"],
		finalStats["min_latency"],
		finalStats["max_latency"],
		finalStats["p95_latency"],
		finalStats["p99_latency"],
		finalStats["success_rate"],
		finalStats["duration"])

	return nil
}

// Close closes the Milvus client connection
func (shp *SearchHorizonPerf) Close() error {
	ctx := context.Background()
	return shp.client.Close(ctx)
}

func main() {
	// Command line arguments
	var (
		searchType       = flag.String("search-type", "normal", "Search type: normal or hybrid")
		fileWorkers      = flag.String("file-workers", "all:1", "File and worker pairs: 'qc_1_exprs.json:10,qc_2_exprs.json:20,qc_3_s_exprs.json:30,qc_3_m_exprs.json:20,qc_3_l_exprs.json:10' or 'all:10' for all files")
		testTimeout      = flag.Int("timeout", 300, "Test timeout in seconds")
		searchTimeoutSec = flag.Int("search-timeout", 30, "Individual search timeout in seconds")
		queryVectorFile  = flag.String("vector-file", "/root/horizon/horizonPoc/data/merged_query_vectors.json", "Path to JSON file containing query vectors")
		exprDir          = flag.String("expr-dir", "/root/horizon/horizonPoc/data/query_expressions", "Directory containing expression JSON files")
		topK             = flag.Int("top-k", 15000, "Top K for search")
		sleepSec         = flag.Int("sleep-sec", 120, "Sleep seconds between tests")
		outputFieldsStr  = flag.String("output-fields", "timestamp,device_id,expert_collected,sensor_lidar_type,gcj02_lon,gcj02_lat", "Comma-separated output fields")
		resultRatio      = flag.Float64("result-ratio", 1.0, "Print warning if result count is less than this ratio of topK (e.g., 0.9 means warn if results < 90% of topK)")

		// Hardcoded values
		host           = "https://in01-3ecccccxx3c28817d.ali-cn-hangzhou.cloud-uat.zilliz.cn:19530"
		collectionName = "horizon_test_collection"
		vectorField    = "feature"
		nq             = 1
		apiKey         = "token"
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
			// Expand "all" to create a separate test for each expression file
			entries, err := os.ReadDir(exprDirPath)
			if err != nil {
				log.Fatalf("❌ Failed to read expression directory: %v", err)
			}

			var allFiles []string
			for _, entry := range entries {
				if !entry.IsDir() && strings.HasSuffix(entry.Name(), "_exprs.json") {
					fullPath := filepath.Join(exprDirPath, entry.Name())
					allFiles = append(allFiles, fullPath)
				}
			}

			if len(allFiles) == 0 {
				log.Fatalf("❌ No expression files found in %s", exprDirPath)
			}

			// Create a separate test config for each file
			for _, filePath := range allFiles {
				baseName := filepath.Base(filePath)
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

		// Create search performance tester for this specific test
		shp, err := NewSearchHorizonPerf(host, apiKey, collectionName, *queryVectorFile, config.ExpressionFiles)
		if err != nil {
			log.Fatalf("❌ Failed to create search performance tester: %v", err)
		}

		ctx := context.Background()
		err = shp.RunSearchTest(ctx, *searchType, vectorField, nq, *topK,
			config.Workers, time.Duration(*testTimeout)*time.Second,
			outputFields, "pre_generated", time.Duration(*searchTimeoutSec)*time.Second, *resultRatio)

		if err != nil {
			log.Printf("❌ Test %d/%d failed: %s: %v", i+1, len(testConfigs), config.TestName, err)
		} else {
			log.Printf("✅ Test %d/%d completed: %s", i+1, len(testConfigs), config.TestName)
		}

		// Close the client
		shp.Close()

		// Brief pause between tests (skip for last test)
		if i < len(testConfigs)-1 {
			log.Printf("⏸️  Pausing %d seconds before next test...", *sleepSec)
			time.Sleep(time.Duration(*sleepSec) * time.Second)
		}
	}

	log.Printf("🎉 All tests completed!")
}
