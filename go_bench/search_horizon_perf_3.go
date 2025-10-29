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

// QueryItem represents a complete query with vectors and expression
type QueryItem struct {
	PicVector  []float32 // pic_embedding
	TextVector []float32 // text_embedding (for hybrid search)
	Expression string    // milvus_expression
}

// QueryPool manages query vectors and expressions from JSON file
// Ensures vectors and expressions maintain their 1-to-1 correspondence
type QueryPool struct {
	queries    []QueryItem
	currentIdx int64
	mu         sync.RWMutex
	isHybrid   bool
}

// NewQueryPool creates a new QueryPool
func NewQueryPool(filePath string, isHybrid bool) (*QueryPool, error) {
	pool := &QueryPool{
		queries:    make([]QueryItem, 0),
		currentIdx: 0,
		isHybrid:   isHybrid,
	}

	if err := pool.LoadQueriesFromFile(filePath); err != nil {
		return nil, err
	}

	return pool, nil
}

// LoadQueriesFromFile loads queries (vectors + expressions) from JSON file
func (qp *QueryPool) LoadQueriesFromFile(filePath string) error {
	log.Printf("📖 Loading queries (vectors + expressions) from %s", filePath)

	// Expand ~ path to full path
	expandedPath, err := expandPath(filePath)
	if err != nil {
		return fmt.Errorf("failed to expand file path %s: %v", filePath, err)
	}

	// Check if file exists
	if _, err := os.Stat(expandedPath); os.IsNotExist(err) {
		return fmt.Errorf("❌ Query file does not exist: %s", expandedPath)
	} else if err != nil {
		return fmt.Errorf("❌ Failed to check file status: %s, error: %v", expandedPath, err)
	}

	log.Printf("✅ Found query file: %s", expandedPath)

	// Only support JSON format
	ext := strings.ToLower(filepath.Ext(expandedPath))
	if ext != ".json" {
		return fmt.Errorf("❌ Only JSON format is supported. Please use .json file extension")
	}

	qp.mu.Lock()
	defer qp.mu.Unlock()

	return qp.loadFromJSON(expandedPath)
}

// QueryData represents a single query with vectors and expression
type QueryData struct {
	QueryID          int                    `json:"query_id"`
	OriginalQuery    map[string]interface{} `json:"original_query"`
	MilvusExpression string                 `json:"milvus_expression"`
}

// ExpressionFileData represents the structure of expression JSON files
type ExpressionFileData struct {
	SourceFile   string      `json:"source_file"`
	TotalQueries int         `json:"total_queries"`
	Queries      []QueryData `json:"queries"`
}

// loadFromJSON reads queries (vectors + expressions) from JSON file
func (qp *QueryPool) loadFromJSON(filePath string) error {
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

	// Parse JSON as ExpressionFileData
	var exprData ExpressionFileData
	if err := json.Unmarshal(data, &exprData); err != nil {
		return fmt.Errorf("failed to parse JSON: %v", err)
	}

	if len(exprData.Queries) == 0 {
		return fmt.Errorf("❌ No queries found in JSON file")
	}

	// Pre-allocate slice with known capacity
	qp.queries = make([]QueryItem, 0, len(exprData.Queries))

	// Extract queries (vectors + expressions) together
	for _, queryData := range exprData.Queries {
		item := QueryItem{
			Expression: queryData.MilvusExpression,
		}

		// Get pic_embedding
		if picEmbedding, ok := queryData.OriginalQuery["pic_embedding"].([]interface{}); ok {
			item.PicVector = make([]float32, len(picEmbedding))
			for i, val := range picEmbedding {
				if floatVal, ok := val.(float64); ok {
					item.PicVector[i] = float32(floatVal)
				}
			}
		} else {
			log.Printf("⚠️ Warning: Query %d missing pic_embedding, skipping", queryData.QueryID)
			continue
		}

		// For hybrid search, also get text_embedding
		if qp.isHybrid {
			if textEmbedding, ok := queryData.OriginalQuery["text_embedding"].([]interface{}); ok {
				item.TextVector = make([]float32, len(textEmbedding))
				for i, val := range textEmbedding {
					if floatVal, ok := val.(float64); ok {
						item.TextVector[i] = float32(floatVal)
					}
				}
			} else {
				log.Printf("⚠️ Warning: Query %d missing text_embedding for hybrid search, skipping", queryData.QueryID)
				continue
			}
		}

		qp.queries = append(qp.queries, item)
	}

	if len(qp.queries) == 0 {
		return fmt.Errorf("❌ No valid queries loaded from JSON file")
	}

	if qp.isHybrid {
		log.Printf("✅ Loaded %d queries with pic_embedding (dim: %d), text_embedding (dim: %d), and expressions",
			len(qp.queries), len(qp.queries[0].PicVector), len(qp.queries[0].TextVector))
	} else {
		log.Printf("✅ Loaded %d queries with pic_embedding (dim: %d) and expressions",
			len(qp.queries), len(qp.queries[0].PicVector))
	}

	return nil
}

// GetQuery returns nq query items (vectors + expression) starting from current index
// This ensures vectors and expressions stay paired as defined in the JSON file
// Optimized: Returns references instead of copies to reduce memory allocation
func (qp *QueryPool) GetQuery(nq int) []*QueryItem {
	qp.mu.Lock()
	defer qp.mu.Unlock()

	if len(qp.queries) == 0 {
		log.Printf("❌ ERROR: No queries loaded from JSON file!")
		panic("Query pool is empty - ensure JSON file was loaded correctly")
	}

	result := make([]*QueryItem, nq)
	poolSize := int64(len(qp.queries))

	for i := 0; i < nq; i++ {
		idx := (qp.currentIdx + int64(i)) % poolSize
		result[i] = &qp.queries[idx] // 返回指针，避免复制
	}

	qp.currentIdx = (qp.currentIdx + int64(nq)) % poolSize
	return result
}

// GetQueryLockFree returns a query item using atomic operations (lock-free for better concurrency)
// Use this for single query fetches in high-concurrency scenarios
func (qp *QueryPool) GetQueryLockFree() *QueryItem {
	if len(qp.queries) == 0 {
		log.Printf("❌ ERROR: No queries loaded from JSON file!")
		panic("Query pool is empty - ensure JSON file was loaded correctly")
	}

	poolSize := int64(len(qp.queries))
	idx := atomic.AddInt64(&qp.currentIdx, 1) % poolSize
	return &qp.queries[idx]
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
	queryPool  *QueryPool
	stats      *SearchStats
}

// NewSearchWorker creates a new search worker
func NewSearchWorker(milvusClient *milvusclient.Client, collectionName string,
	queryPool *QueryPool, stats *SearchStats) *SearchWorker {
	return &SearchWorker{
		client:     milvusClient,
		collection: collectionName,
		queryPool:  queryPool,
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
		// Get query item (vectors + expression together) using lock-free method
		queryItem := sw.queryPool.GetQueryLockFree()

		// Use references directly - no memory copy
		vector1 := entity.FloatVector(queryItem.PicVector)
		vector2 := entity.FloatVector(queryItem.TextVector)

		// Use the expression that corresponds to these vectors
		expr := queryItem.Expression

		// Hybrid search using v2.6 milvusclient API
		annReq1 := milvusclient.NewAnnRequest(task.VectorField, task.TopK, vector1)
		annReq2 := milvusclient.NewAnnRequest(task.VectorField, task.TopK, vector2)

		// Set the same expression for each ANN request
		if expr != "" {
			annReq1 = annReq1.WithFilter(expr)
			annReq2 = annReq2.WithFilter(expr)
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
						log.Printf("⚠️ Warning: Hybrid search returned %d results, less than requested topK=%d * %.2f = %.0f (expr: %s)",
							result.ResultCount, task.TopK, task.ResultRatio, float64(task.TopK)*task.ResultRatio, expr)
					}
				}
			}
		}
	} else {
		// Normal search - optimized for single query (nq=1)
		if task.NQ == 1 {
			// Optimized path for single query - use lock-free method
			queryItem := sw.queryPool.GetQueryLockFree()

			// Use reference directly - no memory copy
			vector := entity.FloatVector(queryItem.PicVector)
			filter := queryItem.Expression

			// Normal search using v2.6 milvusclient API
			searchOpt := milvusclient.NewSearchOption(sw.collection, task.TopK, []entity.Vector{vector}).
				WithANNSField(task.VectorField)

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
		} else {
			// Multiple queries (nq > 1) - use batch method
			queryItems := sw.queryPool.GetQuery(task.NQ)

			// Pre-allocate vectors array once
			vectors := make([]entity.Vector, len(queryItems))
			for i, item := range queryItems {
				vectors[i] = entity.FloatVector(item.PicVector)
			}

			// Use the expression that corresponds to the first vector
			filter := queryItems[0].Expression

			// Normal search using v2.6 milvusclient API
			searchOpt := milvusclient.NewSearchOption(sw.collection, task.TopK, vectors).WithANNSField(task.VectorField)

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
	}

	result.Latency = time.Since(startTime)
	return result
}

// SearchHorizonPerf main search performance testing struct
type SearchHorizonPerf struct {
	client     *milvusclient.Client
	collection string
	queryPool  *QueryPool
	stats      *SearchStats
}

// NewSearchHorizonPerf creates a new search performance tester
func NewSearchHorizonPerf(uri, token, collectionName, queryFile string, searchType string) (*SearchHorizonPerf, error) {
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

	// Determine if hybrid search
	isHybrid := (searchType == "hybrid")

	// Load queries (vectors + expressions together)
	queryPool, err := NewQueryPool(queryFile, isHybrid)
	if err != nil {
		return nil, fmt.Errorf("failed to load queries: %v", err)
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

	return &SearchHorizonPerf{
		client:     milvusClient,
		collection: collectionName,
		queryPool:  queryPool,
		stats:      NewSearchStats(),
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
		workers[i] = NewSearchWorker(shp.client, shp.collection, shp.queryPool, shp.stats)
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
		fileWorkers      = flag.String("file-workers", "all:1", "File and worker pairs: 'qc_1_exprs_pic.json:10,qc_2_exprs_pic.json:20' or 'all:10' for all files")
		testTimeout      = flag.Int("timeout", 300, "Test timeout in seconds")
		searchTimeoutSec = flag.Int("search-timeout", 30, "Individual search timeout in seconds")
		exprDir          = flag.String("expr-dir", "/root/horizon/horizonPoc/data/query_expressions_merged_vector", "Directory containing expression JSON files")
		topKStr          = flag.String("top-k", "15000", "Top K for search (comma-separated for multiple values, e.g., '100,500,1000,15000')")
		sleepSec         = flag.Int("sleep-sec", 120, "Sleep seconds between tests")
		outputFieldsStr  = flag.String("output-fields", "timestamp,device_id,expert_collected,sensor_lidar_type,gcj02_lon,gcj02_lat", "Comma-separated output fields")
		resultRatio      = flag.Float64("result-ratio", 1.0, "Print warning if result count is less than this ratio of topK (e.g., 0.9 means warn if results < 90% of topK)")

		// Hardcoded values
		host           = "https://in01-3e1xxxxx28817d.ali-cn-hangzhou.cloud-uat.zilliz.cn:19530"
		collectionName = "horizon_test_collection"
		vectorField    = "feature"
		nq             = 1
		apiKey         = "token"
	)

	flag.Parse()

	log.SetFlags(log.LstdFlags | log.Lshortfile)

	// Parse top-k values
	topKValues := []int{}
	for _, topKPart := range strings.Split(*topKStr, ",") {
		topKPart = strings.TrimSpace(topKPart)
		if topKPart == "" {
			continue
		}
		topKVal, err := strconv.Atoi(topKPart)
		if err != nil {
			log.Fatalf("❌ Invalid top-k value: %s, error: %v", topKPart, err)
		}
		if topKVal <= 0 {
			log.Fatalf("❌ Invalid top-k value: %d (must be positive)", topKVal)
		}
		topKValues = append(topKValues, topKVal)
	}

	if len(topKValues) == 0 {
		log.Fatalf("❌ No valid top-k values provided")
	}

	log.Printf("📊 Top-K values to test: %v", topKValues)

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
			// Determine file suffix based on search type
			var fileSuffix string
			if *searchType == "hybrid" {
				fileSuffix = "_pic_txt.json"
			} else {
				fileSuffix = "_pic.json"
			}

			// Expand "all" to create a separate test for each expression file
			entries, err := os.ReadDir(exprDirPath)
			if err != nil {
				log.Fatalf("❌ Failed to read expression directory: %v", err)
			}

			var allFiles []string
			for _, entry := range entries {
				if !entry.IsDir() && strings.HasSuffix(entry.Name(), fileSuffix) {
					fullPath := filepath.Join(exprDirPath, entry.Name())
					allFiles = append(allFiles, fullPath)
				}
			}

			if len(allFiles) == 0 {
				log.Fatalf("❌ No expression files found with suffix %s in %s", fileSuffix, exprDirPath)
			}

			// Create a separate test config for each file
			for _, filePath := range allFiles {
				baseName := filepath.Base(filePath)
				baseName = strings.TrimSuffix(baseName, fileSuffix)
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

			// Extract base name without suffix for cleaner test name
			baseName := filepath.Base(fileName)
			baseName = strings.TrimSuffix(baseName, "_pic_txt.json")
			baseName = strings.TrimSuffix(baseName, "_pic.json")
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
	log.Printf("📋 Test configurations (%d file-workers × %d top-k values = %d total tests):",
		len(testConfigs), len(topKValues), len(testConfigs)*len(topKValues))
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
	log.Printf("   Top-K values: %v", topKValues)

	// Run tests sequentially for each configuration and each top-k value
	testCounter := 0
	totalTests := len(testConfigs) * len(topKValues)

	for configIdx, config := range testConfigs {
		// For each file-worker configuration, test all top-k values
		for topKIdx, currentTopK := range topKValues {
			testCounter++

			log.Printf("")
			separatorLine := "=" + strings.Repeat("=", 60)
			log.Printf("%s", separatorLine)
			log.Printf("🎯 Test %d/%d: %s (TopK=%d)", testCounter, totalTests, config.TestName, currentTopK)
			log.Printf("%s", separatorLine)

			// Log expression files being loaded (only on first top-k iteration)
			if topKIdx == 0 {
				if len(config.ExpressionFiles) == 1 {
					log.Printf("📖 Expression file: %s", filepath.Base(config.ExpressionFiles[0]))
				} else {
					log.Printf("📖 Expression files (%d):", len(config.ExpressionFiles))
					for _, f := range config.ExpressionFiles {
						log.Printf("   - %s", filepath.Base(f))
					}
				}
			}

			// Create search performance tester for this specific test
			// Use the expression file as query file (contains vectors + expressions)
			shp, err := NewSearchHorizonPerf(host, apiKey, collectionName, config.ExpressionFiles[0], *searchType)
			if err != nil {
				log.Fatalf("❌ Failed to create search performance tester: %v", err)
			}

			ctx := context.Background()
			err = shp.RunSearchTest(ctx, *searchType, vectorField, nq, currentTopK,
				config.Workers, time.Duration(*testTimeout)*time.Second,
				outputFields, "pre_generated", time.Duration(*searchTimeoutSec)*time.Second, *resultRatio)

			if err != nil {
				log.Printf("❌ Test %d/%d failed: %s (TopK=%d): %v",
					testCounter, totalTests, config.TestName, currentTopK, err)
			} else {
				log.Printf("✅ Test %d/%d completed: %s (TopK=%d)",
					testCounter, totalTests, config.TestName, currentTopK)
			}

			// Close the client
			shp.Close()

			// Brief pause between tests (skip for last test)
			if testCounter < totalTests {
				log.Printf("⏸️  Pausing %d seconds before next test...", *sleepSec)
				time.Sleep(time.Duration(*sleepSec) * time.Second)
			}
		}

		// Log completion of all top-k tests for this config
		log.Printf("")
		log.Printf("✅ Completed all %d top-k tests for config %d/%d: %s",
			len(topKValues), configIdx+1, len(testConfigs), config.TestName)
	}

	log.Printf("")
	log.Printf("🎉 All %d tests completed!", totalTests)
}
