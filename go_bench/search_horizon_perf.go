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

		// Simple sort for percentiles
		for i := 0; i < len(sortedLatencies)-1; i++ {
			for j := i + 1; j < len(sortedLatencies); j++ {
				if sortedLatencies[i] > sortedLatencies[j] {
					sortedLatencies[i], sortedLatencies[j] = sortedLatencies[j], sortedLatencies[i]
				}
			}
		}

		p95Index := int(float64(len(sortedLatencies)) * 0.95)
		p99Index := int(float64(len(sortedLatencies)) * 0.99)

		if p95Index < len(sortedLatencies) {
			p95Latency = float64(sortedLatencies[p95Index])
		}
		if p99Index < len(sortedLatencies) {
			p99Latency = float64(sortedLatencies[p99Index])
		}
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

// ExpressionGenerator generates random filter expressions
type ExpressionGenerator struct {
	deviceIDs            []string
	polygons             []string
	sensors              []string
	jsonContainsPatterns []string
}

// NewExpressionGenerator creates a new expression generator
func NewExpressionGenerator() *ExpressionGenerator {
	return &ExpressionGenerator{
		deviceIDs: []string{
			"SENSOR_A123", "SENSOR_A233", "SENSOR_A108", "SENSOR_A172", "CAM_B112",
			"CAM_B177", "DV348", "DV378", "DV081", "DV349",
		},
		polygons: []string{
			// "'POLYGON((-74.0 40.7, -73.9 40.7, -73.9 40.8, -74.0 40.8, -74.0 40.7))'",
			// "'POLYGON((-74.1 40.6, -73.8 40.6, -73.8 40.9, -74.1 40.9, -74.1 40.6))'",
			// "'POLYGON((-74.05 40.75, -73.95 40.75, -73.95 40.85, -74.05 40.85, -74.05 40.75))'",
			// 2 square kilometers
			// "'POLYGON((-73.990494 40.729934, -73.973710 40.729934, -73.973710 40.742646, -73.990494 40.742646, -73.990494 40.729934))'",
			// "'POLYGON((-74.010980 40.733392, -73.994194 40.733392, -73.994194 40.746104, -74.010980 40.746104, -74.010980 40.733392))'",
			// "'POLYGON((-73.982670 40.784599, -73.965864 40.784599, -73.965864 40.797311, -73.982670 40.797311, -73.982670 40.784599))'",
			// "'POLYGON((-74.009971 40.713026, -73.993189 40.713026, -73.993189 40.725738, -74.009971 40.725738, -74.009971 40.713026))'",
			// "'POLYGON((-73.992444 40.737188, -73.975656 40.737188, -73.975656 40.749900, -73.992444 40.749900, -73.992444 40.737188))'",
			// "'POLYGON((-73.978085 40.742888, -73.961295 40.742888, -73.961295 40.755600, -73.978085 40.755600, -73.978085 40.742888))'",
			// "'POLYGON((-74.000351 40.715211, -73.983563 40.715211, -73.983563 40.727923, -74.000351 40.727923, -74.000351 40.715211))'",

			// 1 square kilometer
			"'POLYGON((-73.98803637970583 40.73179339493682, -73.97616762029416 40.73179339493682, -73.97616762029416 40.74078660506317, -73.98803637970583 40.74078660506317, -73.98803637970583 40.73179339493682))'",
			"'POLYGON((-74.00852168819422 40.735251394936824, -73.99665231180579 40.735251394936824, -73.99665231180579 40.74424460506317, -74.00852168819422 40.74424460506317, -74.00852168819422 40.735251394936824))'",
			"'POLYGON((-73.98020626266393 40.78645839493682, -73.96832773733607 40.78645839493682, -73.96832773733607 40.79545160506317, -73.98020626266393 40.79545160506317, -73.98020626266393 40.78645839493682))'",
			"'POLYGON((-74.00751287211497 40.71488539493683, -73.99564712788504 40.71488539493683, -73.99564712788504 40.72387860506318, -74.00751287211497 40.72387860506318, -74.00751287211497 40.71488539493683))'",
			"'POLYGON((-73.98998502689739 40.739047394936826, -73.97811497310258 40.739047394936826, -73.97811497310258 40.748040605063174, -73.98998502689739 40.748040605063174, -73.98998502689739 40.739047394936826))'",
			"'POLYGON((-73.97562553560911 40.74474739493682, -73.96375446439089 40.74474739493682, -73.96375446439089 40.75374060506317, -73.97562553560911 40.75374060506317, -73.97562553560911 40.74474739493682))'",
		},
		sensors: []string{
			"Thor_Trucks", "WeRide_Robobus", "Delphi_ESR", "Aptiv_SRR4", "AEye_iDAR", "DiDi_Gemini", "ADAS_Eyes",
			"Embark_Guardian", "Hella_24GHz", "ST_VL53L1X", "TuSimple_AFV", "Locomation_AutonomousRelay", "Voyage_Telessport",
			"Livox_Horizon", "Infineon_BGT24", "Aurora_FirstLight", "Ibeo_LUX", "Ouster_OS1_64", "Delphi_ESR",
		},
		jsonContainsPatterns: []string{
			"JSON_CONTAINS_ALL(sensor_lidar_type,['Analog_ADAU1761', 'AEye_iDAR', 'ADAS_Eyes', 'Argo_Geiger']) AND NOT JSON_CONTAINS(sensor_lidar_type, 'Delphi_ESR')",
			"JSON_CONTAINS_ALL(sensor_lidar_type,['Gatik_B2B', 'Bosch_LRR4', 'Mobileye_EyeQ4', 'Motional_Ioniq5']) AND NOT JSON_CONTAINS(sensor_lidar_type, 'Delphi_ESR')",
			"JSON_CONTAINS_ALL(sensor_lidar_type,['TI_AWR1843', 'Continental_HFL110', 'Velodyne_VLS128', 'Analog_ADAU1761']) AND NOT JSON_CONTAINS(sensor_lidar_type, 'Delphi_ESR')",
			"JSON_CONTAINS_ALL(sensor_lidar_type,['Thor_Trucks', 'Aptiv_SRR4', 'Leishen_C32', 'Pony_PonyAlpha']) AND NOT JSON_CONTAINS(sensor_lidar_type, 'Delphi_ESR')",
			"JSON_CONTAINS_ALL(sensor_lidar_type,['Magna_Icon', 'ZF_AC1000', 'NXP_TEF810X', 'Ibeo_LUX']) AND NOT JSON_CONTAINS(sensor_lidar_type, 'Delphi_ESR')",
			"JSON_CONTAINS_ALL(sensor_lidar_type,['Innoviz_One', 'ST_VL53L1X', 'May_Mobility', 'Embark_Guardian']) AND NOT JSON_CONTAINS(sensor_lidar_type, 'Delphi_ESR')",
		},
	}
}

// GenerateExpression generates a random filter expression
func (eg *ExpressionGenerator) GenerateExpression(exprType string) string {
	switch strings.ToLower(exprType) {
	case "equal":
		deviceID := eg.deviceIDs[rand.Intn(len(eg.deviceIDs))]
		return fmt.Sprintf(`device_id == "%s"`, deviceID)

	case "equal_and_expert_collected":
		device_id_keyword := eg.deviceIDs[rand.Intn(len(eg.deviceIDs))]
		return fmt.Sprintf(`device_id == "%s" and expert_collected == True`, device_id_keyword)

	case "equal_and_timestamp_week":
		deviceID := eg.deviceIDs[rand.Intn(len(eg.deviceIDs))]
		// Generate 7-day window
		startDate := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)
		endDate := time.Date(2025, 8, 23, 0, 0, 0, 0, time.UTC)
		days := int(endDate.Sub(startDate).Hours() / 24)
		randomOffset := rand.Intn(days)
		windowStart := startDate.AddDate(0, 0, randomOffset)
		windowEnd := windowStart.AddDate(0, 0, 6)

		startTS := windowStart.Unix() * 1000
		endTS := windowEnd.Unix() * 1000
		return fmt.Sprintf(`device_id == "%s" and timestamp >= %d and timestamp <= %d`,
			deviceID, startTS, endTS)

	case "equal_and_timestamp_month":
		deviceID := eg.deviceIDs[rand.Intn(len(eg.deviceIDs))]
		// Generate 30-day window
		startDate := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)
		endDate := time.Date(2025, 8, 1, 0, 0, 0, 0, time.UTC)
		days := int(endDate.Sub(startDate).Hours() / 24)
		randomOffset := rand.Intn(days)
		windowStart := startDate.AddDate(0, 0, randomOffset)
		windowEnd := windowStart.AddDate(0, 0, 29)

		startTS := windowStart.Unix() * 1000
		endTS := windowEnd.Unix() * 1000
		return fmt.Sprintf(`device_id == "%s" and timestamp >= %d and timestamp <= %d`,
			deviceID, startTS, endTS)

	case "device_id_in_and_timestamp_1_month":
		// Generate 30-day window
		startDate := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)
		endDate := time.Date(2025, 8, 1, 0, 0, 0, 0, time.UTC)
		days := int(endDate.Sub(startDate).Hours() / 24)
		randomOffset := rand.Intn(days)
		windowStart := startDate.AddDate(0, 0, randomOffset)
		windowEnd := windowStart.AddDate(0, 0, 29)

		startTS := windowStart.Unix() * 1000
		endTS := windowEnd.Unix() * 1000
		// get 2 device_ids
		idxs := rand.Perm(len(eg.deviceIDs))[:2]
		device_ids := []string{
			eg.deviceIDs[idxs[0]],
			eg.deviceIDs[idxs[1]],
		}
		device_ids_str := strings.Join(device_ids, `","`)
		return fmt.Sprintf(`device_id in ["%s"] and timestamp >= %d and timestamp <= %d`,
			device_ids_str, startTS, endTS)

	case "equal_and_timestamp_2_months":
		deviceID := eg.deviceIDs[rand.Intn(len(eg.deviceIDs))]
		// Generate 60-day window
		startDate := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)
		endDate := time.Date(2025, 8, 1, 0, 0, 0, 0, time.UTC)
		days := int(endDate.Sub(startDate).Hours() / 24)
		randomOffset := rand.Intn(days)
		windowStart := startDate.AddDate(0, 0, randomOffset)
		windowEnd := windowStart.AddDate(0, 0, 59)

		startTS := windowStart.Unix() * 1000
		endTS := windowEnd.Unix() * 1000
		return fmt.Sprintf(`device_id == "%s" and timestamp >= %d and timestamp <= %d`,
			deviceID, startTS, endTS)

	case "geo_within":
		polygon := eg.polygons[rand.Intn(len(eg.polygons))]
		return fmt.Sprintf("ST_WITHIN(location, %s)", polygon)

	case "sensor_contains":
		sensor := eg.sensors[rand.Intn(len(eg.sensors))]
		return fmt.Sprintf(`ARRAY_CONTAINS(sensor_lidar_type, "%s")`, sensor)
	case "device_id_in":
		idxs := rand.Perm(len(eg.deviceIDs))[:2]
		device_ids := []string{
			eg.deviceIDs[idxs[0]],
			eg.deviceIDs[idxs[1]],
		}
		device_ids_str := strings.Join(device_ids, `","`)
		return fmt.Sprintf(`device_id in ["%s"]`, device_ids_str)
	case "sensor_json_contains":
		// 随机返回一种json_contains_patterns
		return eg.jsonContainsPatterns[rand.Intn(len(eg.jsonContainsPatterns))]

	default:
		return ""
	}
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
	vectorPool *QueryVectorPool, stats *SearchStats) *SearchWorker {
	return &SearchWorker{
		client:     milvusClient,
		collection: collectionName,
		vectorPool: vectorPool,
		exprGen:    NewExpressionGenerator(),
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
		expr1 := sw.exprGen.GenerateExpression(task.ExpressionType)
		// expr2 := sw.exprGen.GenerateExpression(task.ExpressionType)

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
					if result.ResultCount < int(float64(task.TopK)*0.9) {
						log.Printf("⚠️ Warning: Hybrid search returned %d results, less than requested topK=%d (expr1: %s, expr2: %s)",
							result.ResultCount, task.TopK, expr1, expr1)
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
		searchOpt := milvusclient.NewSearchOption(
			sw.collection,
			task.TopK,
			vectors,
		)

		// Generate fresh filter expression for each search (same as hybrid search)
		filter := ""
		if task.ExpressionType != "" {
			filter = sw.exprGen.GenerateExpression(task.ExpressionType)
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
					if result.ResultCount < int(float64(task.TopK)*0.9) {
						log.Printf("⚠️ Warning: Normal search returned %d results, less than requested topK=%d (filter: %s)",
							result.ResultCount, task.TopK, filter)
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
func NewSearchHorizonPerf(uri, token, collectionName, queryVectorFile string) (*SearchHorizonPerf, error) {
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

	return &SearchHorizonPerf{
		client:     milvusClient,
		collection: collectionName,
		vectorPool: vectorPool,
		stats:      NewSearchStats(),
		exprGen:    NewExpressionGenerator(),
	}, nil
}

// RunSearchTest runs the search performance test
func (shp *SearchHorizonPerf) RunSearchTest(ctx context.Context, searchType, vectorField string,
	nq, topK, maxWorkers int, timeout time.Duration, outputFields []string,
	exprType string, searchTimeout time.Duration) error {

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
		workers[i] = NewSearchWorker(shp.client, shp.collection, shp.vectorPool, shp.stats)
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
	// "equal:10,equal_and_expert_collected:20,equal_and_timestamp_week:30,
	// equal_and_timestamp_month:30,geo_contains:10,sensor_contains:10,
	// device_id_in:10,sensor_json_contains:10"
	var (
		searchType       = flag.String("search-type", "normal", "Search type: normal or hybrid")
		exprWorkers      = flag.String("expr-workers", "equal:10,equal_and_expert_collected:20,equal_and_timestamp_2_months:30,geo_within:10,sensor_contains:10,device_id_in:10,sensor_json_contains:10", "Sequential test configurations: 'expr1:workers1,expr2:workers2'. Each pair runs as independent test (e.g., 'equal:10,equal:20,device_id_in:30' → 3 tests)")
		testTimeout      = flag.Int("timeout", 300, "Test timeout in seconds")
		searchTimeoutSec = flag.Int("search-timeout", 30, "Individual search timeout in seconds")
		queryVectorFile  = flag.String("vector-file", "/root/test/data/query_vectors.json", "Path to JSON file containing query vectors")

		// Hardcoded values as in Python version
		host           = "https://in01-9028520cb1d63cf.ali-cn-hangzhou.cloud-uat.zilliz.cn:19530"
		collectionName = "horizon_test_collection"
		vectorField    = "feature"
		nq             = 1
		topK           = 15000
		apiKey         = "cc5bf695ea9236e2c64617e9407a26cf0953034485d27216f8b3f145e3eb72396e042db2abb91c4ef6fde723af70e754d68ca787"
		outputFields   = []string{"timestamp", "url", "device_id", "expert_collected", "sensor_lidar_type", "p_url"}
	)

	flag.Parse()

	log.SetFlags(log.LstdFlags | log.Lshortfile)

	// Define test configuration structure
	type TestConfig struct {
		ExprType string
		Workers  int
	}

	// Parse expression and worker configurations
	var testConfigs []TestConfig

	// Parse expr-workers parameter
	// Format: "expr1:workers1,expr2:workers2" e.g., "equal:10,device_id_in:20"
	// Supports multiple tests: "equal:10,equal:20,device_id_in:30" → 3 independent tests
	configs := strings.Split(*exprWorkers, ",")
	for _, config := range configs {
		config = strings.TrimSpace(config)
		parts := strings.Split(config, ":")
		if len(parts) != 2 {
			log.Fatalf("❌ Invalid expr-workers format: %s. Expected format: 'expr:workers'", config)
		}

		exprName := strings.TrimSpace(parts[0])
		workersStr := strings.TrimSpace(parts[1])

		// Parse worker count and add to test list
		if workers, err := strconv.Atoi(workersStr); err == nil {
			testConfigs = append(testConfigs, TestConfig{
				ExprType: exprName,
				Workers:  workers,
			})
		} else {
			log.Fatalf("❌ Invalid worker count in %s: %s", config, workersStr)
		}
	}

	// Log the test configurations
	log.Printf("📋 Test configurations (%d tests):", len(testConfigs))
	for i, config := range testConfigs {
		log.Printf("   %d. %s: %d workers", i+1, config.ExprType, config.Workers)
	}

	// Create search performance tester
	shp, err := NewSearchHorizonPerf(host, apiKey, collectionName, *queryVectorFile)
	if err != nil {
		log.Fatalf("❌ Failed to create search performance tester: %v", err)
	}
	defer shp.Close()

	// Run tests sequentially for each configuration
	for i, config := range testConfigs {
		log.Printf("🎯 Running test %d/%d: %s with %d workers", i+1, len(testConfigs), config.ExprType, config.Workers)

		ctx := context.Background()
		err := shp.RunSearchTest(ctx, *searchType, vectorField, nq, topK,
			config.Workers, time.Duration(*testTimeout)*time.Second,
			outputFields, config.ExprType, time.Duration(*searchTimeoutSec)*time.Second)

		if err != nil {
			log.Printf("❌ Test %d/%d failed: %s with %d workers: %v", i+1, len(testConfigs), config.ExprType, config.Workers, err)
		} else {
			log.Printf("✅ Test %d/%d completed: %s with %d workers", i+1, len(testConfigs), config.ExprType, config.Workers)
		}

		// Brief pause between tests (skip for last test)
		if i < len(testConfigs)-1 {
			time.Sleep(120 * time.Second)
		}
	}

	log.Printf("🎉 All tests completed!")
}
