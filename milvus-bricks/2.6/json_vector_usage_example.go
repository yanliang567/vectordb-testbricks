package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"sync"
	"time"
)

// Queries represents query vectors for Go applications
type Queries [][]float32

// QueryVectorManager manages query vectors with thread-safe operations
type QueryVectorManager struct {
	vectors     Queries
	currentIdx  int64
	mu          sync.RWMutex
	initialized bool
}

// NewQueryVectorManager creates a new query vector manager
func NewQueryVectorManager() *QueryVectorManager {
	return &QueryVectorManager{
		vectors:     make(Queries, 0),
		currentIdx:  0,
		initialized: false,
	}
}

// LoadFromJSON loads query vectors from JSON file
func (qvm *QueryVectorManager) LoadFromJSON(filePath string) error {
	qvm.mu.Lock()
	defer qvm.mu.Unlock()
	
	log.Printf("📖 Loading query vectors from: %s", filePath)
	
	// Read JSON file
	jsonData, err := ioutil.ReadFile(filePath)
	if err != nil {
		return fmt.Errorf("failed to read JSON file: %v", err)
	}
	
	// Parse JSON
	var vectors Queries
	err = json.Unmarshal(jsonData, &vectors)
	if err != nil {
		return fmt.Errorf("failed to parse JSON: %v", err)
	}
	
	if len(vectors) == 0 {
		return fmt.Errorf("no vectors found in JSON file")
	}
	
	qvm.vectors = vectors
	qvm.initialized = true
	
	log.Printf("✅ Loaded %d query vectors successfully", len(vectors))
	return nil
}

// GetVectors returns nq vectors with cycling (thread-safe)
func (qvm *QueryVectorManager) GetVectors(nq int) [][]float32 {
	qvm.mu.Lock()
	defer qvm.mu.Unlock()
	
	if !qvm.initialized || len(qvm.vectors) == 0 {
		log.Printf("⚠️ No vectors loaded, returning empty slice")
		return nil
	}
	
	result := make([][]float32, nq)
	vectorCount := int64(len(qvm.vectors))
	
	for i := 0; i < nq; i++ {
		idx := (qvm.currentIdx + int64(i)) % vectorCount
		result[i] = make([]float32, len(qvm.vectors[idx]))
		copy(result[i], qvm.vectors[idx])
	}
	
	qvm.currentIdx = (qvm.currentIdx + int64(nq)) % vectorCount
	return result
}

// GetStats returns statistics about loaded vectors (thread-safe)
func (qvm *QueryVectorManager) GetStats() map[string]interface{} {
	qvm.mu.RLock()
	defer qvm.mu.RUnlock()
	
	if !qvm.initialized {
		return map[string]interface{}{
			"initialized":   false,
			"total_vectors": 0,
		}
	}
	
	// Calculate dimension statistics
	dimCount := make(map[int]int)
	for _, vector := range qvm.vectors {
		dimCount[len(vector)]++
	}
	
	return map[string]interface{}{
		"initialized":      true,
		"total_vectors":    len(qvm.vectors),
		"current_index":    qvm.currentIdx,
		"dimension_counts": dimCount,
	}
}

// SearchTaskSimulator simulates using query vectors in search tasks
type SearchTaskSimulator struct {
	vectorManager *QueryVectorManager
	taskCount     int64
	mu            sync.RWMutex
}

// NewSearchTaskSimulator creates a new search task simulator
func NewSearchTaskSimulator(vectorManager *QueryVectorManager) *SearchTaskSimulator {
	return &SearchTaskSimulator{
		vectorManager: vectorManager,
		taskCount:     0,
	}
}

// SimulateSearch simulates a search operation using query vectors
func (sts *SearchTaskSimulator) SimulateSearch(searchID int, nq int) {
	start := time.Now()
	
	// Get query vectors
	queryVectors := sts.vectorManager.GetVectors(nq)
	if queryVectors == nil {
		log.Printf("❌ Search %d: Failed to get query vectors", searchID)
		return
	}
	
	// Simulate processing time
	time.Sleep(time.Millisecond * 10) // Simulate 10ms processing
	
	// Update statistics
	sts.mu.Lock()
	sts.taskCount++
	currentCount := sts.taskCount
	sts.mu.Unlock()
	
	duration := time.Since(start)
	log.Printf("✅ Search %d completed: %d vectors, %d dims, took %v (total: %d)", 
		searchID, len(queryVectors), len(queryVectors[0]), duration, currentCount)
}

// RunConcurrentSearches runs multiple concurrent search simulations
func (sts *SearchTaskSimulator) RunConcurrentSearches(numWorkers, tasksPerWorker int) {
	var wg sync.WaitGroup
	
	log.Printf("🚀 Starting %d workers, %d tasks each...", numWorkers, tasksPerWorker)
	
	startTime := time.Now()
	
	for worker := 0; worker < numWorkers; worker++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			
			for task := 0; task < tasksPerWorker; task++ {
				searchID := workerID*tasksPerWorker + task + 1
				sts.SimulateSearch(searchID, 1) // 1 query vector per search
			}
		}(worker)
	}
	
	wg.Wait()
	
	duration := time.Since(startTime)
	totalTasks := numWorkers * tasksPerWorker
	qps := float64(totalTasks) / duration.Seconds()
	
	log.Printf("📊 Completed %d searches in %v (%.2f QPS)", totalTasks, duration, qps)
}

func main() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	
	// Example usage with the converted JSON file
	jsonFile := "/tmp/query_vectors.json"
	
	log.Printf("🎯 Query Vector Usage Example")
	log.Printf("==============================")
	
	// 1. Create vector manager and load vectors
	vectorManager := NewQueryVectorManager()
	
	err := vectorManager.LoadFromJSON(jsonFile)
	if err != nil {
		log.Fatalf("❌ Failed to load vectors: %v", err)
	}
	
	// 2. Display statistics
	stats := vectorManager.GetStats()
	log.Printf("📊 Vector Manager Statistics:")
	for key, value := range stats {
		log.Printf("   %s: %v", key, value)
	}
	
	// 3. Demonstrate basic usage
	log.Printf("\n🔍 Basic Usage Examples:")
	
	// Get some vectors
	vectors1 := vectorManager.GetVectors(3)
	log.Printf("   Retrieved %d vectors, first vector dim: %d", len(vectors1), len(vectors1[0]))
	
	vectors2 := vectorManager.GetVectors(5)
	log.Printf("   Retrieved %d more vectors (cycling)", len(vectors2))
	
	// 4. Simulate concurrent search operations
	log.Printf("\n🏃 Concurrent Search Simulation:")
	
	simulator := NewSearchTaskSimulator(vectorManager)
	
	// Test with different concurrency levels
	testCases := []struct {
		workers int
		tasks   int
	}{
		{5, 10},   // 5 workers, 10 tasks each = 50 total
		{10, 20},  // 10 workers, 20 tasks each = 200 total
		{20, 25},  // 20 workers, 25 tasks each = 500 total
	}
	
	for i, testCase := range testCases {
		log.Printf("\n📈 Test Case %d: %d workers × %d tasks", 
			i+1, testCase.workers, testCase.tasks)
		simulator.RunConcurrentSearches(testCase.workers, testCase.tasks)
	}
	
	// 5. Final statistics
	finalStats := vectorManager.GetStats()
	log.Printf("\n📊 Final Statistics:")
	log.Printf("   Current index: %v", finalStats["current_index"])
	log.Printf("   Total vectors: %v", finalStats["total_vectors"])
	
	log.Printf("\n✅ Example completed successfully!")
	log.Printf("💡 Integration tips:")
	log.Printf("   - Load vectors once at startup")
	log.Printf("   - Use GetVectors() in your search loops")
	log.Printf("   - Vectors automatically cycle for continuous testing")
	log.Printf("   - Thread-safe for concurrent access")
}
