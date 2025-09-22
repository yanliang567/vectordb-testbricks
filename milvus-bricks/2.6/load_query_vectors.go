package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"os"
)

// Queries represents the query vectors type for Go
type Queries [][]float32

// LoadQueryVectorsFromJSON loads query vectors from JSON file
func LoadQueryVectorsFromJSON(filePath string) (Queries, error) {
	log.Printf("📖 Loading query vectors from JSON: %s", filePath)
	
	// Read JSON file
	jsonData, err := ioutil.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read JSON file: %v", err)
	}
	
	// Parse JSON into Queries type
	var queries Queries
	err = json.Unmarshal(jsonData, &queries)
	if err != nil {
		return nil, fmt.Errorf("failed to parse JSON: %v", err)
	}
	
	log.Printf("✅ Loaded %d query vectors from JSON", len(queries))
	
	// Validate vectors
	if len(queries) == 0 {
		return nil, fmt.Errorf("no vectors found in JSON file")
	}
	
	// Check dimensions consistency (first 10 vectors)
	if len(queries) > 0 {
		firstDim := len(queries[0])
		log.Printf("📊 First vector dimension: %d", firstDim)
		
		inconsistentCount := 0
		checkCount := 10
		if len(queries) < checkCount {
			checkCount = len(queries)
		}
		
		for i := 0; i < checkCount; i++ {
			if len(queries[i]) != firstDim {
				inconsistentCount++
			}
		}
		
		if inconsistentCount > 0 {
			log.Printf("⚠️ Warning: %d/%d sampled vectors have inconsistent dimensions", 
				inconsistentCount, checkCount)
		}
	}
	
	return queries, nil
}

// GetVectorsSlice returns a slice of vectors starting from index
func (q Queries) GetVectorsSlice(startIndex, count int) [][]float32 {
	if len(q) == 0 {
		return nil
	}
	
	result := make([][]float32, 0, count)
	for i := 0; i < count; i++ {
		idx := (startIndex + i) % len(q)
		result = append(result, q[idx])
	}
	
	return result
}

// GetSingleVector returns a single vector at index (with wrapping)
func (q Queries) GetSingleVector(index int) []float32 {
	if len(q) == 0 {
		return nil
	}
	
	idx := index % len(q)
	return q[idx]
}

// GetRandomVectors returns random vectors from the collection
func (q Queries) GetRandomVectors(count int) [][]float32 {
	if len(q) == 0 {
		return nil
	}
	
	result := make([][]float32, 0, count)
	for i := 0; i < count; i++ {
		// Simple pseudo-random selection
		idx := (i * 7919) % len(q) // 7919 is a prime number
		result = append(result, q[idx])
	}
	
	return result
}

// Stats returns statistics about the query vectors
func (q Queries) Stats() map[string]interface{} {
	if len(q) == 0 {
		return map[string]interface{}{
			"total_vectors": 0,
			"dimensions":    0,
		}
	}
	
	// Count dimensions
	dimCount := make(map[int]int)
	for _, vector := range q {
		dimCount[len(vector)]++
	}
	
	// Find most common dimension
	maxCount := 0
	commonDim := 0
	for dim, count := range dimCount {
		if count > maxCount {
			maxCount = count
			commonDim = dim
		}
	}
	
	return map[string]interface{}{
		"total_vectors":    len(q),
		"common_dimension": commonDim,
		"dimension_counts": dimCount,
	}
}

// Example usage and testing
func main() {
	if len(os.Args) < 2 {
		log.Printf("Usage: go run load_query_vectors.go <json_file>")
		log.Printf("Example: go run load_query_vectors.go /tmp/query_vectors.json")
		return
	}
	
	jsonFile := os.Args[1]
	
	// Load query vectors
	queries, err := LoadQueryVectorsFromJSON(jsonFile)
	if err != nil {
		log.Fatalf("❌ Failed to load query vectors: %v", err)
	}
	
	// Display statistics
	stats := queries.Stats()
	log.Printf("📊 Query Vector Statistics:")
	for key, value := range stats {
		log.Printf("   %s: %v", key, value)
	}
	
	// Example usage: Get some vectors
	if len(queries) > 0 {
		log.Printf("\n🔍 Example Usage:")
		
		// Get first 3 vectors
		first3 := queries.GetVectorsSlice(0, 3)
		log.Printf("   First 3 vectors: %d vectors retrieved", len(first3))
		
		// Get a single vector
		singleVector := queries.GetSingleVector(10)
		log.Printf("   Single vector (index 10): dimension %d", len(singleVector))
		
		// Get random vectors
		randomVectors := queries.GetRandomVectors(5)
		log.Printf("   Random 5 vectors: %d vectors retrieved", len(randomVectors))
		
		// Show sample data from first vector
		if len(first3) > 0 && len(first3[0]) > 0 {
			sample := first3[0]
			sampleSize := 5
			if len(sample) < sampleSize {
				sampleSize = len(sample)
			}
			log.Printf("   First vector sample (%d elements): %v", 
				sampleSize, sample[:sampleSize])
		}
	}
	
	log.Printf("✅ Query vectors loaded and ready for use!")
}
