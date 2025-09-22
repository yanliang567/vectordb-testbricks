package main

import (
	"fmt"
	"log"
	"strconv"
	"strings"
)

func main() {
	// Test the new expr-workers parsing logic
	testCases := []string{
		"equal:10,device_id_in:20",
		"equal:5;10;15,geo_contains:8",
		"sensor_contains:1,equal:3;5;7",
	}
	
	for i, testCase := range testCases {
		fmt.Printf("Test case %d: %s\n", i+1, testCase)
		
		exprWorkerConfigs := make(map[string][]int)
		
		// Parse expr-workers parameter
		configs := strings.Split(testCase, ",")
		for _, config := range configs {
			config = strings.TrimSpace(config)
			parts := strings.Split(config, ":")
			if len(parts) != 2 {
				log.Fatalf("❌ Invalid expr-workers format: %s. Expected format: 'expr:workers'", config)
			}
			
			exprName := strings.TrimSpace(parts[0])
			workersStr := strings.TrimSpace(parts[1])
			
			// Parse workers (can be single number or semicolon-separated list)
			var workers []int
			if strings.Contains(workersStr, ";") {
				// Multiple workers: "10;20;30" 
				workerStrs := strings.Split(workersStr, ";")
				for _, ws := range workerStrs {
					ws = strings.TrimSpace(ws)
					if w, err := strconv.Atoi(ws); err == nil {
						workers = append(workers, w)
					} else {
						log.Fatalf("❌ Invalid worker count in %s: %s", config, ws)
					}
				}
			} else {
				// Single worker: "10"
				if w, err := strconv.Atoi(workersStr); err == nil {
					workers = append(workers, w)
				} else {
					log.Fatalf("❌ Invalid worker count in %s: %s", config, workersStr)
				}
			}
			
			exprWorkerConfigs[exprName] = workers
		}
		
		// Print parsed results
		fmt.Printf("  Parsed configurations:\n")
		for expr, workers := range exprWorkerConfigs {
			fmt.Printf("    %s: workers %v\n", expr, workers)
		}
		
		// Simulate test execution
		fmt.Printf("  Would run tests:\n")
		for exprName, workersList := range exprWorkerConfigs {
			for _, workers := range workersList {
				fmt.Printf("    - Expression: %s, Workers: %d\n", exprName, workers)
			}
		}
		fmt.Println()
	}
	
	fmt.Println("✅ All test cases parsed successfully!")
}
