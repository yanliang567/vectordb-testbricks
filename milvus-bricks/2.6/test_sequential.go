package main

import (
	"fmt"
	"strconv"
	"strings"
)

// Define test configuration structure
type TestConfig struct {
	ExprType string
	Workers  int
}

func main() {
	// 测试不同的顺序执行配置
	testCases := []string{
		"equal:10,equal:20,device_id_in:20,device_id_in:30",
		"equal:5,device_id_in:10,geo_contains:15",
		"equal:1,equal:5,equal:10,equal:20",
		"device_id_in:2,device_id_in:4,device_id_in:8",
	}
	
	for i, exprWorkers := range testCases {
		fmt.Printf("\n=== 测试案例 %d: %s ===\n", i+1, exprWorkers)
		
		// 模拟新的解析逻辑
		var testConfigs []TestConfig
		
		configs := strings.Split(exprWorkers, ",")
		for _, config := range configs {
			config = strings.TrimSpace(config)
			parts := strings.Split(config, ":")
			
			exprName := strings.TrimSpace(parts[0])
			workersStr := strings.TrimSpace(parts[1])
			
			if workers, err := strconv.Atoi(workersStr); err == nil {
				testConfigs = append(testConfigs, TestConfig{
					ExprType: exprName,
					Workers:  workers,
				})
			}
		}
		
		// 显示配置
		fmt.Printf("📋 Test configurations (%d tests):\n", len(testConfigs))
		for j, config := range testConfigs {
			fmt.Printf("   %d. %s: %d workers\n", j+1, config.ExprType, config.Workers)
		}
		
		// 显示执行计划
		fmt.Printf("🎯 执行顺序:\n")
		for j, config := range testConfigs {
			fmt.Printf("   Test %d/%d: %s with %d workers\n", j+1, len(testConfigs), config.ExprType, config.Workers)
		}
	}
}
