package main

import (
	"fmt"
	"log"
)

func main() {
	fmt.Println("=== 统计信息数字精度对比 ===\n")
	
	// 模拟数据
	finalStats := map[string]interface{}{
		"total_searches": 1000,
		"total_failures": 5,
		"qps":           125.50,
		"avg_latency":   12.345,
		"min_latency":   5.123,
		"max_latency":   45.678,
		"p95_latency":   25.901,
		"p99_latency":   38.456,
		"success_rate":  99.5,
		"duration":      8.0,
	}
	
	fmt.Println("📋 修改前的数字精度:")
	fmt.Printf("2024/09/19 12:00:01 📊 Final Results:\n")
	fmt.Printf("   Total Searches: %d\n", finalStats["total_searches"])
	fmt.Printf("   Total Failures: %d\n", finalStats["total_failures"])
	fmt.Printf("   QPS: %.2f\n", finalStats["qps"])                      // 2位小数
	fmt.Printf("   Average Latency: %.3f ms\n", finalStats["avg_latency"]) // 3位小数
	fmt.Printf("   Min Latency: %.3f ms\n", finalStats["min_latency"])     // 3位小数
	fmt.Printf("   Max Latency: %.3f ms\n", finalStats["max_latency"])     // 3位小数
	fmt.Printf("   P95 Latency: %.3f ms\n", finalStats["p95_latency"])    // 3位小数
	fmt.Printf("   P99 Latency: %.3f ms\n", finalStats["p99_latency"])    // 3位小数
	fmt.Printf("   Success Rate: %.2f%%\n", finalStats["success_rate"])
	fmt.Printf("   Test Duration: %.2f seconds\n", finalStats["duration"])
	fmt.Println()
	
	fmt.Println("📋 修改后的数字精度:")
	log.Printf("📊 Final Results:\n   Total Searches: %d\n   Total Failures: %d\n   QPS: %.1f\n   Average Latency: %.1f ms\n   Min Latency: %.1f ms\n   Max Latency: %.1f ms\n   P95 Latency: %.1f ms\n   P99 Latency: %.1f ms\n   Success Rate: %.2f%%\n   Test Duration: %.2f seconds",
		finalStats["total_searches"],
		finalStats["total_failures"],
		finalStats["qps"],           // 1位小数
		finalStats["avg_latency"],   // 1位小数
		finalStats["min_latency"],   // 1位小数
		finalStats["max_latency"],   // 1位小数
		finalStats["p95_latency"],   // 1位小数
		finalStats["p99_latency"],   // 1位小数
		finalStats["success_rate"],
		finalStats["duration"])
	
	fmt.Println()
	fmt.Println("🎯 修改内容:")
	fmt.Println("✅ QPS: %.2f → %.1f (从2位改为1位小数)")
	fmt.Println("✅ 延迟数据: %.3f → %.1f (从3位改为1位小数)")
	fmt.Println("✅ 单位保持: ms 不变")
	fmt.Println("✅ 进度日志同样更新为1位小数精度")
	fmt.Println()
	fmt.Println("📊 示例进度日志对比:")
	fmt.Println("修改前: 📊 Progress - Submitted: 1000, QPS: 125.50, Avg: 12.345s, P99: 38.456s, Success: 99.5%")
	fmt.Println("修改后: 📊 Progress - Submitted: 1000, QPS: 125.5, Avg: 12.3 ms, P99: 38.5 ms, Success: 99.5%")
}
