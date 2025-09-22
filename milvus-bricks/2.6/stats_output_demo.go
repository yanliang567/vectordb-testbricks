package main

import (
	"fmt"
	"log"
)

func main() {
	fmt.Println("=== 统计信息输出格式对比 ===\n")
	
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
	
	fmt.Println("📋 修改前的输出格式（每行都有时间戳）:")
	fmt.Println("2024/09/19 12:00:01 ✅ Search test completed!")
	fmt.Println("2024/09/19 12:00:01 📊 Final Results:")
	fmt.Println("2024/09/19 12:00:01    Total Searches: 1000")
	fmt.Println("2024/09/19 12:00:01    Total Failures: 5")
	fmt.Println("2024/09/19 12:00:01    QPS: 125.50")
	fmt.Println("2024/09/19 12:00:01    Average Latency: 12.345 ms")
	fmt.Println("2024/09/19 12:00:01    Min Latency: 5.123 ms")
	fmt.Println("2024/09/19 12:00:01    Max Latency: 45.678 ms")
	fmt.Println("2024/09/19 12:00:01    P95 Latency: 25.901 ms")
	fmt.Println("2024/09/19 12:00:01    P99 Latency: 38.456 ms")
	fmt.Println("2024/09/19 12:00:01    Success Rate: 99.50%")
	fmt.Println("2024/09/19 12:00:01    Test Duration: 8.00 seconds")
	fmt.Println()
	
	fmt.Println("📋 修改后的输出格式（只有一个时间戳，但数据分行）:")
	fmt.Println("2024/09/19 12:00:01 ✅ Search test completed!")
	
	// 模拟新的输出格式
	log.Printf("📊 Final Results:\n   Total Searches: %d\n   Total Failures: %d\n   QPS: %.2f\n   Average Latency: %.3f ms\n   Min Latency: %.3f ms\n   Max Latency: %.3f ms\n   P95 Latency: %.3f ms\n   P99 Latency: %.3f ms\n   Success Rate: %.2f%%\n   Test Duration: %.2f seconds",
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
	
	fmt.Println()
	fmt.Println("🎯 优势:")
	fmt.Println("✅ 只有一个时间戳，方便复制")
	fmt.Println("✅ 所有数据保持分行显示，清晰易读")
	fmt.Println("✅ 在log文件中可以整体选中复制统计信息")
	fmt.Println("✅ 减少了log文件的时间戳冗余")
}
