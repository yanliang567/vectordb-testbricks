package main

import (
	"fmt"
)

func main() {
	fmt.Println("=== Filter 表达式详细分析 ===\n")
	
	// 核心差异
	fmt.Println("🎯 核心差异:")
	fmt.Println("Normal Search:   1次表达式生成（任务创建时）")
	fmt.Println("Hybrid Search:   2次表达式生成（搜索执行时）")
	fmt.Println()
	
	// 执行流程对比
	fmt.Println("📋 执行流程对比:")
	fmt.Println()
	fmt.Println("┌─ Normal Search 流程 ────────────────────────────┐")
	fmt.Println("│ 1. 任务生成器:")
	fmt.Println("│    filter = exprGen.GenerateExpression(exprType)")
	fmt.Println("│    task.Filter = filter  // 固定值")
	fmt.Println("│")
	fmt.Println("│ 2. SearchWorker.PerformSearch:")
	fmt.Println("│    searchOpt.WithFilter(task.Filter)  // 使用固定值")
	fmt.Println("└────────────────────────────────────────────────┘")
	fmt.Println()
	fmt.Println("┌─ Hybrid Search 流程 ────────────────────────────┐")
	fmt.Println("│ 1. 任务生成器:")
	fmt.Println("│    filter = exprGen.GenerateExpression(exprType)")
	fmt.Println("│    task.Filter = filter  // 实际上被忽略！")
	fmt.Println("│    task.ExpressionType = exprType  // 保存类型")
	fmt.Println("│")
	fmt.Println("│ 2. SearchWorker.PerformSearch:")
	fmt.Println("│    expr1 = exprGen.GenerateExpression(task.ExpressionType)")
	fmt.Println("│    expr2 = exprGen.GenerateExpression(task.ExpressionType)")
	fmt.Println("│    annReq1.WithFilter(expr1)  // 新生成的表达式1")
	fmt.Println("│    annReq2.WithFilter(expr2)  // 新生成的表达式2")
	fmt.Println("└────────────────────────────────────────────────┘")
	fmt.Println()
	
	// 随机性分析
	fmt.Println("🎲 随机性分析:")
	fmt.Println("GenerateExpression() 使用 rand.Intn() 生成随机值:")
	fmt.Println("• device_id: 从预定义列表随机选择")
	fmt.Println("• 时间窗口: 随机选择开始日期")
	fmt.Println("• 地理多边形: 从预定义列表随机选择")
	fmt.Println("• 传感器类型: 从预定义列表随机选择")
	fmt.Println()
	
	// 实际差异示例
	fmt.Println("📊 实际差异示例:")
	fmt.Println()
	fmt.Println("Normal Search (一个任务的多次搜索):")
	fmt.Println("  第1次搜索: device_id == \"device_001\"")
	fmt.Println("  第2次搜索: device_id == \"device_001\"  // 相同！")
	fmt.Println("  第3次搜索: device_id == \"device_001\"  // 相同！")
	fmt.Println()
	fmt.Println("Hybrid Search (一个任务的多次搜索):")
	fmt.Println("  第1次搜索: expr1 = device_id == \"device_003\"")
	fmt.Println("             expr2 = device_id == \"device_007\"")
	fmt.Println("  第2次搜索: expr1 = device_id == \"device_015\"")
	fmt.Println("             expr2 = device_id == \"device_002\"")
	fmt.Println("  第3次搜索: expr1 = device_id == \"device_009\"")
	fmt.Println("             expr2 = device_id == \"device_011\"")
	fmt.Println()
	
	// 性能和测试影响
	fmt.Println("💡 性能和测试影响:")
	fmt.Println("1. Normal Search:")
	fmt.Println("   ✓ 表达式固定，测试结果更一致")
	fmt.Println("   ✓ 更好的缓存命中率（相同查询条件）")
	fmt.Println("   ✗ 缺乏表达式多样性")
	fmt.Println()
	fmt.Println("2. Hybrid Search:")
	fmt.Println("   ✓ 表达式多样性更丰富")
	fmt.Println("   ✓ 更真实的随机查询场景")
	fmt.Println("   ✓ 两个ANN请求可能过滤不同数据")
	fmt.Println("   ✗ 结果变异性更大")
	fmt.Println("   ✗ 缓存命中率较低")
	fmt.Println()
	
	// 潜在问题
	fmt.Println("⚠️ 潜在问题:")
	fmt.Println("• task.Filter 在 Hybrid Search 中被浪费")
	fmt.Println("• 两种搜索方式的表达式生成机制不一致")
	fmt.Println("• Hybrid Search 的表达式随机性可能影响基准测试的可重复性")
}
