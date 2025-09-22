package main

import (
	"fmt"
)

func main() {
	fmt.Println("=== Filter 表达式取值机制分析 ===\n")
	
	// 分析任务生成阶段
	fmt.Println("📋 1. 任务生成阶段 (第670-685行):")
	fmt.Println("```go")
	fmt.Println("filter := \"\"")
	fmt.Println("if exprType != \"\" {")
	fmt.Println("    filter = shp.exprGen.GenerateExpression(exprType)")
	fmt.Println("}")
	fmt.Println("task := &SearchTask{")
	fmt.Println("    Filter: filter,           // 设置统一的filter")
	fmt.Println("    ExpressionType: exprType, // 保存表达式类型")
	fmt.Println("    // ... 其他字段")
	fmt.Println("}")
	fmt.Println("```")
	fmt.Println()
	
	// 分析Normal Search
	fmt.Println("🔍 2. Normal Search 阶段 (第490-492行):")
	fmt.Println("```go")
	fmt.Println("if task.Filter != \"\" {")
	fmt.Println("    searchOpt = searchOpt.WithFilter(task.Filter)")
	fmt.Println("}")
	fmt.Println("```")
	fmt.Println("➤ Normal Search 直接使用任务生成时的 task.Filter")
	fmt.Println()
	
	// 分析Hybrid Search
	fmt.Println("🔄 3. Hybrid Search 阶段 (第432-449行):")
	fmt.Println("```go")
	fmt.Println("expr1 := sw.exprGen.GenerateExpression(task.ExpressionType)")
	fmt.Println("expr2 := sw.exprGen.GenerateExpression(task.ExpressionType)")
	fmt.Println("")
	fmt.Println("if expr1 != \"\" {")
	fmt.Println("    annReq1 = annReq1.WithFilter(expr1)")
	fmt.Println("}")
	fmt.Println("if expr2 != \"\" {")
	fmt.Println("    annReq2 = annReq2.WithFilter(expr2)")
	fmt.Println("}")
	fmt.Println("```")
	fmt.Println("➤ Hybrid Search 重新生成两个独立的表达式")
	fmt.Println()
	
	// 总结差异
	fmt.Println("📊 差异总结:")
	fmt.Println("┌─────────────────┬──────────────────────┬─────────────────────┐")
	fmt.Println("│   搜索类型      │    Filter表达式来源   │      生成次数       │")
	fmt.Println("├─────────────────┼──────────────────────┼─────────────────────┤")
	fmt.Println("│ Normal Search   │ task.Filter         │ 1次（任务生成时）    │")
	fmt.Println("│                 │ (任务生成时生成)     │                     │")
	fmt.Println("├─────────────────┼──────────────────────┼─────────────────────┤")
	fmt.Println("│ Hybrid Search   │ expr1, expr2        │ 2次（搜索执行时）    │")
	fmt.Println("│                 │ (搜索时重新生成)     │ 每个ANN请求1次      │")
	fmt.Println("└─────────────────┴──────────────────────┴─────────────────────┘")
	fmt.Println()
	
	// 实际影响
	fmt.Println("💡 实际影响:")
	fmt.Println("1. Normal Search: 同一任务的filter表达式是固定的")
	fmt.Println("2. Hybrid Search: 每次搜索都会生成新的表达式")
	fmt.Println("   - expr1 和 expr2 可能不同（取决于GenerateExpression的实现）")
	fmt.Println("   - 提供了更多的变化性和随机性")
	fmt.Println()
	
	// 注意事项
	fmt.Println("⚠️ 注意事项:")
	fmt.Println("• task.Filter 在 Hybrid Search 中实际上被忽略了")
	fmt.Println("• Hybrid Search 完全依赖 task.ExpressionType 重新生成表达式")
	fmt.Println("• 两种搜索方式的表达式生成时机不同")
}
