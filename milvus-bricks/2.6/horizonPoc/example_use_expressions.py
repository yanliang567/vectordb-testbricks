#!/usr/bin/env python3
"""
Simple script to use pre-generated expression files

This demonstrates how to use the expression JSON files in your search scripts.
"""

import json
from pathlib import Path


def load_expressions(expression_file):
    """Load pre-generated expressions from JSON file"""
    with open(expression_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def main():
    # Example: Load query_condition_1 expressions
    expr_dir = Path("query_expressions")
    
    print("=" * 80)
    print("Example: Using Pre-Generated Expression Files")
    print("=" * 80)
    print()
    
    # List all expression files
    expr_files = sorted(expr_dir.glob("*_expressions.json"))
    
    print(f"📂 Found {len(expr_files)} expression files:")
    for f in expr_files:
        data = load_expressions(f)
        print(f"   ✅ {f.name}: {data['total_queries']} expressions")
    
    print()
    print("=" * 80)
    print("Example Usage in Your Search Script:")
    print("=" * 80)
    print()
    
    # Example 1: Load and use expressions
    print("1️⃣  Load expression file:")
    print("   ```python")
    print("   import json")
    print("   data = json.load(open('query_expressions/query_condition_1_expressions.json'))")
    print("   expressions = [q['milvus_expression'] for q in data['queries']]")
    print("   ```")
    print()
    
    # Example 2: Use in search
    print("2️⃣  Use in Milvus search:")
    print("   ```python")
    print("   # Pick an expression (e.g., round-robin)")
    print("   expr_idx = search_count % len(expressions)")
    print("   filter_expr = expressions[expr_idx]")
    print("   ")
    print("   # Use in search")
    print("   results = client.search(")
    print("       collection_name='my_collection',")
    print("       data=query_vectors,")
    print("       filter=filter_expr,  # Pre-generated expression")
    print("       limit=topk")
    print("   )")
    print("   ```")
    print()
    
    # Example 3: Benefits
    print("3️⃣  Benefits:")
    print("   ✅ No runtime conversion overhead")
    print("   ✅ Expressions validated at generation time")
    print("   ✅ Easy to inspect and debug")
    print("   ✅ Can be version controlled")
    print("   ✅ Faster search performance")
    print()
    
    # Show actual example
    print("=" * 80)
    print("📊 Actual Expression Examples:")
    print("=" * 80)
    print()
    
    # Example from query_condition_1
    data1 = load_expressions(expr_dir / "query_condition_1_expressions.json")
    print(f"Example from {data1['source_file']}:")
    print(f"   Expression: {data1['queries'][0]['milvus_expression']}")
    print()
    
    # Example from query_condition_2
    data2 = load_expressions(expr_dir / "query_condition_2_expressions.json")
    print(f"Example from {data2['source_file']}:")
    print(f"   Expression: {data2['queries'][0]['milvus_expression'][:150]}...")
    print()
    
    # Example from query_condition_3
    data3 = load_expressions(expr_dir / "query_condition_3_small_expressions.json")
    print(f"Example from {data3['source_file']}:")
    print(f"   Expression: {data3['queries'][0]['milvus_expression'][:150]}...")
    print()


if __name__ == "__main__":
    main()

