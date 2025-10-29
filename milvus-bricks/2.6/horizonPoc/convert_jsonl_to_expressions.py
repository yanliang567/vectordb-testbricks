#!/usr/bin/env python3
"""
Convert JSONL query files to Milvus Expression format

This script reads JSONL query files from query_scenarios directory,
converts each query to Milvus expression syntax, and saves them as JSON files.

Features:
- Automatically converts range queries to combined range syntax
  (e.g., "min_value <= column <= max_value" instead of "column >= min and column <= max")
- Supports timestamp, gcj02_lon, gcj02_lat range fields
- Handles array operations (ARRAY_CONTAINS_ANY, ARRAY_CONTAINS_ALL)
- Handles scalar operations (IN, ==, !=)

Field Mappings (Source → Milvus):
- longitude → gcj02_lon  (coordinate system conversion)
- latitude  → gcj02_lat  (coordinate system conversion)
- tag_id    → timeline_tags  (field name standardization)

Usage:
    python3 convert_jsonl_to_expressions.py
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List


def convert_query_to_expression(query: Dict[str, Any]) -> str:
    """
    Convert a JSON query condition to Milvus expression string
    
    Args:
        query: Dictionary containing query conditions
        
    Returns:
        Milvus expression string with combined range syntax
    """
    conditions = []
    
    # Handle timestamp (range query) - Use combined range syntax
    if 'timestamp' in query:
        has_gte = '$gte' in query['timestamp']
        has_lte = '$lte' in query['timestamp']
        
        if has_gte and has_lte:
            # Use combined range: min_value <= timestamp <= max_value
            min_val = int(query['timestamp']['$gte'])
            max_val = int(query['timestamp']['$lte'])
            conditions.append(f"{min_val} <= timestamp <= {max_val}")
        elif has_gte:
            conditions.append(f"timestamp >= {int(query['timestamp']['$gte'])}")
        elif has_lte:
            conditions.append(f"timestamp <= {int(query['timestamp']['$lte'])}")
    
    # Handle type_model (exact match)
    if 'type_model' in query:
        conditions.append(f'type_model == "{query["type_model"]}"')
    
    # Handle expert_collected (boolean)
    if 'expert_collected' in query:
        value = "true" if query['expert_collected'] else "false"
        conditions.append(f'expert_collected == {value}')
    
    # Handle device_id ($in operator for scalar field)
    if 'device_id' in query and '$in' in query['device_id']:
        device_ids = '","'.join(query['device_id']['$in'])
        conditions.append(f'device_id in ["{device_ids}"]')
    
    # Handle tag_id (array field operations) - Map to timeline_tags in Milvus
    # Source file uses 'tag_id', but Milvus collection uses 'timeline_tags'
    if 'tag_id' in query:
        # $in operator -> ARRAY_CONTAINS_ANY
        if '$in' in query['tag_id']:
            tag_ids = '","'.join(query['tag_id']['$in'])
            conditions.append(f'ARRAY_CONTAINS_ANY(timeline_tags, ["{tag_ids}"])')
        
        # contains_all operator -> ARRAY_CONTAINS_ALL
        if 'contains_all' in query['tag_id']:
            tag_ids = '","'.join(query['tag_id']['contains_all'])
            conditions.append(f'ARRAY_CONTAINS_ALL(timeline_tags, ["{tag_ids}"])')
    
    # Handle sensor_lidar_type (for scalar IN and NOT IN)
    if 'sensor_lidar_type' in query:
        # $in operator -> IN operator for scalar field
        if '$in' in query['sensor_lidar_type']:
            sensors = '","'.join(query['sensor_lidar_type']['$in'])
            conditions.append(f'sensor_lidar_type IN ["{sensors}"]')
        
        # $not_in operator -> != for each value
        if '$not_in' in query['sensor_lidar_type']:
            for sensor in query['sensor_lidar_type']['$not_in']:
                conditions.append(f'sensor_lidar_type != "{sensor}"')
    
    # Handle longitude (range query) - Map to gcj02_lon in Milvus
    # Source file uses 'longitude', but Milvus collection uses 'gcj02_lon'
    if 'longitude' in query:
        has_gte = '$gte' in query['longitude']
        has_lte = '$lte' in query['longitude']
        
        if has_gte and has_lte:
            # Use combined range: min_value <= gcj02_lon <= max_value
            min_val = query['longitude']['$gte']
            max_val = query['longitude']['$lte']
            conditions.append(f"{min_val} <= gcj02_lon <= {max_val}")
        elif has_gte:
            conditions.append(f"gcj02_lon >= {query['longitude']['$gte']}")
        elif has_lte:
            conditions.append(f"gcj02_lon <= {query['longitude']['$lte']}")
    
    # Handle latitude (range query) - Map to gcj02_lat in Milvus
    # Source file uses 'latitude', but Milvus collection uses 'gcj02_lat'
    if 'latitude' in query:
        has_gte = '$gte' in query['latitude']
        has_lte = '$lte' in query['latitude']
        
        if has_gte and has_lte:
            # Use combined range: min_value <= gcj02_lat <= max_value
            min_val = query['latitude']['$gte']
            max_val = query['latitude']['$lte']
            conditions.append(f"{min_val} <= gcj02_lat <= {max_val}")
        elif has_gte:
            conditions.append(f"gcj02_lat >= {query['latitude']['$gte']}")
        elif has_lte:
            conditions.append(f"gcj02_lat <= {query['latitude']['$lte']}")
    
    # Join all conditions with " and "
    if len(conditions) == 0:
        return ""
    
    return ' and '.join(conditions)


def process_jsonl_file(input_file: Path, output_dir: Path) -> Dict[str, Any]:
    """
    Process a single JSONL file and convert to expressions
    
    Args:
        input_file: Path to input JSONL file
        output_dir: Directory to save output JSON file
        
    Returns:
        Dictionary with statistics about the conversion
    """
    print(f"\n📖 Processing: {input_file.name}")
    print("-" * 80)
    
    queries_with_expressions = []
    total_lines = 0
    valid_queries = 0
    invalid_queries = 0
    
    # Read JSONL file
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            total_lines += 1
            line = line.strip()
            
            if not line:
                continue
            
            try:
                # Parse JSON query
                query = json.loads(line)
                
                # Convert to Milvus expression
                expression = convert_query_to_expression(query)
                
                if expression:
                    queries_with_expressions.append({
                        'query_id': valid_queries + 1,
                        'original_query': query,
                        'milvus_expression': expression
                    })
                    valid_queries += 1
                else:
                    print(f"⚠️  Line {line_num}: Empty expression generated")
                    invalid_queries += 1
                    
            except json.JSONDecodeError as e:
                print(f"❌ Line {line_num}: Invalid JSON - {e}")
                invalid_queries += 1
            except Exception as e:
                print(f"❌ Line {line_num}: Conversion error - {e}")
                invalid_queries += 1
    
    # Generate output filename
    output_file = output_dir / f"{input_file.stem}_expressions.json"
    
    # Save to JSON file
    output_data = {
        'source_file': input_file.name,
        'total_queries': valid_queries,
        'queries': queries_with_expressions
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Converted: {valid_queries} queries")
    print(f"⚠️  Skipped: {invalid_queries} invalid entries")
    print(f"💾 Saved to: {output_file.name}")
    
    return {
        'file': input_file.name,
        'total_lines': total_lines,
        'valid_queries': valid_queries,
        'invalid_queries': invalid_queries,
        'output_file': output_file.name
    }


def main():
    """Main function to process all JSONL files"""
    print("=" * 80)
    print("JSONL Query to Milvus Expression Converter")
    print("=" * 80)
    
    # Define paths
    query_dir = Path("data/query_scenarios_merged_vector")
    output_dir = Path("data/query_expressions_merged_vector")
    # query_dir = Path("data/query_scenarios")
    # output_dir = Path("data/query_expressions")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📂 Input directory: {query_dir}")
    print(f"📂 Output directory: {output_dir}")
    
    # Find all .jsonl files
    jsonl_files = sorted(query_dir.glob("*.jsonl"))
    
    if not jsonl_files:
        print(f"\n❌ No .jsonl files found in {query_dir}")
        return
    
    print(f"\n🔍 Found {len(jsonl_files)} JSONL files:")
    for f in jsonl_files:
        print(f"   - {f.name}")
    
    # Process each file
    results = []
    for jsonl_file in jsonl_files:
        try:
            result = process_jsonl_file(jsonl_file, output_dir)
            results.append(result)
        except Exception as e:
            print(f"\n❌ Failed to process {jsonl_file.name}: {e}")
            results.append({
                'file': jsonl_file.name,
                'error': str(e)
            })
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 Conversion Summary")
    print("=" * 80)
    print(f"\n{'File':<40} {'Queries':<10} {'Skipped':<10} {'Output':<30}")
    print("-" * 80)
    
    total_queries = 0
    total_skipped = 0
    
    for result in results:
        if 'error' in result:
            print(f"{result['file']:<40} {'ERROR':<10} {'-':<10} {'-':<30}")
        else:
            print(f"{result['file']:<40} {result['valid_queries']:<10} {result['invalid_queries']:<10} {result['output_file']:<30}")
            total_queries += result['valid_queries']
            total_skipped += result['invalid_queries']
    
    print("-" * 80)
    print(f"{'TOTAL':<40} {total_queries:<10} {total_skipped:<10}")
    
    print("\n✅ Conversion completed!")
    print(f"\n💡 Output files saved in: {output_dir}")
    print("\nYou can now use these expression files in your Go search script:")
    print("   - Load the JSON file directly")
    print("   - Use the 'milvus_expression' field for each query")
    print("   - No need to convert on-the-fly during performance testing")


if __name__ == "__main__":
    main()

