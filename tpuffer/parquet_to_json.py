#!/usr/bin/env python3
"""
Convert parquet file columns to JSON format

This script reads specified columns from a parquet file and converts them
to a JSON format.

Expected input format (parquet):
- Parquet file with one or more columns

Output format (json):
- Array of objects: [{"col1": val1, "col2": val2, ...}, ...]
- Or array of arrays if single column: [[val1], [val2], ...]

Usage:
    python3 parquet_to_json.py input.parquet --columns col1 col2 --rows 1000 --output output.json
    python3 parquet_to_json.py input.parquet --columns col1 --rows 100
"""

import sys
import json
import pandas as pd
import numpy as np
import argparse


def convert_to_native(obj):
    """
    Convert numpy types to Python native types for JSON serialization
    
    Args:
        obj: Object that may contain numpy types
        
    Returns:
        Object with all numpy types converted to Python native types
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_native(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_native(item) for item in obj]
    else:
        return obj


def parquet_to_json(parquet_file: str, columns: list, num_rows: int, json_file: str) -> None:
    """
    Convert parquet file columns to JSON file
    
    Args:
        parquet_file: Path to input .parquet file
        columns: List of column names to extract
        num_rows: Number of rows to extract (None for all rows)
        json_file: Path to output .json file
    """
    print(f"📖 Loading data from {parquet_file}")
    
    # Load parquet file
    try:
        df = pd.read_parquet(parquet_file)
    except Exception as e:
        print(f"❌ Error loading .parquet file: {e}")
        sys.exit(1)
    
    total_rows = len(df)
    print(f"✅ Loaded parquet file with {total_rows} rows, {len(df.columns)} columns")
    print(f"   Available columns: {', '.join(df.columns.tolist())}")
    
    # Validate columns
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        print(f"❌ Error: Column(s) not found: {', '.join(missing_columns)}")
        print(f"   Available columns: {', '.join(df.columns.tolist())}")
        sys.exit(1)
    
    # Select columns
    selected_df = df[columns]
    
    # Limit rows if specified
    if num_rows is not None:
        if num_rows > total_rows:
            print(f"⚠️  Warning: Requested {num_rows} rows, but only {total_rows} available")
            num_rows = total_rows
        selected_df = selected_df.head(num_rows)
        rows_to_write = num_rows
    else:
        rows_to_write = total_rows
    
    print(f"✅ Selected {len(columns)} column(s): {', '.join(columns)}")
    print(f"✅ Will write {rows_to_write} rows")
    
    # Write to JSON file
    print(f"💾 Writing to {json_file}")
    print(f"   Converting {rows_to_write} rows to JSON format...")
    
    try:
        with open(json_file, 'w') as f:
            # Write opening bracket
            f.write('[')
            
            # Write rows one by one to save memory
            for i in range(rows_to_write):
                if i > 0:
                    f.write(',')
                
                # Convert row to dict or list
                row = selected_df.iloc[i]
                
                # If single column, output as array of arrays
                # If multiple columns, output as array of objects
                if len(columns) == 1:
                    row_data = [row.iloc[0]]
                else:
                    row_data = row.to_dict()
                
                # Convert numpy types to Python native types for JSON serialization
                row_data = convert_to_native(row_data)
                
                json.dump(row_data, f)
                
                # Progress indicator
                if (i + 1) % 1000 == 0 or (i + 1) == rows_to_write:
                    progress = (i + 1) / rows_to_write * 100
                    print(f"   Progress: {i + 1}/{rows_to_write} ({progress:.1f}%)")
            
            # Write closing bracket
            f.write(']')
        
        print(f"✅ Successfully converted {parquet_file} → {json_file}")
        if len(columns) == 1:
            print(f"   Format: Array of arrays [{rows_to_write} rows × 1 column]")
        else:
            print(f"   Format: Array of objects [{rows_to_write} rows × {len(columns)} columns]")
    except Exception as e:
        print(f"❌ Error writing JSON file: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Convert parquet file columns to JSON format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 parquet_to_json.py input.parquet --columns col1 col2 --rows 1000 --output output.json
  python3 parquet_to_json.py input.parquet --columns col1 --rows 100
  python3 parquet_to_json.py input.parquet --columns col1 col2 col3 --output output.json
        """
    )
    
    parser.add_argument('parquet_file', type=str,
                       help='Input .parquet file')
    parser.add_argument('--columns', '-c', type=str, nargs='+', required=True,
                       help='Column name(s) to extract (one or more)')
    parser.add_argument('--rows', '-r', type=int, default=None,
                       help='Number of rows to extract (default: all rows)')
    parser.add_argument('--output', '-o', type=str, default='output.json',
                       help='Output .json file (default: output.json)')
    
    args = parser.parse_args()
    
    # Convert
    parquet_to_json(args.parquet_file, args.columns, args.rows, args.output)


if __name__ == '__main__':
    main()

