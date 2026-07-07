#!/usr/bin/env python3
"""
Milvus Expression Rewriter (In-Place Modification)

This script rewrites Milvus expressions in JSON files to use combined range syntax.
It transforms expressions from:
    column_name >= min_value and column_name <= max_value
to:
    min_value <= column_name <= max_value

⚠️  WARNING: This script modifies files IN-PLACE. Original files will be overwritten!
    Make sure you have backups before running this script.

Supported columns: timestamp, gcj02_lon, gcj02_lat (and any other numeric range columns)

Usage:
    # Process all *_exprs.json files in current directory
    python rewrite_expressions.py

    # Process specific file
    python rewrite_expressions.py --file qc_1_exprs.json

    # Process files in specific directory
    python rewrite_expressions.py --dir /path/to/query_expressions

    # Process multiple directories
    python rewrite_expressions.py --dir ./query_expressions --dir ./query_expressions_merged_vector

    # Custom file pattern
    python rewrite_expressions.py --pattern "*_exprs*.json" --dir ./data
"""

import json
import re
import os
import sys
import argparse
from pathlib import Path


def rewrite_range_expression(expr):
    """
    Rewrite range expressions from 'column >= min and column <= max' 
    to 'min <= column <= max' format.
    
    Args:
        expr (str): Original Milvus expression string
        
    Returns:
        str: Rewritten expression string
    """
    # Pattern to match: column_name >= VALUE1 and column_name <= VALUE2
    # This regex captures:
    # - Group 1: column name (word characters)
    # - Group 2: minimum value (digits and optional decimal point)
    # - Group 3: maximum value (same column name via backreference \1)
    # - Group 4: maximum value (digits and optional decimal point)
    pattern = r'(\w+) >= ([\d.]+) and \1 <= ([\d.]+)'
    
    def replacement(match):
        column_name = match.group(1)
        min_val = match.group(2)
        max_val = match.group(3)
        return f'{min_val} <= {column_name} <= {max_val}'
    
    return re.sub(pattern, replacement, expr)


def process_json_file(input_file, verbose=True):
    """
    Process a single JSON file and rewrite expressions in-place.
    
    Args:
        input_file (str): Path to JSON file to modify
        verbose (bool): Whether to print progress information
        
    Returns:
        dict: Processing statistics (total_queries, modified_count)
    """
    try:
        # Read the original JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Process all queries
        modified_count = 0
        total_queries = len(data.get('queries', []))
        
        for query in data.get('queries', []):
            if 'milvus_expression' in query:
                original_expr = query['milvus_expression']
                new_expr = rewrite_range_expression(original_expr)
                
                if original_expr != new_expr:
                    modified_count += 1
                
                query['milvus_expression'] = new_expr
        
        # Write back to the same file (in-place modification)
        with open(input_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        if verbose:
            print(f"✓ {os.path.basename(input_file)}")
            print(f"  Total queries: {total_queries}")
            print(f"  Modified: {modified_count}")
            
            # Show one example if there were modifications
            if modified_count > 0 and data.get('queries'):
                example = data['queries'][0]['milvus_expression']
                if len(example) > 120:
                    example = example[:120] + "..."
                print(f"  Example: {example}")
            print()
        
        return {
            'total_queries': total_queries,
            'modified_count': modified_count,
            'success': True
        }
        
    except Exception as e:
        if verbose:
            print(f"✗ Error processing {input_file}: {str(e)}")
        return {
            'total_queries': 0,
            'modified_count': 0,
            'success': False,
            'error': str(e)
        }


def process_directory(directory, file_pattern='*_exprs.json', verbose=True):
    """
    Process all matching JSON files in a directory (in-place modification).
    
    Args:
        directory (str): Directory containing JSON files
        file_pattern (str): Glob pattern for input files (default: '*_exprs.json')
        verbose (bool): Whether to print progress information
        
    Returns:
        dict: Overall statistics
    """
    directory = Path(directory)
    
    # Find all matching JSON files
    json_files = list(directory.glob(file_pattern))
    
    if not json_files:
        if verbose:
            print(f"No matching files found in {directory}")
        return None
    
    if verbose:
        print(f"Found {len(json_files)} JSON files to process in {directory}\n")
        print("=" * 80)
    
    # Process each file
    total_stats = {
        'files_processed': 0,
        'files_failed': 0,
        'total_queries': 0,
        'total_modified': 0
    }
    
    for json_file in sorted(json_files):
        # Process the file in-place
        stats = process_json_file(str(json_file), verbose=verbose)
        
        if stats['success']:
            total_stats['files_processed'] += 1
            total_stats['total_queries'] += stats['total_queries']
            total_stats['total_modified'] += stats['modified_count']
        else:
            total_stats['files_failed'] += 1
    
    if verbose:
        print("=" * 80)
        print(f"Summary:")
        print(f"  Files processed: {total_stats['files_processed']}")
        print(f"  Files failed: {total_stats['files_failed']}")
        print(f"  Total queries: {total_stats['total_queries']}")
        print(f"  Total modified: {total_stats['total_modified']}")
    
    return total_stats


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Rewrite Milvus expressions to use combined range syntax (in-place modification)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all files in current directory
  python rewrite_expressions.py
  
  # Process specific file
  python rewrite_expressions.py --file qc_1_exprs.json
  
  # Process files in specific directory
  python rewrite_expressions.py --dir ./query_expressions
  
  # Process multiple directories
  python rewrite_expressions.py --dir ./query_expressions --dir ./query_expressions_merged_vector
  
  # Process all subdirectories under data/
  python rewrite_expressions.py --all-subdirs --dir ./data
  
  # Use custom file pattern
  python rewrite_expressions.py --pattern "*_exprs*.json" --dir ./query_expressions_merged_vector

WARNING: This script modifies files in-place. Make sure you have backups!
        """
    )
    
    parser.add_argument(
        '--file', '-f',
        help='Process a specific JSON file (in-place)',
        type=str
    )
    
    parser.add_argument(
        '--dir', '-d',
        help='Directory containing JSON files (can be specified multiple times)',
        type=str,
        action='append',
        dest='directories'
    )
    
    parser.add_argument(
        '--all-subdirs',
        help='Process all subdirectories under the specified directory',
        action='store_true'
    )
    
    parser.add_argument(
        '--pattern', '-p',
        help='File pattern to match (default: *_exprs.json)',
        type=str,
        default='*_exprs.json'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        help='Suppress output messages',
        action='store_true'
    )
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    try:
        if args.file:
            # Process single file in-place
            input_file = Path(args.file)
            if not input_file.exists():
                print(f"Error: File not found: {input_file}")
                return 1
            
            if verbose:
                print(f"Processing file: {input_file}")
                print(f"WARNING: File will be modified in-place!\n")
            
            stats = process_json_file(str(input_file), verbose=verbose)
            return 0 if stats['success'] else 1
        else:
            # Process directory or directories
            directories = args.directories if args.directories else ['.']
            
            if verbose:
                print("WARNING: Files will be modified in-place!")
                print("=" * 80 + "\n")
            
            # If --all-subdirs is specified, find all subdirectories
            if args.all_subdirs:
                all_dirs = []
                for directory in directories:
                    dir_path = Path(directory)
                    if not dir_path.exists():
                        print(f"Warning: Directory not found: {directory}")
                        continue
                    
                    # Add current directory
                    all_dirs.append(directory)
                    
                    # Find all subdirectories
                    for subdir in dir_path.rglob('*'):
                        if subdir.is_dir():
                            all_dirs.append(str(subdir))
                
                directories = all_dirs
                if verbose and len(directories) > 1:
                    print(f"Found {len(directories)} directories to process\n")
            
            # Process all directories
            total_stats = {
                'files_processed': 0,
                'files_failed': 0,
                'total_queries': 0,
                'total_modified': 0
            }
            
            for directory in directories:
                if not Path(directory).exists():
                    print(f"Warning: Directory not found: {directory}")
                    continue
                
                if verbose and len(directories) > 1:
                    print(f"\n{'='*80}")
                    print(f"Processing directory: {directory}")
                    print('='*80)
                
                stats = process_directory(
                    directory,
                    file_pattern=args.pattern,
                    verbose=verbose
                )
                
                if stats:
                    total_stats['files_processed'] += stats['files_processed']
                    total_stats['files_failed'] += stats['files_failed']
                    total_stats['total_queries'] += stats['total_queries']
                    total_stats['total_modified'] += stats['total_modified']
            
            # Print overall summary if processing multiple directories
            if len(directories) > 1 and verbose:
                print("\n" + "="*80)
                print("Overall Summary:")
                print(f"  Directories processed: {len(directories)}")
                print(f"  Files processed: {total_stats['files_processed']}")
                print(f"  Files failed: {total_stats['files_failed']}")
                print(f"  Total queries: {total_stats['total_queries']}")
                print(f"  Total modified: {total_stats['total_modified']}")
                print("="*80)
            
            return 0 if total_stats['files_failed'] == 0 else 1
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 130
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1


if __name__ == '__main__':
    sys.exit(main())

