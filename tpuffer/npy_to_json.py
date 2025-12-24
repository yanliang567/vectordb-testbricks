#!/usr/bin/env python3
"""
Convert query.npy to query.json for use with query.go

This script reads a numpy .npy file containing query vectors and converts it
to a JSON format that query.go can read.

Expected input format (npy):
- 2D numpy array: shape (N, D) where N is number of vectors, D is dimension

Output format (json):
- 2D array: [[v1_d1, v1_d2, ...], [v2_d1, v2_d2, ...], ...]

Usage:
    python3 npy_to_json.py query.npy query.json
    python3 npy_to_json.py query.npy  # output to query.json by default
"""

import sys
import json
import numpy as np
import argparse


def npy_to_json(npy_file: str, json_file: str) -> None:
    """
    Convert .npy file to .json file
    
    Args:
        npy_file: Path to input .npy file
        json_file: Path to output .json file
    """
    print(f"📖 Loading vectors from {npy_file}")
    
    # Load numpy array
    try:
        vectors = np.load(npy_file)
    except Exception as e:
        print(f"❌ Error loading .npy file: {e}")
        sys.exit(1)
    
    # Validate shape
    if vectors.ndim != 2:
        print(f"❌ Error: Expected 2D array, got {vectors.ndim}D array")
        print(f"   Shape: {vectors.shape}")
        sys.exit(1)
    
    num_vectors, dimension = vectors.shape
    print(f"✅ Loaded {num_vectors} vectors, dimension: {dimension}")
    
    # Write to JSON file
    print(f"💾 Writing to {json_file}")
    print(f"   Converting {num_vectors} vectors to JSON format...")
    
    try:
        with open(json_file, 'w') as f:
            # Write opening bracket
            f.write('[')
            
            # Write vectors one by one to save memory
            for i in range(num_vectors):
                if i > 0:
                    f.write(',')
                
                # Convert single vector to list and write
                vector_list = vectors[i].tolist()
                json.dump(vector_list, f)
                
                # Progress indicator
                if (i + 1) % 1000 == 0 or (i + 1) == num_vectors:
                    progress = (i + 1) / num_vectors * 100
                    print(f"   Progress: {i + 1}/{num_vectors} ({progress:.1f}%)")
            
            # Write closing bracket
            f.write(']')
        
        print(f"✅ Successfully converted {npy_file} → {json_file}")
        print(f"   Format: 2D array [{num_vectors} vectors × {dimension} dimensions]")
    except Exception as e:
        print(f"❌ Error writing JSON file: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Convert query.npy to query.json for query.go',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 npy_to_json.py query.npy query.json
  python3 npy_to_json.py query.npy  # defaults to query.json
  python3 npy_to_json.py input.npy output.json
        """
    )
    
    parser.add_argument('npy_file', type=str,
                       help='Input .npy file containing query vectors')
    parser.add_argument('json_file', type=str, nargs='?', default='query.json',
                       help='Output .json file (default: query.json)')
    
    args = parser.parse_args()
    
    # Convert
    npy_to_json(args.npy_file, args.json_file)


if __name__ == '__main__':
    main()

