#!/usr/bin/env python3
"""
Convert parquet query vectors to Go-compatible JSON format

This script:
1. Reads query vectors from parquet file
2. Converts to [][]float32 format 
3. Saves as JSON file for Go consumption

Usage:
    python3 convert_parquet_to_json.py <input_parquet> <output_json>

Example:
    python3 convert_parquet_to_json.py /tmp/query.parquet /tmp/query_vectors.json
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p'
)


class ParquetToJsonConverter:
    """Convert parquet query vectors to Go JSON format"""
    
    def __init__(self, input_parquet_path, output_json_path, max_vectors=None):
        """
        Initialize the converter
        
        Args:
            input_parquet_path: Path to input parquet file
            output_json_path: Path to output JSON file
            max_vectors: Maximum number of vectors to convert (None for all)
        """
        self.input_parquet_path = Path(input_parquet_path)
        self.output_json_path = Path(output_json_path)
        self.max_vectors = max_vectors
        
        if not self.input_parquet_path.exists():
            raise ValueError(f"Input parquet file does not exist: {input_parquet_path}")
    
    def load_vectors_from_parquet(self):
        """Load vectors from parquet file"""
        try:
            logging.info(f"📖 Loading vectors from {self.input_parquet_path}")
            df = pd.read_parquet(self.input_parquet_path)
            
            # Check if 'feature' column exists
            if 'feature' not in df.columns:
                raise ValueError(f"Required 'feature' column not found. Available columns: {list(df.columns)}")
            
            logging.info(f"✅ Found {len(df)} vectors in parquet file")
            
            # Limit vectors if max_vectors is specified
            if self.max_vectors is not None and len(df) > self.max_vectors:
                logging.info(f"🔧 Limiting to first {self.max_vectors} vectors (out of {len(df)})")
                df = df.head(self.max_vectors)
            
            # Extract feature vectors
            feature_vectors = df['feature'].tolist()
            
            # Convert to proper format and validate
            converted_vectors = []
            skipped_count = 0
            
            for i, vector in enumerate(feature_vectors):
                try:
                    if isinstance(vector, np.ndarray):
                        # Convert numpy array to list of float32
                        converted_vector = vector.astype(np.float32).tolist()
                    elif isinstance(vector, list):
                        # Convert list elements to float32
                        converted_vector = [float(x) for x in vector]
                    else:
                        logging.warning(f"⚠️ Skipping invalid vector type at row {i}: {type(vector)}")
                        skipped_count += 1
                        continue
                    
                    # Validate vector dimensions
                    if len(converted_vector) == 0:
                        logging.warning(f"⚠️ Skipping empty vector at row {i}")
                        skipped_count += 1
                        continue
                    
                    converted_vectors.append(converted_vector)
                    
                except Exception as e:
                    logging.warning(f"⚠️ Error converting vector at row {i}: {e}")
                    skipped_count += 1
                    continue
            
            if skipped_count > 0:
                logging.warning(f"⚠️ Skipped {skipped_count} invalid vectors")
            
            logging.info(f"✅ Successfully converted {len(converted_vectors)} vectors")
            
            if len(converted_vectors) == 0:
                raise ValueError("No valid vectors found in the parquet file")
            
            return converted_vectors
            
        except Exception as e:
            logging.error(f"❌ Failed to load vectors from parquet: {e}")
            raise
    
    def save_vectors_to_json(self, vectors):
        """Save vectors to JSON file in Go-compatible format"""
        try:
            # Ensure output directory exists
            self.output_json_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create the JSON structure
            # Go expects: type Queries [][]float32
            json_data = vectors
            
            logging.info(f"💾 Saving {len(vectors)} vectors to {self.output_json_path}")
            
            # Save to JSON file with proper formatting
            with open(self.output_json_path, 'w') as f:
                json.dump(json_data, f, indent=2, separators=(',', ': '))
            
            # Get file size
            file_size = self.output_json_path.stat().st_size / 1024 / 1024
            logging.info(f"📁 JSON file saved, size: {file_size:.2f} MB")
            
            return self.output_json_path
            
        except Exception as e:
            logging.error(f"❌ Failed to save vectors to JSON: {e}")
            raise
    
    def verify_json_file(self):
        """Verify the saved JSON file"""
        try:
            logging.info(f"🔍 Verifying JSON file...")
            
            with open(self.output_json_path, 'r') as f:
                loaded_data = json.load(f)
            
            if not isinstance(loaded_data, list):
                raise ValueError("JSON data should be a list (array)")
            
            if len(loaded_data) == 0:
                raise ValueError("JSON data is empty")
            
            # Check first few vectors
            sample_size = min(3, len(loaded_data))
            for i in range(sample_size):
                vector = loaded_data[i]
                if not isinstance(vector, list):
                    raise ValueError(f"Vector {i} is not a list: {type(vector)}")
                
                if len(vector) == 0:
                    raise ValueError(f"Vector {i} is empty")
                
                # Check if all elements are numbers
                for j, val in enumerate(vector[:3]):  # Check first 3 elements
                    if not isinstance(val, (int, float)):
                        raise ValueError(f"Vector {i}, element {j} is not a number: {type(val)}")
            
            # Report statistics
            vector_count = len(loaded_data)
            dimensions = set(len(v) for v in loaded_data[:100])  # Check first 100 vectors
            
            logging.info(f"✅ Verification successful:")
            logging.info(f"   Total vectors: {vector_count}")
            logging.info(f"   Dimensions found: {sorted(dimensions)}")
            
            if len(dimensions) == 1:
                dim = list(dimensions)[0]
                logging.info(f"   All vectors have {dim} dimensions")
            else:
                logging.warning(f"⚠️ Multiple dimensions detected: {sorted(dimensions)}")
            
            # Show sample vectors
            for i in range(min(2, len(loaded_data))):
                vector = loaded_data[i]
                logging.info(f"   Sample vector {i+1}: dimension {len(vector)}, first 3 values: {vector[:3]}")
            
            return True
            
        except Exception as e:
            logging.error(f"❌ JSON verification failed: {e}")
            return False
    
    def convert(self):
        """Main conversion method"""
        try:
            logging.info(f"🚀 Starting parquet to JSON conversion...")
            
            # Load vectors from parquet
            vectors = self.load_vectors_from_parquet()
            
            # Save to JSON
            json_path = self.save_vectors_to_json(vectors)
            
            # Verify the result
            if self.verify_json_file():
                logging.info(f"🎉 Conversion completed successfully!")
                logging.info(f"📁 Output file: {json_path}")
                return json_path
            else:
                raise RuntimeError("JSON verification failed")
                
        except Exception as e:
            logging.error(f"❌ Conversion failed: {e}")
            raise


def main():
    """Main function"""
    
    # Default values
    # input_parquet = "/root/test/data/query.parquet"
    # output_json = "/root/test/data/query_vectors_1.json"
    input_parquet = "~/Downloads/query.parquet"
    output_json = "~/Downloads/query_vectors_10.json"
    max_vectors = 10
    
    try:
        # Create converter and run conversion
        converter = ParquetToJsonConverter(input_parquet, output_json, max_vectors)
        result_path = converter.convert()
        
        logging.info(f"✅ Success! JSON file created: {result_path}")
        
    except Exception as e:
        logging.error(f"❌ Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
