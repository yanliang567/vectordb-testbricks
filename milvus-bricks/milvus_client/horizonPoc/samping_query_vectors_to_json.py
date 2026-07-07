#!/usr/bin/env python3
"""
Convert parquet query vectors to Go-compatible JSON format

This script:
1. Reads vector field from all parquet files in a directory
2. Randomly samples 50 vectors from each file
3. Merges all sampled vectors and saves as JSON file for Go consumption

Usage:
    python3 samping_query_vectors_to_json.py <input_directory> <output_json>

Example:
    python3 samping_query_vectors_to_json.py /tmp/parquet_data /tmp/query_vectors.json
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple
import multiprocessing

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p'
)


def process_single_parquet_file(args: Tuple[Path, int]) -> Tuple[List, int, str]:
    """
    Process a single parquet file and return sampled vectors.
    This function is designed to be called in parallel processes.
    
    Args:
        args: Tuple of (parquet_file_path, vectors_per_file)
    
    Returns:
        Tuple of (converted_vectors, skipped_count, filename)
    """
    parquet_file, vectors_per_file = args
    
    try:
        import pyarrow.parquet as pq
        
        # Get file metadata
        parquet_file_obj = pq.ParquetFile(parquet_file)
        parquet_metadata = parquet_file_obj.schema_arrow
        column_names = parquet_metadata.names
        
        if 'vector' not in column_names:
            return [], 0, str(parquet_file.name)
        
        total_rows = parquet_file_obj.metadata.num_rows
        num_row_groups = parquet_file_obj.num_row_groups
        sample_size = min(vectors_per_file, total_rows)
        
        # 🚀 Performance optimization: Smart sampling based on file size
        if total_rows <= vectors_per_file:
            # Small file: read all
            df = pd.read_parquet(parquet_file, columns=['vector'])
            vectors = df['vector'].tolist()
            del df
            
        elif total_rows <= 50000:
            # Medium file: load and sample
            df = pd.read_parquet(parquet_file, columns=['vector'])
            random_indices = np.random.choice(len(df), size=sample_size, replace=False)
            vectors = [df['vector'].iloc[i] for i in random_indices]
            del df
            
        else:
            # 🚀 Large file optimization: Random row group sampling
            # Instead of reading ALL data, randomly select row groups
            # This can skip 90%+ of the file!
            
            # Calculate how many row groups to read
            rows_per_group = total_rows / num_row_groups
            # Read extra row groups to ensure we get enough samples
            target_row_groups = min(
                num_row_groups,
                max(3, int(np.ceil(sample_size / rows_per_group * 1.5)))
            )
            
            # Randomly select row groups
            selected_row_groups = np.random.choice(
                num_row_groups, 
                size=target_row_groups, 
                replace=False
            )
            selected_row_groups = sorted(selected_row_groups)
            
            # Read only selected row groups
            vectors_collected = []
            for rg_idx in selected_row_groups:
                rg_table = parquet_file_obj.read_row_group(rg_idx, columns=['vector'])
                rg_df = rg_table.to_pandas()
                vectors_collected.extend(rg_df['vector'].tolist())
                del rg_df, rg_table
                
                # Early termination if we have enough samples
                if len(vectors_collected) >= sample_size * 2:
                    break
            
            # Final random sampling from collected vectors
            if len(vectors_collected) > sample_size:
                random_indices = np.random.choice(len(vectors_collected), size=sample_size, replace=False)
                vectors = [vectors_collected[i] for i in random_indices]
            else:
                vectors = vectors_collected[:sample_size]
            
            del vectors_collected
        
        # Convert and validate vectors
        converted_vectors = []
        skipped_count = 0
        
        for vector in vectors:
            try:
                if isinstance(vector, np.ndarray):
                    converted_vector = vector.astype(np.float32).tolist()
                elif isinstance(vector, list):
                    converted_vector = [float(x) for x in vector]
                else:
                    skipped_count += 1
                    continue
                
                if len(converted_vector) == 0:
                    skipped_count += 1
                    continue
                
                converted_vectors.append(converted_vector)
                
            except Exception:
                skipped_count += 1
                continue
        
        del vectors
        gc.collect()
        
        return converted_vectors, skipped_count, str(parquet_file.name)
        
    except Exception as e:
        logging.error(f"❌ Failed to process {parquet_file.name}: {e}")
        return [], 0, str(parquet_file.name)


class ParquetToJsonConverter:
    """Convert parquet query vectors to Go JSON format"""
    
    def __init__(self, input_directory_path, output_json_path, vectors_per_file=50, 
                 use_parallel=True, max_workers=None):
        """
        Initialize the converter
        
        Args:
            input_directory_path: Path to directory containing parquet files
            output_json_path: Path to output JSON file
            vectors_per_file: Number of vectors to randomly sample from each parquet file (default: 50)
            use_parallel: Whether to use parallel processing (default: True)
            max_workers: Maximum number of parallel workers (default: CPU count - 1)
        """
        self.input_directory_path = Path(input_directory_path)
        self.output_json_path = Path(output_json_path)
        self.vectors_per_file = vectors_per_file
        self.use_parallel = use_parallel
        
        # Set max workers for parallel processing
        if max_workers is None:
            cpu_count = multiprocessing.cpu_count()
            self.max_workers = max(1, cpu_count - 1)  # Leave 1 CPU free
        else:
            self.max_workers = max_workers
        
        if not self.input_directory_path.exists():
            raise ValueError(f"Input directory does not exist: {input_directory_path}")
        
        if not self.input_directory_path.is_dir():
            raise ValueError(f"Input path is not a directory: {input_directory_path}")
    
    def load_vectors_from_parquet(self):
        """Load vectors from all parquet files in directory (with optional parallel processing)"""
        try:
            # Find all parquet files in the directory
            parquet_files = list(self.input_directory_path.glob('*.parquet'))
            
            if not parquet_files:
                raise ValueError(f"No parquet files found in directory: {self.input_directory_path}")
            
            logging.info(f"📖 Found {len(parquet_files)} parquet files in {self.input_directory_path}")
            
            if self.use_parallel and len(parquet_files) > 1:
                # 🚀 PARALLEL PROCESSING MODE
                logging.info(f"⚡ Using parallel processing with {self.max_workers} workers")
                return self._load_vectors_parallel(parquet_files)
            else:
                # Serial processing mode
                logging.info(f"📝 Using serial processing")
                return self._load_vectors_serial(parquet_files)
                
        except Exception as e:
            logging.error(f"❌ Failed to load vectors from parquet files: {e}")
            raise
    
    def _load_vectors_parallel(self, parquet_files):
        """Load vectors using parallel processing"""
        all_converted_vectors = []
        total_skipped = 0
        
        # Prepare arguments for parallel processing
        args_list = [(file, self.vectors_per_file) for file in parquet_files]
        
        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(process_single_parquet_file, args): args[0] 
                for args in args_list
            }
            
            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_file):
                completed += 1
                parquet_file = future_to_file[future]
                
                try:
                    vectors, skipped, filename = future.result()
                    
                    logging.info(f"✅ [{completed}/{len(parquet_files)}] Processed {filename}: "
                               f"{len(vectors)} vectors, {skipped} skipped")
                    
                    all_converted_vectors.extend(vectors)
                    total_skipped += skipped
                    
                except Exception as e:
                    logging.error(f"❌ Failed to process {parquet_file.name}: {e}")
        
        if total_skipped > 0:
            logging.warning(f"⚠️ Total skipped vectors across all files: {total_skipped}")
        
        logging.info(f"✅ Successfully merged {len(all_converted_vectors)} vectors from {len(parquet_files)} files")
        
        if len(all_converted_vectors) == 0:
            raise ValueError("No valid vectors found in any parquet files")
        
        return all_converted_vectors
    
    def _load_vectors_serial(self, parquet_files):
        """Load vectors using serial processing"""
        all_converted_vectors = []
        total_skipped = 0
        
        # Process each parquet file
        for file_idx, parquet_file in enumerate(parquet_files, 1):
            logging.info(f"📂 Processing file {file_idx}/{len(parquet_files)}: {parquet_file.name}")
            
            vectors, skipped, filename = process_single_parquet_file((parquet_file, self.vectors_per_file))
            
            logging.info(f"   ✅ Successfully processed {filename}: {len(vectors)} vectors, {skipped} skipped")
            
            all_converted_vectors.extend(vectors)
            total_skipped += skipped
        
        if total_skipped > 0:
            logging.warning(f"⚠️ Total skipped vectors across all files: {total_skipped}")
        
        logging.info(f"✅ Successfully merged {len(all_converted_vectors)} vectors from {len(parquet_files)} files")
        
        if len(all_converted_vectors) == 0:
            raise ValueError("No valid vectors found in any parquet files")
        
        return all_converted_vectors
    
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
    input_directory = "/root/horizon/horizonPoc/data/expanded_dataset_100M"
    output_json = "/root/horizon/horizonPoc/data/merged_query_vectors.json"
    # input_directory = "/Users/yanliang/Downloads/horizonPoc/data"
    # output_json = "/Users/yanliang/Downloads/horizonPoc/data/merged_query_vectors.json"
    vectors_per_file = 50  # Number of random vectors to sample from each file
    
    # Performance settings
    use_parallel = True  # Enable parallel processing for large directories
    max_workers = None  # None = auto-detect (CPU count - 1)
    
    try:
        # Create converter and run conversion
        converter = ParquetToJsonConverter(
            input_directory, 
            output_json, 
            vectors_per_file,
            use_parallel=use_parallel,
            max_workers=max_workers
        )
        result_path = converter.convert()
        
        logging.info(f"✅ Success! JSON file created: {result_path}")
        
    except Exception as e:
        logging.error(f"❌ Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
