#!/usr/bin/env python3
"""
Query Iterator Testing Script - Single-threaded query iterator performance testing

This script performs query iterator operations on Milvus collections:
1. Tests query_iterator performance with configurable batch sizes
2. Measures iteration latency, QPS, and statistics
3. Supports checkpoint files for large dataset iterations
4. Single-threaded operation (no concurrency)
5. Strict collection validation (fail fast if collection doesn't exist)

Key Features:
- MilvusClient API (Milvus 2.6 compatible)
- Single-threaded query iterator testing
- Configurable batch size and iteration limits
- Comprehensive performance metrics (QPS, latency, P99, etc.)
- Checkpoint file support for resumable iterations
- Strict collection validation (exit immediately if not found)
- Smart parquet file export: 100 batches per file to avoid large files
- Memory-efficient batch processing and file saving
- Uses common.py utility functions for schema analysis
- All English comments and documentation

Usage:
    python3 query_iterator.py <host> <collection_name> <iter_times> <output_fields> <expr> <batch_size> <save_in_file> [api_key]

Examples:
    # Single iteration test with all fields, no file save
    python3 query_iterator.py localhost my_collection 1 "*" "" 2000 FALSE None
    
    # Multiple iterations with specific fields and save to parquet
    python3 query_iterator.py localhost test_collection 5 "id,vector" "id > 0" 1000 TRUE None
    
    # Cloud Milvus with API key and parquet export
    python3 query_iterator.py your-host.com collection_name 3 "*" "" 5000 TRUE your_api_key
"""

import time
import sys
import numpy as np
import logging
import pandas as pd
from pymilvus import MilvusClient, DataType

# Import common utility functions from 2.6 directory
from common import (
    get_float_vec_field_names,
    get_primary_field_name,
    get_float_vec_field_name
)

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"


class QueryIteratorTester:
    """
    Query Iterator testing manager using MilvusClient API
    
    This class provides single-threaded query iterator testing with comprehensive
    performance metrics and checkpoint support for large datasets.
    """
    
    def __init__(self, client, collection_name):
        self.client = client
        self.collection_name = collection_name
        self.checkpoint_file = f"/tmp/query_iterator_{collection_name}.checkpoint"
    
    def validate_collection(self):
        """
        Validate collection for query iterator operations
        Returns: (is_valid, error_message)
        """
        # Check if collection exists
        if not self.client.has_collection(self.collection_name):
            return False, f"Collection '{self.collection_name}' does not exist"
        
        load_state = self.client.get_load_state(self.collection_name)
        if load_state.get('state').name != 'Loaded':
            return False, f"Collection '{self.collection_name}' is not loaded"
        
        return True, "Collection validation passed"
    
    
    def run_single_iteration(self, batch_size, output_fields, expr, iteration_num, save_in_file=False):
        """
        Run a single complete iteration through the collection
        
        :param batch_size: Number of entities per batch
        :param output_fields: Fields to return
        :param expr: Query expression filter
        :param iteration_num: Current iteration number
        :param save_in_file: If True, save results to parquet files (100 batches per file)
        """
        iteration_start = time.time()
        
        batch_latencies = []
        batch_count = 0
        total_entities = 0
        failures = 0
        current_batch_results = []  # Store current batch results for parquet export
        file_batch_count = 0  # Count batches for current file
        total_files_saved = 0  # Track total files saved
        
        try:
            # Create query iterator
            logging.info(f"Running query iterator...")
            
            iterator = self.client.query_iterator(
                collection_name=self.collection_name,
                filter=expr,
                output_fields=output_fields,
                batch_size=batch_size
            )
            
            # Iterate through all batches
            while True and batch_count < iteration_num:
                batch_start = time.time()
                
                try:
                    # Get next batch
                    batch_result = iterator.next()
                    
                    if not batch_result:
                        # No more data
                        break
                    
                    batch_latency = time.time() - batch_start
                    batch_latencies.append(batch_latency)
                    batch_count += 1
                    total_entities += len(batch_result)

                    # Store results if saving to file
                    if save_in_file:
                        current_batch_results.extend(batch_result)
                        file_batch_count += 1
                        
                        # Save every 100 batches to avoid large files
                        if file_batch_count >= 5:
                            batch_start = batch_count - file_batch_count + 1
                            batch_end = batch_count
                            success, filename, file_size = self._save_batch_results_to_parquet(
                                current_batch_results, iteration_num, batch_start, batch_end
                            )
                            if success:
                                total_files_saved += 1
                            
                            # Reset for next file
                            current_batch_results = []
                            file_batch_count = 0

                    logging.info(f"Batch {batch_count} result: {batch_result[:5]}...")  # Show only first 5 results
                    
                    # Log progress periodically
                    if batch_count % 100 == 0:
                        logging.info(f"  📊 Processed {batch_count} batches, {total_entities} entities...")
                    
                except Exception as e:
                    failures += 1
                    logging.error(f"❌ Batch {batch_count + 1} failed: {e}")
                    break
            
            # Close iterator
            iterator.close()
            
        except Exception as e:
            failures += 1
            logging.error(f"❌ Iterator creation failed: {e}")
            return
        
        # Calculate statistics
        iteration_duration = time.time() - iteration_start
        
        if batch_latencies:
            avg_batch_latency = np.mean(batch_latencies)
            p99_batch_latency = np.percentile(batch_latencies, 99)
            p95_batch_latency = np.percentile(batch_latencies, 95)
            min_batch_latency = np.min(batch_latencies)
            max_batch_latency = np.max(batch_latencies)
            
            # Calculate effective QPS (entities per second)
            entities_per_second = total_entities / max(iteration_duration, 0.001)
            batches_per_second = batch_count / max(iteration_duration, 0.001)
        else:
            avg_batch_latency = p99_batch_latency = p95_batch_latency = 0
            min_batch_latency = max_batch_latency = 0
            entities_per_second = batches_per_second = 0
        
        # Log detailed results
        logging.info("=" * 80)
        logging.info(f"📈 ITERATION {iteration_num} RESULTS:")
        logging.info(f"  ⏱️  Total Duration: {iteration_duration:.3f}s")
        logging.info(f"  📦 Batches Processed: {batch_count}")
        logging.info(f"  📊 Entities Retrieved: {total_entities}")
        logging.info(f"  📋 Expected Total: {batch_size * iteration_num}")
        if failures > 0:
            logging.info(f"  ❌ Failures: {failures}")
        
        logging.info(f"  🚀 Performance Metrics:")
        logging.info(f"    - Entities/Second: {entities_per_second:.2f}")
        logging.info(f"    - Batches/Second: {batches_per_second:.2f}")
        logging.info(f"    - Avg Batch Latency: {avg_batch_latency:.4f}s")
        logging.info(f"    - P95 Batch Latency: {p95_batch_latency:.4f}s")
        logging.info(f"    - P99 Batch Latency: {p99_batch_latency:.4f}s")
        logging.info(f"    - Min Batch Latency: {min_batch_latency:.4f}s")
        logging.info(f"    - Max Batch Latency: {max_batch_latency:.4f}s")
        
        # Save remaining results if any
        if save_in_file and current_batch_results:
            batch_start = batch_count - file_batch_count + 1
            batch_end = batch_count
            success, filename, file_size = self._save_batch_results_to_parquet(
                current_batch_results, iteration_num, batch_start, batch_end
            )
            if success:
                total_files_saved += 1
        
        # Log file saving summary
        if save_in_file:
            logging.info("")
            logging.info(f"📂 File Export Summary:")
            logging.info(f"    - Total files saved: {total_files_saved}")
            logging.info(f"    - Files saved to: /tmp/query_iterator_{self.collection_name}_iter{iteration_num}_*")
        
        logging.info("=" * 80)
    
    def _save_batch_results_to_parquet(self, results, iteration_num, batch_start, batch_end):
        """
        Save batch results to parquet file
        
        :param results: List of result dictionaries to save
        :param iteration_num: Current iteration number
        :param batch_start: Starting batch number for this file
        :param batch_end: Ending batch number for this file
        :return: (success, filename, file_size)
        """
        if not results:
            return False, None, 0
        
        timestamp = int(time.time())
        parquet_filename = f"/tmp/query_iterator_{self.collection_name}_iter{iteration_num}_batch{batch_start}-{batch_end}_{timestamp}.parquet"
        
        try:
            # Convert results to DataFrame
            df = pd.DataFrame(results)
            
            # Save to parquet file
            df.to_parquet(parquet_filename, index=False)
            
            file_size = self._get_file_size(parquet_filename)
            
            logging.info(f"💾 Saved batch results to: {parquet_filename}")
            logging.info(f"    - Batches: {batch_start}-{batch_end} ({batch_end - batch_start + 1} batches)")
            logging.info(f"    - Records: {len(results)}")
            logging.info(f"    - File size: {file_size}")
            logging.info(f"    - Columns: {list(df.columns)}")
            
            return True, parquet_filename, file_size
            
        except Exception as e:
            logging.error(f"❌ Failed to save batch results to parquet: {e}")
            return False, None, 0
    
    def _get_file_size(self, filepath):
        """Get human-readable file size"""
        try:
            import os
            size_bytes = os.path.getsize(filepath)
            
            # Convert to human readable format
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size_bytes < 1024.0:
                    return f"{size_bytes:.2f} {unit}"
                size_bytes /= 1024.0
            return f"{size_bytes:.2f} TB"
        except:
            return "Unknown"


def main():
    """Main query iterator testing function"""
    # Parse command line arguments
    try:
        host = '10.104.14.244'
        collection_name = 'test_aa'
        iter_times = 7
        output_fields = 'embedding_0'
        expr = ''
        batch_size = 2
        save_in_file = True
        api_key = 'none'

        # host = sys.argv[1]
        # collection_name = sys.argv[2]
        # iter_times = int(sys.argv[3])
        # output_fields = str(sys.argv[4]).strip()
        # expr = str(sys.argv[5]).strip()
        # batch_size = int(sys.argv[6])
        # save_in_file = str(sys.argv[7]).upper() == "TRUE"
        # api_key = sys.argv[8]
        
    except (IndexError, ValueError) as e:
        print("Usage: python3 query_iterator.py <host> <collection_name> <iter_times> <output_fields> <expr> <batch_size> <save_in_file> [api_key]")
        print("\nDescription:")
        print("  Single-threaded query iterator performance testing with configurable batch sizes.")
        print("  Optionally save all results to parquet files (100 batches per file) for analysis.")
        print("\nParameters:")
        print("  host            : Milvus server host")
        print("  collection_name : Collection name")
        print("  iter_times      : Number of complete iterations")
        print("  output_fields   : Output fields (comma-separated or '*' for all)")
        print("  expr            : Query expression filter (empty string for no filter)")
        print("  batch_size      : Number of entities per batch")
        print("  save_in_file    : Save results to parquet file (TRUE/FALSE)")
        print("  api_key         : API key (optional, use 'None' for local)")
        print("\nExamples:")
        print("  # Single iteration test with all fields, no file save")
        print("  python3 query_iterator.py localhost my_collection 1 '*' '' 2000 FALSE None")
        print()
        print("  # Multiple iterations with specific fields and save to parquet (100 batches per file)")
        print("  python3 query_iterator.py localhost test_collection 5 'id,vector' 'id > 0' 1000 TRUE None")
        print()
        print("  # Cloud Milvus with API key and smart parquet export")
        print("  python3 query_iterator.py host.com collection_name 3 '*' '' 5000 TRUE api_key")
        sys.exit(1)
    
    port = 19530
    
    # Parameter processing
    if output_fields in ["None", "none", "NONE"] or output_fields == "":
        output_fields = ["*"]
    else:
        output_fields = output_fields.split(",")
    
    if expr in ["None", "none", "NONE"] or expr == "":
        expr = None
    
    # Validate parameters
    if iter_times <= 0:
        print("Error: iter_times must be positive")
        sys.exit(1)
    
    if batch_size <= 0:
        print("Error: batch_size must be positive")
        sys.exit(1)
    
    # Setup logging
    log_filename = f"/tmp/query_iterator_{collection_name}_{int(time.time())}.log"
    file_handler = logging.FileHandler(filename=log_filename)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    
    logging.info("🚀 Starting Query Iterator Testing")
    logging.info(f"  Host: {host}")
    logging.info(f"  Collection: {collection_name}")
    logging.info(f"  Iterations: {iter_times}")
    logging.info(f"  Output Fields: {output_fields}")
    logging.info(f"  Expression: {expr or 'None (all entities)'}")
    logging.info(f"  Batch Size: {batch_size}")
    logging.info(f"  Save to File: {save_in_file}")
    logging.info(f"  API Key: {'***' if api_key and api_key.upper() != 'NONE' else 'None (local)'}")
    
    # Create MilvusClient
    try:
        if api_key is None or api_key == "" or api_key.upper() == "NONE":
            client = MilvusClient(uri=f"http://{host}:{port}")
        else:
            client = MilvusClient(uri=host, token=api_key)
        
        logging.info(f"✅ Connected to MilvusClient at {host}")
        
    except Exception as e:
        logging.error(f"Failed to create MilvusClient: {e}")
        sys.exit(1)
    
    # Create query iterator tester
    tester = QueryIteratorTester(client, collection_name)
    
    # Validate collection (fail fast if invalid)
    is_valid, error_message = tester.validate_collection()
    if not is_valid:
        logging.error(f"❌ Collection validation failed: {error_message}")
        sys.exit(1)
    
    logging.info(f"✅ Collection '{collection_name}' validation passed")
    
    
    # Run query iterator test
    try:
        tester.run_single_iteration(
            batch_size=batch_size,
            output_fields=output_fields,
            expr=expr,
            iteration_num=iter_times,
            save_in_file=save_in_file
        )
        
        logging.info("🎯 Query Iterator Testing Completed Successfully")
        logging.info(f"📁 Log file: {log_filename}")
        
    except Exception as e:
        logging.error(f"❌ Query iterator testing failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
