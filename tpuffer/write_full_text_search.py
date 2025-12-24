#!/usr/bin/env python3
"""
Turbopuffer Full Text Search Write Performance Test Script (Serial)

Features:
- Full Text Search (FTS) writes with BM25 indexing
- Serial writes (single process, sequential execution)
- Row-based writes (upsert_rows) - standard turbopuffer API
- Pandas for parquet reading - faster than pyarrow
- Pre-assembled upsert_rows format - no repeated conversion
- Memory-efficient: can preload multiple files
- Writes: id, content (text), title, paragraph_id, vector (emb) columns
- MANDATORY PRE-CHECK: Validates data availability before starting writes
- Automatic data scanning: writes ALL data in file range

Data Validation:
- Pre-scans ALL parquet files to count total available rows
- Automatically calculates total writes needed
- FAILS IMMEDIATELY if no data found (prevents wasted writes)

Schema:
- content: string field with full_text_search enabled (BM25)
- title: string field
- paragraph_id: integer field
- vector: float array field (from parquet 'emb' column)

Usage:
    python write_full_text_search.py --parquet-dir data/ \
                    --file-id-start 0 \
                    --file-id-end 10 \
                    --user-id 0 \
                    --batch-size 10000 \
                    --preload-files 3
    
    # Files expected: parquet-train-00000-of-00252.parquet to parquet-train-00010-of-00252.parquet
"""

import argparse
import os
import sys
import time
import logging
import gc
from typing import List, Dict, Any
import pandas as pd
import numpy as np

try:
    import turbopuffer
except ImportError:
    print("Error: turbopuffer package not found. Install with: pip install 'turbopuffer[fast]'")
    sys.exit(1)

# Constants
ROWS_PER_FILE_ESTIMATE = 130000  # Estimated average rows per file (actual varies)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class WriteStatistics:
    """Statistics collector for write performance metrics"""
    
    def __init__(self):
        self.latencies: List[float] = []
        self.batch_sizes: List[int] = []
        self.timestamps: List[float] = []
        self.total_writes = 0
        self.total_failures = 0
        self.total_rows = 0
        self.total_bytes = 0  # Total data size in bytes
        self.start_time = time.time()
        self.end_time = None
    
    def record_success(self, latency: float, batch_size: int, batch_bytes: int = 0):
        """Record a successful write
        
        Args:
            latency: Write latency in seconds
            batch_size: Number of rows in the batch
            batch_bytes: Actual data size of the batch in bytes (optional)
        """
        self.latencies.append(latency)
        self.batch_sizes.append(batch_size)
        self.timestamps.append(time.time())
        self.total_rows += batch_size
        if batch_bytes > 0:
            self.total_bytes += batch_bytes
    
    def record_failure(self):
        """Record a failed write"""
        self.total_failures += 1
    
    def increment_writes(self):
        """Increment total write count"""
        self.total_writes += 1
    
    def print_stats(self):
        """Print comprehensive final statistics"""
        self.end_time = time.time()
        total_duration = self.end_time - self.start_time
        
        # Read all stats
        total_writes = self.total_writes
        total_failures = self.total_failures
        total_rows = self.total_rows
        total_bytes = self.total_bytes
        latencies = self.latencies.copy()
        
        if total_writes == 0:
            logger.info("No writes executed.")
            return
        
        if not latencies:
            logger.info("No latency data recorded.")
            return
        
        # Calculate success rate
        success_rate = (total_writes - total_failures) / total_writes * 100
        
        # Sort latencies for percentile calculations
        sorted_latencies = sorted(latencies)
        
        # Calculate statistics
        avg_latency = np.mean(sorted_latencies)
        min_latency = np.min(sorted_latencies)
        max_latency = np.max(sorted_latencies)
        p50_latency = np.percentile(sorted_latencies, 50)
        p95_latency = np.percentile(sorted_latencies, 95)
        p99_latency = np.percentile(sorted_latencies, 99)
        
        # Calculate QPS and Throughput
        qps = total_writes / total_duration if total_duration > 0 else 0
        throughput = total_rows / total_duration if total_duration > 0 else 0  # rows per second
        
        # Print results
        logger.info("✅ Write test completed!")
        logger.info(f"📊 Final Results:")
        logger.info(f"   Total Writes: {total_writes}")
        logger.info(f"   Total Rows: {total_rows:,}")
        logger.info(f"   Total Failures: {total_failures}")
        logger.info(f"   Average Latency: {avg_latency:.4f}s")
        logger.info(f"   Min Latency: {min_latency:.4f}s")
        logger.info(f"   Max Latency: {max_latency:.4f}s")
        logger.info(f"   P50 Latency: {p50_latency:.4f}s")
        logger.info(f"   P95 Latency: {p95_latency:.4f}s")
        logger.info(f"   P99 Latency: {p99_latency:.4f}s")
        logger.info(f"   Success Rate: {success_rate:.2f}%")
        logger.info(f"   Test Duration: {total_duration:.2f}s")
        
        # Calculate and print throughput if data size available
        if total_bytes > 0 and total_duration > 0:
            throughput_mbps = (total_bytes / total_duration) / (1024 * 1024)
            logger.info(f"   Throughput: {throughput:.2f} rows/s, {throughput_mbps:.2f} MB/s")
        elif total_duration > 0:
            throughput_mbps_est = throughput * 0.5 / 1024
            logger.info(f"   Throughput: {throughput:.2f} rows/s, {throughput_mbps_est:.2f} MB/s (estimated)")


class DataPool:
    """Manages text data loading from parquet files (can preload multiple files)"""
    
    def __init__(self, parquet_dir: str, file_id_start: int, file_id_end: int, 
                 file_pattern: str = "parquet-train-{:05d}-of-00252.parquet",
                 preload_count: int = 3):
        self.parquet_dir = parquet_dir
        self.file_id_start = file_id_start
        self.file_id_end = file_id_end
        self.current_file_id = file_id_start
        self.file_pattern = file_pattern
        self.preload_count = preload_count
        
        # Store all loaded records
        self.all_records: List[dict] = []
        self.next_index = 0
        self.total_rows_loaded = 0
        self.files_loaded = 0
        
        logger.info(f"📖 Initializing DataPool from directory: {parquet_dir}")
        logger.info(f"   File ID range: {file_id_start} to {file_id_end}")
        logger.info(f"   💾 Preload strategy: {preload_count} files at a time")
        
        # Preload initial files (only if preload_count > 0)
        if preload_count > 0:
            self.preload_files()
    
    def preload_files(self):
        """Preload multiple files into memory"""
        files_to_load = min(self.preload_count, 
                           self.file_id_end - self.file_id_start + 1 - self.files_loaded)
        current_files_loaded = self.files_loaded
        
        if files_to_load <= 0:
            return
        
        for i in range(files_to_load):
            file_id = self.file_id_start + current_files_loaded + i
            if file_id > self.file_id_end:
                break
            
            self.load_file(file_id)
    
    def load_file(self, file_id: int):
        """Load a specific parquet file"""
        file_name = self.file_pattern.format(file_id)
        file_path = os.path.join(self.parquet_dir, file_name)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Parquet file not found: {file_path}")
        
        logger.info(f"📖 Loading parquet file: {file_name}")
        
        # Get file size
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        
        df = pd.read_parquet(file_path)
        
        # Check required columns
        required_columns = ['id', 'text', 'title', 'paragraph_id', 'emb']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in parquet file: {missing_columns}. "
                           f"Available columns: {df.columns.tolist()}")
        
        # Extract columns
        ids = df['id'].tolist()
        texts = df['text'].tolist()
        titles = df['title'].tolist()
        paragraph_ids = df['paragraph_id'].tolist()
        embs = df['emb'].tolist()  # Extract embedding vectors
        
        # Convert to WriteRecord format
        new_records = []
        for row_id, text, title, paragraph_id, emb in zip(ids, texts, titles, paragraph_ids, embs):
            # Convert emb to list if it's numpy array or other types
            if isinstance(emb, np.ndarray):
                vector = emb.tolist()
            elif isinstance(emb, (list, tuple)):
                vector = list(emb)
            else:
                # Try to convert to list
                vector = list(emb) if hasattr(emb, '__iter__') else [emb]
            
            new_records.append({
                'id': str(row_id),  # Ensure ID is string
                'content': text,
                'title': title,
                'paragraph_id': int(paragraph_id),
                'vector': vector  # Add vector field from emb column
            })
        
        # Append to all_records
        self.all_records.extend(new_records)
        self.total_rows_loaded += len(new_records)
        self.files_loaded += 1
        
        logger.info(f"✅ Loaded {len(new_records):,} records")
        
        # Clean up
        del df, ids, texts, titles, paragraph_ids, embs, new_records
        gc.collect()
    
    def preload_all_files_and_count(self) -> int:
        """
        Pre-scan all parquet files to count total available rows.
        Does NOT load the actual data, only counts rows for validation.
        Returns total row count across all files.
        """
        total_rows = 0
        files_scanned = 0
        
        for file_id in range(self.file_id_start, self.file_id_end + 1):
            file_name = self.file_pattern.format(file_id)
            file_path = os.path.join(self.parquet_dir, file_name)
            
            if not os.path.exists(file_path):
                logger.warning(f"⚠️  File not found: {file_name} (skipping)")
                continue
            
            try:
                # Read only the parquet metadata, not the actual data
                df = pd.read_parquet(file_path, columns=['id'])  # Only read one column for speed
                row_count = len(df)
                total_rows += row_count
                files_scanned += 1
                
                del df
                gc.collect()
                
            except Exception as e:
                logger.error(f"❌ Failed to read {file_name}: {e}")
                raise
        
        return total_rows
    
    def get_batch(self, batch_size: int) -> List[dict]:
        """Returns a batch of records"""
        total_records = len(self.all_records)
        current_index = self.next_index
        
        # Check if we have enough data
        if current_index >= total_records:
            # Check if we can load more files
            if self.files_loaded < (self.file_id_end - self.file_id_start + 1):
                # Need to load more files
                self.preload_files()
                total_records = len(self.all_records)
            
            # Check again after loading
            if current_index >= total_records:
                # No more data available
                return []
        
        # Get batch from available data
        start_idx = current_index
        end_idx = min(start_idx + batch_size, total_records)
        
        if start_idx >= total_records:
            return []
        
        # Copy batch (this handles the remainder case correctly - if end_idx < start_idx + batch_size,
        # we get a smaller batch which is the correct behavior for the last batch)
        batch = self.all_records[start_idx:end_idx].copy()
        
        # Clean up consumed records to free memory
        # Keep recent records, but clean up old ones
        MAX_RECORDS_IN_MEMORY = 200000  # Keep at most 200K records in memory
        actual_deleted = 0
        
        if total_records > MAX_RECORDS_IN_MEMORY and end_idx > 50000:
            # Only clean up if we have consumed at least 50K records
            keep_count = MAX_RECORDS_IN_MEMORY
            # Calculate how many records to remove from the beginning
            # We want to keep the last keep_count records
            # Remove records from index 0 to (end_idx - keep_count), but only if end_idx > keep_count
            if end_idx > keep_count:
                records_to_remove = end_idx - keep_count
                
                if records_to_remove > 10000:  # Only clean up if significant amount
                    try:
                        old_count = len(self.all_records)
                        # Remove old records from the beginning
                        for _ in range(records_to_remove):
                            if len(self.all_records) > keep_count:
                                self.all_records.pop(0)
                            else:
                                break
                        
                        actual_deleted = old_count - len(self.all_records)
                    except Exception as e:
                        logger.warning(f"⚠️  Failed to clean up memory: {e} (continuing without cleanup)")
        
        # Update next_index AFTER cleanup, adjusting for deleted records
        # Since we deleted records from the beginning, we need to adjust the index
        if actual_deleted > 0:
            # next_index should be relative to the new list length
            # If we deleted records before end_idx, adjust accordingly
            # After deletion, the records that were at end_idx are now at (end_idx - actual_deleted)
            self.next_index = end_idx - actual_deleted
            # Ensure next_index doesn't exceed the new list length
            self.next_index = min(self.next_index, len(self.all_records))
        else:
            self.next_index = end_idx
        
        return batch


def calculate_batch_size_bytes(batch: List[dict]) -> int:
    """Calculate actual size of batch in bytes (estimated)
    
    This is an estimation based on the data structure size.
    Actual network payload may vary due to serialization overhead.
    """
    total_size = 0
    for record in batch:
        # Estimate size of each field
        # Note: sys.getsizeof() includes Python object overhead,
        # so this is an approximation
        total_size += len(str(record.get('id', '')))
        total_size += len(str(record.get('content', '')))
        total_size += len(str(record.get('title', '')))
        total_size += 8  # paragraph_id (integer, ~8 bytes)
        
        # Calculate vector size (float array)
        vector = record.get('vector', [])
        if isinstance(vector, (list, tuple)):
            # Each float is typically 4 bytes (float32) or 8 bytes (float64)
            # We'll use 4 bytes as default (float32 is more common for embeddings)
            total_size += len(vector) * 4
        elif isinstance(vector, np.ndarray):
            total_size += vector.nbytes
        # Add overhead for JSON serialization (roughly 20% overhead)
    return int(total_size * 1.2)  # Add 20% for JSON/HTTP overhead


def perform_write(tpuf: turbopuffer.Turbopuffer, namespace: str, 
                  upsert_rows: List[dict]) -> float:
    """Write batch using row-based API with FTS schema, returns write latency in seconds
    
    This measures the complete write API call latency, including:
    - Data serialization (SDK internal)
    - Network round-trip time (request + response)
    - Server processing time
    
    Returns:
        float: Write latency in seconds
    """
    if not upsert_rows:
        raise ValueError("Empty batch")
    
    # Define schema for Full Text Search and Vector
    schema = {
        'content': {
            'type': 'string',
            'full_text_search': True  # Enable BM25 with default settings
        }
    }
    
    # Measure only the write API call latency
    # This includes serialization, network, and server processing
    ns = tpuf.namespace(namespace)
    start_time = time.time()
    ns.write(
        upsert_rows=upsert_rows, 
        distance_metric="cosine_distance", 
        schema=schema)
    latency = time.time() - start_time
    
    return latency


def main():
    parser = argparse.ArgumentParser(description='Turbopuffer Full Text Search Write Performance Test (Concurrent)')
    parser.add_argument('--parquet-dir', type=str, default='data/wikipedia/', 
                       help='Directory containing text parquet files')
    parser.add_argument('--file-id-start', type=int, default=0,
                       help='Start file ID (e.g., 0 for parquet-train-00000-of-00252.parquet)')
    parser.add_argument('--file-id-end', type=int, default=0,
                       help='End file ID (e.g., 10 for parquet-train-00010-of-00252.parquet)')
    parser.add_argument('--file-pattern', type=str, default='parquet-train-{:05d}-of-00252.parquet',
                       help='File pattern with {:05d} placeholder for file ID')
    parser.add_argument('--key', type=str, default='',
                       help='Turbopuffer API key (or set TURBOPUFFER_API_KEY env var)')
    parser.add_argument('--region', type=str, default='aws-us-west-2',
                       help='Turbopuffer region (e.g., aws-us-west-2, gcp-us-central1)')
    parser.add_argument('--batch-size', type=int, default=10000,
                       help='Number of rows per write batch')
    parser.add_argument('--user-id', type=int, default=0,
                       help='User ID for namespace (e.g., 0 for namespace fts_id_0)')
    parser.add_argument('--preload-files', type=int, default=1,
                       help='Number of parquet files to preload in memory')
    
    args = parser.parse_args()
    
    # Validate parameters
    if not args.parquet_dir:
        logger.error("Error: --parquet-dir is required")
        sys.exit(1)
    
    if args.file_id_start < 0 or args.file_id_end < 0:
        logger.error("Error: file IDs must be non-negative")
        sys.exit(1)
    
    if args.file_id_start > args.file_id_end:
        logger.error("Error: file-id-start must be <= file-id-end")
        sys.exit(1)
    
    # Get API key
    api_key = args.key or os.getenv("TURBOPUFFER_API_KEY")
    if not api_key:
        logger.error("Error: --key or TURBOPUFFER_API_KEY environment variable is required")
        sys.exit(1)
    
    if args.batch_size <= 0:
        logger.error("Error: batch-size must be positive")
        sys.exit(1)
    
    if args.preload_files <= 0:
        logger.error("Error: preload-files must be positive")
        sys.exit(1)
    
    # Create namespace
    namespace = f"fts_id_{args.user_id}"
    
    # Calculate file count
    num_files = args.file_id_end - args.file_id_start + 1
    
    # Create temporary data pool for counting only
    temp_data_pool = DataPool(
        args.parquet_dir, 
        args.file_id_start, 
        args.file_id_end, 
        args.file_pattern,
        preload_count=0  # Don't load any files, just count
    )
    
    # Calculate available data from parquet files
    actual_total_rows = temp_data_pool.preload_all_files_and_count()
    if actual_total_rows == 0:
        logger.error("❌ No data found in the specified file range")
        sys.exit(1)
    
    # Calculate total writes needed (write ALL available data)
    total_writes = (actual_total_rows + args.batch_size - 1) // args.batch_size  # Round up
    
    logger.info("📊 Data Requirements:")
    logger.info(f"   Namespace: {namespace}")
    logger.info(f"   Batch size: {args.batch_size} rows")
    logger.info(f"   Total writes needed: {total_writes}")
    logger.info(f"   Total rows needed: {actual_total_rows:,}")
    logger.info("")
    logger.info("📁 Data Availability:")
    logger.info(f"   Parquet files: {num_files} ({args.file_pattern.format(args.file_id_start)} "
               f"to {args.file_pattern.format(args.file_id_end)})")
    logger.info(f"   Total rows available: {actual_total_rows:,}")
    logger.info("")
    logger.info(f"✅ Data validation passed: {actual_total_rows:,} rows available")
    logger.info("")
    
    
    # Initialize data pool with preloading
    data_pool = DataPool(
        args.parquet_dir, 
        args.file_id_start, 
        args.file_id_end, 
        args.file_pattern,
        preload_count=args.preload_files
    )
    
    # Initialize statistics
    stats = WriteStatistics()
    
    # Load data from parquet files
    try:
        data_pool = DataPool(
            args.parquet_dir, 
            args.file_id_start, 
            args.file_id_end, 
            args.file_pattern,
            preload_count=args.preload_files
        )
    except Exception as e:
        logger.error(f"Failed to initialize data pool: {e}")
        sys.exit(1)
    
    # Create Turbopuffer client
    tpuf = turbopuffer.Turbopuffer(api_key=api_key, region=args.region)
    
    logger.info(f"Region: {args.region}")
    logger.info(f"💡 Mode: Serial streaming, row-based writes")
    logger.info("")
    
    # Initialize statistics
    stats = WriteStatistics()
    
    # Streaming write loop
    logger.info("🚀 Starting streaming writes...")
    logger.info("")
    
    write_count = 0
    
    while write_count < total_writes:
        # Get batch data
        batch = data_pool.get_batch(args.batch_size)
        if not batch:
            logger.warning(f"⚠️ Ran out of data at write {write_count+1}/{total_writes}")
            break
        
        write_count += 1
        
        # Log if this is a partial batch (remainder rows)
        if len(batch) < args.batch_size:
            logger.info(f"📝 Processing final batch: {len(batch)} rows (remainder from {args.batch_size} batch size)")
        
        if write_count % 10 == 0 or write_count == total_writes:
            logger.info(f"📝 Progress: {write_count}/{total_writes} ({write_count/total_writes*100:.1f}%)")
        
        try:
            # Calculate actual batch size in bytes for accurate throughput calculation
            batch_bytes = calculate_batch_size_bytes(batch)
            
            # Perform write and measure latency
            latency = perform_write(tpuf, namespace, batch)
            
            # Record success with actual data size
            stats.record_success(latency, len(batch), batch_bytes)
            stats.increment_writes()
        except Exception as e:
            stats.record_failure()
            stats.increment_writes()
            logger.error(f"Write error: {e}")
    
    # Final verification
    logger.info("\n🔍 Final Verification:")
    final_rows = stats.total_rows
    
    if final_rows == actual_total_rows:
        logger.info(f"✅ All {actual_total_rows:,} rows written successfully")
    elif final_rows < actual_total_rows:
        missing_rows = actual_total_rows - final_rows
        logger.error(f"❌ DATA INTEGRITY ERROR: Only {final_rows:,} of {actual_total_rows:,} rows were written")
        logger.error(f"   Missing {missing_rows:,} rows ({missing_rows/actual_total_rows*100:.2f}%)")
        sys.exit(1)
    else:
        logger.warning(f"⚠️  Unexpected: More rows written ({final_rows:,}) than available ({actual_total_rows:,})")
    
    # Print final statistics
    stats.print_stats()


if __name__ == '__main__':
    main()
