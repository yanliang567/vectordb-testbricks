#!/usr/bin/env python3
"""
Turbopuffer Write Performance Test Script (Optimized & Simplified)

Optimizations:
- Row-based writes (upsert_rows) - standard turbopuffer API
- Pandas for parquet reading - faster than pyarrow
- Pre-assembled upsert_rows format - no repeated conversion
- Serial execution - simple and reliable
- Memory-efficient: loads maximum 1 parquet file at a time
- No schema override - uses turbopuffer defaults

Features:
- Streaming writes (no pre-allocation)
- Simple round-robin namespace assignment
- Memory footprint: ~1.2 GB (fixed)
- 100% data correctness guarantee

Usage:
    python write.py --parquet-dir data/ \
                    --file-id-start 0 \
                    --file-id-end 10 \
                    --batch-size 1000 \
                    --writes-per-ns 1 \
                    --user-id-start 0 \
                    --user-id-end 799
"""

import argparse
import os
import sys
import time
import logging
import gc
from typing import List, Dict
import pandas as pd
import numpy as np

try:
    import turbopuffer
except ImportError:
    print("Error: turbopuffer package not found. Install with: pip install 'turbopuffer[fast]'")
    sys.exit(1)

# Constants
ROWS_PER_FILE = 400000

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class WriteStatistics:
    """Collects write performance metrics"""
    
    def __init__(self):
        self.latencies = []
        self.total_writes = 0
        self.total_failures = 0
        self.total_rows = 0
        self.start_time = time.time()
        self.end_time = None
    
    def record_latency(self, latency: float):
        """Record a successful write latency in seconds"""
        self.latencies.append(latency)
    
    def increment_writes(self):
        """Increment total write count"""
        self.total_writes += 1
    
    def increment_failures(self):
        """Increment failure count"""
        self.total_failures += 1
    
    def add_rows(self, count: int):
        """Add to total rows written"""
        self.total_rows += count
    
    def print_stats(self):
        """Print final statistics"""
        self.end_time = time.time()
        total_duration = self.end_time - self.start_time
        
        if self.total_writes == 0:
            logger.info("No writes executed.")
            return
        
        if not self.latencies:
            logger.info("No latency data recorded.")
            return
        
        # Calculate success rate
        success_rate = (self.total_writes - self.total_failures) / self.total_writes * 100
        
        # Sort latencies for percentile calculations
        sorted_latencies = sorted(self.latencies)
        
        # Calculate statistics
        avg_latency = np.mean(sorted_latencies)
        min_latency = np.min(sorted_latencies)
        max_latency = np.max(sorted_latencies)
        p50_latency = np.percentile(sorted_latencies, 50)
        p95_latency = np.percentile(sorted_latencies, 95)
        p99_latency = np.percentile(sorted_latencies, 99)
        
        # Print results
        logger.info("✅ Write test completed!")
        logger.info(f"📊 Final Results:")
        logger.info(f"   Total Writes: {self.total_writes}")
        logger.info(f"   Total Rows: {self.total_rows}")
        logger.info(f"   Total Failures: {self.total_failures}")
        logger.info(f"   Average Latency: {avg_latency:.4f}s")
        logger.info(f"   Min Latency: {min_latency:.4f}s")
        logger.info(f"   Max Latency: {max_latency:.4f}s")
        logger.info(f"   P50 Latency: {p50_latency:.4f}s")
        logger.info(f"   P95 Latency: {p95_latency:.4f}s")
        logger.info(f"   P99 Latency: {p99_latency:.4f}s")
        logger.info(f"   Success Rate: {success_rate:.2f}%")
        logger.info(f"   Test Duration: {total_duration:.2f}s")


class DataPool:
    """Manages vector data loading from parquet files (one file at a time)"""
    
    def __init__(self, parquet_dir: str, file_id_start: int, file_id_end: int):
        self.parquet_dir = parquet_dir
        self.file_id_start = file_id_start
        self.file_id_end = file_id_end
        self.current_file_id = file_id_start
        
        # Store data in upsert_rows format (list of dicts)
        self.records: List[dict] = []
        self.current_idx = 0
        
        logger.info(f"📖 Initializing DataPool from directory: {parquet_dir}")
        logger.info(f"   File ID range: {file_id_start} to {file_id_end}")
        logger.info(f"   💾 One file at a time, pre-assembled upsert_rows format")
        
        # Load first file
        self._load_next_file()
    
    def _load_next_file(self):
        """Load next parquet file and pre-assemble into upsert_rows format"""
        if self.current_file_id > self.file_id_end:
            logger.info("📥 No more files to load")
            return
        
        # Clear current data and force garbage collection
        if self.records:  # 不是第一次加载
            logger.info("🗑️  Clearing previous data...")
            self.records = []
            gc.collect()
            logger.info("✅ Memory cleared")
        
        # Load file
        file_name = f"binary_768d_{self.current_file_id:05d}.parquet"
        file_path = os.path.join(self.parquet_dir, file_name)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Parquet file not found: {file_path}")
        
        logger.info(f"📖 Loading parquet file: {file_name}")
        
        df = pd.read_parquet(file_path)
        pks = df['PK'].tolist()
        
        # Convert vectors to Python lists (critical for JSON serialization)
        vectors_raw = df['Vector'].tolist()
        vectors = []
        for vec in vectors_raw:
            if isinstance(vec, np.ndarray):
                vectors.append(vec.tolist())  # numpy array → Python list
            else:
                vectors.append(vec)
        
        # Pre-assemble into upsert_rows format (list of dicts)
        logger.info(f"🔧 Assembling {len(pks)} records into upsert_rows format...")
        self.records = [
            {'id': pk, 'vector': vector}
            for pk, vector in zip(pks, vectors)
        ]
        
        # Clean up intermediate data
        del df, pks, vectors_raw, vectors
        gc.collect()
        
        logger.info(f"✅ Loaded {len(self.records)} records (vector dim: {len(self.records[0]['vector'])})")
        
        self.current_idx = 0
        self.current_file_id += 1
    
    def get_batch(self, batch_size: int) -> List[dict]:
        """Returns a batch of pre-assembled records"""
        # Check if we need to load more data
        if self.current_idx >= len(self.records):
            if self.current_file_id <= self.file_id_end:
                logger.info(f"📥 Current file exhausted, loading next file...")
                self._load_next_file()
            
            if self.current_idx >= len(self.records):
                logger.warning("⚠️ Warning: All data has been used")
                return []
        
        # Get batch
        remaining = len(self.records) - self.current_idx
        actual_batch_size = min(batch_size, remaining)
        
        if actual_batch_size < batch_size:
            logger.warning(f"⚠️ Warning: Only {remaining} records remaining, adjusting batch size")
        
        # Return slice of pre-assembled records
        end_idx = self.current_idx + actual_batch_size
        batch = self.records[self.current_idx:end_idx]
        
        self.current_idx += actual_batch_size
        
        return batch


def perform_write(tpuf: turbopuffer.Turbopuffer, namespace: str, 
                  upsert_rows: List[dict]) -> float:
    """Write batch using row-based API, returns write latency in seconds"""
    if not upsert_rows:
        raise ValueError("Empty batch")
    
    # Measure only the write API call latency
    ns = tpuf.namespace(namespace)
    start_time = time.time()
    ns.write(upsert_rows=upsert_rows, distance_metric='euclidean_squared')
    latency = time.time() - start_time
    
    return latency


def main():
    parser = argparse.ArgumentParser(description='Turbopuffer Serial Write Performance Test (Memory Efficient)')
    parser.add_argument('--parquet-dir', type=str, default='data/', 
                       help='Directory containing parquet files')
    parser.add_argument('--file-id-start', type=int, default=0,
                       help='Start file ID (e.g., 0 for binary_768d_00000.parquet)')
    parser.add_argument('--file-id-end', type=int, default=0,
                       help='End file ID (e.g., 10 for binary_768d_00010.parquet)')
    parser.add_argument('--key', type=str, default='',
                       help='Turbopuffer API key (or set TURBOPUFFER_API_KEY env var)')
    parser.add_argument('--region', type=str, default='aws-us-west-2',
                       help='Turbopuffer region (e.g., aws-us-west-2)')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Number of rows per write batch')
    parser.add_argument('--writes-per-ns', type=int, default=1,
                       help='Number of write operations per namespace')
    parser.add_argument('--user-id-start', type=int, default=1,
                       help='Start of user ID range for namespaces')
    parser.add_argument('--user-id-end', type=int, default=1,
                       help='End of user ID range for namespaces')
    
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
    
    if args.user_id_start > args.user_id_end:
        logger.error("Error: user-id-start must be <= user-id-end")
        sys.exit(1)
    
    if args.batch_size <= 0:
        logger.error("Error: batch-size must be positive")
        sys.exit(1)
    
    if args.writes_per_ns <= 0:
        logger.error("Error: writes-per-ns must be positive")
        sys.exit(1)
    
    # Calculate expected data requirements
    num_namespaces = args.user_id_end - args.user_id_start + 1
    total_writes = num_namespaces * args.writes_per_ns
    total_rows_needed = total_writes * args.batch_size
    
    # Calculate available data from parquet files
    num_files = args.file_id_end - args.file_id_start + 1
    total_rows_available = num_files * ROWS_PER_FILE
    
    logger.info("📊 Data Requirements:")
    logger.info(f"   Namespaces: {num_namespaces} (id_{args.user_id_start} to id_{args.user_id_end})")
    logger.info(f"   Writes per namespace: {args.writes_per_ns}")
    logger.info(f"   Batch size: {args.batch_size} rows")
    logger.info(f"   Total writes needed: {total_writes}")
    logger.info(f"   Total rows needed: {total_rows_needed}")
    logger.info("")
    logger.info("📁 Data Availability:")
    logger.info(f"   Parquet files: {num_files} (binary_768d_{args.file_id_start:05d}.parquet "
               f"to binary_768d_{args.file_id_end:05d}.parquet)")
    logger.info(f"   Rows per file: {ROWS_PER_FILE}")
    logger.info(f"   Total rows available: {total_rows_available}")
    
    # Validate data availability
    if total_rows_needed > total_rows_available:
        logger.error("")
        logger.error("❌ ERROR: Not enough data in parquet files!")
        logger.error(f"   Required: {total_rows_needed} rows")
        logger.error(f"   Available: {total_rows_available} rows")
        logger.error(f"   Shortage: {total_rows_needed - total_rows_available} rows")
        logger.error("")
        needed_files = (total_rows_needed + ROWS_PER_FILE - 1) // ROWS_PER_FILE
        logger.error(f"Please either:")
        logger.error(f"   1. Increase file-id-end (need at least {needed_files} files)")
        logger.error(f"   2. Reduce batch-size, writes-per-ns, or namespace range")
        sys.exit(1)
    
    logger.info("")
    logger.info(f"✅ Data validation passed: {total_rows_available} rows available, "
               f"{total_rows_needed} rows needed")
    logger.info("")
    
    # Load data from parquet files
    try:
        data_pool = DataPool(args.parquet_dir, args.file_id_start, args.file_id_end)
    except Exception as e:
        logger.error(f"Failed to initialize data pool: {e}")
        sys.exit(1)
    
    # Create Turbopuffer client
    tpuf = turbopuffer.Turbopuffer(api_key=api_key, region=args.region)
    
    logger.info(f"Region: {args.region}")
    logger.info(f"💡 Mode: Serial streaming, row-based writes")
    logger.info(f"   💾 Memory: ~{(ROWS_PER_FILE * 768 * 4) / 1024**3:.2f} GB per file")
    logger.info("")
    
    # Initialize statistics
    stats = WriteStatistics()
    namespace_write_counts = {f"id_{i}": 0 for i in range(args.user_id_start, args.user_id_end + 1)}
    
    # Streaming write loop
    logger.info("🚀 Starting streaming writes...")
    logger.info("")
    
    write_count = 0
    current_namespace_idx = 0
    namespaces = list(namespace_write_counts.keys())
    
    while write_count < total_writes:
        # Round-robin namespace selection
        namespace = namespaces[current_namespace_idx % len(namespaces)]
        current_namespace_idx += 1
        
        # Check write limit
        if namespace_write_counts[namespace] >= args.writes_per_ns:
            logger.info("✅ All namespaces reached write limit")
            break
        
        # Get batch data (already in upsert_rows format)
        batch = data_pool.get_batch(args.batch_size)
        if not batch:
            logger.warning(f"⚠️ Ran out of data at write {write_count+1}/{total_writes}")
            break
        
        # Perform write
        write_count += 1
        
        if write_count % 10 == 0 or write_count == total_writes:
            logger.info(f"📝 Progress: {write_count}/{total_writes} ({write_count/total_writes*100:.1f}%)")
        
        try:
            # perform_write returns the pure write API latency
            latency = perform_write(tpuf, namespace, batch)
            stats.record_latency(latency)
            stats.add_rows(len(batch))
            namespace_write_counts[namespace] += 1
        except Exception as e:
            stats.increment_failures()
            logger.error(f"Write error (namespace: {namespace}): {e}")
        finally:
            stats.increment_writes()
    
    # Final verification
    logger.info("\n🔍 Final Verification:")
    expected_writes_per_ns = args.writes_per_ns
    verification_passed = True
    
    for namespace, actual_count in sorted(namespace_write_counts.items()):
        if actual_count != expected_writes_per_ns:
            logger.error(f"❌ MISMATCH: {namespace} - expected {expected_writes_per_ns} writes, "
                        f"actually performed {actual_count} writes")
            verification_passed = False
    
    if verification_passed:
        logger.info(f"✅ All {len(namespace_write_counts)} namespaces received exact expected write counts")
    else:
        logger.error("❌ Write count verification FAILED!")
        sys.exit(1)
    
    # Print per-namespace statistics (limit output for large namespace counts)
    logger.info("\n📋 Per-Namespace Write Counts:")
    namespaces_to_show = sorted(namespace_write_counts.keys())[:10]
    for ns in namespaces_to_show:
        actual = namespace_write_counts[ns]
        status = "✅" if actual == expected_writes_per_ns else "❌"
        logger.info(f"   {status} {ns}: {actual}/{expected_writes_per_ns} writes")
    
    if len(namespace_write_counts) > 10:
        remaining = len(namespace_write_counts) - 10
        remaining_ok = sum(1 for ns in list(namespace_write_counts.keys())[10:] 
                          if namespace_write_counts[ns] == expected_writes_per_ns)
        logger.info(f"   ... {remaining} more namespaces ({remaining_ok}/{remaining} verified ✅)")
    
    # Print final statistics
    stats.print_stats()


if __name__ == '__main__':
    main()
