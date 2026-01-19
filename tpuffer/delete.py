#!/usr/bin/env python3
"""
Turbopuffer Delete Performance Test Script

Tests the delete_by_filter API performance with:
- User-specified namespace (user_id)
- Random ID selection from specified range for each delete
- Configurable number of delete operations
- Row count tracking before and after each delete
- Comprehensive performance metrics: P99, P95, P50, Avg, Min, Max latency

API Tested:
    ns = tpuf.namespace('id_0')
    result = ns.write(
        delete_by_filter=("id", "Eq", random_id)
    )
    print(result.rows_affected)

Usage:
    # Basic usage
    python delete.py --key YOUR_API_KEY \
                     --user-id 0 \
                     --id-start 1 \
                     --id-end 1000 \
                     --delete-count 100
    
    # Using environment variable
    export TURBOPUFFER_API_KEY=your_api_key
    python delete.py --user-id 0 \
                     --id-start 1 \
                     --id-end 1000 \
                     --delete-count 50
"""

import argparse
import os
import sys
import time
import logging
import random
from typing import List, Dict, Any
import numpy as np

try:
    import turbopuffer
except ImportError:
    print("Error: turbopuffer package not found. Install with: pip install 'turbopuffer[fast]'")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class DeleteStatistics:
    """Collects delete performance metrics"""
    
    def __init__(self):
        self.latencies: List[float] = []
        self.first_10_latencies: List[float] = []
        self.total_deletes = 0
        self.total_failures = 0
        self.total_rows_deleted = 0
        self.start_time = time.time()
        self.end_time = None
    
    def record_latency(self, latency: float, delete_index: int):
        """Record a successful delete latency in seconds"""
        self.latencies.append(latency)
        if delete_index <= 10:
            self.first_10_latencies.append(latency)
    
    def increment_deletes(self):
        """Increment total delete count"""
        self.total_deletes += 1
    
    def increment_failures(self):
        """Increment failure count"""
        self.total_failures += 1
    
    def add_rows_deleted(self, count: int):
        """Add to total rows deleted"""
        self.total_rows_deleted += count
    
    def print_stats(self):
        """Print final statistics"""
        self.end_time = time.time()
        total_duration = self.end_time - self.start_time
        
        if self.total_deletes == 0:
            logger.info("No deletes executed.")
            return
        
        if not self.latencies:
            logger.info("No latency data recorded.")
            return
        
        # Calculate success rate
        success_rate = (self.total_deletes - self.total_failures) / self.total_deletes * 100
        
        # Sort latencies for percentile calculations
        sorted_latencies = sorted(self.latencies)
        latencies_ms = [lat * 1000 for lat in sorted_latencies]  # Convert to milliseconds
        
        # Calculate statistics
        avg_latency = np.mean(latencies_ms)
        min_latency = np.min(latencies_ms)
        max_latency = np.max(latencies_ms)
        p50_latency = np.percentile(latencies_ms, 50)
        p95_latency = np.percentile(latencies_ms, 95)
        p99_latency = np.percentile(latencies_ms, 99)
        
        # Print results
        logger.info("\n" + "="*80)
        logger.info("📊 Delete Test Final Results")
        logger.info("="*80)
        logger.info(f"\n📈 Test Summary:")
        logger.info(f"   Total Deletes: {self.total_deletes}")
        logger.info(f"   Total Rows Deleted: {self.total_rows_deleted}")
        logger.info(f"   Total Failures: {self.total_failures}")
        logger.info(f"   Success Rate: {success_rate:.2f}%")
        logger.info(f"   Test Duration: {total_duration:.2f}s")
        
        logger.info(f"\n⏱️  Latency Metrics (milliseconds):")
        logger.info(f"   Average (Mean): {avg_latency:.2f} ms")
        logger.info(f"   Min: {min_latency:.2f} ms")
        logger.info(f"   Max: {max_latency:.2f} ms")
        
        logger.info(f"\n📊 Percentiles:")
        logger.info(f"   P50 (Median): {p50_latency:.2f} ms")
        logger.info(f"   P95: {p95_latency:.2f} ms")
        logger.info(f"   P99: {p99_latency:.2f} ms")
        
        # Print first 10 delete response times
        if self.first_10_latencies:
            logger.info(f"\n🕐 First 10 Delete Response Times:")
            for i, latency_ms in enumerate([lat * 1000 for lat in self.first_10_latencies], 1):
                logger.info(f"   Delete #{i}: {latency_ms:.2f} ms")
        
        logger.info("\n" + "="*80)


def get_namespace_row_count(ns: turbopuffer.Namespace) -> int:
    """Get approximate row count from namespace metadata"""
    try:
        metadata = ns.metadata()
        return metadata.approx_row_count
    except Exception as e:
        logger.warning(f"Failed to get row count: {e}")
        return -1


def perform_delete(ns: turbopuffer.Namespace, delete_id: int) -> tuple[float, int]:
    """
    Perform delete operation using delete_by_filter
    
    Args:
        ns: Turbopuffer namespace object
        delete_id: ID value to delete
    
    Returns:
        Tuple of (latency in seconds, rows_affected)
    """
    # Delete by filter: delete records where id equals the random value
    delete_filter = ("id", "Eq", delete_id)
    
    start_time = time.time()
    result = ns.write(delete_by_filter=delete_filter)
    latency = time.time() - start_time
    
    rows_affected = result.rows_affected if hasattr(result, 'rows_affected') else 0
    
    return latency, rows_affected


def main():
    parser = argparse.ArgumentParser(
        description='Turbopuffer Delete Performance Test',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic test: delete 100 times from namespace id_0, using random IDs from 1-1000
  python delete.py --user-id 0 --id-start 1 --id-end 1000 --delete-count 100
  
  # Using environment variable for API key
  export TURBOPUFFER_API_KEY=your_api_key
  python delete.py --user-id 5 --id-start 100 --id-end 5000 --delete-count 50
        """
    )
    
    parser.add_argument('--key', type=str, default='',
                       help='Turbopuffer API key (or set TURBOPUFFER_API_KEY env var)')
    parser.add_argument('--region', type=str, default='aws-us-west-2',
                       help='Turbopuffer region (default: aws-us-west-2)')
    parser.add_argument('--user-id', type=int, required=True,
                       help='User ID (namespace) to perform delete operations (e.g., 0 for id_0)')
    parser.add_argument('--id-start', type=int, required=True,
                       help='Start of ID range for delete filter (inclusive)')
    parser.add_argument('--id-end', type=int, required=True,
                       help='End of ID range for delete filter (inclusive)')
    parser.add_argument('--delete-count', type=int, required=True,
                       help='Number of delete operations to execute')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging (debug level)')
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Get API key
    api_key = args.key or os.getenv("TURBOPUFFER_API_KEY")
    if not api_key:
        logger.error("❌ Error: --key or TURBOPUFFER_API_KEY environment variable is required")
        logger.error("\nSet API key using one of these methods:")
        logger.error("  1. Command line: --key YOUR_API_KEY")
        logger.error("  2. Environment: export TURBOPUFFER_API_KEY=YOUR_API_KEY")
        sys.exit(1)
    
    # Validate parameters
    if args.user_id < 0:
        logger.error("❌ Error: user-id must be non-negative")
        sys.exit(1)
    
    if args.id_start < 0 or args.id_end < 0:
        logger.error("❌ Error: ID range values must be non-negative")
        sys.exit(1)
    
    if args.id_start > args.id_end:
        logger.error("❌ Error: id-start must be <= id-end")
        sys.exit(1)
    
    if args.delete_count <= 0:
        logger.error("❌ Error: delete-count must be positive")
        sys.exit(1)
    
    # Create namespace name
    namespace = f"id_{args.user_id}"
    
    # Create Turbopuffer client
    logger.info("🔧 Initializing Turbopuffer client...")
    logger.info(f"   Region: {args.region}")
    
    try:
        tpuf = turbopuffer.Turbopuffer(api_key=api_key, region=args.region)
        ns = tpuf.namespace(namespace)
        logger.info(f"✅ Client initialized successfully")
        logger.info(f"   Target Namespace: {namespace}")
        logger.info("")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Turbopuffer client: {e}")
        sys.exit(1)
    
    # Initialize statistics
    stats = DeleteStatistics()
    
    # Get initial row count
    logger.info("📊 Initial Namespace Status:")
    initial_row_count = get_namespace_row_count(ns)
    if initial_row_count >= 0:
        logger.info(f"   Approximate Row Count: {initial_row_count:,}")
    logger.info("")
    
    # Perform delete operations
    logger.info("🚀 Starting delete operations...")
    logger.info(f"   Namespace: {namespace}")
    logger.info(f"   ID Range: {args.id_start} to {args.id_end}")
    logger.info(f"   Delete Count: {args.delete_count}")
    logger.info(f"   Each delete uses a random ID from the range")
    logger.info("")
    
    for i in range(1, args.delete_count + 1):
        # Get row count before delete
        row_count_before = get_namespace_row_count(ns)
        if row_count_before >= 0:
            logger.info(f"📝 Delete #{i}/{args.delete_count} - Row count before: {row_count_before:,}")
        else:
            logger.info(f"📝 Delete #{i}/{args.delete_count}")
        
        # Select random ID from range
        random_id = random.randint(args.id_start, args.id_end)
        logger.info(f"   Using ID: {random_id}")
        
        # Perform delete
        try:
            latency, rows_affected = perform_delete(ns, random_id)
            stats.record_latency(latency, i)
            stats.add_rows_deleted(rows_affected)
            
            # Get row count after delete
            row_count_after = get_namespace_row_count(ns)
            if row_count_after >= 0:
                logger.info(f"   ✅ Delete completed - Rows affected: {rows_affected}, "
                          f"Row count after: {row_count_after:,}, "
                          f"Latency: {latency*1000:.2f} ms")
            else:
                logger.info(f"   ✅ Delete completed - Rows affected: {rows_affected}, "
                          f"Latency: {latency*1000:.2f} ms")
            
        except Exception as e:
            stats.increment_failures()
            logger.error(f"   ❌ Delete failed: {e}")
        
        stats.increment_deletes()
        logger.info("")
    
    # Print final statistics
    logger.info("🎉 Delete test completed!")
    stats.print_stats()
    
    # Get final row count
    logger.info("\n📊 Final Namespace Status:")
    final_row_count = get_namespace_row_count(ns)
    if final_row_count >= 0:
        logger.info(f"   Approximate Row Count: {final_row_count:,}")
        if initial_row_count >= 0:
            rows_deleted_total = initial_row_count - final_row_count
            logger.info(f"   Estimated Total Rows Deleted: {rows_deleted_total:,}")
    logger.info("")


if __name__ == '__main__':
    main()

