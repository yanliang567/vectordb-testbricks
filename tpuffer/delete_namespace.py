#!/usr/bin/env python3
"""
Turbopuffer Delete Namespace Performance Test Script

Tests the namespace.delete_all() API performance with:
- User-specified namespace range (user_id_start to user_id_end)
- Sequential deletion of namespaces
- Response time tracking for each deletion
- Comprehensive performance metrics: P99, P95, P50, Avg, Min, Max latency

API Tested:
    ns = tpuf.namespace('id_0')
    ns.delete_all()  # Deletes all data in the namespace

Usage:
    # Basic usage
    python delete_namespace.py --key YOUR_API_KEY \
                               --user-id-start 0 \
                               --user-id-end 99
    
    # Using environment variable
    export TURBOPUFFER_API_KEY=your_api_key
    python delete_namespace.py --user-id-start 0 \
                               --user-id-end 99
"""

import argparse
import os
import sys
import time
import logging
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


class DeleteNamespaceStatistics:
    """Collects delete namespace performance metrics"""
    
    def __init__(self):
        self.latencies: List[float] = []
        self.total_deletes = 0
        self.total_failures = 0
        self.failed_namespaces: List[str] = []
        self.start_time = time.time()
        self.end_time = None
    
    def record_latency(self, latency: float):
        """Record a successful delete namespace latency in seconds"""
        self.latencies.append(latency)
    
    def increment_deletes(self):
        """Increment total delete count"""
        self.total_deletes += 1
    
    def increment_failures(self, namespace: str):
        """Increment failure count and record failed namespace"""
        self.total_failures += 1
        self.failed_namespaces.append(namespace)
    
    def print_stats(self):
        """Print final statistics"""
        self.end_time = time.time()
        total_duration = self.end_time - self.start_time
        
        if self.total_deletes == 0:
            logger.info("No namespace deletions executed.")
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
        logger.info("📊 Delete Namespace Test Final Results")
        logger.info("="*80)
        logger.info(f"\n📈 Test Summary:")
        logger.info(f"   Total Namespaces Deleted: {self.total_deletes}")
        logger.info(f"   Successful Deletions: {len(self.latencies)}")
        logger.info(f"   Failed Deletions: {self.total_failures}")
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
        
        # Print failed namespaces if any
        if self.failed_namespaces:
            logger.info(f"\n❌ Failed Namespaces ({len(self.failed_namespaces)}):")
            for ns in self.failed_namespaces:
                logger.info(f"   - {ns}")
        
        logger.info("\n" + "="*80)


def delete_namespace(tpuf: turbopuffer.Turbopuffer, namespace: str) -> float:
    """
    Delete all data in a namespace using delete_all()
    
    Args:
        tpuf: Turbopuffer client
        namespace: Namespace name to delete
    
    Returns:
        Latency in seconds
    """
    ns = tpuf.namespace(namespace)
    start_time = time.time()
    ns.delete_all()
    latency = time.time() - start_time
    
    return latency


def main():
    parser = argparse.ArgumentParser(
        description='Turbopuffer Delete Namespace Performance Test',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic test: delete namespaces from id_0 to id_99
  python delete_namespace.py --user-id-start 0 --user-id-end 99
  
  # Delete a single namespace
  python delete_namespace.py --user-id-start 0 --user-id-end 0
  
  # Using environment variable for API key
  export TURBOPUFFER_API_KEY=your_api_key
  python delete_namespace.py --user-id-start 0 --user-id-end 199
        """
    )
    
    parser.add_argument('--key', type=str, default='',
                       help='Turbopuffer API key (or set TURBOPUFFER_API_KEY env var)')
    parser.add_argument('--region', type=str, default='aws-us-west-2',
                       help='Turbopuffer region (default: aws-us-west-2)')
    parser.add_argument('--user-id-start', type=int, required=True,
                       help='Start of user ID range for namespaces to delete (e.g., 0)')
    parser.add_argument('--user-id-end', type=int, required=True,
                       help='End of user ID range for namespaces to delete (e.g., 99)')
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
    if args.user_id_start < 0 or args.user_id_end < 0:
        logger.error("❌ Error: user IDs must be non-negative")
        sys.exit(1)
    
    if args.user_id_start > args.user_id_end:
        logger.error("❌ Error: user-id-start must be <= user-id-end")
        sys.exit(1)
    
    # Generate namespace list
    namespaces = [f"id_{i}" for i in range(args.user_id_start, args.user_id_end + 1)]
    num_namespaces = len(namespaces)
    
    # Create Turbopuffer client
    logger.info("🔧 Initializing Turbopuffer client...")
    logger.info(f"   Region: {args.region}")
    
    try:
        tpuf = turbopuffer.Turbopuffer(api_key=api_key, region=args.region)
        logger.info("✅ Client initialized successfully")
        logger.info("")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Turbopuffer client: {e}")
        sys.exit(1)
    
    # Initialize statistics
    stats = DeleteNamespaceStatistics()
    
    # Perform namespace deletions
    logger.info("🚀 Starting namespace deletion operations...")
    logger.info(f"   Namespace Range: id_{args.user_id_start} to id_{args.user_id_end}")
    logger.info(f"   Total Namespaces: {num_namespaces}")
    logger.info("")
    
    for i, namespace in enumerate(namespaces, 1):
        logger.info(f"🗑️  Deleting namespace #{i}/{num_namespaces}: {namespace}")
        
        try:
            latency = delete_namespace(tpuf, namespace)
            stats.record_latency(latency)
            logger.info(f"   ✅ Deletion completed - Latency: {latency*1000:.2f} ms")
        except turbopuffer.APIError as e:
            stats.increment_failures(namespace)
            logger.error(f"   ❌ Deletion failed (APIError): {e}")
        except Exception as e:
            stats.increment_failures(namespace)
            logger.error(f"   ❌ Deletion failed: {e}")
        
        stats.increment_deletes()
        logger.info("")
    
    # Print final statistics
    logger.info("🎉 Namespace deletion test completed!")
    stats.print_stats()


if __name__ == '__main__':
    main()

