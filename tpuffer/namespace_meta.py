#!/usr/bin/env python3
"""
Turbopuffer Namespace Meta Performance Test Script (Concurrent)

Tests the namespace.metadata() API performance with:
- Concurrent requests using thread pool
- Configurable user ID range for namespace selection
- Duration-based testing (run for specified seconds)
- Comprehensive performance metrics: QPS, P99, P95, P50, Avg, Min, Max latency

API Tested:
    ns = tpuf.namespace('id_0')
    metadata = ns.metadata()  # Returns turbopuffer.NamespaceMetadata object

Usage:
    # Basic usage
    python namespace_meta.py --key YOUR_API_KEY \
                             --user-id-start 0 \
                             --user-id-end 99 \
                             --concurrency 10 \
                             --duration 60
    
    # Using environment variable
    export TURBOPUFFER_API_KEY=your_api_key
    python namespace_meta.py --user-id-start 0 \
                             --user-id-end 99 \
                             --concurrency 20 \
                             --duration 120
"""

import argparse
import os
import sys
import time
import logging
import random
import threading
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
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


class NamespaceMetaStatistics:
    """Thread-safe statistics collector for namespace meta queries"""
    
    def __init__(self):
        self.latencies: List[float] = []
        self.failures: List[Dict[str, Any]] = []
        self.namespace_query_counts: Dict[str, int] = {}
        self.start_time = time.time()
        self.end_time = None
        self.lock = threading.Lock()
        
        # Real-time stats
        self.total_requests = 0
        self.total_failures = 0
    
    def record_success(self, namespace: str, latency: float):
        """Record a successful query (thread-safe)"""
        with self.lock:
            self.latencies.append(latency)
            self.total_requests += 1
            self.namespace_query_counts[namespace] = self.namespace_query_counts.get(namespace, 0) + 1
    
    def record_failure(self, namespace: str, error: str):
        """Record a failed query (thread-safe)"""
        with self.lock:
            self.failures.append({
                'namespace': namespace,
                'error': str(error),
                'timestamp': time.time()
            })
            self.total_requests += 1
            self.total_failures += 1
    
    def get_current_qps(self) -> float:
        """Calculate current QPS"""
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            return self.total_requests / elapsed
        return 0.0
    
    def print_progress(self):
        """Print current progress"""
        elapsed = time.time() - self.start_time
        qps = self.get_current_qps()
        success_rate = ((self.total_requests - self.total_failures) / self.total_requests * 100) if self.total_requests > 0 else 0
        
        logger.info(f"📊 Progress: {self.total_requests} requests | "
                   f"QPS: {qps:.2f} | "
                   f"Success Rate: {success_rate:.2f}% | "
                   f"Elapsed: {elapsed:.1f}s")
    
    def print_final_stats(self):
        """Print comprehensive final statistics"""
        self.end_time = time.time()
        total_duration = self.end_time - self.start_time
        
        logger.info("\n" + "="*80)
        logger.info("📊 Final Performance Statistics")
        logger.info("="*80)
        
        if self.total_requests == 0:
            logger.info("No requests were made.")
            return
        
        # Basic stats
        success_count = self.total_requests - self.total_failures
        success_rate = success_count / self.total_requests * 100
        
        logger.info(f"\n📈 Test Summary:")
        logger.info(f"   Total Requests: {self.total_requests}")
        logger.info(f"   Successful Requests: {success_count}")
        logger.info(f"   Failed Requests: {self.total_failures}")
        logger.info(f"   Success Rate: {success_rate:.2f}%")
        logger.info(f"   Total Duration: {total_duration:.2f}s")
        
        # QPS
        overall_qps = self.total_requests / total_duration if total_duration > 0 else 0
        logger.info(f"\n🚀 Throughput:")
        logger.info(f"   QPS (Queries Per Second): {overall_qps:.2f}")
        logger.info(f"   Average Time Between Requests: {1000/overall_qps:.2f}ms" if overall_qps > 0 else "   N/A")
        
        # Latency statistics
        if self.latencies:
            latencies_ms = [lat * 1000 for lat in self.latencies]
            
            avg_latency = np.mean(latencies_ms)
            median_latency = np.median(latencies_ms)
            min_latency = np.min(latencies_ms)
            max_latency = np.max(latencies_ms)
            p50_latency = np.percentile(latencies_ms, 50)
            p95_latency = np.percentile(latencies_ms, 95)
            p99_latency = np.percentile(latencies_ms, 99)
            std_latency = np.std(latencies_ms)
            
            logger.info(f"\n⏱️  Latency Metrics (milliseconds):")
            logger.info(f"   Average (Mean): {avg_latency:.2f} ms")
            logger.info(f"   Median: {median_latency:.2f} ms")
            logger.info(f"   Min: {min_latency:.2f} ms")
            logger.info(f"   Max: {max_latency:.2f} ms")
            logger.info(f"   Std Dev: {std_latency:.2f} ms")
            
            logger.info(f"\n📊 Percentiles:")
            logger.info(f"   P50 (Median): {p50_latency:.2f} ms")
            logger.info(f"   P95: {p95_latency:.2f} ms")
            logger.info(f"   P99: {p99_latency:.2f} ms")
        else:
            logger.info("\n⏱️  No latency data recorded (all requests failed)")
        
        # Namespace distribution
        if self.namespace_query_counts:
            total_namespaces = len(self.namespace_query_counts)
            queries_per_namespace = [count for count in self.namespace_query_counts.values()]
            avg_queries_per_ns = np.mean(queries_per_namespace)
            min_queries = np.min(queries_per_namespace)
            max_queries = np.max(queries_per_namespace)
            
            logger.info(f"\n📋 Namespace Distribution:")
            logger.info(f"   Unique Namespaces Queried: {total_namespaces}")
            logger.info(f"   Avg Queries per Namespace: {avg_queries_per_ns:.2f}")
            logger.info(f"   Min Queries to a Namespace: {min_queries}")
            logger.info(f"   Max Queries to a Namespace: {max_queries}")
            
            # Show top 10 most queried namespaces
            if total_namespaces > 0:
                sorted_ns = sorted(self.namespace_query_counts.items(), 
                                  key=lambda x: x[1], reverse=True)[:10]
                logger.info(f"\n   Top 10 Most Queried Namespaces:")
                for i, (ns, count) in enumerate(sorted_ns, 1):
                    logger.info(f"      {i}. {ns}: {count} queries")
        
        # Failure details
        if self.failures:
            logger.info(f"\n❌ Failure Details:")
            logger.info(f"   Total Failures: {len(self.failures)}")
            
            # Group failures by error type
            error_types: Dict[str, int] = {}
            for failure in self.failures:
                error_msg = failure['error']
                error_types[error_msg] = error_types.get(error_msg, 0) + 1
            
            logger.info(f"   Unique Error Types: {len(error_types)}")
            logger.info(f"\n   Error Breakdown:")
            for error, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5]:
                logger.info(f"      • {error}: {count} occurrences")
            
            if len(error_types) > 5:
                logger.info(f"      ... and {len(error_types) - 5} more error types")
        
        logger.info("\n" + "="*80)


def query_namespace_meta(tpuf: turbopuffer.Turbopuffer, namespace: str) -> tuple[float, Any]:
    """
    Query namespace metadata
    
    Args:
        tpuf: Turbopuffer client
        namespace: Namespace name
    
    Returns:
        Tuple of (latency in seconds, result)
    """
    start_time = time.time()
    ns = tpuf.namespace(namespace)
    result = ns.metadata()
    latency = time.time() - start_time
    
    return latency, result


def worker_thread(tpuf: turbopuffer.Turbopuffer, 
                 namespaces: List[str],
                 stats: NamespaceMetaStatistics,
                 stop_event: threading.Event,
                 worker_id: int):
    """
    Worker thread that continuously queries namespace meta until stop_event is set
    
    Args:
        tpuf: Turbopuffer client
        namespaces: List of namespace names to query from
        stats: Statistics collector
        stop_event: Event to signal worker to stop
        worker_id: Worker thread ID for logging
    """
    queries_made = 0
    
    while not stop_event.is_set():
        # Randomly select a namespace
        namespace = random.choice(namespaces)
        
        try:
            latency, result = query_namespace_meta(tpuf, namespace)
            stats.record_success(namespace, latency)
            queries_made += 1
            
        except Exception as e:
            stats.record_failure(namespace, str(e))
            logger.debug(f"Worker {worker_id} failed to query {namespace}: {e}")
    
    logger.debug(f"Worker {worker_id} completed {queries_made} queries")


def run_concurrent_test(tpuf: turbopuffer.Turbopuffer,
                       user_id_start: int,
                       user_id_end: int,
                       concurrency: int,
                       duration: int) -> NamespaceMetaStatistics:
    """
    Run concurrent namespace meta queries for specified duration
    
    Args:
        tpuf: Turbopuffer client
        user_id_start: Start of user ID range
        user_id_end: End of user ID range
        concurrency: Number of concurrent workers
        duration: Test duration in seconds
    
    Returns:
        NamespaceMetaStatistics with collected metrics
    """
    # Generate namespace list
    namespaces = [f"id_{i}" for i in range(user_id_start, user_id_end + 1)]
    num_namespaces = len(namespaces)
    
    logger.info(f"🚀 Starting Concurrent Namespace Meta Test")
    logger.info(f"   Namespace Range: id_{user_id_start} to id_{user_id_end} ({num_namespaces} namespaces)")
    logger.info(f"   Concurrency: {concurrency} workers")
    logger.info(f"   Duration: {duration} seconds")
    logger.info("")
    
    stats = NamespaceMetaStatistics()
    stop_event = threading.Event()
    
    # Start worker threads
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        # Submit all workers
        futures = []
        for i in range(concurrency):
            future = executor.submit(worker_thread, tpuf, namespaces, stats, stop_event, i)
            futures.append(future)
        
        logger.info(f"✅ {concurrency} worker threads started")
        logger.info("")
        
        # Monitor progress
        start_time = time.time()
        last_progress_time = start_time
        progress_interval = 10  # Print progress every 10 seconds
        
        while time.time() - start_time < duration:
            time.sleep(1)
            
            # Print progress at intervals
            current_time = time.time()
            if current_time - last_progress_time >= progress_interval:
                stats.print_progress()
                last_progress_time = current_time
        
        # Signal workers to stop
        logger.info("")
        logger.info("⏱️  Time's up! Stopping workers...")
        stop_event.set()
        
        # Wait for all workers to complete
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Worker thread error: {e}")
    
    logger.info("✅ All workers stopped")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Turbopuffer Namespace Meta Performance Test (Concurrent)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic test with 10 concurrent workers for 60 seconds
  python namespace_meta.py --user-id-start 0 --user-id-end 99 --concurrency 10 --duration 60
  
  # High concurrency test
  python namespace_meta.py --user-id-start 0 --user-id-end 799 --concurrency 50 --duration 120
  
  # Using environment variable for API key
  export TURBOPUFFER_API_KEY=your_api_key
  python namespace_meta.py --user-id-start 0 --user-id-end 99 --concurrency 20 --duration 60
        """
    )
    
    parser.add_argument('--key', type=str, default='',
                       help='Turbopuffer API key (or set TURBOPUFFER_API_KEY env var)')
    parser.add_argument('--region', type=str, default='aws-us-west-2',
                       help='Turbopuffer region (default: aws-us-west-2)')
    parser.add_argument('--user-id-start', type=int, required=True,
                       help='Start of user ID range for namespaces (e.g., 0)')
    parser.add_argument('--user-id-end', type=int, required=True,
                       help='End of user ID range for namespaces (e.g., 99)')
    parser.add_argument('--concurrency', type=int, required=True,
                       help='Number of concurrent workers/threads')
    parser.add_argument('--duration', type=int, required=True,
                       help='Test duration in seconds')
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
    
    if args.concurrency <= 0:
        logger.error("❌ Error: concurrency must be positive")
        sys.exit(1)
    
    if args.duration <= 0:
        logger.error("❌ Error: duration must be positive")
        sys.exit(1)
    
    num_namespaces = args.user_id_end - args.user_id_start + 1
    if args.concurrency > num_namespaces * 10:
        logger.warning(f"⚠️  Warning: High concurrency ({args.concurrency}) relative to "
                      f"namespace count ({num_namespaces}). This may cause uneven load distribution.")
    
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
    
    # Run the concurrent test
    try:
        stats = run_concurrent_test(
            tpuf=tpuf,
            user_id_start=args.user_id_start,
            user_id_end=args.user_id_end,
            concurrency=args.concurrency,
            duration=args.duration
        )
        
        # Print final statistics
        logger.info("\n🎉 Test completed!")
        stats.print_final_stats()
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
