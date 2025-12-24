#!/usr/bin/env python3
"""
Turbopuffer List Namespace Performance Test Script

Tests the list namespace API performance by:
- Running list operation 100 times
- Recording the first 10 results
- Calculating performance metrics: P99, P95, P50, Avg, Min, Max

Usage:
    python list_namespace_test.py --key YOUR_API_KEY --region aws-us-west-2
    
    # Or set environment variable:
    export TURBOPUFFER_API_KEY=your_api_key
    python list_namespace_test.py --region aws-us-west-2
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


class ListNamespaceStatistics:
    """Collects list namespace performance metrics"""
    
    def __init__(self, total_iterations: int = 100, record_first_n: int = 10):
        self.latencies: List[float] = []
        self.results: List[Any] = []
        self.total_iterations = total_iterations
        self.record_first_n = record_first_n
        self.total_failures = 0
        self.start_time = time.time()
        self.end_time = None
    
    def record_result(self, latency: float, result: Any, iteration: int):
        """Record a successful list operation"""
        self.latencies.append(latency)
        
        # Only store first N results
        if iteration < self.record_first_n:
            self.results.append({
                'iteration': iteration + 1,
                'latency': latency,
                'namespace_count': len(result) if result else 0,
                'namespaces': result if result else []
            })
    
    def increment_failures(self):
        """Increment failure count"""
        self.total_failures += 1
    
    def print_first_results(self):
        """Print the first N recorded results"""
        logger.info("\n" + "="*80)
        logger.info(f"📋 First {len(self.results)} List Results:")
        logger.info("="*80)
        
        for result in self.results:
            logger.info(f"\nIteration {result['iteration']}:")
            logger.info(f"  Latency: {result['latency']:.4f}s")
            logger.info(f"  Namespace Count: {result['namespace_count']}")
            if result['namespaces'] and len(result['namespaces']) > 0:
                # Show first 5 namespaces as sample
                sample_size = min(5, len(result['namespaces']))
                logger.info(f"  Sample Namespaces (first {sample_size}):")
                for i, ns in enumerate(result['namespaces'][:sample_size], 1):
                    logger.info(f"    {i}. {ns}")
                if len(result['namespaces']) > sample_size:
                    logger.info(f"    ... and {len(result['namespaces']) - sample_size} more")
            else:
                logger.info(f"  No namespaces found")
    
    def print_performance_stats(self):
        """Print performance statistics"""
        self.end_time = time.time()
        total_duration = self.end_time - self.start_time
        
        logger.info("\n" + "="*80)
        logger.info("📊 Performance Statistics")
        logger.info("="*80)
        
        if not self.latencies:
            logger.info("No latency data recorded.")
            return
        
        # Calculate success rate
        total_operations = len(self.latencies) + self.total_failures
        success_rate = len(self.latencies) / total_operations * 100 if total_operations > 0 else 0
        
        # Sort latencies for percentile calculations
        sorted_latencies = sorted(self.latencies)
        
        # Calculate statistics (in milliseconds for better readability)
        latencies_ms = [lat * 1000 for lat in sorted_latencies]
        
        avg_latency = np.mean(latencies_ms)
        median_latency = np.median(latencies_ms)
        min_latency = np.min(latencies_ms)
        max_latency = np.max(latencies_ms)
        p50_latency = np.percentile(latencies_ms, 50)
        p95_latency = np.percentile(latencies_ms, 95)
        p99_latency = np.percentile(latencies_ms, 99)
        
        # Standard deviation
        std_latency = np.std(latencies_ms)
        
        # Print results
        logger.info(f"\n📈 Test Summary:")
        logger.info(f"   Total Iterations: {total_operations}")
        logger.info(f"   Successful Iterations: {len(self.latencies)}")
        logger.info(f"   Failed Iterations: {self.total_failures}")
        logger.info(f"   Success Rate: {success_rate:.2f}%")
        logger.info(f"   Total Test Duration: {total_duration:.2f}s")
        
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
        
        # Calculate throughput
        if total_duration > 0:
            throughput = len(self.latencies) / total_duration
            logger.info(f"\n🚀 Throughput:")
            logger.info(f"   Operations per Second: {throughput:.2f} ops/s")
        
        logger.info("\n" + "="*80)


def test_list_namespaces(tpuf: turbopuffer.Turbopuffer, iterations: int = 100, 
                         record_first_n: int = 10) -> ListNamespaceStatistics:
    """
    Test the list namespaces operation multiple times
    
    Args:
        tpuf: Turbopuffer client instance
        iterations: Number of times to run the list operation
        record_first_n: Number of first results to record in detail
    
    Returns:
        ListNamespaceStatistics object with collected metrics
    """
    stats = ListNamespaceStatistics(iterations, record_first_n)
    
    logger.info(f"🚀 Starting List Namespace Performance Test")
    logger.info(f"   Total Iterations: {iterations}")
    logger.info(f"   Recording First: {record_first_n} results")
    logger.info("")
    
    for i in range(iterations):
        try:
            # Measure list operation latency
            start_time = time.time()
            result = tpuf.namespaces()
            latency = time.time() - start_time
            
            # Record the result
            stats.record_result(latency, result, i)
            
            # Progress logging
            if (i + 1) % 10 == 0 or (i + 1) == iterations:
                logger.info(f"📝 Progress: {i + 1}/{iterations} ({(i + 1)/iterations*100:.1f}%) - "
                           f"Last latency: {latency:.4f}s")
            
        except Exception as e:
            stats.increment_failures()
            logger.error(f"❌ Iteration {i + 1} failed: {e}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Turbopuffer List Namespace Performance Test',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using API key argument
  python list_namespace_test.py --key YOUR_API_KEY --region aws-us-west-2
  
  # Using environment variable
  export TURBOPUFFER_API_KEY=your_api_key
  python list_namespace_test.py --region aws-us-west-2
  
  # Custom iteration count
  python list_namespace_test.py --iterations 200 --record-first 20
        """
    )
    
    parser.add_argument('--key', type=str, default='',
                       help='Turbopuffer API key (or set TURBOPUFFER_API_KEY env var)')
    parser.add_argument('--region', type=str, default='aws-us-west-2',
                       help='Turbopuffer region (default: aws-us-west-2)')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of times to run the list operation (default: 100)')
    parser.add_argument('--record-first', type=int, default=10,
                       help='Number of first results to record in detail (default: 10)')
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.key or os.getenv("TURBOPUFFER_API_KEY")
    if not api_key:
        logger.error("❌ Error: --key or TURBOPUFFER_API_KEY environment variable is required")
        logger.error("\nSet API key using one of these methods:")
        logger.error("  1. Command line: --key YOUR_API_KEY")
        logger.error("  2. Environment: export TURBOPUFFER_API_KEY=YOUR_API_KEY")
        sys.exit(1)
    
    # Validate parameters
    if args.iterations <= 0:
        logger.error("❌ Error: iterations must be positive")
        sys.exit(1)
    
    if args.record_first < 0:
        logger.error("❌ Error: record-first must be non-negative")
        sys.exit(1)
    
    if args.record_first > args.iterations:
        logger.warning(f"⚠️  Warning: record-first ({args.record_first}) is greater than "
                      f"iterations ({args.iterations}), adjusting to {args.iterations}")
        args.record_first = args.iterations
    
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
    
    # Run the test
    stats = test_list_namespaces(tpuf, args.iterations, args.record_first)
    
    # Print results
    logger.info("\n🎉 Test completed!")
    
    # Print first N results
    stats.print_first_results()
    
    # Print performance statistics
    stats.print_performance_stats()


if __name__ == '__main__':
    main()
