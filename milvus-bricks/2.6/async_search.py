import asyncio
import time
import argparse
import numpy as np
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass, field
from pymilvus import AsyncMilvusClient, DataType
import statistics


@dataclass
class SearchMetrics:
    """Store performance metrics for search operations"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeout_requests: int = 0
    latencies: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    start_time: float = 0
    end_time: float = 0

    def add_success(self, latency: float):
        """Record a successful request"""
        self.total_requests += 1
        self.successful_requests += 1
        self.latencies.append(latency)

    def add_failure(self, error_msg: str, is_timeout: bool = False):
        """Record a failed request"""
        self.total_requests += 1
        self.failed_requests += 1
        if is_timeout:
            self.timeout_requests += 1
        self.errors.append(error_msg)

    def calculate_statistics(self) -> Dict[str, Any]:
        """Calculate and return performance statistics"""
        duration = self.end_time - self.start_time
        qps = self.total_requests / duration if duration > 0 else 0

        stats = {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'timeout_requests': self.timeout_requests,
            'success_rate': (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0,
            'duration_seconds': duration,
            'qps': qps,
        }

        if self.latencies:
            stats.update({
                'min_latency_ms': min(self.latencies) * 1000,
                'max_latency_ms': max(self.latencies) * 1000,
                'avg_latency_ms': statistics.mean(self.latencies) * 1000,
                'median_latency_ms': statistics.median(self.latencies) * 1000,
                'p95_latency_ms': np.percentile(self.latencies, 95) * 1000,
                'p99_latency_ms': np.percentile(self.latencies, 99) * 1000,
            })

        return stats

    def print_report(self):
        """Print formatted performance report to console and log"""
        stats = self.calculate_statistics()
        
        # Get logger
        logger = logging.getLogger(__name__)
        
        report_lines = []
        report_lines.append("\n" + "="*60)
        report_lines.append("MILVUS ASYNC SEARCH PERFORMANCE REPORT")
        report_lines.append("="*60)
        
        report_lines.append(f"\n📊 Request Summary:")
        report_lines.append(f"  Total Requests:      {stats['total_requests']}")
        report_lines.append(f"  Successful:          {stats['successful_requests']}")
        report_lines.append(f"  Failed:              {stats['failed_requests']}")
        report_lines.append(f"  Timeout:             {stats['timeout_requests']}")
        report_lines.append(f"  Success Rate:        {stats['success_rate']:.2f}%")
        
        report_lines.append(f"\n⏱️  Performance Metrics:")
        report_lines.append(f"  Total Duration:      {stats['duration_seconds']:.2f}s")
        report_lines.append(f"  QPS (Queries/sec):   {stats['qps']:.2f}")
        
        if self.latencies:
            report_lines.append(f"\n📈 Latency Statistics (ms):")
            report_lines.append(f"  Min:                 {stats['min_latency_ms']:.2f}")
            report_lines.append(f"  Max:                 {stats['max_latency_ms']:.2f}")
            report_lines.append(f"  Average:             {stats['avg_latency_ms']:.2f}")
            report_lines.append(f"  Median:              {stats['median_latency_ms']:.2f}")
            report_lines.append(f"  P95:                 {stats['p95_latency_ms']:.2f}")
            report_lines.append(f"  P99:                 {stats['p99_latency_ms']:.2f}")
        
        if self.errors:
            report_lines.append(f"\n⚠️  Sample Errors (showing first 5):")
            for i, error in enumerate(self.errors[:5], 1):
                report_lines.append(f"  {i}. {error}")
        
        report_lines.append("="*60 + "\n")
        
        # Print to console
        report_text = "\n".join(report_lines)
        print(report_text)
        
        # Also log to file
        for line in report_lines:
            logger.info(line)


class AsyncSearchBenchmark:
    """Async search benchmark runner for Milvus"""
    
    def __init__(self, uri: str, token: str, collection_name: str, 
                 concurrency: int, total_duration: int, timeout: float,
                 vector_field: str = None, top_k: int = 10, log_level: str = "INFO"):
        """
        Initialize benchmark parameters
        
        Args:
            uri: Milvus server URI
            token: Authentication token
            collection_name: Target collection name
            concurrency: Number of concurrent requests
            total_duration: Total test duration in seconds
            timeout: Single query timeout in seconds (supports decimal for milliseconds, e.g., 0.5 = 500ms)
            vector_field: Vector field name to search on (None for auto-detect)
            top_k: Number of top results to return
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.uri = uri
        self.token = token
        self.collection_name = collection_name
        self.concurrency = concurrency
        self.total_duration = total_duration
        self.timeout = timeout
        self.vector_field = vector_field
        self.top_k = top_k
        self.log_level = log_level
        self.vector_dim = None  # Will be determined after connecting
        self.metrics = SearchMetrics()
        self.client = None
        self.running = True
        self.logger = None  # Will be set after logging configuration
        
        # Configure logging
        self._configure_logging()
        self.logger = logging.getLogger(__name__)

    def _configure_logging(self):
        """Configure logging level for pymilvus and the script"""
        # Create log directory if it doesn't exist
        log_dir = Path("/tmp/log")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate log file name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"milvus_async_search_{timestamp}.log"
        
        # Map string level to logging constant
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        
        pymilvus_log_level = level_map.get(self.log_level.upper(), logging.INFO)
        
        # Configure root logger to INFO (for script's own logs)
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all logs
        
        # Clear existing handlers to avoid duplicate logs
        root_logger.handlers.clear()
        
        # Create file handler - logs everything at DEBUG level
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Capture all logs to file
        
        # Create console handler (only for WARNING and above)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to root logger
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # Configure script's own logger (always INFO level)
        script_logger = logging.getLogger(__name__)
        script_logger.setLevel(logging.INFO)
        
        # Configure pymilvus logger with user-specified level
        pymilvus_logger = logging.getLogger('pymilvus')
        pymilvus_logger.setLevel(pymilvus_log_level)
        
        # Also configure specific pymilvus submodules
        for logger_name in ['pymilvus.client', 'pymilvus.grpc_gen', 'pymilvus.milvus_client']:
            logger = logging.getLogger(logger_name)
            logger.setLevel(pymilvus_log_level)
        
        # Print to console (this is one of the few console outputs we keep)
        print(f"📝 Log file: {log_file}")
        if self.log_level.upper() == 'DEBUG':
            print(f"🔍 Pymilvus logging level set to DEBUG")
        else:
            print(f"ℹ️  Pymilvus logging level: {self.log_level.upper()}")

    async def connect(self):
        """Establish connection to Milvus and determine vector field"""
        self.client = AsyncMilvusClient(
            uri=self.uri,
            token=self.token
        )
        self.logger.info(f"Connected to Milvus at {self.uri}")
        
        # Verify collection exists
        collections = await self.client.list_collections()
        if self.collection_name not in collections:
            self.logger.error(f"Collection '{self.collection_name}' not found. Available collections: {collections}")
            raise ValueError(f"Collection '{self.collection_name}' not found. Available collections: {collections}")
        
        self.logger.info(f"Collection '{self.collection_name}' found")
        
        # Get collection schema to determine vector field and dimension
        schema = await self.client.describe_collection(self.collection_name)
        vector_fields = []
        
        for field in schema['fields']:
            field_type = field['type']
            # Check if field is a vector type
            if field_type in [DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR, 
                             DataType.FLOAT16_VECTOR, DataType.BFLOAT16_VECTOR,
                             DataType.SPARSE_FLOAT_VECTOR]:
                field_info = {
                    'name': field['name'],
                    'type': field_type,
                    'dim': field.get('params', {}).get('dim', 'N/A')
                }
                vector_fields.append(field_info)
        
        if not vector_fields:
            self.logger.error(f"No vector fields found in collection '{self.collection_name}'")
            raise ValueError(f"No vector fields found in collection '{self.collection_name}'")
        
        # Determine which vector field to use
        if self.vector_field is None or self.vector_field.lower() in ['none', 'default']:
            # Auto-select the first vector field
            selected_field = vector_fields[0]
            self.vector_field = selected_field['name']
            self.vector_dim = selected_field['dim']
            self.logger.info(f"Auto-selected vector field: '{self.vector_field}' (dim: {self.vector_dim})")
            
            if len(vector_fields) > 1:
                self.logger.info(f"Note: Collection has multiple vector fields: {[f['name'] for f in vector_fields]}")
        else:
            # Find the specified vector field
            selected_field = None
            for field in vector_fields:
                if field['name'] == self.vector_field:
                    selected_field = field
                    break
            
            if selected_field is None:
                available_fields = [f['name'] for f in vector_fields]
                self.logger.error(
                    f"Vector field '{self.vector_field}' not found. "
                    f"Available vector fields: {available_fields}"
                )
                raise ValueError(
                    f"Vector field '{self.vector_field}' not found. "
                    f"Available vector fields: {available_fields}"
                )
            
            self.vector_dim = selected_field['dim']
            self.logger.info(f"Using vector field: '{self.vector_field}' (dim: {self.vector_dim})")

    async def close(self):
        """Close connection to Milvus"""
        if self.client:
            await self.client.close()
            self.logger.info("Connection closed")

    def generate_random_vector(self) -> List[float]:
        """Generate a random search vector"""
        return np.random.random(self.vector_dim).tolist()

    async def single_search(self, worker_id: int) -> None:
        """
        Perform a single search operation
        
        Args:
            worker_id: Worker identifier for logging
        """
        query_vector = self.generate_random_vector()
        
        start = time.time()
        try:
            # Perform search with timeout
            result = await self.client.search(
                    collection_name=self.collection_name,
                    data=[query_vector],
                    anns_field=self.vector_field,
                    limit=self.top_k,
                    output_fields=["*"],
                    timeout=self.timeout
            )
            
            latency = time.time() - start
            self.metrics.add_success(latency)
            
        except asyncio.TimeoutError:
            latency = time.time() - start
            error_msg = f"Worker {worker_id}: Search timeout after {latency:.2f}s"
            self.metrics.add_failure(error_msg, is_timeout=True)
            
        except Exception as e:
            latency = time.time() - start
            error_msg = f"Worker {worker_id}: {type(e).__name__}: {str(e)}"
            self.metrics.add_failure(error_msg, is_timeout=False)

    async def worker(self, worker_id: int):
        """
        Worker coroutine that continuously performs searches
        
        Args:
            worker_id: Worker identifier
        """
        self.logger.debug(f"Worker {worker_id} started")
        
        while self.running:
            await self.single_search(worker_id)
            # Small delay to prevent overwhelming the system
            await asyncio.sleep(0.01)
        
        self.logger.debug(f"Worker {worker_id} stopped")

    async def run_benchmark(self):
        """Run the benchmark with specified parameters"""
        # Log to file and print minimal info to console
        self.logger.info("="*60)
        self.logger.info("STARTING ASYNC SEARCH BENCHMARK")
        self.logger.info("="*60)
        self.logger.info(f"Collection:          {self.collection_name}")
        self.logger.info(f"Concurrency:         {self.concurrency}")
        self.logger.info(f"Duration:            {self.total_duration}s")
        
        # Format timeout display based on value
        if self.timeout < 1:
            timeout_display = f"{self.timeout * 1000:.0f}ms"
        else:
            timeout_display = f"{self.timeout}s"
        self.logger.info(f"Query Timeout:       {timeout_display}")
        
        self.logger.info(f"Top K:               {self.top_k}")
        self.logger.info("="*60)
        
        # Also print to console for user visibility
        print(f"\n🚀 Starting benchmark: {self.collection_name}")
        print(f"   Concurrency: {self.concurrency} | Duration: {self.total_duration}s | Timeout: {timeout_display}")
        
        # Connect to Milvus and determine vector field
        await self.connect()
        
        self.logger.info(f"Vector Field:        {self.vector_field}")
        self.logger.info(f"Vector Dimension:    {self.vector_dim}")
        self.logger.info("="*60)
        
        # Start metrics timer
        self.metrics.start_time = time.time()
        
        # Create worker tasks
        workers = [asyncio.create_task(self.worker(i)) for i in range(self.concurrency)]
        self.logger.info(f"Created {self.concurrency} worker tasks")
        
        # Wait for specified duration
        await asyncio.sleep(self.total_duration)
        
        # Stop workers
        self.running = False
        self.logger.info("Stopping workers...")
        await asyncio.gather(*workers, return_exceptions=True)
        
        # End metrics timer
        self.metrics.end_time = time.time()
        
        # Close connection
        await self.close()
        
        # Print report
        self.metrics.print_report()


async def main():
    """Main entry point for the benchmark script"""
    parser = argparse.ArgumentParser(
        description='Milvus Async Search Performance Benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with 10 concurrent workers for 60 seconds
  python async_search.py --collection my_collection --concurrency 10 --duration 60

  # Specify a vector field to search
  python async_search.py --collection my_collection --concurrency 10 --duration 60 \\
      --vector-field dense_vector

  # Use default/auto-detected vector field
  python async_search.py --collection my_collection --concurrency 10 --duration 60 \\
      --vector-field default

  # Custom Milvus connection and timeout settings
  python async_search.py --uri http://localhost:19530 --token root:Milvus \\
      --collection my_collection --concurrency 20 --duration 120 --timeout 5

  # High concurrency test
  python async_search.py --collection my_collection --concurrency 100 \\
      --duration 300 --timeout 10 --top-k 20

  # Test with millisecond-level timeout (500ms = 0.5s)
  python async_search.py --collection my_collection --concurrency 10 --duration 60 \\
      --timeout 0.5

  # Test with 100ms timeout
  python async_search.py --collection my_collection --concurrency 10 --duration 60 \\
      --timeout 0.1

  # Enable DEBUG logging for detailed pymilvus logs
  python async_search.py --collection my_collection --concurrency 10 --duration 60 \\
      --log-level DEBUG
        """
    )
    
    # Connection parameters
    parser.add_argument('--uri', type=str, default='http://localhost:19530',
                        help='Milvus server URI (default: http://localhost:19530)')
    parser.add_argument('--token', type=str, default='root:Milvus',
                        help='Authentication token (default: root:Milvus)')
    
    # Required parameters
    parser.add_argument('--collection', type=str, required=True,
                        help='Collection name to search against (required)')
    parser.add_argument('--concurrency', type=int, default=10,
                        help='Number of concurrent search workers (required)')
    parser.add_argument('--duration', type=int, default=180,
                        help='Total benchmark duration in seconds (required)')
    
    # Optional parameters
    parser.add_argument('--timeout', type=float, default=10.0,
                        help='Single query timeout in seconds (supports decimals for milliseconds, '
                             'e.g., 0.5 for 500ms, 0.1 for 100ms, default: 10.0)')
    parser.add_argument('--top-k', type=int, default=10,
                        help='Number of top results to return (default: 10)')
    parser.add_argument('--vector-field', type=str, default=None,
                        help='Vector field name to search on (default: auto-detect first vector field, '
                             'use "none" or "default" for auto-detection)')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level for pymilvus SDK (default: INFO, use DEBUG for detailed pymilvus logs). '
                             'Script logs are always at INFO level.')
    
    args = parser.parse_args()
    
    # Validate parameters
    if args.concurrency <= 0:
        print("❌ Error: Concurrency must be positive")
        return
    
    if args.duration <= 0:
        print("❌ Error: Duration must be positive")
        return
    
    if args.timeout <= 0:
        print("❌ Error: Timeout must be positive")
        return
    
    # Create and run benchmark
    benchmark = AsyncSearchBenchmark(
        uri=args.uri,
        token=args.token,
        collection_name=args.collection,
        concurrency=args.concurrency,
        total_duration=args.duration,
        timeout=args.timeout,
        vector_field=args.vector_field,
        top_k=args.top_k,
        log_level=args.log_level
    )
    
    try:
        await benchmark.run_benchmark()
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
        benchmark.logger.warning("Benchmark interrupted by user")
        benchmark.running = False
        await benchmark.close()
    except Exception as e:
        error_msg = f"Error running benchmark: {type(e).__name__}: {str(e)}"
        print(f"\n❌ {error_msg}")
        if benchmark.logger:
            benchmark.logger.error(error_msg, exc_info=True)
        if benchmark.client:
            await benchmark.close()


if __name__ == "__main__":
    asyncio.run(main())

