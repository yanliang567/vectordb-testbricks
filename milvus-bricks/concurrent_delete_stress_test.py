import sys
import logging
import argparse
import os
import time
import random
import threading
from pymilvus import MilvusClient
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta


def setup_logging():
    """Initialize logging configuration"""
    log_dir = "/tmp"
    log_file = os.path.join(log_dir, "concurrent_delete_stress.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Logging initialized. Log file: %s", log_file)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Concurrent delete stress test for Milvus collection')
    parser.add_argument('--uri', type=str, default="http://localhost:19530",
                        help='Milvus server URI')
    parser.add_argument('--token', type=str, default="root:Milvus",
                        help='Milvus authentication token')
    parser.add_argument('--collection-name', type=str, required=True,
                        help='Name of the collection to delete from')
    parser.add_argument('--max-value', type=int, required=True,
                        help='Maximum value for doc_id range (doc_id will be randomly selected from 1 to max-value)')
    parser.add_argument('--concurrency', type=int, default=10,
                        help='Number of concurrent threads')
    parser.add_argument('--duration', type=int, default=60,
                        help='Duration to run the test in seconds')
    parser.add_argument('--interval', type=int, default=0,
                        help='Interval between delete operations in milliseconds (default: 0, no interval)')
    return parser.parse_args()


class DeleteStressTest:
    """Delete stress test class"""
    
    def __init__(self, uri, token, collection_name, max_value, concurrency, duration, interval):
        self.uri = uri
        self.token = token
        self.collection_name = collection_name
        self.max_value = max_value
        self.concurrency = concurrency
        self.duration = duration
        self.interval = interval / 1000.0  # Convert to seconds
        
        # Statistics
        self.total_operations = 0
        self.successful_operations = 0
        self.failed_operations = 0
        self.lock = threading.Lock()
        self.start_time = None
        self.end_time = None
        self.latencies = []
        
        # Control flag
        self.should_stop = False
    
    def create_client(self):
        """Create a new MilvusClient instance"""
        return MilvusClient(uri=self.uri, token=self.token)
    
    def perform_delete(self, thread_id):
        """Perform delete operation with random doc_id filter"""
        client = self.create_client()
        
        while not self.should_stop:
            try:
                # Generate random doc_id value
                random_value = random.randint(1, self.max_value)
                filter_expr = f'doc_id=="doc_{random_value}"'
                
                # Record start time
                op_start = time.time()
                
                # Execute delete operation
                result = client.delete(
                    collection_name=self.collection_name,
                    filter=filter_expr
                )
                
                # Calculate latency
                latency = time.time() - op_start
                
                # Update statistics
                with self.lock:
                    self.total_operations += 1
                    self.successful_operations += 1
                    self.latencies.append(latency)
                
                logging.debug(f"Thread {thread_id}: Delete with filter '{filter_expr}' completed. "
                             f"Delete count: {result.get('delete_count', 0)}, Latency: {latency*1000:.2f}ms")
                
                # Sleep if interval is set
                if self.interval > 0:
                    time.sleep(self.interval)
                    
            except Exception as e:
                with self.lock:
                    self.total_operations += 1
                    self.failed_operations += 1
                logging.error(f"Thread {thread_id}: Delete operation failed - {str(e)}")
        
        logging.info(f"Thread {thread_id}: Stopped")
    
    def run(self):
        """Run the stress test"""
        logging.info(f"Starting concurrent delete stress test")
        logging.info(f"Configuration:")
        logging.info(f"  - URI: {self.uri}")
        logging.info(f"  - Collection: {self.collection_name}")
        logging.info(f"  - Max Value: {self.max_value}")
        logging.info(f"  - Concurrency: {self.concurrency}")
        logging.info(f"  - Duration: {self.duration} seconds")
        logging.info(f"  - Interval: {self.interval*1000:.0f} milliseconds")
        
        # Check collection exists
        client = self.create_client()
        if not client.has_collection(self.collection_name):
            logging.error(f"Collection '{self.collection_name}' does not exist!")
            return
        
        logging.info(f"Collection '{self.collection_name}' found")
        
        # Start timing
        self.start_time = time.time()
        end_time_target = self.start_time + self.duration
        
        # Create thread pool
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            # Submit tasks
            futures = []
            for i in range(self.concurrency):
                future = executor.submit(self.perform_delete, i)
                futures.append(future)
            
            logging.info(f"Started {self.concurrency} concurrent threads")
            
            # Monitor progress and print statistics
            last_report_time = self.start_time
            report_interval = 5  # Report every 5 seconds
            
            while time.time() < end_time_target:
                time.sleep(1)
                current_time = time.time()
                
                # Print periodic report
                if current_time - last_report_time >= report_interval:
                    elapsed = current_time - self.start_time
                    with self.lock:
                        qps = self.successful_operations / elapsed if elapsed > 0 else 0
                        avg_latency = sum(self.latencies) / len(self.latencies) * 1000 if self.latencies else 0
                    
                    logging.info(f"Progress: {elapsed:.0f}s elapsed, "
                               f"Total: {self.total_operations}, "
                               f"Success: {self.successful_operations}, "
                               f"Failed: {self.failed_operations}, "
                               f"QPS: {qps:.2f}, "
                               f"Avg Latency: {avg_latency:.2f}ms")
                    last_report_time = current_time
            
            # Signal threads to stop
            logging.info("Duration reached, stopping threads...")
            self.should_stop = True
            
            # Wait for all threads to complete
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Thread encountered error: {str(e)}")
        
        self.end_time = time.time()
        
        # Print final statistics
        self.print_statistics()
    
    def print_statistics(self):
        """Print final test statistics"""
        logging.info("=" * 80)
        logging.info("DELETE STRESS TEST COMPLETED")
        logging.info("=" * 80)
        
        elapsed_time = self.end_time - self.start_time
        
        logging.info(f"Test Configuration:")
        logging.info(f"  Collection Name: {self.collection_name}")
        logging.info(f"  Max Value Range: 1 - {self.max_value}")
        logging.info(f"  Concurrency: {self.concurrency}")
        logging.info(f"  Duration: {self.duration} seconds")
        logging.info(f"  Interval: {self.interval*1000:.0f} milliseconds")
        
        logging.info(f"\nTest Results:")
        logging.info(f"  Actual Runtime: {elapsed_time:.2f} seconds")
        logging.info(f"  Total Operations: {self.total_operations}")
        logging.info(f"  Successful Operations: {self.successful_operations}")
        logging.info(f"  Failed Operations: {self.failed_operations}")
        
        if self.total_operations > 0:
            success_rate = (self.successful_operations / self.total_operations) * 100
            logging.info(f"  Success Rate: {success_rate:.2f}%")
        
        if elapsed_time > 0:
            qps = self.successful_operations / elapsed_time
            logging.info(f"  Average QPS: {qps:.2f}")
        
        if self.latencies:
            sorted_latencies = sorted(self.latencies)
            count = len(sorted_latencies)
            
            avg_latency = sum(sorted_latencies) / count * 1000
            min_latency = sorted_latencies[0] * 1000
            max_latency = sorted_latencies[-1] * 1000
            p50_latency = sorted_latencies[int(count * 0.5)] * 1000
            p95_latency = sorted_latencies[int(count * 0.95)] * 1000
            p99_latency = sorted_latencies[int(count * 0.99)] * 1000
            
            logging.info(f"\nLatency Statistics (milliseconds):")
            logging.info(f"  Average: {avg_latency:.2f} ms")
            logging.info(f"  Min: {min_latency:.2f} ms")
            logging.info(f"  Max: {max_latency:.2f} ms")
            logging.info(f"  P50: {p50_latency:.2f} ms")
            logging.info(f"  P95: {p95_latency:.2f} ms")
            logging.info(f"  P99: {p99_latency:.2f} ms")
        
        logging.info("=" * 80)


def main():
    """Main function"""
    args = parse_args()
    setup_logging()
    
    # Create and run stress test
    test = DeleteStressTest(
        uri=args.uri,
        token=args.token,
        collection_name=args.collection_name,
        max_value=args.max_value,
        concurrency=args.concurrency,
        duration=args.duration,
        interval=args.interval
    )
    
    test.run()


if __name__ == "__main__":
    main()

