#!/usr/bin/env python3
"""
Ultra-Simplified Concurrent Query Testing - Single Thread Pool Architecture

Key Simplifications:
1. ✅ Remove client connection pool, use single MilvusClient instance
2. ✅ Remove dual-layer thread pools (BatchController + QueryTasks)
3. ✅ max_workers directly equals concurrent query count
4. ✅ Single ThreadPoolExecutor directly manages all query tasks
5. ✅ Rely on Milvus server-side connection reuse, no layered complexity
"""

import time
import sys
import random
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from collections import deque
from pymilvus import MilvusClient, DataType

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"


class OptimizedStats:
    """Optimized statistics system"""
    
    def __init__(self, max_samples=1000):
        self.latencies = deque(maxlen=max_samples)
        self.total_queries = 0
        self.total_failures = 0
        self.start_time = time.time()
        self.lock = Lock()
    
    def record_query(self, latency, success=True):
        """Record query result"""
        with self.lock:
            self.latencies.append(latency)
            self.total_queries += 1
            if not success:
                self.total_failures += 1
    
    def get_stats(self, actual_elapsed_time=None):
        """Get statistics information
        
        :param actual_elapsed_time: Actual test duration for accurate QPS calculation
        """
        with self.lock:
            if not self.latencies:
                return {
                    'total_queries': self.total_queries,
                    'failures': self.total_failures,
                    'success_rate': 100.0 if self.total_queries == 0 else (self.total_queries - self.total_failures) / self.total_queries * 100,
                    'qps': 0,
                    'avg_latency': 0,
                    'p95_latency': 0,
                    'p99_latency': 0,
                    'min_latency': 0,
                    'max_latency': 0
                }
            
            # Prefer provided actual duration, otherwise use internal calculated time
            if actual_elapsed_time is not None:
                elapsed_time = actual_elapsed_time
            else:
                elapsed_time = time.time() - self.start_time
            
            latency_array = np.array(self.latencies)
            
            return {
                'total_queries': self.total_queries,
                'failures': self.total_failures,
                'success_rate': (self.total_queries - self.total_failures) / max(self.total_queries, 1) * 100,
                'qps': self.total_queries / max(elapsed_time, 0.001),  # Use actual duration to calculate QPS
                'avg_latency': float(np.mean(latency_array)),
                'p95_latency': float(np.percentile(latency_array, 95)),
                'p99_latency': float(np.percentile(latency_array, 99)),
                'min_latency': float(np.min(latency_array)),
                'max_latency': float(np.max(latency_array))
            }
    
    def reset_samples(self):
        """Reset sample data (keep total counts)"""
        with self.lock:
            self.latencies.clear()


def generate_random_expression(base_expr):
    """Generate random query expression"""
    keywords = ["con%", "%nt", "%con%", "%content%", "%co%nt", "%con_ent%", "%co%nt%"]
    keyword = random.choice(keywords)
    return f'content like "{keyword}"'


def single_query_task(client, collection_name, base_expr, output_fields, limit, each_query_timeout=10):
    """
    Single query task - directly use shared MilvusClient
    
    Note: Relies on MilvusClient thread safety and Milvus server-side connection reuse
    """
    start_time = time.time()
    
    try:
        current_expr = generate_random_expression(base_expr)
        
        # Directly use shared client instance
        result = client.query(
            collection_name=collection_name,
            filter=current_expr,
            output_fields=output_fields,
            limit=limit,
            timeout=each_query_timeout
        )
        
        latency = time.time() - start_time
        return {
            'success': True,
            'latency': latency,
            'result_count': len(result) if result else 0,
            'expression': current_expr
        }
    
    except Exception as e:
        latency = time.time() - start_time
        return {
            'success': False,
            'latency': latency,
            'error': str(e),
            'expression': current_expr
        }




def query_permanently_simplified(client, collection_name, max_workers, 
                                output_fields, expr, timeout, limit=100, each_query_timeout=10):
    """
    Simplified version of continuous query testing - single thread pool directly controls concurrency
    
    :param client: Single shared MilvusClient instance
    :param max_workers: Directly control concurrent query count
    """
    stats = OptimizedStats()
    end_time = time.time() + timeout
    
    # Log control variables
    last_logged_milestone = 0
    log_interval = min(max_workers * 100, 1000)
    
    # Single thread pool, directly manage all query tasks
    with ThreadPoolExecutor(max_workers=max_workers, 
                           thread_name_prefix="QueryWorker") as executor:
        
        # Continuously submit query tasks until timeout
        submitted_tasks = 0
        pending_futures = set()
        
        while time.time() < end_time:
            current_time = time.time()
            remaining_time = end_time - current_time
            
            if remaining_time <= 0:
                break
            
            # Control number of pending tasks，to avoid infinite memory growth
            max_pending = min(max_workers * 2, 50)
            
            # Submit new tasks（if there's space）
            while len(pending_futures) < max_pending and time.time() < end_time:
                future = executor.submit(
                    single_query_task,
                    client, collection_name, expr, output_fields, limit, each_query_timeout
                )
                pending_futures.add(future)
                submitted_tasks += 1
            
            # Collect completed tasks
            completed_futures = set()
            for future in list(pending_futures):
                if future.done():
                    try:
                        result = future.result(timeout=0.1)
                        stats.record_query(result['latency'], result['success'])
                        completed_futures.add(future)
                    except Exception as e:
                        logging.warning(f"Task failed: {e}")
                        stats.record_query(0.1, False)
                        completed_futures.add(future)
            
            # Remove completed tasks
            pending_futures -= completed_futures
            
            # Periodically output statistics - avoid duplicate printing
            if submitted_tasks >= last_logged_milestone + log_interval:
                current_stats = stats.get_stats()
                logging.info(
                    f"Progress: {submitted_tasks} submitted, "
                    f"{len(pending_futures)} pending | "
                    f"QPS: {current_stats['qps']:.1f}, "
                    f"Avg: {current_stats['avg_latency']:.3f}s, "
                    f"P99: {current_stats['p99_latency']:.3f}s, "
                    f"Success: {current_stats['success_rate']:.1f}%"
                )
                last_logged_milestone = submitted_tasks
                
                # Reset sample data after an interval
                stats.reset_samples()
            
            # # Short break to avoid CPU overload
            # time.sleep(0.001)
        
        # Wait for all remaining tasks to complete
        logging.info(f"Waiting for {len(pending_futures)} remaining tasks to complete...")
        for future in as_completed(pending_futures, timeout=30):
            try:
                result = future.result(timeout=1.0)
                stats.record_query(result['latency'], result['success'])
            except Exception as e:
                logging.warning(f"Final task failed: {e}")
                stats.record_query(0.1, False)
    
    # Final statistics - use actual test time to calculate accurate QPS
    actual_test_time = time.time() - stats.start_time
    final_stats = stats.get_stats(actual_elapsed_time=actual_test_time)
    
    logging.info("=" * 80)
    logging.info("FINAL PERFORMANCE STATISTICS (ULTRA-SIMPLIFIED):")
    logging.info(f"  Actual Test Duration: {actual_test_time:.2f}s")
    logging.info(f"  Total Queries: {final_stats['total_queries']}")
    logging.info(f"  Total Failures: {final_stats['failures']}")
    logging.info(f"  Success Rate: {final_stats['success_rate']:.2f}%")
    logging.info(f"  Overall QPS: {final_stats['qps']:.2f} (total queries ÷ actual duration)")
    logging.info(f"  Average Latency: {final_stats['avg_latency']:.3f}s")
    logging.info(f"  P95 Latency: {final_stats['p95_latency']:.3f}s")
    logging.info(f"  P99 Latency: {final_stats['p99_latency']:.3f}s")
    logging.info("=" * 80)
    
    return final_stats


def verify_collection_setup(client, collection_name):
    """Verify collection setup"""
    if not client.has_collection(collection_name=collection_name):
        logging.error(f"Collection {collection_name} does not exist")
        return False
            
    # Check if collection is loaded
    load_state = client.get_load_state(collection_name=collection_name)
    if load_state.get('state') != 'Loaded':
        logging.info(f"Loading collection {collection_name}...")
        client.load_collection(collection_name=collection_name)
        logging.info(f"Collection {collection_name} loaded successfully")
       
    return True


if __name__ == '__main__':
    try:
        host = sys.argv[1]
        name = str(sys.argv[2])
        max_workers = int(sys.argv[3])
        timeout = int(sys.argv[4])
        output_fields = str(sys.argv[5]).strip()
        expr = str(sys.argv[6]).strip()
        limit = int(sys.argv[7])
        api_key = str(sys.argv[8])
    except Exception as e:
        logging.error(f"Failed to get command line arguments: {e}")
        print("Usage: python3 query_permanently_simplified.py <host> <collection> <max_workers> <timeout> <output_fields> <expression> <limit> <api_key>")
        print("Parameters:")
        print("  host             : Milvus server host")
        print("  collection       : Collection name")
        print("  max_workers      : Concurrent query count (direct control)")
        print("  timeout          : Test timeout in seconds")
        print("  output_fields    : Fields to return (comma-separated or '*')")
        print("  expression       : Query filter expression")
        print("  limit            : Query limit")
        print("  api_key          : API key (or 'None' for local)")
        print()
        print("Examples:")
        print("  # 4 concurrent queries")
        print("  python3 query_permanently_simplified.py localhost test_collection 4 60 'id' 'id>0' 100 None")
        print()
        print("  # 16 concurrent queries (high concurrency)")
        print("  python3 query_permanently_simplified.py localhost test_collection 16 60 'id' 'id>0' 100 None")
        print()
        print("🚀 Ultra-Simplified Architecture:")
        print("  ✅ Single shared MilvusClient")
        print("  ✅ Single ThreadPoolExecutor") 
        print("  ✅ max_workers = concurrent query count")
        print("  ✅ No connection pool, no layering, simplest")
        print("  ✅ Rely on Milvus server-side connection reuse")
        sys.exit(1)
    

    port = 19530
    
    # Parameter processing
    if timeout <= 0:
        timeout = 2 * 3600
    
    if output_fields in ["None", "none", "NONE"] or output_fields == "":
        output_fields = ["*"]
    else:
        output_fields = output_fields.split(",")
    
    if expr in ["None", "none", "NONE"] or expr == "":
        expr = "None"
    
    if limit <= 0:
        limit = 100
    
    each_query_timeout = 10
        
    # Setup logging
    log_filename = f"/tmp/query_ultra_simplified_{name}_{int(time.time())}.log"
    file_handler = logging.FileHandler(filename=log_filename)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    
    logging.info("🚀 Starting ULTRA-SIMPLIFIED query_permanently test:")
    logging.info(f"  Host: {host}")
    logging.info(f"  Collection: {name}")
    logging.info(f"  Max Workers: {max_workers} (= concurrent query count)")
    logging.info(f"  THe Whole Test Timeout: {timeout}s")
    logging.info(f"  Output Fields: {output_fields}")
    logging.info(f"  Expression: {expr}")
    logging.info(f"  Limit: {limit}")
    logging.info(f"  Each Query Timeout: {each_query_timeout}s")

    # Create single shared client - key simplification!
    try:
        if api_key is None or api_key == "" or api_key.upper() == "NONE":
            client = MilvusClient(uri=f"http://{host}:{port}")
        else:
            client = MilvusClient(uri=host, token=api_key)
        
        logging.info(f"✅ Created single shared MilvusClient for {host}")
        
        # Verify collection
        if not verify_collection_setup(client, name):
            logging.error(f"Collection '{name}' setup verification failed")
            sys.exit(1)
        
    except Exception as e:
        logging.error(f"Failed to create MilvusClient: {e}")
        sys.exit(1)
    
    # Run simplified query test
    start_time = time.time()
    final_stats = query_permanently_simplified(
        client=client,  # Pass single client
        collection_name=name,
        max_workers=max_workers,  # Direct concurrency control, no layering
        output_fields=output_fields,
        expr=expr,
        timeout=timeout,
        limit=limit,
        each_query_timeout=each_query_timeout
    )
    end_time = time.time()
    
    actual_duration = end_time - start_time
    # Recalculate accurate final QPS
    accurate_qps = final_stats['total_queries'] / max(actual_duration, 0.001)
    
    logging.info(f"✅ Simplified query test completed in {actual_duration:.2f} seconds")
    logging.info(f"📊 Accurate Final QPS: {accurate_qps:.2f} ({final_stats['total_queries']} queries ÷ {actual_duration:.2f}s)")
    logging.info(f"📁 Log file: {log_filename}")
   
