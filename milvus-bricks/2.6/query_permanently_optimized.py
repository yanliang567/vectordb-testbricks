#!/usr/bin/env python3
"""
优化版本的多线程查询test脚本

主要优化:
1. 连接池管理
2. ThreadPoolExecutor替代原生threading
3. 优化的统计系统
4. 减少锁竞争
5. 批量handle
"""

import time
import sys
import random
import numpy as np
import logging
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from collections import deque
from pymilvus import MilvusClient, DataType

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"


class MilvusClientPool:
    """Milvus客户端连接池"""
    
    def __init__(self, uri, token=None, pool_size=10):
        self.uri = uri
        self.token = token
        self.pool = queue.Queue(maxsize=pool_size)
        self.pool_size = pool_size
        
        # 初始化连接池
        for i in range(pool_size):
            client = MilvusClient(uri=uri, token=token, alias=f"client_{i}")
            self.pool.put(client)
        
        logging.info(f"Created Milvus client pool with {pool_size} connections")
    
    def get_client(self):
        """获取客户端连接"""
        return self.pool.get()
    
    def return_client(self, client):
        """归还客户端连接"""
        self.pool.put(client)
    
    def close_all(self):
        """closed所有连接"""
        while not self.pool.empty():
            try:
                client = self.pool.get_nowait()
                # MilvusClient没有显式的close方法，让GChandle
                client.close()
            except queue.Empty:
                break


class OptimizedStats:
    """优化的统计系统"""
    
    def __init__(self, max_samples=1000):
        self.latencies = deque(maxlen=max_samples)  # 限制内存使用
        self.total_queries = 0
        self.total_failures = 0
        self.start_time = time.time()
        self.lock = Lock()
    
    def record_query(self, latency, success=True):
        """记录查询结果"""
        with self.lock:
            self.latencies.append(latency)
            self.total_queries += 1
            if not success:
                self.total_failures += 1
    
    def get_stats(self):
        """获取统计信息"""
        with self.lock:
            if not self.latencies:
                return {
                    'total_queries': 0,
                    'failures': 0,
                    'qps': 0,
                    'avg_latency': 0,
                    'p99_latency': 0
                }
            
            elapsed_time = time.time() - self.start_time
            latency_array = np.array(self.latencies)
            
            return {
                'total_queries': self.total_queries,
                'failures': self.total_failures,
                'success_rate': (self.total_queries - self.total_failures) / max(self.total_queries, 1) * 100,
                'qps': self.total_queries / max(elapsed_time, 0.001),
                'avg_latency': float(np.mean(latency_array)),
                'p95_latency': float(np.percentile(latency_array, 95)),
                'p99_latency': float(np.percentile(latency_array, 99)),
                'min_latency': float(np.min(latency_array)),
                'max_latency': float(np.max(latency_array))
            }
    
    def reset_samples(self):
        """Reset sample data（保留总计数）"""
        with self.lock:
            self.latencies.clear()


def generate_random_expression(base_expr):
    """Generate random query expression"""

    keywords = ["con%", "%nt", "%con%", "%content%", "%co%nt", "%con_ent%", "%co%nt%"]
    keyword = random.choice(keywords)
    return f'content like "{keyword}"'


def single_query_task(client_pool, collection_name, base_expr, output_fields, limit, timeout=60):
    """单个查询任务"""
    client = None
    start_time = time.time()
    
    try:
        client = client_pool.get_client()
        current_expr = generate_random_expression(base_expr)
        
        result = client.query(
            collection_name=collection_name,
            filter=current_expr,
            output_fields=output_fields,
            limit=limit,
            timeout=timeout
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
    
    finally:
        if client:
            client_pool.return_client(client)


def query_permanently_optimized(client_pool, collection_name, max_workers, 
                               output_fields, expr, timeout, batch_size=100, limit=None):
    """
    优化版本的持续查询test
    
    :param client_pool: MilvusClientPool实例
    :param collection_name: collection name称
    :param max_workers: 最大工作线程数
    :param output_fields: Output Fields
    :param expr: 查询Expression
    :param timeout: 超时时间
    :param batch_size: 批handle大小
    :param limit: 查询限制
    """
    stats = OptimizedStats()
    end_time = time.time() + timeout
    total_batches = 0
    
    logging.info(f"Starting optimized query test with {max_workers} workers, batch_size={batch_size}")
    
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="QueryWorker") as executor:
        
        while time.time() < end_time:
            # 双重检查时间，确保严格控制
            current_time = time.time()
            if current_time >= end_time:
                logging.info("Timeout reached, exiting main loop")
                break
            batch_start_time = time.time()
            
            # 提交批量任务
            futures = []
            remaining_time = end_time - time.time()
            
            if remaining_time <= 0:
                break
            
            # 动态调整批次大小
            current_batch_size = min(batch_size, max(10, int(remaining_time * 10)))
            
            for _ in range(current_batch_size):
                if time.time() >= end_time:
                    break
                
                future = executor.submit(
                    single_query_task,
                    client_pool, collection_name, expr, output_fields, limit
                )
                futures.append(future)
            
            # 收集批次结果 - 严格控制timeout
            batch_results = []
            completed_count = 0
            
            try:
                for future in as_completed(futures, timeout=min(remaining_time, 60)):  # 最多等待60秒
                    # 再次检查时间，确保严格控制
                    if time.time() >= end_time:
                        logging.info(f"Timeout reached, cancelling remaining {len(futures) - completed_count} futures")
                        # 取消剩余的future
                        for f in futures[completed_count:]:
                            f.cancel()
                        break
                
                    try:
                        result = future.result(timeout=60.0)  # 单个任务最多等待60秒
                        batch_results.append(result)
                        stats.record_query(result['latency'], result['success'])
                        completed_count += 1
                    except Exception as e:
                        logging.warning(f"Future execution failed: {e}")
                        stats.record_query(0.1, False)  # 记录Failed to
                        completed_count += 1
                        
            except Exception as e:
                # handle as_completed 的超时异常
                logging.warning(f"Batch timeout or other error: {e}")
                # 取消所有未completed的future
                for f in futures:
                    if not f.done():
                        f.cancel()
                break  # 退出主循环
            
            total_batches += 1
            batch_duration = time.time() - batch_start_time
            
            # Periodically output statistics
            logging_batch = 10
            if total_batches % logging_batch == 0:  # 每10个批次输出一次
                current_stats = stats.get_stats()
                logging.info(
                    f"Batch {total_batches}: {len(batch_results)} queries in {batch_duration:.2f}s, "
                    f"QPS: {current_stats['qps']:.1f}, "
                    f"Avg: {current_stats['avg_latency']:.3f}s, "
                    f"P99: {current_stats['p99_latency']:.3f}s, "
                    f"Success Rate: {current_stats['success_rate']:.1f}%, "
                    f"Total: {current_stats['total_queries']}"
                )
                
                # Reset sample data以to avoid infinite memory growth
                if total_batches % (logging_batch * 100) == 0:
                    stats.reset_samples()
    
    # Final statistics
    final_stats = stats.get_stats()
    logging.info("=" * 80)
    logging.info("FINAL PERFORMANCE STATISTICS:")
    logging.info(f"  Total Queries: {final_stats['total_queries']}")
    logging.info(f"  Total Failures: {final_stats['failures']}")
    logging.info(f"  Success Rate: {final_stats['success_rate']:.2f}%")
    logging.info(f"  Overall QPS: {final_stats['qps']:.2f}")
    logging.info(f"  Average Latency: {final_stats['avg_latency']:.3f}s")
    logging.info(f"  P95 Latency: {final_stats['p95_latency']:.3f}s")
    logging.info(f"  P99 Latency: {final_stats['p99_latency']:.3f}s")
    logging.info(f"  Min Latency: {final_stats['min_latency']:.3f}s")
    logging.info(f"  Max Latency: {final_stats['max_latency']:.3f}s")
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
    if len(sys.argv) not in [10]:
        print("Usage: python3 query_permanently_optimized.py <host> <collection> <threads> <timeout> <output_fields> <expression> <api_key> [batch_size]")
        print("Parameters:")
        print("  host         : Milvus server host")
        print("  collection   : Collection name")
        print("  threads      : Number of concurrent threads/workers")
        print("  timeout      : Test timeout in seconds")
        print("  output_fields: Fields to return (comma-separated or '*')")
        print("  expression   : Query filter expression")
        print("  limit        : Query limit")
        print("  batch_size   : Batch size for task submission")
        print("  api_key      : API key (or 'None' for local)")
        print()
        print("Examples:")
        print("  python3.11 query_permanently_optimized.py 10.104.33.161 test_aa 1 3600 'id,content' 'dd' 50 2 None")
        print("actual len of argv is: ", len(sys.argv), "argv is: ", sys.argv)
        sys.exit(1)
    
    host = sys.argv[1]
    name = str(sys.argv[2])
    max_workers = int(sys.argv[3])
    timeout = int(sys.argv[4])
    output_fields = str(sys.argv[5]).strip()
    expr = str(sys.argv[6]).strip()
    limit = int(sys.argv[7])
    batch_size = int(sys.argv[8]) 
    api_key = str(sys.argv[9])

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
    if limit in [None, "None", "none", "NONE"] or limit == "":
        limit = None
    
    # Setup logging
    log_filename = f"/tmp/query_optimized_{name}_{int(time.time())}.log"
    file_handler = logging.FileHandler(filename=log_filename)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    
    logging.info("🚀 Starting OPTIMIZED query_permanently test:")
    logging.info(f"  Host: {host}")
    logging.info(f"  Collection: {name}")
    logging.info(f"  Max Workers: {max_workers}")
    logging.info(f"  Timeout: {timeout}s")
    logging.info(f"  Output Fields: {output_fields}")
    logging.info(f"  Expression: {expr}")
    logging.info(f"  Limit: {limit}")
    logging.info(f"  Batch Size: {batch_size}")
    logging.info(f"  Connection Pool Size: {max_workers}")

    # Created客户端连接池
    try:
        pool_size = max_workers  
        if api_key is None or api_key == "" or api_key.upper() == "NONE":
            client_pool = MilvusClientPool(uri=f"http://{host}:{port}", pool_size=pool_size)
        else:
            client_pool = MilvusClientPool(uri=host, token=api_key, pool_size=pool_size)
        
        # Verify collection
        test_client = client_pool.get_client()
        if not verify_collection_setup(test_client, name):
            logging.error(f"Collection '{name}' setup verification failed")
            sys.exit(1)
        client_pool.return_client(test_client)
        
    except Exception as e:
        logging.error(f"Failed to create client pool: {e}")
        sys.exit(1)
    
    # 运行优化的查询test
    try:
        start_time = time.time()
        final_stats = query_permanently_optimized(
            client_pool=client_pool,
            collection_name=name,
            max_workers=max_workers,
            output_fields=output_fields,
            expr=expr,
            timeout=timeout,
            batch_size=batch_size,
            limit=limit
        )
        end_time = time.time()
        
        logging.info(f"✅ Optimized query test completed in {end_time - start_time:.2f} seconds")
        logging.info(f"📊 Final QPS: {final_stats['qps']:.2f}")
        logging.info(f"📁 Log file: {log_filename}")
        
    except KeyboardInterrupt:
        logging.info("⚠️ Query test interrupted by user")
    except Exception as e:
        logging.error(f"❌ Query test failed: {e}")
        raise
    finally:
        # 清理连接池
        try:
            client_pool.close_all()
            logging.info("🔌 Client pool connections closed")
        except Exception as e:
            logging.warning(f"Failed to close client pool: {e}")
