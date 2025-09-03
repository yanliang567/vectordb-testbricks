#!/usr/bin/env python3
"""
简化版并发查询测试 - 移除客户端连接池

关键简化:
1. 移除 MilvusClientPool，使用单个 MilvusClient 实例
2. 依赖 Milvus 服务端的连接复用机制
3. 减少代码复杂度和内存开销
4. 保持并发查询能力
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
    """优化的统计系统"""
    
    def __init__(self, max_samples=1000):
        self.latencies = deque(maxlen=max_samples)
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
        """重置样本数据（保留总计数）"""
        with self.lock:
            self.latencies.clear()


def generate_random_expression(base_expr):
    """生成随机查询表达式"""
    keywords = ["con%", "%nt", "%con%", "%content%", "%co%nt", "%con_ent%", "%co%nt%"]
    keyword = random.choice(keywords)
    return f'content like "{keyword}"'


def single_query_task(client, collection_name, base_expr, output_fields, limit, timeout=60):
    """
    单个查询任务 - 直接使用共享的 MilvusClient
    
    注意: 依赖 MilvusClient 的线程安全性和 Milvus 服务端连接复用
    """
    start_time = time.time()
    
    try:
        current_expr = generate_random_expression(base_expr)
        
        # 直接使用共享的客户端实例
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


def execute_concurrent_batch(client, collection_name, expr, output_fields, 
                           batch_size, batch_concurrency, limit):
    """
    执行并发批次 - 使用共享客户端
    """
    with ThreadPoolExecutor(max_workers=batch_concurrency, 
                           thread_name_prefix=f"BatchWorker") as batch_executor:
        
        # 提交批次内的所有任务
        futures = []
        for _ in range(batch_size):
            # 所有线程共享同一个 client 实例
            future = batch_executor.submit(
                single_query_task,
                client, collection_name, expr, output_fields, limit
            )
            futures.append(future)
        
        # 收集结果
        batch_results = []
        for future in as_completed(futures, timeout=60):
            try:
                result = future.result(timeout=60.0)
                batch_results.append(result)
            except Exception as e:
                logging.warning(f"Batch task failed: {e}")
                batch_results.append({
                    'success': False,
                    'latency': 0.1,
                    'error': str(e)
                })
        
        return batch_results


def query_permanently_simplified(client, collection_name, max_workers, 
                                output_fields, expr, timeout, batch_size=100, 
                                batch_concurrency=None, limit=None):
    """
    简化版本的持续查询测试 - 无连接池
    
    :param client: 单个共享的 MilvusClient 实例
    """
    stats = OptimizedStats()
    end_time = time.time() + timeout
    total_batches = 0
    
    # 如果未指定批次并发数，默认等于batch_size
    if batch_concurrency is None:
        batch_concurrency = min(batch_size, 20)  # 合理限制，避免过多线程
    
    logging.info(f"Starting simplified query test:")
    logging.info(f"  Max Workers: {max_workers} (批次控制)")
    logging.info(f"  Batch Size: {batch_size} (每批次任务数)")
    logging.info(f"  Batch Concurrency: {batch_concurrency} (批次内部并发)")
    logging.info(f"  Client: Shared single MilvusClient instance")
    
    # 主控制循环
    with ThreadPoolExecutor(max_workers=max_workers, 
                           thread_name_prefix="BatchController") as main_executor:
        
        while time.time() < end_time:
            if time.time() >= end_time:
                logging.info("Timeout reached, exiting main loop")
                break
            
            batch_start_time = time.time()
            remaining_time = end_time - time.time()
            
            if remaining_time <= 0:
                break
            
            # 动态调整批次大小
            current_batch_size = min(batch_size, max(1, int(remaining_time * 10)))
            
            # 提交批次执行任务
            batch_future = main_executor.submit(
                execute_concurrent_batch,
                client, collection_name, expr, output_fields,
                current_batch_size, batch_concurrency, limit
            )
            
            try:
                # 等待批次完成
                batch_results = batch_future.result(timeout=min(remaining_time, 60))
                
                # 记录统计
                for result in batch_results:
                    stats.record_query(result['latency'], result['success'])
                
                total_batches += 1
                batch_duration = time.time() - batch_start_time
                
                # 定期输出统计信息
                logging_batch = 10
                if total_batches % logging_batch == 0:
                    current_stats = stats.get_stats()
                    logging.info(
                        f"Batch {total_batches}: {len(batch_results)} queries in {batch_duration:.2f}s, "
                        f"QPS: {current_stats['qps']:.1f}, "
                        f"Avg: {current_stats['avg_latency']:.3f}s, "
                        f"P99: {current_stats['p99_latency']:.3f}s, "
                        f"Success Rate: {current_stats['success_rate']:.1f}%, "
                        f"Total: {current_stats['total_queries']}"
                    )
                    
                    # 重置样本数据以避免内存无限增长
                    if total_batches % (logging_batch * 100) == 0:
                        stats.reset_samples()
                        
            except Exception as e:
                logging.warning(f"Batch execution failed: {e}")
                break
    
    # 最终统计
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
    logging.info("=" * 80)
    
    return final_stats


def verify_collection_setup(client, collection_name):
    """验证集合设置"""
    if not client.has_collection(collection_name=collection_name):
        logging.error(f"Collection {collection_name} does not exist")
        return False
            
    # 检查集合是否已加载
    load_state = client.get_load_state(collection_name=collection_name)
    if load_state.get('state') != 'Loaded':
        logging.info(f"Loading collection {collection_name}...")
        client.load_collection(collection_name=collection_name)
        logging.info(f"Collection {collection_name} loaded successfully")
       
    return True


if __name__ == '__main__':
    if len(sys.argv) not in [11]:
        print("Usage: python3 query_permanently_simplified.py <host> <collection> <max_workers> <timeout> <output_fields> <expression> <api_key> <batch_size> [batch_concurrency]")
        print("Parameters:")
        print("  host             : Milvus server host")
        print("  collection       : Collection name")
        print("  max_workers      : Maximum concurrent batches")
        print("  timeout          : Test timeout in seconds")
        print("  output_fields    : Fields to return (comma-separated or '*')")
        print("  expression       : Query filter expression")
        print("  limit            : Query limit")
        print("  api_key          : API key (or 'None' for local)")
        print("  batch_size       : Number of queries per batch")
        print("  batch_concurrency: Concurrent threads within each batch (optional)")
        print()
        print("Examples:")
        print("  # 基础使用 - 单客户端，无连接池")
        print("  python3 query_permanently_simplified.py localhost test_collection 1 60 'id' 'id>0' None 50 5")
        print()
        print("  # 批次内并发")
        print("  python3 query_permanently_simplified.py localhost test_collection 1 60 'id' 'id>0' None 50 10 5")
        print()
        print("  # 高性能配置")
        print("  python3 query_permanently_simplified.py localhost test_collection 2 60 'id' 'id>0' None 50 20 10")
        print()
        print("🔧 关键改进:")
        print("  ✅ 移除了客户端连接池")
        print("  ✅ 使用单个共享的 MilvusClient 实例")
        print("  ✅ 依赖 Milvus 服务端连接复用")
        print("  ✅ 显著降低内存使用和代码复杂度")
        sys.exit(1)
    
    host = sys.argv[1]
    name = str(sys.argv[2])
    max_workers = int(sys.argv[3])
    timeout = int(sys.argv[4])
    output_fields = str(sys.argv[5]).strip()
    expr = str(sys.argv[6]).strip()
    limit = int(sys.argv[7])
    api_key = str(sys.argv[8])
    batch_size = int(sys.argv[9])
    batch_concurrency = int(sys.argv[10])

    port = 19530
    
    # 参数处理
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
        
    # 设置日志
    log_filename = f"/tmp/query_simplified_{name}_{int(time.time())}.log"
    file_handler = logging.FileHandler(filename=log_filename)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    
    logging.info("🚀 Starting SIMPLIFIED query_permanently test:")
    logging.info(f"  Host: {host}")
    logging.info(f"  Collection: {name}")
    logging.info(f"  Max Workers: {max_workers}")
    logging.info(f"  Timeout: {timeout}s")
    logging.info(f"  Output Fields: {output_fields}")
    logging.info(f"  Expression: {expr}")
    logging.info(f"  Limit: {limit}")
    logging.info(f"  Batch Size: {batch_size}")
    logging.info(f"  Batch Concurrency: {batch_concurrency or 'auto'}")

    # 创建单个共享客户端 - 关键简化！
    try:
        if api_key is None or api_key == "" or api_key.upper() == "NONE":
            client = MilvusClient(uri=f"http://{host}:{port}")
        else:
            client = MilvusClient(uri=host, token=api_key)
        
        logging.info(f"✅ Created single shared MilvusClient for {host}")
        
        # 验证集合
        if not verify_collection_setup(client, name):
            logging.error(f"Collection '{name}' setup verification failed")
            sys.exit(1)
        
    except Exception as e:
        logging.error(f"Failed to create MilvusClient: {e}")
        sys.exit(1)
    
    # 运行简化的查询测试
    try:
        start_time = time.time()
        final_stats = query_permanently_simplified(
            client=client,  # 传递单个客户端而不是连接池
            collection_name=name,
            max_workers=max_workers,
            output_fields=output_fields,
            expr=expr,
            timeout=timeout,
            batch_size=batch_size,
            batch_concurrency=batch_concurrency,
            limit=limit
        )
        end_time = time.time()
        
        logging.info(f"✅ Simplified query test completed in {end_time - start_time:.2f} seconds")
        logging.info(f"📊 Final QPS: {final_stats['qps']:.2f}")
        logging.info(f"📁 Log file: {log_filename}")
        
    except KeyboardInterrupt:
        logging.info("⚠️ Query test interrupted by user")
    except Exception as e:
        logging.error(f"❌ Query test failed: {e}")
        raise
    finally:
        # 简化的清理：只需要关闭一个客户端
        try:
            if hasattr(client, 'close'):
                client.close()
            logging.info("🔌 MilvusClient closed")
        except Exception as e:
            logging.warning(f"Failed to close client: {e}")
