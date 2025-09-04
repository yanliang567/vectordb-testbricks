#!/usr/bin/env python3
"""
超级简化版并发查询测试 - 单线程池架构

关键简化:
1. ✅ 移除客户端连接池，使用单个 MilvusClient 实例
2. ✅ 移除双层线程池（BatchController + QueryTasks）
3. ✅ max_workers 直接等于并发查询数量
4. ✅ 单个 ThreadPoolExecutor 直接管理所有查询任务
5. ✅ 依赖 Milvus 服务端连接复用，无分层复杂性
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




def query_permanently_simplified(client, collection_name, max_workers, 
                                output_fields, expr, timeout, limit=100):
    """
    简化版本的持续查询测试 - 单线程池直接控制并发
    
    :param client: 单个共享的 MilvusClient 实例
    :param max_workers: 直接控制并发查询数量
    """
    stats = OptimizedStats()
    end_time = time.time() + timeout
    
    logging.info(f"Starting ULTRA-SIMPLIFIED query test:")
    logging.info(f"  Max Workers: {max_workers} (直接控制并发查询数)")
    logging.info(f"  架构: 单 MilvusClient + 单 ThreadPoolExecutor")
    logging.info(f"  无连接池，无批次分层，最简架构")
    
    # 单一线程池，直接管理所有查询任务
    with ThreadPoolExecutor(max_workers=max_workers, 
                           thread_name_prefix="QueryWorker") as executor:
        
        # 持续提交查询任务直到超时
        submitted_tasks = 0
        pending_futures = set()
        
        while time.time() < end_time:
            current_time = time.time()
            remaining_time = end_time - current_time
            
            if remaining_time <= 0:
                break
            
            # 控制未完成任务数量，避免内存无限增长
            max_pending = max_workers * 2  # 允许一些缓冲
            
            # 提交新任务（如果有空间）
            while len(pending_futures) < max_pending and time.time() < end_time:
                future = executor.submit(
                    single_query_task,
                    client, collection_name, expr, output_fields, limit
                )
                pending_futures.add(future)
                submitted_tasks += 1
            
            # 收集已完成的任务
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
            
            # 移除已完成的任务
            pending_futures -= completed_futures
            
            # 定期输出统计信息
            if submitted_tasks % (max_workers * 10) == 0:
                current_stats = stats.get_stats()
                logging.info(
                    f"Submitted: {submitted_tasks}, "
                    f"Pending: {len(pending_futures)}, "
                    f"QPS: {current_stats['qps']:.1f}, "
                    f"Avg: {current_stats['avg_latency']:.3f}s, "
                    f"P99: {current_stats['p99_latency']:.3f}s, "
                    f"Success Rate: {current_stats['success_rate']:.1f}%"
                )
                
                # 重置样本数据
                if submitted_tasks % (max_workers * 1000) == 0:
                    stats.reset_samples()
            
            # 短暂休息，避免CPU过载
            time.sleep(0.001)
        
        # 等待所有剩余任务完成
        logging.info(f"Waiting for {len(pending_futures)} remaining tasks to complete...")
        for future in as_completed(pending_futures, timeout=30):
            try:
                result = future.result(timeout=1.0)
                stats.record_query(result['latency'], result['success'])
            except Exception as e:
                logging.warning(f"Final task failed: {e}")
                stats.record_query(0.1, False)
    
    # 最终统计
    final_stats = stats.get_stats()
    logging.info("=" * 80)
    logging.info("FINAL PERFORMANCE STATISTICS (ULTRA-SIMPLIFIED):")
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
    if len(sys.argv) != 9:
        print("Usage: python3 query_permanently_simplified.py <host> <collection> <max_workers> <timeout> <output_fields> <expression> <limit> <api_key>")
        print("Parameters:")
        print("  host             : Milvus server host")
        print("  collection       : Collection name")
        print("  max_workers      : 并发查询数量 (直接控制)")
        print("  timeout          : Test timeout in seconds")
        print("  output_fields    : Fields to return (comma-separated or '*')")
        print("  expression       : Query filter expression")
        print("  limit            : Query limit")
        print("  api_key          : API key (or 'None' for local)")
        print()
        print("Examples:")
        print("  # 4 个并发查询")
        print("  python3 query_permanently_simplified.py localhost test_collection 4 60 'id' 'id>0' 100 None")
        print()
        print("  # 16 个并发查询 (高并发)")
        print("  python3 query_permanently_simplified.py localhost test_collection 16 60 'id' 'id>0' 100 None")
        print()
        print("🚀 超级简化架构:")
        print("  ✅ 单个共享 MilvusClient")
        print("  ✅ 单个 ThreadPoolExecutor") 
        print("  ✅ max_workers = 并发查询数")
        print("  ✅ 无连接池，无分层，最简单")
        print("  ✅ 依赖 Milvus 服务端连接复用")
        sys.exit(1)
    
    host = sys.argv[1]
    name = str(sys.argv[2])
    max_workers = int(sys.argv[3])
    timeout = int(sys.argv[4])
    output_fields = str(sys.argv[5]).strip()
    expr = str(sys.argv[6]).strip()
    limit = int(sys.argv[7])
    api_key = str(sys.argv[8])
    

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
    
    if limit <= 0:
        limit = 100
        
    # 设置日志
    log_filename = f"/tmp/query_ultra_simplified_{name}_{int(time.time())}.log"
    file_handler = logging.FileHandler(filename=log_filename)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    
    logging.info("🚀 Starting ULTRA-SIMPLIFIED query_permanently test:")
    logging.info(f"  Host: {host}")
    logging.info(f"  Collection: {name}")
    logging.info(f"  Max Workers: {max_workers} (= 并发查询数)")
    logging.info(f"  Timeout: {timeout}s")
    logging.info(f"  Output Fields: {output_fields}")
    logging.info(f"  Expression: {expr}")
    logging.info(f"  Limit: {limit}")
    logging.info(f"  架构: 单客户端 + 单线程池，最简单!")

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
            client=client,  # 传递单个客户端
            collection_name=name,
            max_workers=max_workers,  # 直接控制并发数，无分层
            output_fields=output_fields,
            expr=expr,
            timeout=timeout,
            limit=limit
        )
        end_time = time.time()
        
        logging.info(f"✅ Ultra-simplified query test completed in {end_time - start_time:.2f} seconds")
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
