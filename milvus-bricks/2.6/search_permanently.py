#!/usr/bin/env python3
"""
超级简化版向量搜索测试 - 支持普通搜索和混合搜索

关键简化:
1. ✅ 使用 MilvusClient 替代 connections + Collection
2. ✅ 单线程池架构，max_workers 直接控制并发数
3. ✅ 支持普通搜索和混合搜索模式
4. ✅ 统一的性能统计和日志系统
5. ✅ 依赖 Milvus 服务端连接复用
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

# 导入公共工具函数 - 所有方法现在只需要schema参数
from common import (
    get_float_vec_field_names,
    get_dim_by_field_name,
    get_vector_field_info_from_schema,
    get_primary_field_name
)


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"


class SimpleStats:
    """简化的统计系统"""
    
    def __init__(self, max_samples=1000):
        self.latencies = deque(maxlen=max_samples)
        self.total_searches = 0
        self.total_failures = 0
        self.start_time = time.time()
        self.lock = Lock()
    
    def record_search(self, latency, success=True):
        """记录搜索结果"""
        with self.lock:
            self.latencies.append(latency)
            self.total_searches += 1
            if not success:
                self.total_failures += 1
    
    def get_stats(self, actual_elapsed_time=None):
        """获取统计信息
        
        :param actual_elapsed_time: 实际测试耗时，用于准确计算QPS
        """
        with self.lock:
            if not self.latencies:
                return {
                    'total_searches': self.total_searches,
                    'failures': self.total_failures,
                    'success_rate': 100.0 if self.total_searches == 0 else (self.total_searches - self.total_failures) / self.total_searches * 100,
                    'qps': 0,
                    'avg_latency': 0,
                    'p95_latency': 0,
                    'p99_latency': 0,
                    'min_latency': 0,
                    'max_latency': 0
                }
            
            # 优先使用传入的实际耗时，否则使用内部计算的时间
            if actual_elapsed_time is not None:
                elapsed_time = actual_elapsed_time
            else:
                elapsed_time = time.time() - self.start_time
            
            latency_array = np.array(self.latencies)
            
            return {
                'total_searches': self.total_searches,
                'failures': self.total_failures,
                'success_rate': (self.total_searches - self.total_failures) / max(self.total_searches, 1) * 100,
                'qps': self.total_searches / max(elapsed_time, 0.001),  # 使用实际耗时计算QPS
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


def get_collection_info(client, collection_name):
    """获取集合信息 - 优化为使用公共方法"""
    try:
        # 获取集合 schema （只调用一次）
        schema = client.describe_collection(collection_name)
        
        # 使用简化的公共方法提取向量字段信息 - 只需传入schema
        vector_fields = get_vector_field_info_from_schema(schema)
        
        # 获取索引信息
        indexes = client.list_indexes(collection_name)
        
        return {
            'schema': schema,
            'vector_fields': vector_fields,
            'indexes': indexes
        }
    except Exception as e:
        logging.error(f"Failed to get collection info: {e}")
        return None


def generate_random_vectors(dim, nq):
    """生成随机向量"""
    return [[random.random() for _ in range(dim)] for _ in range(nq)]


def generate_random_expression():
    """生成随机查询表达式"""
    keywords = ["con%", "%nt", "%con%", "%content%", "%co%nt", "%con_ent%", "%co%nt%"]
    keyword = random.choice(keywords)
    return f'content like "{keyword}"'


def single_search_task(client, collection_name, search_params, timeout=60):
    """
    单个搜索任务 - 支持普通搜索和混合搜索
    
    :param search_params: 包含搜索参数的字典
    """
    start_time = time.time()
    
    try:
        if search_params['search_type'] == 'hybrid':
            # 混合搜索
            result = client.hybrid_search(
                collection_name=collection_name,
                reqs=search_params['reqs'],
                ranker=search_params.get('ranker'),
                limit=search_params['limit'],
                output_fields=search_params.get('output_fields'),
                timeout=timeout
            )
        else:
            # 普通搜索
            result = client.search(
                collection_name=collection_name,
                data=search_params['data'],
                anns_field=search_params['anns_field'],
                search_params=search_params.get('search_params', {}),
                limit=search_params['limit'],
                expr=search_params.get('expr'),
                output_fields=search_params.get('output_fields'),
                group_by_field=search_params.get('group_by_field'),
                partition_names=search_params.get('partition_names'),
                timeout=timeout
            )
        
        latency = time.time() - start_time
        result_count = len(result[0]) if result and len(result) > 0 else 0
        
        return {
            'success': True,
            'latency': latency,
            'result_count': result_count,
            'search_type': search_params['search_type']
        }
    
    except Exception as e:
        latency = time.time() - start_time
        return {
            'success': False,
            'latency': latency,
            'error': str(e),
            'search_type': search_params['search_type']
        }


def create_search_params(search_type, collection_info, vec_field_names, nq, topk, 
                        output_fields, expr, group_by_field):
    """创建搜索参数 - 使用简化的公共方法（只需schema参数）"""
        
    if search_type == 'hybrid':
        # 混合搜索参数
        reqs = []
        for field_name in vec_field_names:
            # 简化调用：只需传入schema和field_name
            dim = get_dim_by_field_name(schema=collection_info, field_name=field_name)
            if not dim:
                logging.warning(f"Vector field {field_name} not found or has no dimension")
                continue
            
            search_vectors = generate_random_vectors(dim, nq)
            
            # 创建 AnnSearchRequest (用字典表示)
            req = {
                "data": search_vectors,
                "anns_field": field_name,
                "param": {},
                "limit": topk,
                "expr": expr
            }
            reqs.append(req)
        
        return {
            'search_type': 'hybrid',
            'reqs': reqs,
            'ranker': {'strategy': 'rrf'},  # RRFRanker
            'limit': topk,
            'output_fields': output_fields
        }
    
    else:
        # 普通搜索参数
        if not vec_field_names:
            # 简化调用：只需传入schema
            all_vector_fields = get_float_vec_field_names(schema=collection_info)
            field_name = all_vector_fields[0] if all_vector_fields else None
        else:
            field_name = vec_field_names[0]
        
        if not field_name:
            raise ValueError("No vector fields found in collection")
        
        # 简化调用：只需传入schema和field_name
        dim = get_dim_by_field_name(schema=collection_info, field_name=field_name)
        if not dim:
            raise ValueError(f"Vector field {field_name} not found or has no dimension")
        
        search_vectors = generate_random_vectors(dim, nq)
        
        return {
            'search_type': 'normal',
            'data': search_vectors,
            'anns_field': field_name,
            'search_params': {"metric_type": "L2", "params": {"nprobe": 10}},
            'limit': topk,
            'expr': expr,
            'output_fields': output_fields,
            'group_by_field': group_by_field,
            'partition_names': None  # 可以后续扩展支持分区
        }


def search_permanently_simplified(client, collection_name, max_workers, search_type,
                                 vec_field_names, nq, topk, output_fields, expr, 
                                 group_by_field, timeout):
    """
    简化版本的持续搜索测试
    
    :param client: 单个共享的 MilvusClient 实例
    :param search_type: 'normal' 或 'hybrid'
    """
    stats = SimpleStats()
    end_time = time.time() + timeout
    
    # 获取集合信息
    collection_info = get_collection_info(client, collection_name)
    if not collection_info:
        raise ValueError(f"Failed to get collection info for {collection_name}")
    
    # 确定向量字段 - 使用简化的公共方法
    if not vec_field_names:
        # 简化调用：只需传入schema
        vec_field_names = get_float_vec_field_names(collection_info['schema'])
    
    if not vec_field_names:
        raise ValueError("No vector fields found in collection")
    
    logging.info(f"Starting SEARCH test:")
    logging.info(f"  Search Type: {search_type}")
    logging.info(f"  Max Workers: {max_workers} (直接控制并发搜索数)")
    logging.info(f"  Vector Fields: {vec_field_names}")
    logging.info(f"  nq: {nq}, topk: {topk}")
    logging.info(f"  Output Fields: {output_fields}")
    logging.info(f"  Expression: {expr}")
    
    # 日志控制变量
    last_logged_milestone = 0
    log_interval = max_workers * 100
    
    # 单一线程池，直接管理所有搜索任务
    with ThreadPoolExecutor(max_workers=max_workers, 
                           thread_name_prefix="SearchWorker") as executor:
        
        # 持续提交搜索任务直到超时
        submitted_tasks = 0
        pending_futures = set()
        
        while time.time() < end_time:
            current_time = time.time()
            remaining_time = end_time - current_time
            
            if remaining_time <= 0:
                break
            
            # 控制未完成任务数量，避免内存无限增长
            max_pending = max_workers * 2
            
            # 提交新任务（如果有空间）
            while len(pending_futures) < max_pending and time.time() < end_time:
                # 创建搜索参数
                expr = generate_random_expression()
                search_params = create_search_params(
                    search_type, collection_info, vec_field_names, 
                    nq, topk, output_fields, expr, group_by_field
                )
                
                future = executor.submit(
                    single_search_task,
                    client, collection_name, search_params
                )
                pending_futures.add(future)
                submitted_tasks += 1
            
            # 收集已完成的任务
            completed_futures = set()
            for future in list(pending_futures):
                if future.done():
                    try:
                        result = future.result(timeout=0.1)
                        stats.record_search(result['latency'], result['success'])
                        completed_futures.add(future)
                        
                        # 检查结果数量
                        if result['success'] and 'result_count' in result:
                            if result['result_count'] != topk:
                                logging.debug(f"Search results do not meet topk, expected:{topk}, actual:{result['result_count']}")
                                
                    except Exception as e:
                        logging.warning(f"Task failed: {e}")
                        stats.record_search(0.1, False)
                        completed_futures.add(future)
            
            # 移除已完成的任务
            pending_futures -= completed_futures
            
            # 定期输出统计信息 - 避免重复打印
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
                
                # 重置样本数据
                if submitted_tasks % (max_workers * 1000) == 0:
                    stats.reset_samples()
        
        # 等待所有剩余任务完成
        logging.info(f"Waiting for {len(pending_futures)} remaining tasks to complete...")
        for future in as_completed(pending_futures, timeout=30):
            try:
                result = future.result(timeout=1.0)
                stats.record_search(result['latency'], result['success'])
            except Exception as e:
                logging.warning(f"Final task failed: {e}")
                stats.record_search(0.1, False)
    
    # 最终统计 - 使用实际测试时间计算准确的QPS
    actual_test_time = time.time() - stats.start_time
    final_stats = stats.get_stats(actual_elapsed_time=actual_test_time)
    
    logging.info("=" * 80)
    logging.info(f"FINAL SEARCH PERFORMANCE STATISTICS ({search_type.upper()}):") 
    logging.info(f"  Actual Test Duration: {actual_test_time:.2f}s")
    logging.info(f"  Total Searches: {final_stats['total_searches']}")
    logging.info(f"  Total Failures: {final_stats['failures']}")
    logging.info(f"  Success Rate: {final_stats['success_rate']:.2f}%")
    logging.info(f"  Overall QPS: {final_stats['qps']:.2f} (总搜索数 ÷ 实际耗时)")
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
    if len(sys.argv) != 13:
        print("Usage: python3 search_permanently.py <host> <collection> <vec_field_names> <use_hybrid_search> <max_workers> <timeout> <output_fields> <expression> <nq> <topk> <group_by_field> <api_key>")
        print("Parameters:")
        print("  host               : Milvus server host")
        print("  collection         : Collection name")
        print("  vec_field_names    : Vector field names (comma-separated, or 'None' for all)")
        print("  use_hybrid_search  : Use hybrid search (True/False)")
        print("  max_workers        : 并发搜索数量 (直接控制)")
        print("  timeout            : Test timeout in seconds")
        print("  output_fields      : Output fields (comma-separated or '*')")
        print("  expression         : Search filter expression")
        print("  nq                 : Number of query vectors")
        print("  topk               : Top K results")
        print("  group_by_field     : Group by field (or 'None')")
        print("  api_key            : API key (or 'None' for local)")
        print()
        print("Examples:")
        print("  # 普通向量搜索，4个并发")
        print("  python3 search_permanently.py localhost test_collection vector_field False 4 60 'id' 'None' 1 10 None None")
        print()
        print("  # 混合搜索，8个并发")
        print("  python3 search_permanently.py localhost test_collection 'field1,field2' True 8 60 'id' 'None' 1 10 None None")
        print()
        print("🚀 超级简化架构:")
        print("  ✅ 单个共享 MilvusClient")
        print("  ✅ 单个 ThreadPoolExecutor")
        print("  ✅ max_workers = 并发搜索数")
        print("  ✅ 支持普通搜索和混合搜索")
        print("  ✅ 依赖 Milvus 服务端连接复用")
        sys.exit(1)
    
    host = sys.argv[1]
    name = str(sys.argv[2])
    vec_field_names = str(sys.argv[3]).strip()
    use_hybrid_search = str(sys.argv[4]).upper()
    max_workers = int(sys.argv[5])
    timeout = int(sys.argv[6])
    output_fields = str(sys.argv[7]).strip()
    expr = str(sys.argv[8]).strip()
    nq = int(sys.argv[9])
    topk = int(sys.argv[10])
    group_by_field = str(sys.argv[11]).strip()
    api_key = str(sys.argv[12])

    port = 19530
    
    # 参数处理
    if timeout <= 0:
        timeout = 2 * 3600
    
    use_hybrid_search = use_hybrid_search in ["TRUE", "YES"]
    search_type = 'hybrid' if use_hybrid_search else 'normal'
    
    if vec_field_names in ["None", "none", "NONE"] or vec_field_names == "":
        vec_field_names = None
    else:
        vec_field_names = vec_field_names.split(",")
    
    if output_fields in ["None", "none", "NONE"] or output_fields == "":
        output_fields = None
    else:
        output_fields = output_fields.split(",")
    
    if expr in ["None", "none", "NONE"] or expr == "":
        expr = None
    
    if group_by_field in ["None", "none", "NONE"] or group_by_field == "":
        group_by_field = None
    
    if nq <= 0:
        nq = 1
    if topk <= 0:
        topk = 10
    
    # 设置日志
    log_filename = f"/tmp/search_{name}_{int(time.time())}.log"
    file_handler = logging.FileHandler(filename=log_filename)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    
    logging.info("🚀 Starting ULTRA-SIMPLIFIED search_permanently test:")
    logging.info(f"  Host: {host}")
    logging.info(f"  Collection: {name}")
    logging.info(f"  Search Type: {search_type}")
    logging.info(f"  Vector Fields: {vec_field_names or 'all'}")
    logging.info(f"  Max Workers: {max_workers} (= 并发搜索数)")
    logging.info(f"  Timeout: {timeout}s")
    logging.info(f"  nq: {nq}, topk: {topk}")
    logging.info(f"  Output Fields: {output_fields}")
    logging.info(f"  Expression: {expr}")
    logging.info(f"  Group By: {group_by_field}")
    logging.info(f"  架构: 单客户端 + 单线程池，最简单!")

    # 创建单个共享客户端
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
    
    # 运行搜索测试
    start_time = time.time()
    final_stats = search_permanently_simplified(
        client=client,
        collection_name=name,
        max_workers=max_workers,
        search_type=search_type,
        vec_field_names=vec_field_names,
        nq=nq,
        topk=topk,
        output_fields=output_fields,
        expr=expr,
        group_by_field=group_by_field,
        timeout=timeout
    )
    end_time = time.time()
    
    actual_duration = end_time - start_time
    # 重新计算准确的最终QPS
    accurate_qps = final_stats['total_searches'] / max(actual_duration, 0.001)
    
    logging.info(f"✅ Search test completed in {actual_duration:.2f} seconds")
    logging.info(f"📊 Accurate Final QPS: {accurate_qps:.2f} ({final_stats['total_searches']} searches ÷ {actual_duration:.2f}s)")
    logging.info(f"📁 Log file: {log_filename}")
    
