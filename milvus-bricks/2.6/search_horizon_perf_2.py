#!/usr/bin/env python3
"""
Simplified Vector Search Testing - Support Normal Search and Hybrid Search

Key Features:
1. ✅ Use MilvusClient instead of connections + Collection
2. ✅ Single thread pool architecture, max_workers directly controls concurrency
3. ✅ Support both normal search and hybrid search modes
4. ✅ Unified performance statistics and logging system
5. ✅ Rely on Milvus server-side connection reuse
6. ✅ Load query vectors from local parquet file (/root/test/data/query.parquet)
7. ✅ Automatic vector dimension matching and cycling

Query Vector Management:
- Loads query vectors from parquet file (expects 'feature' column only)
- Supports multiple vector dimensions automatically  
- Cycles through available vectors for continuous testing
- Falls back to random vectors if file loading fails
- File format: /root/test/data/query.parquet with 'feature' column containing vector arrays
"""

import time
import sys
import random
import numpy as np
import pandas as pd
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from collections import deque
from pymilvus import MilvusClient, DataType
from pymilvus import AnnSearchRequest, WeightedRanker


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"


class SimpleStats:
    """Simplified statistics system"""
    
    def __init__(self, max_samples=1000):
        self.latencies = deque(maxlen=max_samples)
        self.total_searches = 0
        self.total_failures = 0
        self.start_time = time.time()
        self.lock = Lock()
    
    def record_search(self, latency, success=True):
        """Record search result"""
        with self.lock:
            self.latencies.append(latency)
            self.total_searches += 1
            if not success:
                self.total_failures += 1
    
    def get_stats(self, actual_elapsed_time=None):
        """Get statistics information
        
        :param actual_elapsed_time: Actual test duration for accurate QPS calculation
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
            
            # Prefer provided actual duration, otherwise use internal calculated time
            if actual_elapsed_time is not None:
                elapsed_time = actual_elapsed_time
            else:
                elapsed_time = time.time() - self.start_time
            
            latency_array = np.array(self.latencies)
            
            return {
                'total_searches': self.total_searches,
                'failures': self.total_failures,
                'success_rate': (self.total_searches - self.total_failures) / max(self.total_searches, 1) * 100,
                'qps': self.total_searches / max(elapsed_time, 0.001),  # Use actual duration to calculate QPS
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


class QueryVectorPool:
    """Query vector pool for searching vectors from file"""
    
    def __init__(self, query_file_path="/root/test/data/query.parquet"):
        self.query_file_path = query_file_path
        self.vector_pool = []  # [vectors]
        self.current_idx = 0
        
    def load_vectors_from_file(self):
        """Load the first 10000 query vectors from parquet file - expects 'feature' column only"""

        logging.info(f"📖 Loading query vectors from {self.query_file_path}")
        df = pd.read_parquet(self.query_file_path)

        # Directly use 'feature' column as specified by user
        if 'feature' not in df.columns:
            raise ValueError(f"Required 'feature' column not found in {self.query_file_path}. Available columns: {list(df.columns)}")

        # Extract the first 10000 vectors from 'feature' column
        vectors = df['feature'].iloc[:10000].values.tolist()
        logging.info(f"✅ Found 'feature' column with {len(vectors)} vectors (loaded up to 10000)")
        self.vector_pool = vectors
        return True

    
    def get_vectors(self, nq):
        """Get nq vectors starting from current_idx"""
        if len(self.vector_pool) == 0:
            raise ValueError(f"⚠️ No vectors found, generating random vectors")
        
        result_vectors = self.vector_pool

        # return nq vectors starting from current_idx
        if self.current_idx + nq > len(self.vector_pool):
            self.current_idx = 0
        result_vectors = result_vectors[self.current_idx:self.current_idx + nq]
        self.current_idx = self.current_idx + nq
        
        return result_vectors


def generate_random_expression(expr_key):
    """Generate random query expression"""

    device_id_keywords = [
        'SENSOR_A123', 'SENSOR_A233', 'SENSOR_A108', 'SENSOR_A172', 'CAM_B112', 
        'CAM_B177', 'DV348', 'DV378', 'DV081', 'DV349']
    
    polygon_keywords = [
        "'POLYGON((-74.0 40.7, -73.9 40.7, -73.9 40.8, -74.0 40.8, -74.0 40.7))'",  # NYC area
        "'POLYGON((-74.1 40.6, -73.8 40.6, -73.8 40.9, -74.1 40.9, -74.1 40.6))'",  # Larger NYC area
        "'POLYGON((-74.05 40.75, -73.95 40.75, -73.95 40.85, -74.05 40.85, -74.05 40.75))'"  # Central NYC
    ]
    json_contains_pattens = [
        "JSON_CONTAINS_ALL(sensor_lidar_type,['Analog_ADAU1761', 'AEye_iDAR', 'ADAS_Eyes', 'Argo_Geiger']) AND NOT JSON_CONTAINS(sensor_lidar_type, 'Delphi_ESR')",
        "JSON_CONTAINS_ALL(sensor_lidar_type,['Gatik_B2B', 'Bosch_LRR4', 'Mobileye_EyeQ4', 'Motional_Ioniq5']) AND NOT JSON_CONTAINS(sensor_lidar_type, 'Delphi_ESR')",
        "JSON_CONTAINS_ALL(sensor_lidar_type,['TI_AWR1843', 'Continental_HFL110', 'Velodyne_VLS128', 'Analog_ADAU1761']) AND NOT JSON_CONTAINS(sensor_lidar_type, 'Delphi_ESR')",
        "JSON_CONTAINS_ALL(sensor_lidar_type,['Thor_Trucks', 'Aptiv_SRR4', 'Leishen_C32', 'Pony_PonyAlpha']) AND NOT JSON_CONTAINS(sensor_lidar_type, 'Delphi_ESR')",
        "JSON_CONTAINS_ALL(sensor_lidar_type,['Magna_Icon', 'ZF_AC1000', 'NXP_TEF810X', 'Ibeo_LUX']) AND NOT JSON_CONTAINS(sensor_lidar_type, 'Delphi_ESR')",
        "JSON_CONTAINS_ALL(sensor_lidar_type,['Innoviz_One', 'ST_VL53L1X', 'May_Mobility', 'Embark_Guardian']) AND NOT JSON_CONTAINS(sensor_lidar_type, 'Delphi_ESR')"
    ]

    if expr_key is None:
        return None

    if expr_key.lower() == "equal":
        device_id_keyword = random.choice(device_id_keywords)
        return f'device_id == \"{device_id_keyword}\"'
    elif expr_key.lower() == "equal_and_expert_collected":
        device_id_keyword = random.choice(device_id_keywords)
        return f'device_id == \"{device_id_keyword}\" and expert_collected == True'
    elif expr_key.lower() == "equal_and_timestamp_week":
        device_id_keyword = random.choice(device_id_keywords)
        # Generate a random 7-day window between 2025-01-01 and 2025-08-30
        import datetime

        # Define the start and end date range
        start_date = datetime.datetime(2025, 1, 1)
        end_date = datetime.datetime(2025, 8, 23)  # So that start + 7 days <= 2025-08-30

        # Calculate the total days between start and end
        total_days = (end_date - start_date).days

        # Randomly pick a start day offset
        random_offset = random.randint(0, total_days)
        window_start_date = start_date + datetime.timedelta(days=random_offset)
        window_end_date = window_start_date + datetime.timedelta(days=6)  # 7 days window

        # Convert to timestamp (assume UTC)
        left_timestamp_keyword = int(window_start_date.replace(tzinfo=datetime.timezone.utc).timestamp())
        right_timestamp_keyword = int(window_end_date.replace(tzinfo=datetime.timezone.utc).timestamp())
        return f'device_id == \"{device_id_keyword}\" and timestamp >= {left_timestamp_keyword} and timestamp <= {right_timestamp_keyword}'

    elif expr_key.lower() == "equal_and_timestamp_month":
        device_id_keyword = random.choice(device_id_keywords)
        # Generate a random 30-day window between 2025-01-01 and 2025-08-30
        import datetime

        # Define the start and end date range
        start_date = datetime.datetime(2025, 1, 1)
        end_date = datetime.datetime(2025, 8, 1)  # So that start + 30 days <= 2025-08-30

        # Calculate the total days between start and end
        total_days = (end_date - start_date).days

        # Randomly pick a start day offset
        random_offset = random.randint(0, total_days)
        window_start_date = start_date + datetime.timedelta(days=random_offset)
        window_end_date = window_start_date + datetime.timedelta(days=29)  # 30 days window

        # Convert to timestamp (assume UTC)
        left_timestamp_keyword = int(window_start_date.replace(tzinfo=datetime.timezone.utc).timestamp())
        right_timestamp_keyword = int(window_end_date.replace(tzinfo=datetime.timezone.utc).timestamp())

        return f'device_id == \"{device_id_keyword}\" and timestamp >= {left_timestamp_keyword} and timestamp <= {right_timestamp_keyword}'

    elif expr_key.lower() == "geo_contains":
        polygon = random.choice(polygon_keywords)
        polygon = "'POLYGON((-73.991957 40.721567, -73.982102 40.73629, -74.002587 40.739748, -73.974267 40.790955, -73.991957 40.721567))'"
        return f'ST_CONTAINS(location, {polygon})'
        
    elif expr_key.lower() == "sensor_contains":
        keywords = [
            'Thor_Trucks', 'WeRide_Robobus', 'Delphi_ESR', 'Aptiv_SRR4', 'AEye_iDAR', 'DiDi_Gemini', 'ADAS_Eyes', 
            'Embark_Guardian', 'Hella_24GHz', 'ST_VL53L1X', 'TuSimple_AFV', 'Locomation_AutonomousRelay', 'Voyage_Telessport', 
            'Livox_Horizon', 'Infineon_BGT24', 'Aurora_FirstLight', 'Ibeo_LUX', 'Ouster_OS1_64', 'Delphi_ESR']
        keyword = random.choice(keywords)
        return f'ARRAY_CONTAINS(sensor_lidar_type, \"{keyword}\")'
    elif expr_key.lower() == "sensor_json_contains":
        return random.choice(json_contains_pattens)
    else:
        return None


def single_search_task(client, collection_name, search_params, each_search_timeout=10):
    """
    Single search task - Support normal search and hybrid search
    
    :param search_params: Dictionary containing search parameters
    """
    start_time = time.time()
    
    try:
        if search_params['search_type'] == 'hybrid':
            # Hybrid search
            result = client.hybrid_search(
                collection_name=collection_name,
                reqs=search_params['reqs'],
                ranker=search_params.get('ranker'),
                limit=search_params['limit'],
                output_fields=search_params.get('output_fields'),
                timeout=each_search_timeout
            )
        else:
            # Normal search
            result = client.search(
                collection_name=collection_name,
                data=search_params['data'],
                anns_field=search_params['anns_field'],
                search_params=search_params.get('search_params', {}),
                limit=search_params['limit'],
                filter=search_params.get('filter'),
                output_fields=search_params.get('output_fields'),
                group_by_field=search_params.get('group_by_field'),
                partition_names=search_params.get('partition_names'),
                timeout=each_search_timeout
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


def create_search_params(search_type, vec_field_name, nq, topk, 
                        output_fields, filter):
    """Create search parameters - Use simplified common methods（only need schema parameter）"""
        
    if search_type == 'hybrid':
        # Hybrid searchParameters
        reqs = []
        field_name = vec_field_name
        search_vectors_1 = query_vector_pool.get_vectors(nq)
        req_text = AnnSearchRequest(
            data=search_vectors_1,
            anns_field=field_name,
            param={},
            limit=topk,
            expr=filter
        )
        reqs.append(req_text)
        search_vectors_2 = query_vector_pool.get_vectors(nq)
        req_image = AnnSearchRequest(
            data=search_vectors_2,
            anns_field=field_name,
            param={},
            limit=topk,
            expr=filter
        )
        reqs.append(req_image)
        
        weight = random.random()
        rerank = WeightedRanker(weight, 1 - weight)
        return {
            'search_type': 'hybrid',
            'reqs': reqs,
            'ranker': rerank,  
            'limit': topk,
            'output_fields': output_fields
        }
    
    else:
        # Normal searchParameters
        # Simplified call：only need to pass schema and field_name
        field_name = vec_field_name
        
        search_vectors = query_vector_pool.get_vectors(nq)
        
        return {
            'search_type': 'normal',
            'data': search_vectors,
            'anns_field': field_name,
            'search_params': {},
            'limit': topk,
            'filter': filter,
            'output_fields': output_fields,
            'group_by_field': None,
            'partition_names': None  # Can be extended to support partitions later
        }


def search_permanently_simplified(client, collection_name, max_workers, search_type,
                                 vec_field_name, nq, topk, output_fields, expr, 
                                 timeout, each_search_timeout):
    """
    Simplified version of continuous search testing
    
    :param client: Single shared MilvusClient instance
    :param search_type: 'normal' or 'hybrid'
    """
    stats = SimpleStats()
    end_time = time.time() + timeout
    
    # Log control variables
    last_logged_milestone = 0
    log_interval = min(max_workers * 100, 1000)
    
    # Single thread pool，directly manage all search tasks
    with ThreadPoolExecutor(max_workers=max_workers, 
                           thread_name_prefix="SearchWorker") as executor:
        
        # Continuously submit search tasks until timeout
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
                # Create search parameters
                filter = generate_random_expression(expr_key=expr) 
                search_params = create_search_params(
                    search_type, vec_field_name, 
                    nq, topk, output_fields, filter
                )
                
                future = executor.submit(
                    single_search_task,
                    client, collection_name, search_params, each_search_timeout
                )
                pending_futures.add(future)
                submitted_tasks += 1
            
            # Collect completed tasks
            completed_futures = set()
            for future in list(pending_futures):
                if future.done():
                    try:
                        result = future.result(timeout=0.1)
                        stats.record_search(result['latency'], result['success'])
                        completed_futures.add(future)
                        
                        # Check result count
                        if result['success'] and 'result_count' in result:
                            if result['result_count'] != topk:
                                logging.debug(f"Search results do not meet topk, expected:{topk}, actual:{result['result_count']}")
                                
                    except Exception as e:
                        logging.warning(f"Task failed: {e}")
                        stats.record_search(0.1, False)
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
        
        # Wait for all remaining tasks to complete
        logging.info(f"Waiting for {len(pending_futures)} remaining tasks to complete...")
        for future in as_completed(pending_futures, timeout=30):
            try:
                result = future.result(timeout=1.0)
                stats.record_search(result['latency'], result['success'])
            except Exception as e:
                logging.warning(f"Final task failed: {e}")
                stats.record_search(0.1, False)
    
    # Final statistics - use actual test time to calculate accurate QPS
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


def search(client, collection_name, vector_field_name, nq, topk, threads_num, output_fields, expr, timeout):
    threads_num = int(threads_num)

    search_params = {}
    logging.info(f"search on vector_field_name:{vector_field_name}")
    logging.info(f"search_param: {search_params}")
    logging.info(f"output_fields: {output_fields}")
    logging.info(f"expr: {expr}")

    def search_th(c, thread_no):
        search_latency = []
        searched_partitions = []
        count = 0
        failures = 0
        interval_count = 500
        start_time = time.time()
        while time.time() < start_time + timeout:
            count += 1
            search_vectors = query_vector_pool.get_vectors(nq)
            filter = generate_random_expression(expr_key=expr)
            t1 = time.time()
            try:
                res = c.search(
                    collection_name=collection_name,
                    data=search_vectors,
                    anns_field=vector_field_name,
                    search_params={},
                    limit=topk,
                    filter=filter,
                    output_fields=output_fields,
                    timeout=30
                )
                if len(res[0]) < topk:
                    logging.info(f"search results do not meet topk, expected:{topk}, actual:{len(res[0])}")
            except Exception as e:
                failures += 1
                logging.error(e)
            t2 = round(time.time() - t1, 4)
            search_latency.append(t2)
            if count == interval_count:
                total = round(np.sum(search_latency), 4)
                p99 = round(np.percentile(search_latency, 99), 4)
                avg = round(np.mean(search_latency), 4)
                qps = round(interval_count / total, 4)
                max_latency = round(np.max(search_latency), 4)
                distict_partitions_num = len(set(searched_partitions))
                logging.info(f"collection {collection_name} total failures: {failures}, search {interval_count} times "
                             f"in thread{thread_no}: cost {total}, qps {qps}, avg {avg}, p99 {p99}, max {max_latency}")
                count = 0
                search_latency = []
                searched_partitions = []

    threads = []
    if threads_num > 1:
        for i in range(threads_num):
            t = threading.Thread(target=search_th, args=(client, i))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
    else:
        interval_count = 100
        search_latency = []
        count = 0
        failures = 0
        start_time = time.time()
        while time.time() < start_time + timeout:
            count += 1
            search_vectors = query_vector_pool.get_vectors(nq)
            filter = generate_random_expression(expr_key=expr)

            t1 = time.time()
            try:
                res = client.search(
                    collection_name=collection_name,
                    data=search_vectors,
                    anns_field=vector_field_name,
                    search_params={},
                    limit=topk,
                    filter=filter,
                    output_fields=output_fields,
                    timeout=30
                )
                if len(res[0]) < topk - (topk * 0.1):
                    logging.info(f"search results do not meet topk, expected:{topk}, actual:{len(res[0])}")
            except Exception as e:
                failures += 1
                logging.error(e)
            t2 = round(time.time() - t1, 4)
            search_latency.append(t2)
            if count == interval_count:
                total = round(np.sum(search_latency), 4)
                p99 = round(np.percentile(search_latency, 99), 4)
                avg = round(np.mean(search_latency), 4)
                qps = round(interval_count / total, 4)
                max_latency = round(np.max(search_latency), 4)
                logging.info(f"collection {collection_name} total failures: {failures}, search {interval_count}"
                             f" times single thread: cost {total}, qps {qps}, avg {avg}, p99 {p99}, max {max_latency}")
                count = 0
                search_latency = []


def hybrid_search(client, collection_name, vec_field_names, nq, topk, threads_num, output_fields, expr, timeout):
    threads_num = int(threads_num)
    interval_count = 100

    def hybrid_search_th(c, thread_no):
        search_latency = []
        count = 0
        failures = 0
        start_time = time.time()
        while time.time() < start_time + timeout:
            count += 1
            reqs = []
            filter = generate_random_expression(expr_key=expr)
            search_vectors_1 = query_vector_pool.get_vectors(nq)
            req_text = AnnSearchRequest(
                data=search_vectors_1,
                anns_field=vec_field_names[0],
                limit=topk,
                param={},
                expr=filter
            )
            reqs.append(req_text)
            search_vectors_2 = query_vector_pool.get_vectors(nq)
            req_image = AnnSearchRequest(
                data=search_vectors_2,
                anns_field=vec_field_names[1],
                limit=topk,
                param={},
                expr=filter
            )
            reqs.append(req_image)
            
            # random weight between 0 and 1
            weight = random.random()
            rerank = WeightedRanker(weight, 1 - weight)
            t1 = time.time()
            try:
                c.hybrid_search(collection_name=collection_name, 
                                reqs=reqs, 
                                ranker=rerank, 
                                limit=topk, 
                                timeout=30,
                                output_fields=output_fields)
                if len(res[0]) < topk:
                    logging.info(f"hybrid search results do not meet topk, expected:{topk}, actual:{len(res[0])}")
            except Exception as e:
                failures += 1
                logging.error(e)
            t2 = round(time.time() - t1, 4)
            search_latency.append(t2)
            if count == interval_count:
                total = round(np.sum(search_latency), 4)
                p99 = round(np.percentile(search_latency, 99), 4)
                avg = round(np.mean(search_latency), 4)
                qps = round(interval_count / total, 4)
                logging.info(f"collection {collection_name} hybrid_search {interval_count} times in thread{thread_no}: cost {total}, qps {qps}, avg {avg}, p99 {p99} ")
                count = 0
                search_latency = []

    threads = []
    if threads_num > 1:
        for i in range(threads_num):
            t = threading.Thread(target=hybrid_search_th, args=(client, i))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
    else:
        interval_count = 100
        search_latency = []
        count = 0
        start_time = time.time()
        while time.time() < start_time + timeout:
            count += 1
            search_vectors_1 = query_vector_pool.get_vectors(nq)
            filter = generate_random_expression(expr_key=expr)
            reqs = []
            req_text = AnnSearchRequest(
                data=search_vectors_1,
                anns_field=vec_field_names[0],
                param={},
                limit=topk,
                expr=filter
            )
            reqs.append(req_text)
            search_vectors_2 = query_vector_pool.get_vectors(nq)
            req_image = AnnSearchRequest(
                data=search_vectors_2,
                anns_field=vec_field_names[1],
                param={},
                limit=topk,
                expr=filter
            )
            reqs.append(req_image)
            
            weight = random.random()
            rerank = WeightedRanker(weight, 1 - weight)
            t1 = time.time()
            try:
                client.hybrid_search(collection_name=collection_name, reqs=reqs, ranker=rerank, limit=topk,
                                         output_fields=output_fields, timeout=30)
            except Exception as e:
                logging.error(e)
            t2 = round(time.time() - t1, 4)
            search_latency.append(t2)
            if count == interval_count:
                total = round(np.sum(search_latency), 4)
                p99 = round(np.percentile(search_latency, 99), 4)
                avg = round(np.mean(search_latency), 4)
                qps = round(interval_count / total, 4)
                logging.info(
                    f"collection {collection_name} hybrid_search {interval_count} times single thread: cost {total}, qps {qps}, avg {avg}, p99 {p99} ")
                count = 0
                search_latency = []


if __name__ == '__main__':

    use_hybrid_search = 'false'     # str(sys.argv[1]).upper()
    max_workers = 10  # int(sys.argv[2])
    timeout = 300    # int(sys.argv[3])
    expr = 'sensor_json_contains'  # str(sys.argv[5]).strip()

    # use_hybrid_search = str(sys.argv[1]).upper()
    # max_workers = int(sys.argv[2])
    # timeout = int(sys.argv[3])
    # expr = str(sys.argv[4]).strip()

    host = 'https://in01-9028520cb1d63cf.ali-cn-hangzhou.cloud-uat.zilliz.cn:19530'
    name = 'horizon_test_collection'
    vec_field_name = 'feature'
    nq = 1
    topk = 15000
    api_key = 'cc5bf695ea9236e2c64617e9407a26cf0953034485d27216f8b3f145e3eb72396e042db2abb91c4ef6fde723af70e754d68ca787'

    port = 19530
    
    # Parameter processing
    if timeout <= 0:
        timeout = 2 * 3600
    
    search_type = 'hybrid' if use_hybrid_search.upper() in ["TRUE", "YES"] else 'normal'
    
    if expr in ["None", "none", "NONE"] or expr == "":
        expr = None

    # output_fields = ['timestamp', 'url', 'device_id', 'location', 'expert_collected', 'sensor_lidar_type', 'p_url']
    output_fields = None
    
    if nq <= 0:
        nq = 1
    if topk <= 0:
        topk = 15000
    
    each_search_timeout = 30
    
    # Setup logging
    log_filename = f"/tmp/search_{name}_{int(time.time())}.log"
    file_handler = logging.FileHandler(filename=log_filename)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    
    logging.info("🚀 Starting search_permanently test:")
    logging.info(f"  Host: {host}")
    logging.info(f"  Collection: {name}")
    logging.info(f"  Search Type: {search_type}")
    logging.info(f"  Vector Fields: {vec_field_name}")
    logging.info(f"  Max Workers: {max_workers} (= 并发搜索数)")
    logging.info(f"  The whole Test Timeout: {timeout}s")
    logging.info(f"  nq: {nq}, topk: {topk}")
    logging.info(f"  Output Fields: {output_fields}")
    logging.info(f"  Expression: {expr}")
    logging.info(f"  Each Search Timeout: {each_search_timeout}s")

    # Initialize query vector pool
    query_vector_file_path = "/root/test/data/query.parquet"
    # query_vector_file_path = '~/Downloads/query.parquet'
    query_vector_pool = QueryVectorPool(query_vector_file_path)
    
    logging.info("📖 Initializing query vector pool...")
    if query_vector_pool.load_vectors_from_file():
        logging.info("✅ Query vector pool initialized successfully")
    else:
        logging.warning("⚠️ Failed to load query vectors, will use random vectors as fallback")

    # Create single shared client
    if api_key is None or api_key == "" or api_key.upper() == "NONE":
        client = MilvusClient(uri=f"http://{host}:{port}")
    else:
        client = MilvusClient(uri=host, token=api_key)
    
    logging.info(f"✅ Created single shared MilvusClient for {host}")
    
    # Verify collection
    if not verify_collection_setup(client, name):
        logging.error(f"Collection '{name}' setup verification failed")
        sys.exit(1)
    
    # Run search test
    if expr is None:
        expr_keys = [None]
    else:
        expr_keys = [expr]
        if len(expr_keys) == 1 and expr_keys[0].upper() == "ALL":
            expr_keys = [
                'equal',
                'equal_and_expert_collected',
                'equal_and_timestamp_week',
                'equal_and_timestamp_month',
                'geo_contains',
                'sensor_contains',
                'device_id_in',
                'sensor_json_contains'
                ]
    logging.info(f"expr_keys: {expr_keys}")
    for expr_key in expr_keys:
        logging.info(f"✅ Search test on expr {expr_key} started")
    #     start_time = time.time()
    #     final_stats = search_permanently_simplified(
    #         client=client,
    #         collection_name=name,
    #         max_workers=max_workers,
    #         search_type=search_type,
    #         vec_field_name=vec_field_name,
    #         nq=nq,
    #         topk=topk,
    #         output_fields=output_fields,
    #         expr=expr_key,
    #         timeout=timeout,
    #         each_search_timeout=each_search_timeout
    #     )
    #     end_time = time.time()
    #
    #     actual_duration = end_time - start_time
    #     # Recalculate accurate final QPS
    #     accurate_qps = final_stats['total_searches'] / max(actual_duration, 0.001)
    #
    #     logging.info(f"✅ Search test on expr {expr_key} completed in {actual_duration:.2f} seconds")
    #     logging.info(f"📊 Accurate Final QPS: {accurate_qps:.2f} ({final_stats['total_searches']} searches ÷ {actual_duration:.2f}s)")

        if use_hybrid_search:
            hybrid_search(client, collection_name=name, vec_field_names=[vec_field_name, vec_field_name], 
                nq=nq, topk=topk, threads_num=max_workers, output_fields=output_fields, expr=expr, timeout=timeout)
        else:
            search(client, collection_name=name, vector_field_name=vec_field_name,
                nq=nq, topk=topk, threads_num=max_workers, output_fields=output_fields,
                expr=expr, timeout=timeout)
        
        logging.info(f"✅ Search test on expr {expr_key} completed")
        time.sleep(20)

    logging.info(f"📁 Log file: {log_filename}")
