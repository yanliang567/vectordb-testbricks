#!/usr/bin/env python3
"""
Insert Performance Testing - Updated for MilvusClient API 2.6

This script performs multi-threaded insert performance testing on Milvus collections.
It measures insert throughput and latency under concurrent load.

Key Features:
- Uses MilvusClient API (v2.6+)
- Multi-threaded concurrent insert testing
- Configurable batch size and thread count
- Pre-load collection option for performance testing
- Detailed performance metrics (requests/sec, entities/sec)
- Auto-creates collection if not exists

Updated to use MilvusClient API instead of the older connections-based API.
"""

import time
import sys
import random
import threading
import logging
from pymilvus import MilvusClient, DataType

from common import gen_row_data_by_schema, create_n_insert, create_collection_schema


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"


def do_insert_concurrent(client, collection_name, schema, nb, threads_num, 
                         ins_times_per_thread, sleep_interval, start_id=0):
    """
    Perform concurrent insert testing using multiple threads
    
    :param client: MilvusClient instance
    :param collection_name: str, collection name
    :param schema: dict, collection schema from describe_collection
    :param nb: int, number of entities per batch
    :param threads_num: int, number of concurrent threads
    :param ins_times_per_thread: int, insert times per thread
    :param sleep_interval: int, sleep seconds between inserts
    :param start_id: int, starting ID value for primary key generation
    """
    
    # Thread-safe counter for tracking progress
    insert_count = {'success': 0, 'failed': 0}
    lock = threading.Lock()
    
    def insert_thread_worker(thread_id, rounds):
        """Worker function for each insert thread"""
        for r in range(rounds):
            try:
                # Generate data for this batch
                batch_start_id = start_id + thread_id * rounds * nb + r * nb
                data = gen_row_data_by_schema(schema=schema, nb=nb, start=batch_start_id)
                
                # Validate data length
                if isinstance(data, list) and len(data) > 0:
                    first_field_len = len(data[0]) if isinstance(data[0], list) else len(data)
                    if first_field_len != nb:
                        logging.warning(
                            f"Thread {thread_id} Round {r}: Generated data length mismatch. "
                            f"Expected: {nb}, Actual: {first_field_len}"
                        )
                
                # Execute insert
                t1 = time.time()
                result = client.insert(
                    collection_name=collection_name,
                    data=data
                )
                t2 = round(time.time() - t1, 3)
                
                # Log and track success
                with lock:
                    insert_count['success'] += 1
                
                logging.info(
                    f"Thread {thread_id} Round {r}: Insert completed in {t2}s, "
                    f"inserted {result.get('insert_count', nb)} entities"
                )
                
                # Sleep between inserts if specified
                if sleep_interval > 0:
                    time.sleep(sleep_interval)
                    
            except Exception as e:
                with lock:
                    insert_count['failed'] += 1
                logging.error(f"Thread {thread_id} Round {r}: Insert failed - {e}")
    
    # Start concurrent insert threads
    threads = []
    logging.info(
        f"Starting concurrent insert test on '{collection_name}': "
        f"{threads_num} threads × {ins_times_per_thread} inserts × {nb} entities/batch"
    )
    
    if threads_num > 1:
        # Multi-threaded mode
        for i in range(threads_num):
            t = threading.Thread(
                target=insert_thread_worker, 
                args=(i, ins_times_per_thread),
                name=f"InsertThread-{i}"
            )
            threads.append(t)
            t.start()
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
    else:
        # Single-threaded mode
        insert_thread_worker(0, ins_times_per_thread)
    
    logging.info(
        f"Insert test completed: Success={insert_count['success']}, "
        f"Failed={insert_count['failed']}"
    )
    
    return insert_count


if __name__ == '__main__':
    if len(sys.argv) != 11:
        print("Usage: python3 insert_perf_1.py <host> <collection_name> <dim> <nb> <num_threads> <ins_per_thread> <sleep_interval> <pre_load> <start_id> <api_key>")
        print("\nParameters:")
        print("  host            : Milvus server host")
        print("  collection_name : Collection name")
        print("  dim             : Vector dimension")
        print("  nb              : Number of vectors per insert request")
        print("  num_threads     : Number of concurrent insert threads")
        print("  ins_per_thread  : Insert times per thread")
        print("  sleep_interval  : Sleep seconds between each insert (0 for no sleep)")
        print("  pre_load        : Pre-load collection before insert (TRUE/FALSE)")
        print("  start_id        : Starting ID value for primary key (default: 0)")
        print("  api_key         : API key for cloud instances (None for local)")
        print("\nExample:")
        print("  python3 insert_perf_1.py localhost test_collection 128 1000 4 10 0 FALSE 0 None")
        print("  # This will: 4 threads × 10 inserts × 1000 entities = 40,000 total entities, starting from ID 0")
        print()
        print("  python3 insert_perf_1.py localhost test_collection 128 1000 4 10 0 FALSE 100000 None")
        print("  # This will: Insert 40,000 entities starting from ID 100,000")
        sys.exit(1)
    
    # Parse command line arguments
    host = sys.argv[1]
    collection_name = sys.argv[2]
    dim = int(sys.argv[3])
    nb = int(sys.argv[4])
    num_threads = int(sys.argv[5])
    ins_per_thread = int(sys.argv[6])
    sleep_interval = int(sys.argv[7])
    pre_load = str(sys.argv[8]).upper()
    start_id = int(sys.argv[9])
    api_key = str(sys.argv[10])
    
    port = 19530
    
    # Process boolean parameters
    pre_load = True if pre_load == "TRUE" else False
    
    # Setup logging
    log_filename = f"/tmp/insert_perf_{collection_name}.log"
    file_handler = logging.FileHandler(filename=log_filename)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    logger = logging.getLogger('INSERT_PERF')
    
    logging.info("🚀 Starting insert_perf_1 (v2.6):")
    logging.info(f"  Host: {host}")
    logging.info(f"  Collection: {collection_name}")
    logging.info(f"  Vector dimension: {dim}")
    logging.info(f"  Batch size (nb): {nb}")
    logging.info(f"  Threads: {num_threads}")
    logging.info(f"  Inserts per thread: {ins_per_thread}")
    logging.info(f"  Sleep interval: {sleep_interval}s")
    logging.info(f"  Pre-load: {pre_load}")
    logging.info(f"  Start ID: {start_id}")
    
    # Create MilvusClient
    try:
        if api_key is None or api_key == "" or api_key.upper() == "NONE":
            client = MilvusClient(uri=f"http://{host}:{port}")
        else:
            client = MilvusClient(uri=host, token=api_key)
        logging.info(f"✅ Created MilvusClient for {host}")
    except Exception as e:
        logging.error(f"Failed to create MilvusClient: {e}")
        sys.exit(1)
    
    # Check if collection exists, create if not
    if not client.has_collection(collection_name=collection_name):
        logging.warning(f"Collection '{collection_name}' does not exist. Creating with default schema...")
        try:
            # Create default schema
            default_schema = create_collection_schema(
                dims=[dim],
                vector_types=[DataType.FLOAT_VECTOR],
                auto_id=True,
                use_str_pk=False
            )
            
            create_n_insert(
                collection_name=collection_name,
                schema=default_schema,
                nb=nb,
                insert_times=0,  # Don't insert during creation
                index_types=['HNSW'],
                dims=[dim],
                metric_types=['COSINE'],
                build_index=False,
                clients=[client]
            )
            logging.info(f"✅ Collection '{collection_name}' created successfully")
        except Exception as e:
            logging.error(f"Failed to create collection: {e}")
            sys.exit(1)
    
    # Get collection schema
    try:
        schema = client.describe_collection(collection_name)
        logging.info(f"Collection schema: {len(schema.get('fields', []))} fields")
    except Exception as e:
        logging.error(f"Failed to get collection schema: {e}")
        sys.exit(1)
    
    # Pre-load collection if requested
    if pre_load:
        try:
            logging.info(f"Pre-loading collection '{collection_name}'...")
            t1 = time.time()
            client.load_collection(collection_name=collection_name)
            t2 = round(time.time() - t1, 3)
            logging.info(f"✅ Collection loaded in {t2}s")
        except Exception as e:
            logging.error(f"Failed to load collection: {e}")
            # Continue anyway, as some operations work without loading
    
    # Start insert performance test
    logging.info("=" * 80)
    logging.info("STARTING INSERT PERFORMANCE TEST")
    logging.info("=" * 80)
    
    test_start_time = time.time()
    
    # Execute concurrent inserts
    insert_stats = do_insert_concurrent(
        client=client,
        collection_name=collection_name,
        schema=schema,
        nb=nb,
        threads_num=num_threads,
        ins_times_per_thread=ins_per_thread,
        sleep_interval=sleep_interval,
        start_id=start_id
    )
    
    test_end_time = time.time()
    total_time = test_end_time - test_start_time
    
    # Calculate performance metrics
    total_inserts = num_threads * ins_per_thread
    successful_inserts = insert_stats['success']
    failed_inserts = insert_stats['failed']
    
    # Requests per second (RPS)
    req_per_sec = round(successful_inserts / max(total_time, 0.001), 3)
    
    # Entities throughput (entities per second)
    total_entities = successful_inserts * nb
    entities_throughput = round(total_entities / max(total_time, 0.001), 3)
    
    # Average latency per request
    avg_latency = round(total_time / max(successful_inserts, 1), 3)
    
    # Calculate ID range
    end_id = start_id + total_entities
    
    # Final summary
    logging.info("=" * 80)
    logging.info("INSERT PERFORMANCE TEST COMPLETED")
    logging.info("=" * 80)
    logging.info(f"Collection: {collection_name}")
    logging.info(f"Total time: {round(total_time, 3)}s")
    logging.info(f"Total insert requests: {total_inserts}")
    logging.info(f"  - Successful: {successful_inserts}")
    logging.info(f"  - Failed: {failed_inserts}")
    logging.info(f"  - Success rate: {round(successful_inserts/max(total_inserts,1)*100, 2)}%")
    logging.info(f"Batch size: {nb} entities/request")
    logging.info(f"Total entities inserted: {total_entities:,}")
    logging.info(f"ID range: {start_id:,} to {end_id-1:,} (start_id={start_id})")
    logging.info("Performance Metrics:")
    logging.info(f"  - Requests per second (RPS): {req_per_sec}")
    logging.info(f"  - Entities throughput: {entities_throughput:,} entities/sec")
    logging.info(f"  - Average latency: {avg_latency}s/request")
    logging.info(f"Log file: {log_filename}")
    logging.info("=" * 80)
    
    # Flush collection to ensure data is persisted
    try:
        logging.info("Flushing collection to persist data...")
        t1 = time.time()
        client.flush(collection_name=collection_name)
        t2 = round(time.time() - t1, 3)
        
        # Get final entity count
        stats = client.get_collection_stats(collection_name)
        final_count = stats.get('row_count', 0)
        
        logging.info(f"✅ Collection flushed in {t2}s")
        logging.info(f"Final entity count: {final_count:,}")
    except Exception as e:
        logging.warning(f"Failed to flush collection: {e}")

