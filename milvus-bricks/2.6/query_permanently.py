import time
import sys
import random
import numpy as np
import threading
import logging
from pymilvus import MilvusClient, DataType
from common import get_float_vec_field_names, create_n_insert, create_collection_schema


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"


def generate_random_expression():
    """Generate random query expression"""
    keywords = ["con%", "%nt", "%con%", "%content%", "%co%nt", "%con_ent%", "%co%nt%"]
    keyword = random.choice(keywords)
    return f'content like "{keyword}"'


def query_permanently(client, collection_name, threads_num, output_fields, expr, timeout, limit):
    """
    Execute queries permanently using MilvusClient
    :param client: MilvusClient object
    :param collection_name: str, collection name
    :param threads_num: int, number of threads
    :param output_fields: list, output fields
    :param expr: str, query expression
    :param timeout: int, timeout in seconds
    :param limit: int, query limit
    """
    threads_num = int(threads_num)
    interval_count = 100
    
    def query_thread(thread_no):
        """Query thread function"""
        query_latency = []
        count = 0
        failures = 0
        start_time = time.time()
        
        while time.time() < start_time + timeout:
            count += 1
            t1 = time.time()
            
            # Handle random expressions
            current_expr = generate_random_expression()
            
            try:
                # Execute query
                res = client.query(
                    collection_name=collection_name,
                    filter=current_expr,
                    output_fields=output_fields,
                    limit=limit,
                )
                result_count = len(res) if res else 0

            except Exception as e:
                failures += 1
                logging.error(f"Thread {thread_no} query failed: {e}")
            
            t2 = round(time.time() - t1, 4)
            query_latency.append(t2)
            
            # Report statistics
            if count == interval_count:
                total = round(np.sum(query_latency), 4)
                p99 = round(np.percentile(query_latency, 99), 4)
                avg = round(np.mean(query_latency), 4)
                qps = round(interval_count / total, 4) if total > 0 else 0
                
                logging.info(f"Thread {thread_no}: query {interval_count} times, "
                           f"failures: {failures}, cost: {total}s, qps: {qps}, avg: {avg}s, p99: {p99}s")
                
                count = 0
                query_latency = []
                failures = 0

    # Start threads
    threads = []
    
    if threads_num > 1:
        # Multi-threaded execution
        for i in range(threads_num):
            t = threading.Thread(target=query_thread, args=(i,))
            threads.append(t)
            t.start()
        # Wait for all threads to complete
        for t in threads:
            t.join()
    else:
        query_latency = []
        count = 0
        start_time = time.time()
        while time.time() < start_time + timeout:
            count += 1
            t1 = time.time()
            current_expr = generate_random_expression()
            try:
                res = client.query(collection_name=collection_name, filter=current_expr, output_fields=output_fields, limit=limit, timeout=60)
                # logging.info(f"res: {res}")
            except Exception as e:
                logging.error(e)
            t2 = round(time.time() - t1, 4)
            query_latency.append(t2)
            if count == interval_count:
                total = round(np.sum(query_latency), 4)
                p99 = round(np.percentile(query_latency, 99), 4)
                avg = round(np.mean(query_latency), 4)
                qps = round(interval_count / total, 4)
                logging.info(f"collection {collection_name} query {interval_count} times single thread: cost {total}, qps {qps}, avg {avg}, p99 {p99} ")
                count = 0
                query_latency = []


def verify_collection_setup(client, collection_name):
    """
    Verify collection exists and is properly set up
    :param client: MilvusClient object
    :param collection_name: str, collection name
    :return: bool, True if ready
    """
    # Check if collection exists
    if not client.has_collection(collection_name=collection_name):
        logging.error(f"Collection {collection_name} does not exist")
        return False
            
    # Check if collection is loaded
    load_state = client.get_load_state(collection_name=collection_name)
    if load_state.get('state') != 'Loaded':
        client.load_collection(collection_name=collection_name)
        logging.warning(f"Collection {collection_name} is not loaded, loaded")        
       
    return True
    

if __name__ == '__main__':
    host = sys.argv[1]                                 # host ip or uri
    name = str(sys.argv[2])                            # collection name/alias
    th = int(sys.argv[3])                              # query thread num
    timeout = int(sys.argv[4])                         # query timeout, permanently if 0
    output_fields = str(sys.argv[5]).strip()           # output fields, default is None
    expr = str(sys.argv[6]).strip()                    # query expression, default is None
    limit = int(sys.argv[7])                           # query limit, default is None
    api_key = str(sys.argv[8])                         # api key for cloud instances

    port = 19530
    
    # Timeout handling
    if timeout <= 0:
        timeout = 2 * 3600  # Default to 2 hours for "permanent" testing
    
    # Parse output fields
    if output_fields in ["None", "none", "NONE"] or output_fields == "":
        output_fields = ["*"]  # Default to all fields
    else:
        output_fields = output_fields.split(",")
    
    # Parse expression
    if expr in ["None", "none", "NONE"] or expr == "":
        expr = None  # Default expression
    if limit in [None, "None", "none", "NONE"] or limit == "":
        limit = None
    
    # Setup logging
    log_filename = f"/tmp/query_permanently_{name}.log"
    file_handler = logging.FileHandler(filename=log_filename)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    logger = logging.getLogger('QUERY_PERMANENTLY')

    logging.info(f"Starting query_permanently test:")
    logging.info(f"  Host: {host}")
    logging.info(f"  Collection: {name}")
    logging.info(f"  Threads: {th}")
    logging.info(f"  Timeout: {timeout}s")
    logging.info(f"  Output fields: {output_fields}")
    logging.info(f"  Expression: {expr}")
    logging.info(f"  Limit: {limit}")
    logging.info(f"  API key: {'***' if api_key != 'None' else 'None'}")

    # Create MilvusClient instance
    try:
        if api_key is None or api_key == "" or api_key.upper() == "NONE":
            client = MilvusClient(uri=f"http://{host}:{port}")
        else:
            client = MilvusClient(uri=host, token=api_key)
        logging.info(f"Created client for host: {host}")
    
    except Exception as e:
        logging.error(f"Failed to create client: {e}")
        sys.exit(1)
    
    # Verify collection setup
    if not verify_collection_setup(client, name):
        logging.error(f"Collection '{name}' setup verification failed")
        logging.error("Please ensure the collection is properly loaded and indexed.")
        sys.exit(1)
        
    # Start query test
    logging.info(f"Starting query test with {th} threads for {timeout} seconds...")
    
    start_time = time.time()
    query_permanently(client, name, th, output_fields, expr, timeout, limit)
    end_time = time.time()
    total_time = round(end_time - start_time, 2)
    
    logging.info(f"Query test completed in {total_time} seconds")
    logging.info(f"Log file: {log_filename}")
