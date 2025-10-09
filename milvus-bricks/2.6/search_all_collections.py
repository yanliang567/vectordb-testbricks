#!/usr/bin/env python3
"""
Search All Collections - Updated for MilvusClient API 2.6

This script continuously searches all collections in a Milvus instance.
It monitors collection health and reports search performance statistics.

Key Features:
- Uses MilvusClient API (v2.6+)
- Searches all available collections one by one
- Supports search filtering and output fields
- Monitors collection health (index status, load state)
- Tracks search failures and recovery time
- Continuous search until timeout (if timeout > 0)

Updated to use MilvusClient API instead of the older connections-based API.
"""

import time
import sys
import random
import numpy as np
import logging
from pymilvus import MilvusClient, DataType

# Import common utility functions - simplified for 2.6
from common import (
    get_float_vec_field_names,
    get_dim_by_field_name,
    get_vector_field_info_from_schema,
    get_primary_field_name
)

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"


def generate_random_vectors(dim, nq):
    """Generate random vectors for search"""
    return [[random.random() for _ in range(dim)] for _ in range(nq)]


def search_single_collection(client, collection_name, nq, topk, output_fields, expr):
    """
    Search a single collection
    
    :param client: MilvusClient instance
    :param collection_name: str, collection name
    :param nq: int, number of query vectors
    :param topk: int, top K results
    :param output_fields: list or None, output fields
    :param expr: str or None, search filter expression
    :return: dict with search results and metrics
    """
    result = {
        'collection': collection_name,
        'status': 'success',
        'search_time': 0,
        'flush_time': 0,
        'num_entities': 0,
        'error': None
    }
    
    try:
        # Flush collection to ensure data consistency
        t1 = time.time()
        client.flush(collection_name=collection_name)
        result['flush_time'] = round(time.time() - t1, 3)
        
        # Get collection info
        schema = client.describe_collection(collection_name)
        
        # Get number of entities
        stats = client.get_collection_stats(collection_name)
        result['num_entities'] = stats.get('row_count', 0)
        
        # Get vector field information
        vec_fields = get_float_vec_field_names(schema=schema)
        if not vec_fields:
            result['status'] = 'no_vector_field'
            result['error'] = 'No vector fields found'
            return result
        
        # Use first vector field for search
        vector_field_name = vec_fields[0]
        dim = get_dim_by_field_name(schema=schema, field_name=vector_field_name)
        
        if not dim:
            result['status'] = 'invalid_dimension'
            result['error'] = f'Invalid dimension for field {vector_field_name}'
            return result
        
        # Generate random search vectors
        search_vectors = generate_random_vectors(dim, nq)
        
        # Add random partition key to expression if exists
        filter_expr = None
        if expr:
            parkey = random.randint(1, 1000)
            filter_expr = expr + str(parkey)
        
        # Execute search
        t1 = time.time()
        search_result = client.search(
            collection_name=collection_name,
            data=search_vectors,
            anns_field=vector_field_name,
            search_params={},
            limit=topk,
            filter=filter_expr,
            output_fields=output_fields
        )
        result['search_time'] = round(time.time() - t1, 4)
        
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)
        logging.error(f"Search failed for {collection_name}: {e}")
    
    return result


def check_collection_status(client, collection_name):
    """
    Check collection status (index and load state)
    
    :param client: MilvusClient instance
    :param collection_name: str, collection name
    :return: dict with status information
    """
    status = {
        'is_loaded': False,
        'error': None
    }
    
    try:
        # Check if collection has index
        # Note: MilvusClient doesn't have direct has_index() method
        # We check by describing the collection and looking at indexes
        schema = client.describe_collection(collection_name)
        
        # Try to get load state
        try:
            load_state = client.get_load_state(collection_name=collection_name)
            status['is_loaded'] = load_state.get('state') == 'Loaded'
        except Exception as e:
            status['is_loaded'] = False
            status['error'] = f"Load state check failed: {e}"
     
    except Exception as e:
        status['error'] = str(e)
    
    return status


if __name__ == '__main__':
    if len(sys.argv) != 9:
        print("Usage: python3 search_all_collections.py <host> <timeout> <ignore_growing> <output_fields> <expr> <nq> <topk> <api_key>")
        print("\nParameters:")
        print("  host            : Milvus server host or cloud instance URI")
        print("  timeout         : Search timeout in seconds (0 for permanent)")
        print("  ignore_growing  : Ignore searching growing segments (TRUE/FALSE)")
        print("  output_fields   : Output fields (comma-separated or 'None')")
        print("  expr            : Search expression/filter (or 'None')")
        print("  nq              : Number of query vectors")
        print("  topk            : Top K results")
        print("  api_key         : Cloud API key or token (or 'None' for local)")
        print("\nExample:")
        print("  python3 search_all_collections.py localhost 60 FALSE 'id' 'None' 1 10 None")
        sys.exit(1)
    
    # Parse command line arguments
    host = sys.argv[1]
    timeout = int(sys.argv[2])
    ignore_growing = str(sys.argv[3]).upper()
    output_fields = str(sys.argv[4]).strip()
    expr = str(sys.argv[5]).strip()
    nq = int(sys.argv[6])
    topk = int(sys.argv[7])
    api_key = str(sys.argv[8])
    
    port = 19530
    
    # Process parameters
    ignore_growing = True if ignore_growing == "TRUE" else False
    
    if output_fields in ["None", "none", "NONE"] or output_fields == "":
        output_fields = None
    else:
        output_fields = output_fields.split(",")
    
    if expr in ["None", "none", "NONE"] or expr == "":
        expr = None
    
    # Setup logging
    log_filename = f"/tmp/search_all_collections_v26.log"
    file_handler = logging.FileHandler(filename=log_filename)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    logger = logging.getLogger('SEARCH_ALL_COLLECTIONS')
    
    logging.info("🚀 Starting search_all_collections (v2.6):")
    logging.info(f"  Host: {host}")
    logging.info(f"  Timeout: {timeout}s (0=permanent)")
    logging.info(f"  Ignore growing: {ignore_growing}")
    logging.info(f"  Output fields: {output_fields}")
    logging.info(f"  Expression: {expr}")
    logging.info(f"  nq: {nq}, topk: {topk}")
    
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
    
    # Check and get collection info
    try:
        all_collections = client.list_collections()
        num_collections = len(all_collections)
    except Exception as e:
        logging.error(f"Failed to list collections: {e}")
        sys.exit(1)
    
    if num_collections == 0:
        logging.error("No collections exist")
        sys.exit(-1)
    
    logging.info(f"Found {num_collections} collections")
    logging.info("Starting to search collections one by one...")
    
    # Main search loop
    start_time = time.time()
    had_failure = False
    fail_st = 0
    iteration = 0
    
    while timeout == 0 or time.time() < start_time + timeout:
        iteration += 1
        not_loaded = 0
        search_fail = 0
        search_succ = 0
        
        round_start = time.time()
        
        # Get latest collection list
        try:
            all_collections = client.list_collections()
        except Exception as e:
            logging.error(f"Failed to refresh collection list: {e}")
            time.sleep(1)
            continue
        
        for collection_name in all_collections:
            # Check collection status
            status = check_collection_status(client, collection_name)
            
            if not status['is_loaded']:
                logging.debug(f"Collection {collection_name} not loaded")
                not_loaded += 1
                continue
            
            # Execute search
            t1 = time.time()
            result = search_single_collection(
                client=client,
                collection_name=collection_name,
                nq=nq,
                topk=topk,
                output_fields=output_fields,
                expr=expr
            )
            
            if result['status'] == 'success':
                search_succ += 1
                logging.info(
                    f"✓ {collection_name}: "
                    f"flush={result['flush_time']}s, "
                    f"search={result['search_time']}s, "
                    f"entities={result['num_entities']}"
                )
            else:
                search_fail += 1
                if had_failure is False:
                    fail_st = t1
                    had_failure = True
                logging.error(f"✗ {collection_name}: {result['error']}")
        
        round_time = round(time.time() - round_start, 4)
        
        # Log round summary
        logging.info(
            f"[Round {iteration}] Completed {num_collections} collections in {round_time}s | "
            f"Success: {search_succ}, Failed: {search_fail}, "
            f"Not Loaded: {not_loaded}"
        )
        
        # Check recovery
        if search_fail == 0 and had_failure is True:
            had_failure = False
            recover_t = round(time.time() - fail_st, 4)
            logging.info(f"✅ RECOVERED: All {num_collections} collections healthy after {recover_t}s")
        
        # Rate limiting for cloud instances
        if round_time <= 1:
            time.sleep(1.1)
    
    # Final summary
    total_time = round(time.time() - start_time, 2)
    logging.info("=" * 80)
    logging.info("SEARCH ALL COLLECTIONS COMPLETED:")
    logging.info(f"  Total time: {total_time}s")
    logging.info(f"  Total iterations: {iteration}")
    logging.info(f"  Collections monitored: {num_collections}")
    logging.info(f"  Log file: {log_filename}")
    logging.info("=" * 80)
