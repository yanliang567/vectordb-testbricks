# write a test of pymilvus client, which is used to run in the AWS Lambda

import os
from pymilvus import MilvusClient, DataType
import random
# import numpy as np
# import preprocessing
import time

def gen_vectors(nb, dim, vector_data_type=DataType.FLOAT_VECTOR):
    vectors = []
    if vector_data_type == DataType.FLOAT_VECTOR:
        vectors = [[random.random() for _ in range(dim)] for _ in range(nb)]
    else:
        raise Exception(f"Invalid vector data type: {vector_data_type}")
    # if dim > 1:
    #     if vector_data_type == DataType.FLOAT_VECTOR:
    #         vectors = preprocessing.normalize(vectors, axis=1, norm='l2')
    #         vectors = vectors.tolist()
    return vectors


def lambda_handler(event, context):
    print(event)    
    print(context)

    # connect to milvus
    # uri = os.environ.get('MILVUS_URI')
    # token = os.environ.get('MILVUS_TOKEN')
    uri = "https://in01-17a2861836e7fee.aws-us-west-2.vectordb.zillizcloud.com:19541"
    token = "e86aa06e367d53c8c1546aee76efc8bf8dd5448fc19a4c7af7ca7cd5e42a2c165b0ffebe030e7a7f0854a6bb88441ff1d1cca179"
    client = MilvusClient(uri=uri, token=token)
    
    collection_name = 'test_collection'
    default_dim = 64
   
    # Search
    import concurrent.futures
    from concurrent.futures import ThreadPoolExecutor

    default_search_exp = "id >= 0"
    search_params = {}
    timeout = 898
    start_time = time.time()
    
    def do_search():
        while time.time() - start_time < timeout:
            t0 = time.time()
            vectors_to_search = gen_vectors(1, default_dim)
            search_res = client.search(
                collection_name,
                vectors_to_search,
                anns_field="embeddings", 
                search_params=search_params,
                limit=10)
            t1 = time.time()
            
            # Verify count
            query_res = client.query(
                collection_name,
                filter=default_search_exp,
                output_fields=['count(*)'],
            )

    # 使用线程池并发执行search
    workers = 4
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(do_search) for _ in range(workers)]
        concurrent.futures.wait(futures)
        
    print(f"completed concurrent search and query for {timeout}s")

    return {
        'statusCode': 200,
        'body': 'success'
    }