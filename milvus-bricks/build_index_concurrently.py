import time
import sys
import random
import logging
from pymilvus import MilvusClient, utility, connections, DataType, \
    Collection, FieldSchema, CollectionSchema, Partition
from create_n_insert import create_n_insert
from create_n_parkey_insert import create_n_insert_parkey
from common import gen_data_by_collection
from concurrent.futures import ThreadPoolExecutor
from common import insert_entities, get_float_vec_field_names, get_default_params_by_index_type

import threading

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"


if __name__ == '__main__':
    host = sys.argv[1]
    collection_name = sys.argv[2]                       # collection mame prefix
    build_times = int(sys.argv[3]).upper()              # build index times
    pool_size = int(sys.argv[4])                       # thread pool size
    api_key = str(sys.argv[5])                         # api key to connect to milvu

    port = 19530

    file_handler = logging.FileHandler(filename=f"/tmp/build_index_{collection_name}_{build_times}.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    logger = logging.getLogger('LOGGER_NAME')

    if api_key is None or api_key == "" or api_key.upper() == "NONE":
        conn = connections.connect('default', host=host, port=port)
    else:
        conn = connections.connect('default', uri=host, token=api_key)

    pool = ThreadPoolExecutor(max_workers=pool_size)


    def execute(h, name, j):
        # build index
        supported_vector_types = [DataType.FLOAT_VECTOR, DataType.FLOAT16_VECTOR, DataType.BFLOAT16_VECTOR]
        index_type = "HNSW"
        metric_type = "COSINE"
        index_params = get_default_params_by_index_type(index_type, metric_type)
        # build index for all the scalar fields
        client = MilvusClient(uri=f"http://{h}:19530")
        c_info = client.describe_collection(name)
        fields = c_info.get('fields', None)
        for field in fields:
            if field['type'] not in supported_vector_types and field['type'] != DataType.JSON:
                index_params = client.prepare_index_params()
                index_params.add_index(field_name=field['name'])
                try:
                    client.create_index(collection_name=name, index_params=index_params)
                    logging.info(f"build index_{j} for scalar field: {field['name']}")
                except Exception as e:
                    logging.error(e)
                    pass
                continue
            if field['type'] in supported_vector_types:
                index_params = client.prepare_index_params()
                index_params.add_index(field_name=field['name'], metric_type=metric_type)
                try:
                    t0 = time.time()
                    client.create_index(collection_name=name, index_params=index_params)
                    tt = round(time.time() - t0, 3)
                    logging.info(f"build index_{j} {index_params} costs {tt}")
                except Exception as e:
                    logging.error(e)
                    pass

    # check and get the collection info
    if not utility.has_collection(collection_name=collection_name):
        create_n_insert(collection_name=collection_name, vector_types=[DataType.FLOAT_VECTOR],
                        dims=[128], nb=2000, insert_times=10, auto_id=False,
                        index_types=["HNSW"], metric_types=['COSINE'], build_index=False,
                        use_insert=True)

    # c = Collection(collection_name)
    logging.info(f"start")
    futures=[]
    for i in range(build_times):
        future=pool.submit(execute, host, collection_name, i)
        futures.append(future)
    for fu in futures:
        fu.result()

    logging.info(f"build index for {build_times} times completed")
