import time
import sys
import random
import logging
from pymilvus import utility, connections, DataType, \
    Collection, FieldSchema, CollectionSchema, Partition
from create_n_insert import create_n_insert
from create_n_parkey_insert import create_n_insert_parkey
from common import gen_data_by_collection, gen_str_by_length, get_dim, get_vector_field_name, get_search_params
from concurrent.futures import ThreadPoolExecutor
import threading

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"


if __name__ == '__main__':
    host = sys.argv[1]
    collection_prefix = sys.argv[2]                     # collection mame prefix
    total_requests_num = int(sys.argv[3])               # how many requests to send totally
    partition_num = int(sys.argv[4])                    # how many customized partitions(except _default) to create
    shards_num = int(sys.argv[5])                       # how many shards to create
    dim = int(sys.argv[6])                              # dim for vectors
    nb = int(sys.argv[7])                               # how many entities to insert each time
    insert_times_per_partition = int(sys.argv[8])       # how many times to insert for each partition
    partition_key_field = str(sys.argv[9]).upper()     # partition key field name, set None to disable it
    api_key = str(sys.argv[10])                         # api key to connect to milvus
    pool_size = int(sys.argv[11])                       # thread pool size
    index_type = str(sys.argv[12]).upper()              # index type

    port = 19530

    file_handler = logging.FileHandler(filename=f"/tmp/send{total_requests_num}_requests.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    logger = logging.getLogger('LOGGER_NAME')

    if api_key is None or api_key == "" or api_key.upper() == "NONE":
        conn = connections.connect('default', host=host, port=port)
    else:
        conn = connections.connect('default', uri=host, token=api_key)

    shards_num = 1 if shards_num == 0 else shards_num
    if partition_key_field == "" or partition_key_field == "NONE":
        partition_key_enabled = False
    else:
        partition_key_enabled = True

    pool = ThreadPoolExecutor(max_workers=pool_size)


    def execute(seed, dim, nb, insert_times_per_partition, index_type, shards_num, partition_key_enabled, partition_num):
        if seed % 2 == 0 or seed % 5 == 0:
            # create a collection -50%
            auto_id = random.choice([True, False])
            metric_type = random.choice(["COSINE", "L2", "IP"])
            collection_name = f"{collection_prefix}_{gen_str_by_length(8, letters_only=True)}"
            if not partition_key_enabled:
                create_n_insert(collection_name=collection_name,
                                dim=dim, nb=nb, insert_times=0, auto_id=auto_id,
                                index_type=index_type, metric_type=metric_type, build_index=True,
                                shards_num=shards_num)
            else:
                num_partitions = 64 if partition_num == 0 else partition_num
                create_n_insert_parkey(collection_name=collection_name, dim=dim, nb=nb,
                                       insert_times=0,  index_type=index_type, metric_type=metric_type,
                                       parkey_collection_only=True,
                                       parkey_values_evenly=True, num_partitions=num_partitions,
                                       pre_load=True, shards_num=shards_num)
                collection_name = f"{collection_name}_parkey"
            c = Collection(collection_name)
            c.load()
            logging.info(f"create {collection_name}  successfully")
        elif seed % 3 == 0:
            # insert -20%
            collection_name = utility.list_collections()[random.randint(0, len(utility.list_collections()) - 1)]
            c = Collection(collection_name)
            for r in range(insert_times_per_partition):
                data = gen_data_by_collection(collection=c, nb=nb, r=r)
                t1 = time.time()
                c.insert(data)
                t2 = round(time.time() - t1, 3)
                logging.info(f"{c.name} insert {r} costs {t2}")
            c.flush()

        elif seed % 5 == 0:
            pass
            logging.info(f"seed=5, pass")
        else:
            # search -20%
            collection_name = utility.list_collections()[random.randint(0, len(utility.list_collections()) - 1)]
            c = Collection(collection_name)
            dim1 = get_dim(collection=c)
            vector_field_name = get_vector_field_name(collection=c)
            topk = 10
            search_params = get_search_params(collection=c, topk=topk)
            for k in range(2):
                search_vectors = [[random.random() for _ in range(dim1)] for _ in range(1)]
                t1 = time.time()
                try:
                    c.search(data=search_vectors, anns_field=vector_field_name,
                             # output_fields=output_fields,
                             param=search_params, limit=topk)
                except Exception as e:
                    logging.error(e)
                t2 = round(time.time() - t1, 4)
                logging.info(f"{c.name} search costs {t2}")


    # check and get the collection info
    logging.info(f"start to send {total_requests_num} requests in pool")
    futures = []
    for _ in range(total_requests_num):
        seed = random.randint(1, 10)
        future = pool.submit(execute, seed=seed, dim=dim, nb=nb,
                             insert_times_per_partition=insert_times_per_partition,
                             index_type=index_type, shards_num=shards_num,
                             partition_key_enabled=partition_key_enabled, partition_num=partition_num)
        futures.append(future)
    for fu in futures:
        fu.result()

    logging.info(f"run completed")
