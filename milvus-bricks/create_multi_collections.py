import time
import sys
import random
import logging
from pymilvus import utility, connections, DataType, \
    Collection, FieldSchema, CollectionSchema, Partition
from create_n_insert import create_n_insert
from create_n_parkey_insert import create_n_insert_parkey
from common import gen_data_by_collection
from concurrent.futures import ThreadPoolExecutor
import threading

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"


if __name__ == '__main__':
    host = sys.argv[1]
    collection_prefix = sys.argv[2]                     # collection mame prefix
    collection_num = int(sys.argv[3])                   # how many collections to create
    partition_num = int(sys.argv[4])                    # how many customized partitions(except _default) to create
    shards_num = int(sys.argv[5])                       # how many shards to create
    dim = int(sys.argv[6])                              # dim for vectors
    nb = int(sys.argv[7])                               # how many entities to insert each time
    insert_times_per_partition = int(sys.argv[8])       # how many times to insert for each partition
    need_insert = str(sys.argv[9]).upper()              # insert or not, if yes, it inserts random vectors
    need_build_index = str(sys.argv[10]).upper()        # build index or not after insert
    need_load = str(sys.argv[11]).upper()               # load the collection or not at the end
    partition_key_field = str(sys.argv[12]).upper()     # partition key field name, set None to disable it
    api_key = str(sys.argv[13])                         # api key to connect to milvus
    pool_size = int(sys.argv[14])                       # thread pool size
    index_type = str(sys.argv[15])                      # index type

    port = 19530

    file_handler = logging.FileHandler(filename=f"/tmp/create_{collection_num}_collections.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    logger = logging.getLogger('LOGGER_NAME')

    if api_key is None or api_key == "" or api_key.upper() == "NONE":
        conn = connections.connect('default', host=host, port=port)
    else:
        conn = connections.connect('default', uri=host, token=api_key)

    shards_num = 1 if shards_num == 0 else shards_num
    need_insert = True if need_insert == "TRUE" else False
    need_build_index = True if need_build_index == "TRUE" else False
    need_load = True if need_load == "TRUE" else False
    if partition_key_field == "" or partition_key_field == "NONE":
        partition_key_enabled = False
    else:
        partition_key_enabled = True

    pool = ThreadPoolExecutor(max_workers=pool_size)


    def execute(collection_name):
        if not utility.has_collection(collection_name=collection_name):
            auto_id = random.choice([True, False])
            metric_type = random.choice(["COSINE", "L2", "IP"])
            if not partition_key_enabled:
                create_n_insert(collection_name=collection_name,
                                dim=dim, nb=nb, insert_times=insert_times_per_partition, auto_id=auto_id,
                                index_type=index_type, metric_type=metric_type, build_index=need_build_index,
                                shards_num=shards_num)
            else:
                num_partitions = 64 if partition_num == 0 else partition_num
                create_n_insert_parkey(collection_name=collection_name, dim=dim, nb=nb, insert_times=insert_times_per_partition,
                                       index_type=index_type, metric_type=metric_type,
                                       parkey_collection_only=True,
                                       parkey_values_evenly=True, num_partitions=num_partitions,
                                       pre_load=False, shards_num=shards_num)
                collection_name = f"{collection_name}_parkey"
            logging.info(f"create {collection_name}  successfully")
        else:
            logging.info(f"{collection_name} already exists")

        c = Collection(collection_name)
        if not partition_key_enabled:
            for j in range(partition_num):
                partition_name = f"partition_{j}"
                p = Partition(collection=c, name=partition_name)
                for r in range(insert_times_per_partition):
                    data = gen_data_by_collection(collection=c, nb=nb, r=r)
                    t1 = time.time()
                    p.insert(data)
                    t2 = round(time.time() - t1, 3)
                    logging.info(f"{partition_name} insert {r} costs {t2}")

        if need_load:
            c = Collection(name=collection_name)
            t1 = time.time()
            c.load()
            t2 = round(time.time() - t1, 3)
            logging.info(f"{collection_name} load in {t2}")

    # check and get the collection info
    logging.info(f"start to create {collection_num} collections")
    futures=[]
    for i in range(collection_num):
        collection_name = f"{collection_prefix}_{i}"
        future=pool.submit(execute, collection_name)
        futures.append(future)
    for fu in futures:
        fu.result()

    logging.info(f"create multi collections and partitions completed")
