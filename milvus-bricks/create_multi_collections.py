import time
import sys
import random
import logging
from pymilvus import utility, connections, DataType, \
    Collection, FieldSchema, CollectionSchema, Partition
from create_n_insert import create_n_insert
from common import gen_data_by_collection


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"


if __name__ == '__main__':
    host = sys.argv[1]
    collection_prefix = sys.argv[2]                 # collection mame prefix
    collection_num = int(sys.argv[3])               # how many collections to create
    partition_num = int(sys.argv[4])                # how many customized partitions to create
    need_insert = str(sys.argv[5]).upper()          # insert or not, if yes, it inserts random number of entities
    need_build_index = str(sys.argv[6]).upper()     # build index or not after insert
    need_load = str(sys.argv[7]).upper()            # load the collection or not at the end
    api_key = str(sys.argv[8])                      # api key to connect to milvus

    port = 19530

    file_handler = logging.FileHandler(filename=f"/tmp/create_{collection_num}_collections.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    logger = logging.getLogger('LOGGER_NAME')

    if api_key is None or api_key == "":
        conn = connections.connect('default', host=host, port=port)
    else:
        conn = connections.connect('default', uri=host, token=api_key)

    need_insert = True if need_insert == "TRUE" else False
    need_build_index = True if need_build_index == "TRUE" else False
    need_load = True if need_load == "TRUE" else False

    # check and get the collection info
    logging.info(f"start to create {collection_num} collections")
    for i in range(collection_num):
        collection_name = f"{collection_prefix}_{i}"
        insert_times = random.randint(2, 10) if need_insert else 0
        if not utility.has_collection(collection_name=collection_name):
            dim = random.randint(100, 1000)
            auto_id = random.choice([True, False])
            create_n_insert(collection_name=collection_name,
                            dim=dim, nb=1000, insert_times=insert_times, auto_id=auto_id,
                            index_type="AUTOINDEX", metric_type="L2", build_index=need_build_index)
            logging.info(f"create {collection_name}  successfully")
        else:
            logging.info(f"{collection_name} already exists")

        c = Collection(collection_name)
        for j in range(partition_num):
            partition_name = f"partition_{j}"
            p = Partition(collection=c, name=partition_name)
            for r in range(insert_times):
                data = gen_data_by_collection(collection=c, nb=200, r=r)
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

    logging.info(f"create multi collections and partitions completed")
