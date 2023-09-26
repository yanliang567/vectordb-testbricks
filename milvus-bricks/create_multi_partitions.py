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
    collection_name = sys.argv[2]                   # collection mame
    numbers = int(sys.argv[3])                      # how many partitions to create
    need_build_index = str(sys.argv[4]).upper()     # build index or not after insert
    need_load = str(sys.argv[5]).upper()            # load the collection or not at the end

    port = 19530

    file_handler = logging.FileHandler(filename=f"/tmp/create_{numbers}_partitions.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    logger = logging.getLogger('LOGGER_NAME')

    conn = connections.connect('default', host=host, port=port)

    need_build_index = True if need_build_index == "TRUE" else False
    need_load = True if need_load == "TRUE" else False

    # check and get the collection info
    nb = 100
    insert_times = 5
    if not utility.has_collection(collection_name=collection_name):
        logging.info(f"start to create default collection")
        dim = 768
        create_n_insert(collection_name=collection_name,
                        dim=dim, nb=nb, insert_times=insert_times, auto_id=False,
                        index_type="HNSW", metric_type="L2", build_index=need_build_index)
        logging.info(f"create {collection_name}  successfully")

    c = Collection(collection_name)
    logging.info(f"start to create {numbers} partitions")
    for i in range(numbers):
        partition_name = f"partition_{i}"
        p = Partition(collection=c, name=partition_name)
        for r in range(insert_times):
            data = gen_data_by_collection(collection=c, nb=nb, r=r)
            t1 = time.time()
            p.insert(data)
            t2 = round(time.time() - t1, 3)
            logging.info(f"{partition_name} insert {r} costs {t2}")

        if need_load:
            t1 = time.time()
            p.load()
            t2 = round(time.time() - t1, 3)
            logging.info(f"{partition_name} load in {t2}")

    logging.info(f"create multi partitions completed")