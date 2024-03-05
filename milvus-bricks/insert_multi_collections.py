import time
import sys
import random
from sklearn import preprocessing
import threading
import logging
from pymilvus import utility, connections, DataType, \
    Collection, FieldSchema, CollectionSchema
from common import gen_data_by_collection
from create_n_insert import create_n_insert
from concurrent.futures import ThreadPoolExecutor


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"


if __name__ == '__main__':
    host = sys.argv[1]                                  # host address
    nb = int(sys.argv[2])                               # number of vectors per insert request
    pool_size = int(sys.argv[3])                        # insert thread pool size
    ins_times = int(sys.argv[4])                        # total insert times for all collections
    flush_after_insert = str(sys.argv[5]).upper()       # if flush after every insert request
    port = 19530

    file_handler = logging.FileHandler(filename=f"/tmp/insert_multi_collections_concurrently.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    logger = logging.getLogger('LOGGER_NAME')

    conn = connections.connect('default', host=host, port=port)
    flush = True if flush_after_insert == "TRUE" else False
    logging.info(f"host={host}, nb={nb}, pool_size={pool_size}, ins_times={ins_times}, flush={flush}")

    # check and get the collection info
    collection_names = utility.list_collections()
    if len(collection_names) == 0:
        logging.error(f"found 0 collections, exit")
        exit(0)

    logging.info(f"found {len(collection_names)} collections ready for inserting")

    pool = ThreadPoolExecutor(max_workers=pool_size)

    def do_insert(c_names, i, flush):
        collection_name = collection_names[random.randint(0, len(c_names) - 1)]
        c = Collection(collection_name)
        data = gen_data_by_collection(c, nb, i)
        t1 = time.time()
        c.insert(data)
        t2 = round(time.time() - t1, 3)
        logging.info(f"insert times: {i}, into collection {c.name} costs {t2}")
        if flush is True
            t1 = time.time()
            c.flush()
            t2 = round(time.time() - t1, 3)
            logging.info(f"insert times: {i},  collection {c.name} flush costs {t2}")

    futures = []
    t1 = time.time()
    for i in range(ins_times):
        future = pool.submit(do_insert, collection_names, i, flush)
        futures.append(future)
    for fu in futures:
        fu.result()
    t2 = round(time.time() - t1, 3)
    logging.info(f"insert multi collections concurrently completed")

