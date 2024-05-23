import time
import sys
import random
import numpy as np
import threading
import logging
from pymilvus import utility, connections, DataType, \
    Collection, FieldSchema, CollectionSchema
from common import get_index_params, get_float_vec_dim
from create_n_insert import create_n_insert


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"


def query(collection,  threads_num, output_fields, expr, timeout):
    threads_num = int(threads_num)
    interval_count = 100

    def query_th(col, thread_no):
        query_latency = []
        count = 0
        start_time = time.time()
        while time.time() < start_time + timeout:
            count += 1
            t1 = time.time()
            if expr in ["random", "RANDOM", "Random"]:
                seed = random.randint(0, 2000)
                expr_seed = f"{seed-100}<=age<={seed+100}"
            try:
                res = col.query(expr=expr_seed, output_fields=output_fields, timeout=5)
            except Exception as e:
                logging.error(e)
            t2 = round(time.time() - t1, 4)
            query_latency.append(t2)
            if count == interval_count:
                total = round(np.sum(query_latency), 4)
                p99 = round(np.percentile(query_latency, 99), 4)
                avg = round(np.mean(query_latency), 4)
                qps = round(interval_count / total, 4)
                logging.info(f"collection {col.description} query {interval_count} times in thread{thread_no}: cost {total}, qps {qps}, avg {avg}, p99 {p99} ")
                count = 0
                query_latency = []

    threads = []
    if threads_num > 1:
        for i in range(threads_num):
            t = threading.Thread(target=query_th, args=(collection, i))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
    else:
        query_latency = []
        count = 0
        start_time = time.time()
        while time.time() < start_time + timeout:
            count += 1
            t1 = time.time()
            if expr in ["random", "RANDOM", "Random"]:
                seed = random.randint(0, 2000)
                expr_seed = f"{seed-100}<=age<={seed+100}"
            try:
                res = collection.query(expr=expr_seed, output_fields=output_fields, timeout=3)
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
                logging.info(f"collection {collection.description} query {interval_count} times single thread: cost {total}, qps {qps}, avg {avg}, p99 {p99} ")
                count = 0
                query_latency = []


if __name__ == '__main__':
    host = sys.argv[1]
    name = sys.argv[2]                  # collection mame/alias
    th = int(sys.argv[3])               # search thread num
    timeout = int(sys.argv[4])          # search timeout, permanently if 0
    output_fields = str(sys.argv[5]).strip()       # output fields, default is None
    expr = str(sys.argv[6]).strip()                # query expression, default is None
    port = 19530

    if output_fields in ["None", "none", "NONE"] or output_fields == "":
        output_fields = None
    else:
        output_fields = output_fields.split(",")
    if expr in ["None", "none", "NONE"] or expr == "":
        expr = None
    file_handler = logging.FileHandler(filename=f"/tmp/search_{name}.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    logger = logging.getLogger('LOGGER_NAME')

    conn = connections.connect('default', host=host, port=port)

    # check and get the collection info
    if not utility.has_collection(collection_name=name):
        logging.error(f"collection: {name} does not exit, create 10m-128d as default")
        create_n_insert(collection_name=name, dim=128, nb=20000, insert_times=50,
                        index_type="AUTOINDEX", metric_type="L2")

    collection = Collection(name=name)
    dim = get_float_vec_dim(collection)

    if not collection.has_index():
        logging.error(f"collection: {name} has no index")
        exit(-1)

    index_params = get_index_params(collection)
    logging.info(f"index param: {index_params}")
    logging.info(f"output_fields: {output_fields}")
    logging.info(f"expr: {expr}")

    # flush before indexing
    t1 = time.time()
    num = collection.num_entities
    t2 = round(time.time() - t1, 3)
    logging.info(f"assert {name} flushed num_entities {num}: {t2}")

    logging.info(utility.index_building_progress(name))

    # load collection
    t1 = time.time()
    collection.load()
    t2 = round(time.time() - t1, 3)
    logging.info(f"assert load {name}: {t2}")

    logging.info(f"query start: _threads{th}")
    query(collection, th, output_fields, expr, timeout)
    logging.info(f"search completed")
