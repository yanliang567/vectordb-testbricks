import time
import sys
import random
import numpy as np
import threading
import logging
from pymilvus import utility, connections, DataType, \
    Collection, FieldSchema, CollectionSchema
from common import get_dim, get_vector_field_name, get_search_params, get_index_params
from create_n_insert import create_n_insert


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"


def search(collection, search_params, nq, topk, threads_num,
           output_fields, expr, timeout):
    threads_num = int(threads_num)
    interval_count = 1000
    dim = get_dim(collection)
    vector_field_name = get_vector_field_name(collection)

    def search_th(col, thread_no):
        search_latency = []
        count = 0
        start_time = time.time()
        while time.time() < start_time + timeout:
            count += 1
            search_vectors = [[random.random() for _ in range(dim)] for _ in range(nq)]
            parkey = random.randint(1, 1000)
            exact_expr = None if expr is None else expr+str(parkey)
            t1 = time.time()
            try:
                col.search(data=search_vectors, anns_field=vector_field_name,
                           param=search_params, limit=topk,
                           expr=exact_expr,
                           output_fields=output_fields)
            except Exception as e:
                logging.error(e)
            t2 = round(time.time() - t1, 4)
            search_latency.append(t2)
            if count == interval_count:
                total = round(np.sum(search_latency), 4)
                p99 = round(np.percentile(search_latency, 99), 4)
                avg = round(np.mean(search_latency), 4)
                qps = round(interval_count / total, 4)
                logging.info(f"collection {col.description} search {interval_count} times in thread{thread_no}: cost {total}, qps {qps}, avg {avg}, p99 {p99} ")
                count = 0
                search_latency = []

    threads = []
    if threads_num > 1:
        for i in range(threads_num):
            t = threading.Thread(target=search_th, args=(collection, i))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
    else:
        interval_count = 100
        search_latency = []
        count = 0
        start_time = time.time()
        while time.time() < start_time + timeout:
            count += 1
            search_vectors = [[random.random() for _ in range(dim)] for _ in range(nq)]
            parkey = random.randint(1, 1000)
            exact_expr = None if expr is None else expr+str(parkey)
            t1 = time.time()
            try:
                collection.search(data=search_vectors, anns_field=vector_field_name,
                                  output_fields=output_fields, expr=exact_expr,
                                  param=search_params, limit=topk)

            except Exception as e:
                logging.error(e)
            t2 = round(time.time() - t1, 4)
            search_latency.append(t2)
            if count == interval_count:
                total = round(np.sum(search_latency), 4)
                p99 = round(np.percentile(search_latency, 99), 4)
                avg = round(np.mean(search_latency), 4)
                qps = round(interval_count / total, 4)
                logging.info(f"collection {collection.description} search {interval_count} times single thread: cost {total}, qps {qps}, avg {avg}, p99 {p99} ")
                count = 0
                search_latency = []


if __name__ == '__main__':
    host = sys.argv[1]
    name = sys.argv[2]                  # collection mame/alias
    th = int(sys.argv[3])               # search thread num
    timeout = int(sys.argv[4])          # search timeout, permanently if 0
    ignore_growing = str(sys.argv[5]).upper()   # ignore searching growing segments if True
    output_fields = str(sys.argv[6]).strip()       # output fields, default is None
    expr = str(sys.argv[7]).strip()                # search expression, default is None
    nq = int(sys.argv[8])               # search nq
    topk = int(sys.argv[9])             # search topk
    api_key = str(sys.argv[10])         # api key for cloud instances
    port = 19530

    ignore_growing = True if ignore_growing == "TRUE" else False
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

    if api_key is None or api_key == "" or api_key.upper() == "NONE":
        conn = connections.connect('default', host=host, port=port)
    else:
        conn = connections.connect('default', uri=host, token=api_key)

    # check and get the collection info
    if not utility.has_collection(collection_name=name):
        logging.error(f"collection: {name} does not exit, create 10m-128d as default")
        create_n_insert(collection_name=name, dim=128, nb=20000, insert_times=50,
                        index_type="HNSW", metric_type="L2")

    collection = Collection(name=name)
    if not collection.has_index():
        logging.error(f"collection: {name} has no index")
        exit(0)

    index_params = get_index_params(collection)
    search_params = get_search_params(collection, topk)
    logging.info(f"index param: {index_params}")
    logging.info(f"search_param: {search_params}")
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

    logging.info(f"search start: nq{nq}_top{topk}_threads{th}")
    search(collection, search_params, nq, topk, th, output_fields, expr, timeout)
    logging.info(f"search completed")
