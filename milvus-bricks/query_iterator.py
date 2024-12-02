import time
import sys
import random
import numpy as np
import threading
import logging
from pymilvus import utility, connections, DataType, \
    Collection, FieldSchema, CollectionSchema, AnnSearchRequest, RRFRanker, WeightedRanker
from common import get_float_vec_dim, get_float_vec_field_name, get_search_params, get_index_params, \
    get_float_vec_field_names, get_dim_by_field_name, is_vector_field, get_index_by_field_name
from create_n_insert import create_n_insert


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"


def query_iterator(collection, batch_size, limit, threads_num, output_fields, expr, iter_times, offset=0):
    threads_num = int(threads_num)
    interval_count = 1000

    res = collection.query(expr="", output_fields=["count(*)"])
    count_star = res[0].get("count(*)", 0)

    def search_th(col, thread_no):
        search_latency = []
        count = 0
        failures = 0
        start_times = 0
        while start_times < iter_times:
            start_times += 1
            count += 1
            t1 = time.time()
            try:
                res = col.query_iterator(expr=expr, limit=limit, batch_size=batch_size, output_fields=output_fields)
            except Exception as e:
                failures += 1
                logging.error(e)
            t2 = round(time.time() - t1, 4)
            search_latency.append(t2)
            if count == interval_count:
                total = round(np.sum(search_latency), 4)
                p99 = round(np.percentile(search_latency, 99), 4)
                avg = round(np.mean(search_latency), 4)
                qps = round(interval_count / total, 4)
                logging.info(f"collection {col.description} total failures: {failures}, query_iterator {interval_count} times "
                             f"in thread{thread_no}: cost {total}, qps {qps}, avg {avg}, p99 {p99} ")
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
        start_times = 0
        while start_times < iter_times:
            start_times += 1
            every_iter_latency = []
            iter_count = 0
            iter_ids_len = 0
            failures = 0
            t1 = time.time()
            try:
                iterator = collection.query_iterator(expr=expr, limit=limit,
                                                     iterator_cp_file="/tmp/it_cp",
                                                     batch_size=batch_size, output_fields=['id'])
                while True:
                    t1_i = time.time()
                    res = iterator.next()
                    t2_i = round(time.time() - t1_i, 4)
                    if not res:
                        iterator.close()
                        break
                    every_iter_latency.append(t2_i)
                    iter_count += 1
                    # logging.info(res)
                    iter_ids_len += len(res)
            except Exception as e:
                failures += 1
                iterator.close()
                logging.error(e)
            t2 = round(time.time() - t1, 3)
            total = t2
            p99 = round(np.percentile(every_iter_latency, 99), 3)
            avg = round(np.mean(every_iter_latency), 3)
            logging.info(f"collection {collection.description} query_iter {start_times} total failures: {failures}, "
                         f"single thread iterate the whole collection cost {total}, iterator_count: {iter_count}, "
                         f"per iterator cost: avg  {avg}, p99 {p99}")
            logging.info(f"collection {collection.description} query_iter {start_times} row count: {iter_ids_len}, "
                         f"query count(*): {count_star}, num_entities: {collection.num_entities}")


if __name__ == '__main__':
    # host = sys.argv[1]
    # name = sys.argv[2]                              # collection mame/alias
    # th = int(sys.argv[3])                           # query thread num
    # iter_times = int(sys.argv[4])                    # query iterator times
    # output_fields = str(sys.argv[5]).strip()        # output fields, default is None
    # expr = str(sys.argv[6]).strip()                 # query_iterator expression, default is None
    # batch_size = int(sys.argv[7])                   # query iterator batch size
    # limit = int(sys.argv[8])                        # query limit, -1 for the whole collection
    # api_key = str(sys.argv[9])                     # api key for cloud instances

    host = '10.104.26.205'
    name = 'test_33'
    th = 1
    iter_times = 1
    output_fields = "*"
    expr = ''
    batch_size = 2000
    limit = -1
    api_key = ""
    port = 19530

    if output_fields in ["None", "none", "NONE"] or output_fields == "":
        output_fields = None
    else:
        output_fields = output_fields.split(",")
    if expr in ["None", "none", "NONE"] or expr == "":
        expr = None
    limit = -1 if limit is None or limit == "" or limit < 1 else int(limit)
    file_handler = logging.FileHandler(filename=f"/tmp/query_iterator_{name}.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    logger = logging.getLogger('LOGGER_NAME')

    logging.info(f"query_iterator collection={name}, host={host}, thread={th}, iter_times={iter_times}, "
                 f"output_fields={output_fields}, expr={expr}, batch_size={batch_size}, "
                 f"limit={limit}, api_key={api_key}")

    if api_key is None or api_key == "" or api_key.upper() == "NONE":
        conn = connections.connect('default', host=host, port=port)
    else:
        conn = connections.connect('default', uri=host, token=api_key)

    logging.info(f"milvus collections: {utility.list_collections()}")
    # check and get the collection info
    if not utility.has_collection(collection_name=name):
        logging.error(f"collection: {name} does not exit, create 1m-768d as default")
        create_n_insert(collection_name=name, dims=[32], nb=2000, insert_times=100, auto_id=False,
                        index_types=["HNSW"], metric_types=["L2"])

    collection = Collection(name=name)
    if len(collection.indexes) == 0:
        logging.error(f"collection: {name} has no index")
        exit(-1)

    # flush before indexing
    t1 = time.time()
    num = collection.num_entities
    t2 = round(time.time() - t1, 3)
    logging.info(f"assert {name} flushed num_entities {num}: {t2}")

    # load collection
    if utility.load_state(collection.name).name.upper() != "LOADED":
        t1 = time.time()
        collection.load()
        t2 = round(time.time() - t1, 3)
        logging.info(f"assert load {name}: {t2}")

    logging.info(f"query_iterator start: batch_size{batch_size}_limit{limit}_threads{th}")
    query_iterator(collection, batch_size, limit, th, output_fields, expr, iter_times)

    logging.info(f"query_iterator completed")
