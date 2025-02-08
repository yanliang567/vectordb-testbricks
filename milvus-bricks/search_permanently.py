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


def hybrid_search(collection, vec_field_names, nq, topk, threads_num, output_fields, expr, group_by_field, timeout):
    threads_num = int(threads_num)
    interval_count = 1000
    req_list = []
    # ranker = WeightedRanker(*weights)
    ranker = RRFRanker()
    # vec_field_names = get_float_vec_field_names(collection)
    for vec_field_name in vec_field_names:
        dim = get_dim_by_field_name(collection, vec_field_name)
        search_vectors = [[random.random() for _ in range(dim)] for _ in range(nq)]
        search_param = {
            "data": search_vectors,
            "anns_field": vec_field_name,
            "param": {},
            "limit": topk,
            "expr": None}
        req = AnnSearchRequest(**search_param)
        req_list.append(req)
        logging.info(f"hybrid_search on field:{vec_field_name}, dim:{dim}")

    def hybrid_search_th(col, thread_no):
        search_latency = []
        count = 0
        start_time = time.time()
        while time.time() < start_time + timeout:
            count += 1
            t1 = time.time()
            try:
                col.hybrid_search(reqs=req_list, rerank=ranker, limit=topk,
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
                logging.info(f"collection {col.description} hybrid_search {interval_count} times in thread{thread_no}: cost {total}, qps {qps}, avg {avg}, p99 {p99} ")
                count = 0
                search_latency = []

    threads = []
    if threads_num > 1:
        for i in range(threads_num):
            t = threading.Thread(target=hybrid_search_th, args=(collection, i))
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
            t1 = time.time()
            try:
                collection.hybrid_search(reqs=req_list, rerank=ranker, limit=topk,
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
                logging.info(
                    f"collection {collection.description} hybrid_search {interval_count} times single thread: cost {total}, qps {qps}, avg {avg}, p99 {p99} ")
                count = 0
                search_latency = []


def search(collection, partitions, index_0, nq, topk, threads_num, output_fields, expr, group_by_field, timeout):
    threads_num = int(threads_num)
    interval_count = 1000
    # vector_field_name = get_float_vec_field_name(collection)

    # index_0 = collection.indexes[0]
    index_name = index_0.index_name
    vector_field_name = index_0.field_name
    dim = get_dim_by_field_name(collection, vector_field_name)
    index_params = get_index_params(collection, index_name=index_name)
    search_params = get_search_params(collection, topk, index_name=index_name)
    num_partitions = len(collection.partitions)
    logging.info(f"search on vector_field_name:{vector_field_name}, dim:{dim}")
    logging.info(f"index_name:{index_name}, index_param: {index_params}")
    logging.info(f"search_param: {search_params}")
    logging.info(f"output_fields: {output_fields}")
    logging.info(f"expr: {expr}")
    logging.info(utility.index_building_progress(collection.name, index_name=index_name))

    def search_th(col, thread_no):
        search_latency = []
        count = 0
        failures = 0
        start_time = time.time()
        while time.time() < start_time + timeout:
            count += 1
            search_vectors = [[random.random() for _ in range(dim)] for _ in range(nq)]
            parkey = random.randint(1, 1000)
            exact_expr = None if expr is None else expr+str(parkey)
            partition_names = None
            if partitions is None:
                partition_names = None
            elif partitions.__class__ == "str" and partitions.upper() == "RANDOM" and num_partitions > 1:
                partition_names = [f"partition_{random.randint(0, num_partitions-2)}"]
                logging.info(f"search on partition: {partition_names}")
            elif partitions.__class__ == "list":
                partition_names = partitions
            t1 = time.time()
            try:
                res = col.search(data=search_vectors, anns_field=vector_field_name,
                                 partition_names=partition_names,
                                 param=search_params, limit=topk,
                                 expr=exact_expr, group_by_field=group_by_field,
                                 output_fields=output_fields)
                if len(res[0]) != topk:
                    logging.info(f"search results do not meet topk, expected:{topk}, actual:{len(res[0])}")
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
                logging.info(f"collection {col.description} total failures: {failures}, search {interval_count} times "
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
        interval_count = 100
        search_latency = []
        count = 0
        failures = 0
        start_time = time.time()
        while time.time() < start_time + timeout:
            count += 1
            search_vectors = [[random.random() for _ in range(dim)] for _ in range(nq)]
            parkey = random.randint(1, 1000)
            exact_expr = None if expr is None else expr+str(parkey)
            partition_names = None
            if partitions is None:
                partition_names = None
            elif partitions.__class__ == "str" and partitions.upper() == "RANDOM" and num_partitions > 1:
                partition_names = [f"partition_{random.randint(0, num_partitions - 2)}"]
            elif partitions.__class__ == "list":
                partition_names = partitions
            t1 = time.time()
            try:
                res = collection.search(data=search_vectors, anns_field=vector_field_name,
                                        partition_names=partition_names,
                                        output_fields=output_fields, expr=exact_expr,
                                        group_by_field=group_by_field,
                                        param=search_params, limit=topk)
                if len(res[0]) != topk:
                    logging.info(f"search results do not meet topk, expected:{topk}, actual:{len(res[0])}")
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
                logging.info(f"collection {collection.description} total failures: {failures}, search {interval_count}"
                             f" times single thread: cost {total}, qps {qps}, avg {avg}, p99 {p99}")
                count = 0
                search_latency = []


if __name__ == '__main__':
    host = sys.argv[1]
    name = sys.argv[2]                              # collection mame/alias
    vec_field_names = str(sys.argv[3]).strip()      # vector field names, default is None, meaning all vector fields
    use_hybrid_search = str(sys.argv[4]).upper()    # run hybrid search if True
    th = int(sys.argv[5])                           # search thread num
    timeout = int(sys.argv[6])                      # search timeout, permanently if 0
    ignore_growing = str(sys.argv[7]).upper()       # ignore searching growing segments if True
    output_fields = str(sys.argv[8]).strip()        # output fields, default is None
    expr = str(sys.argv[9]).strip()                 # search expression, default is None
    nq = int(sys.argv[10])                          # search nq
    topk = int(sys.argv[11])                        # search topk
    group_by_field = str(sys.argv[12]).strip()      # group by field, default is None
    api_key = str(sys.argv[13])                     # api key for cloud instances
    port = 19530

    ignore_growing = True if ignore_growing == "TRUE" else False
    if output_fields in ["None", "none", "NONE"] or output_fields == "":
        output_fields = None
    else:
        output_fields = output_fields.split(",")
    if expr in ["None", "none", "NONE"] or expr == "":
        expr = None
    if group_by_field in ["None", "none", "NONE"] or group_by_field == "":
        group_by_field = None
    use_hybrid_search = True if use_hybrid_search in ["TRUE", "YES"] else False
    if vec_field_names in ["None", "none", "NONE"] or vec_field_names == "":
        vec_field_names = None
    else:
        vec_field_names = vec_field_names.split(",")
    file_handler = logging.FileHandler(filename=f"/tmp/search_{name}.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    logger = logging.getLogger('LOGGER_NAME')

    logging.info(f"searching collection={name}, host={host}, vec_fields={vec_field_names}, "
                 f"use_hybrid={use_hybrid_search}, thread={th}, timeout={timeout}, "
                 f"ignore_growing={ignore_growing}, output_fields={output_fields}, expr={expr}, nq={nq}, "
                 f"topk={topk}, group_by_filed={group_by_field}, api_key={api_key}")

    if api_key is None or api_key == "" or api_key.upper() == "NONE":
        conn = connections.connect('default', host=host, port=port)
    else:
        conn = connections.connect('default', uri=host, token=api_key)

    # check and get the collection info
    if not utility.has_collection(collection_name=name):
        logging.error(f"collection: {name} does not exit, create 10m-128d as default")
        exit(-1)

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

    logging.info(f"search start: nq{nq}_top{topk}_threads{th}")
    if use_hybrid_search:
        if vec_field_names is None:
            vec_field_names = get_float_vec_field_names(collection)
        for name in vec_field_names:
            if name not in get_float_vec_field_names(collection):
                logging.error(f"field {name} is not a vector field")
                exit(-1)
        vec_field_names = [name for name in vec_field_names if name in get_float_vec_field_names(collection)]
        hybrid_search(collection, vec_field_names, nq, topk, th, output_fields, expr, group_by_field, timeout)
    else:
        if vec_field_names is None:
            vec_field_names = get_float_vec_field_names(collection)
        index = get_index_by_field_name(collection, vec_field_names[0])
        partitions = "RANDOM"
        search(collection, partitions, index, nq, topk, th, output_fields, expr, group_by_field, timeout)

    logging.info(f"search completed")
