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
           output_fields, expr, group_by_field, timeout):
    threads_num = int(threads_num)
    interval_count = 1000
    dim = get_dim(collection)
    vector_field_name = get_vector_field_name(collection)

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
                                  group_by_field=group_by_field,
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
    name = sys.argv[2]                              # collection mame/alias
    timeout = int(sys.argv[3])                      # search timeout, permanently if 0
    output_fields = str(sys.argv[4]).strip()        # output fields, default is None
    expr = str(sys.argv[5]).strip()                 # search expression, default is None
    nq = int(sys.argv[6])                           # search nq
    topk = int(sys.argv[7])                         # search topk
    reload = str(sys.argv[8]).upper()               # reload collection before search
    api_key = str(sys.argv[9])                      # api key for cloud instances
    port = 19530

    if output_fields.upper() == "NONE" or output_fields == "":
        output_fields = None
    else:
        output_fields = output_fields.split(",")
    if expr.upper() == "NONE" or expr == "":
        expr = None

    reload = True if reload == "TRUE" else False

    file_handler = logging.FileHandler(filename=f"/tmp/cold_search_{name}.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    logger = logging.getLogger('LOGGER_NAME')

    logging.info(f"searching collection={name}, host={host}, timeout={timeout}, "
                 f"output_fields={output_fields}, expr={expr}, nq={nq}, "
                 f"topk={topk}, reload={reload}, api_key={api_key}")

    if api_key is None or api_key == "" or api_key.upper() == "NONE":
        conn = connections.connect('default', host=host, port=port)
    else:
        conn = connections.connect('default', uri=host, token=api_key)

    if name.upper() == "NONE" or name == "":
        name = utility.list_collections()[0]
        logging.info(f"collection name is not specified, use the first one: {name}")

    # check and get the collection info
    if not utility.has_collection(collection_name=name):
        logging.error(f"collection: {name} does not exit, create 10m-128d as default")
        create_n_insert(collection_name=name, dim=128, nb=20000, insert_times=50,
                        index_type="HNSW", metric_type="L2")

    collection = Collection(name=name)
    if not collection.has_index():
        logging.error(f"collection: {name} has no index")
        exit(-1)

    index_params = get_index_params(collection)
    search_params = get_search_params(collection, topk)
    logging.info(f"index param: {index_params}")
    logging.info(f"search_param: {search_params}")
    logging.info(f"output_fields: {output_fields}")
    logging.info(f"expr: {expr}")

    # flush before indexing
    num = collection.num_entities
    logging.info(f"{name} num_entities: {num}")
    logging.info(f"{name} index progress: {utility.index_building_progress(name)}")

    # release collection
    if reload:
        t1 = time.time()
        collection.release()
        t2 = round(time.time() - t1, 3)
        logging.info(f"assert release {name}: {t2}")
        time.sleep(5)

        # load collection
        t1 = time.time()
        collection.load()
        t2 = round(time.time() - t1, 3)
        logging.info(f"assert re-load {name}: {t2}")

    logging.info(f"cold search start: nq{nq}_top{topk}")
    dim = get_dim(collection)
    vector_field_name = get_vector_field_name(collection)
    start_time = time.time()
    i = 0
    while time.time() < start_time + timeout:
        search_vectors = [[random.random() for _ in range(dim)] for _ in range(nq)]
        parkey = random.randint(1, 1000)
        # exact_expr = "category==27"
        exact_expr = None if expr is None else expr + str(parkey)
        t1 = time.time()
        try:
            collection.search(data=search_vectors, anns_field=vector_field_name, output_fields=output_fields, expr=exact_expr, param=search_params, limit=topk)
        except Exception as e:
            logging.error(e)
        t2 = round(time.time() - t1, 4)
        i += 1
        logging.info(f"collection {collection.description} {i} search: cost {t2}")

    logging.info(f"cold search completed")
