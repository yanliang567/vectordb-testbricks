import time
import sys
import random
import numpy as np
import threading
import logging
from pymilvus import utility, connections, DataType, \
    Collection, FieldSchema, CollectionSchema
from common import get_float_vec_dim, get_float_vec_field_name, get_search_params, get_index_params
from create_n_insert import create_n_insert


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"

if __name__ == '__main__':
    host = sys.argv[1]                              # milvus host or cloud instance uri
    timeout = int(sys.argv[2])                      # search timeout, permanently if 0
    ignore_growing = str(sys.argv[3]).upper()       # ignore searching growing segments if True
    output_fields = str(sys.argv[4]).strip()        # output fields, default is None
    expr = str(sys.argv[5]).strip()                 # search expression, default is None
    nq = int(sys.argv[6])                           # search nq
    topk = int(sys.argv[7])                         # search topk
    api_key = str(sys.argv[8])                     # cloud api key or token
    port = 19530

    ignore_growing = True if ignore_growing == "TRUE" else False
    if output_fields in ["None", "none", "NONE"] or output_fields == "":
        output_fields = None
    else:
        output_fields = output_fields.split(",")
    if expr in ["None", "none", "NONE"] or expr == "":
        expr = None
    file_handler = logging.FileHandler(filename=f"/tmp/search_all_collections.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    logger = logging.getLogger('LOGGER_NAME')

    logging.info(f"host: {host}, timeout: {timeout}, ignore_growing: {ignore_growing}, output_fields: {output_fields},"
                 f" expr: {expr}, nq: {nq}, topk: {topk}, api_key: {api_key}")

    if api_key is None or api_key == "" or api_key.upper() == "NONE":
        conn = connections.connect('default', host=host, port=port)
    else:
        conn = connections.connect('default', uri=host, token=api_key)
    # check and get the collection info
    num_collections = len(utility.list_collections())
    if num_collections == 0:
        logging.error(f"no collections exist")
        exit(-1)
    logging.info(f"there are {num_collections} collections")
    logging.info(f"start to search collections one by one...")
    logging.info(f"nq: {nq}, topk: {topk}, output_fields: {output_fields}, expr: {expr}")

    start_time = time.time()
    had_failure = False
    fail_st = 0
    while time.time() < start_time + timeout:
        no_index = 0
        not_loaded = 0
        search_fail = 0
        search_succ = 0
        tt = time.time()
        for name in utility.list_collections():
            collection = Collection(name=name)
            # logging.info(f"collection: {name}")
            if not collection.has_index():
                logging.info(f"collection: {name} has no index")
                no_index += 1
                continue
            try:
                if not utility.load_state(name).name == "Loaded":
                    logging.info(f"collection: {name} not loaded")
                    not_loaded += 1
                    continue
            except Exception as e:
                not_loaded += 1
                continue
            try:
                t1 = time.time()
                collection.flush()
                num = collection.num_entities
                logging.info(f"collection {name} flushed in {round(time.time() - t1, 3)}, num_entities {num}")
                t1 = time.time()
                dim = get_float_vec_dim(collection)
                index_params = get_index_params(collection)
                search_params = get_search_params(collection, topk)
                vector_field_name = get_float_vec_field_name(collection)
                search_vectors = [[random.random() for _ in range(dim)] for _ in range(nq)]
                parkey = random.randint(1, 1000)
                exact_expr = None if expr is None else expr + str(parkey)
                collection.search(data=search_vectors, anns_field=vector_field_name,
                                  output_fields=output_fields, expr=exact_expr,
                                  param=search_params, limit=topk)
                search_succ += 1
            except Exception as e:
                if had_failure is False:
                    fail_st = t1
                    had_failure = True
                search_fail += 1
                logging.error(e)
            t2 = round(time.time() - t1, 4)
            # logging.info(f"collection {name} search cost {t2}")
        tt2 = round(time.time() - tt, 4)
        logging.info(f"complete {num_collections} collections in {tt2}, search_succ: {search_succ}, "
                     f"search_fail: {search_fail}，no_index: {no_index}, not_loaded: {not_loaded}")
        if search_fail == 0 and had_failure is True:
            had_failure = False
            recover_t = round(time.time() - fail_st, 4)
            logging.info(f"recover time for {num_collections} collections is {recover_t}")
        if tt2 <= 1:
            time.sleep(1.1)       # sleep to avoid too frequent requests on cloud instances

