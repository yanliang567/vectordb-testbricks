import time
import sys
import random
import numpy as np
import threading
import logging
from pymilvus import utility, connections, DataType, \
    Collection, FieldSchema, CollectionSchema

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"


# def query(collection, expr, output_fields=None, threads_num=1, timeout=300):
#     threads_num = int(threads_num)
#     interval_count = 1000
#
#     def query_th(col, thread_no):
#         query_latency = []
#         count = 0
#         start_time = time.time()
#         while time.time() < start_time + timeout:
#             count += 1
#             t1 = time.time()
#             try:
#                 col.query(expr, output_fields=output_fields)
#             except Exception as e:
#                 logging.error(e)
#             t2 = round(time.time() - t1, 4)
#             query_latency.append(t2)
#             if count == interval_count:
#                 total = round(np.sum(query_latency), 4)
#                 p99 = round(np.percentile(query_latency, 99), 4)
#                 avg = round(np.mean(query_latency), 4)
#                 qps = round(interval_count / total, 4)
#                 logging.info(f"collection {col.description} query {interval_count} times in thread{thread_no}: cost {total}, qps {qps}, avg {avg}, p99 {p99} ")
#                 count = 0
#                 query_latency = []
#
#     threads = []
#     if threads_num > 1:
#         for i in range(threads_num):
#             t = threading.Thread(target=query_th, args=(collection, i))
#             threads.append(t)
#             t.start()
#         for t in threads:
#             t.join()
#     else:
#         query_latency = []
#         count = 0
#         start_time = time.time()
#         while time.time() < start_time + timeout:
#             count += 1
#             t1 = time.time()
#             try:
#                 collection.query(expr, output_fields=output_fields)
#             except Exception as e:
#                 logging.error(e)
#             t2 = round(time.time() - t1, 4)
#             query_latency.append(t2)
#             if count == interval_count:
#                 total = round(np.sum(query_latency), 4)
#                 p99 = round(np.percentile(query_latency, 99), 4)
#                 avg = round(np.mean(query_latency), 4)
#                 qps = round(interval_count / total, 4)
#                 logging.info(f"collection {collection.description} query {interval_count} times single thread: cost {total}, qps {qps}, avg {avg}, p99 {p99} ")
#                 count = 0
#                 query_latency = []


if __name__ == '__main__':
    host = sys.argv[1]
    name = sys.argv[2]                  # collection mame/alias
    # expr = "category i"   # str(sys.argv[3])             # query filter expression
    # output_fields = "i" # str(sys.argv[4])    # query output fields
    # th = 1    #  int(sys.argv[5])               # search thread num
    # timeout = 10 # int(sys.argv[6])          # search timeout, permanently if 0
    port = 19530

    file_handler = logging.FileHandler(filename=f"/tmp/parkey_search_{name}.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    logger = logging.getLogger('LOGGER_NAME')

    conn = connections.connect('default', host=host, port=port)

    nq = 1
    topk = 1
    ef = 32
    nprobe = 16

    # check and get the collection info
    if not utility.has_collection(collection_name=name):
        logging.error(f"collection: {name} does not exit")
        exit(0)

    collection = Collection(name=name)
    fields = collection.schema.fields
    for field in fields:
        if field.dtype == DataType.FLOAT_VECTOR:
            vector_field_name = field.name
            dim = field.params.get("dim")
            break

    if not collection.has_index():
        logging.error(f"collection: {name} has no index")
        exit(0)
    idx = collection.index()
    metric_type = idx.params.get("metric_type")
    index_type = idx.params.get("index_type")
    if index_type == "HNSW":
        search_params = {"metric_type": metric_type, "params": {"ef": ef}}
    elif index_type in ["IVF_SQ8", "IVF_FLAT"]:
        search_params = {"metric_type": metric_type, "params": {"nprobe": nprobe}}
    elif index_type == "DISKANN":
        search_params = {"metric_type": metric_type, "params": {"search_list": 100}}
    else:
        logging.error(f"index: {index_type} does not support yet")
        exit(0)

    logging.info(f"index param: {idx.params}")
    logging.info(f"search_param: {search_params}")

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

    # logging.info(f"search start: nq{nq}_top{topk}_threads{th}")
    # query(collection, expr=expr, output_fields=output_fields, th=th, timeout=timeout)
    # logging.info(f"search completed ")

    collection_parkey = Collection(name=f"{name}_parkey")
    collection_parkey.load()

    output_fields_list = [None, ["id", "category"], ["id", "category", "embedding"]]
    round_times = [101, 1001]
    for output_fields in output_fields_list:
        for round_time in round_times:
            total_count = 0
            total_time_query = 0
            total_time_query_parkey = 0
            total_time_search = 0
            total_time_search_parkey = 0
            logging.info(f"start {round_time} partition keys, output_fields: {output_fields}")
            # parkey query
            for i in range(round_time):
                t2 = time.time()
                res2 = collection_parkey.query(expr=f"category == {i}", output_fields=output_fields)
                # embedding = res[0].get("embedding")
                t3 = round(time.time() - t2, 4)
                logging.info(f"category {i}: parkey query: {t3}")
                total_time_query_parkey += t3
                total_count += len(res2)
            # parkey search
            if "embedding" in output_fields:
                logging.info("search does not support retrieve embedding in 2.2")
                total_time_search_parkey = 1
            else:
                for i in range(round_time):
                    search_vectors = [[random.random() for _ in range(dim)] for _ in range(nq)]
                    t2 = time.time()
                    res = collection_parkey.search(data=search_vectors, anns_field="embedding",
                                                   param=search_params, limit=10, expr=f"category == {i}",
                                                   output_fields=output_fields)
                    t3 = round(time.time() - t2, 4)
                    logging.info(f"category {i}: parkey search: {t3}")
                    total_time_search_parkey += t3

            # sleep to wait for cpu to cool down
            time.sleep(120)

            # non-parkey query
            for i in range(round_time):
                t2 = time.time()
                res2 = collection.query(expr=f"category == {i}", output_fields=output_fields)
                t3 = round(time.time() - t2, 4)
                logging.info(f"category {i}: query: {t3}")
                total_time_query += t3
            # non-parkey search
            # parkey search
            if "embedding" in output_fields:
                logging.info("search does not support retrieve embedding in 2.2")
                total_time_search = 1
            else:
                for i in range(round_time):
                    search_vectors = [[random.random() for _ in range(dim)] for _ in range(nq)]
                    t2 = time.time()
                    res = collection.serach(data=search_vectors, anns_field="embedding",
                                            param=search_params, limit=10, expr=f"category == {i}",
                                            output_fields=output_fields)
                    t3 = round(time.time() - t2, 4)
                    logging.info(f"category {i}: search: {t3}")
                    total_time_search += t3

            logging.info(f"total count: {total_count}, query time: {round(total_time_query,4)}, parkey time: {round(total_time_query_parkey,4)}, "
                         f"saved:{round(total_time_query-total_time_query_parkey,4)} @{round(total_time_query/total_time_query_parkey/0.01,2)}%")
            logging.info(f"total search time: {round(total_time_search, 4)}, parkey time: {round(total_time_search_parkey, 4)}, "
                         f"saved:{round(total_time_search - total_time_search_parkey, 4)} @{round((total_time_search) / total_time_search_parkey / 0.01, 2)}%")
