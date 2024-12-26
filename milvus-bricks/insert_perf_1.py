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


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"


def do_insert(collection, threads_num, ins_times_per_thread):

    def insert_th(collection, rounds, thread_no):
        for r in range(rounds):
            data = gen_data_by_collection(collection, nb, r)
            t1 = time.time()
            res = collection.insert(data)
            t2 = round(time.time() - t1, 3)
            logging.info(f"assert insert thread{thread_no} round{r}: {t2}")

    # insert
    threads = []
    logging.info(f"ready to insert {collection.name}, insert {ins_times_per_thread} times per thread")
    if threads_num > 1:
        for i in range(threads_num):
            t = threading.Thread(target=insert_th, args=(collection, int(ins_times_per_thread), i))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
    else:
        for r in range(ins_times_per_thread):
            data = gen_data_by_collection(collection, nb, r)
            t1 = time.time()
            res = collection.insert(data)
            t2 = round(time.time() - t1, 3)
            logging.info(f"assert insert thread0 round{r}: {t2}")


if __name__ == '__main__':
    host = sys.argv[1]                          # host address
    collection_name = sys.argv[2]               # collection name
    dim = int(sys.argv[3])                      # vector dimension
    nb = int(sys.argv[4])                       # number of vectors per insert request
    num_threads = int(sys.argv[5])              # insert thread num
    ins_per_thread = int(sys.argv[6])           # insert times per thread
    port = 19530

    file_handler = logging.FileHandler(filename=f"/tmp/insert_perf_{collection_name}.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    logger = logging.getLogger('LOGGER_NAME')

    conn = connections.connect('default', host=host, port=port)
    logging.info(f"host={host}, collection_name={collection_name}, dim={dim}, "
                 f"nb={nb}, num_threads={num_threads}, ins_times_per_thread={ins_per_thread}")
    logging.info("Insert perf start... ...")

    # check and get the collection info
    if not utility.has_collection(collection_name=collection_name):
        logging.error(f"collection: {collection_name} does not exit, create an empty one as default")
        create_n_insert(collection_name=collection_name, dims=[dim], nb=nb, insert_times=0, build_index=False)

    c = Collection(collection_name)
    # insert
    t1 = time.time()
    do_insert(c, num_threads, ins_per_thread)
    t2 = time.time() - t1
    req_per_sec = round(ins_per_thread * num_threads / t2, 3)  # how many insert requests response per second
    entities_throughput = round(nb * req_per_sec, 3)  # how many entities inserted per second
    logging.info(f"Insert  {collection_name} cost {round(t2, 3)}, "
                 f"req_per_second {req_per_sec}, entities_throughput {entities_throughput}")
