import time
import sys
import random

import logging
from pymilvus import connections, DataType, \
    Collection, FieldSchema, CollectionSchema, utility
import create_n_insert
import string


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"

default_rg = "__default_resource_group"


def gen_unique_str(str_value=None):
    prefix = "".join(random.choice(string.ascii_letters + string.digits) for _ in range(8))
    return prefix if str_value is None else str_value + "_" + prefix


if __name__ == '__main__':
    host = sys.argv[1]  # host address
    name = str(sys.argv[2])  # collection name
    dim = int(sys.argv[3])  # collection dimension
    nb = int(sys.argv[4])  # collection insert batch size
    insert_times = int(sys.argv[5])  # collection insert times
    index = str(sys.argv[6]).upper()  # index type
    metric = str(sys.argv[7]).upper()  # metric type, L2 or IP
    port = 19530
    log_name = f"create_n_drop_e2e_{name}"

    file_handler = logging.FileHandler(filename=f"/tmp/{log_name}.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    logger = logging.getLogger('LOGGER_NAME')

    logging.info("start")
    connections.add_connection(default={"host": host, "port": 19530})
    connections.connect('default')

    # create and insert
    tmp_name = gen_unique_str(name)
    create_n_insert.create_n_insert(collection_name=tmp_name, dim=dim, nb=nb, insert_times=insert_times,
                                    index_type=index, metric_type=metric)
    # create and drop
    for i in range(10):
        # create and insert
        tmp_name = gen_unique_str(name)
        create_n_insert.create_n_insert(collection_name=tmp_name, dim=dim, nb=nb, insert_times=insert_times,
                                        index_type=index, metric_type=metric)
        # drop the collections
        collection = Collection(name=tmp_name)
        collection.drop()
        logging.info(f"collection {tmp_name} dropped")

    logging.info("create and drop completed")

