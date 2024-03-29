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
    rg_num = int(sys.argv[8])  # resource group num == replica num
    node_num_each_rg = int(sys.argv[9])  # query node num for each rg
    port = 19530
    log_name = f"prepare_{name}"

    file_handler = logging.FileHandler(filename=f"/tmp/{log_name}.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    logger = logging.getLogger('LOGGER_NAME')

    logging.info("start")
    connections.add_connection(default={"host": host, "port": 19530})
    connections.connect('default')

    default_rg_info = utility.describe_resource_group(name=default_rg)
    if default_rg_info.num_available_node < rg_num * node_num_each_rg:
        logging.error(f"there is no available nodes in default rg, expected: {rg_num} x {node_num_each_rg}")
        exit(-1)

    prefix = gen_unique_str()
    rgs = [f"rg_{i}_{prefix}" for i in range(rg_num)]
    for rg_name in rgs:
        utility.create_resource_group(name=rg_name)
        utility.transfer_node(source_group=default_rg, target_group=rg_name, num_nodes=node_num_each_rg)
        logging.info(f"create and transfer node to resource group {rg_name}")

    # create an insert
    create_n_insert.create_n_insert(collection_name=name, dim=dim, nb=nb, insert_times=insert_times,
                                    index_type=index, metric_type=metric)

    # load collection
    c = Collection(name=name)
    c.load(replica_number=rg_num, _resource_groups=rgs)
    logging.info(f"collection loaded to resource groups: {rgs}")

    logging.info("collection prepared completed")

