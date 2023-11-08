import time
import sys
import random
import sklearn.preprocessing
import numpy as np
import logging
from pymilvus import connections, DataType, \
    Collection, FieldSchema, CollectionSchema
from common import get_default_params_by_index_type


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"

auto_id = True


def build(collection, index_type, metric_type):

    index_params = get_default_params_by_index_type(index_type.upper(), metric_type)

    if not collection.has_index():
        t0 = time.time()
        collection.create_index(field_name="embedding", index_params=index_params)
        tt = round(time.time() - t0, 3)
        logging.info(f"{collection.name} build index {index_params} costs {tt}")
    else:
        idx = collection.index()
        logging.info(f"{collection.name} index {idx.params} already exists")


def create_n_insert_parkey(collection_name, dim, nb, insert_times, index_type, metric_type="L2",
                           parkey_num=10000, parkey_collection_only=False, parkey_values_evenly=False,
                           num_partitions=64, pre_load=False):
    id_field = FieldSchema(name="id", dtype=DataType.INT64, description="auto primary id")
    category_field = FieldSchema(name="category", dtype=DataType.INT64, description="age")
    embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    if not parkey_collection_only:
        schema = CollectionSchema(fields=[id_field, category_field, embedding_field],
                                  auto_id=auto_id, primary_field=id_field.name,
                                  description=f"{collection_name}")
        collection = Collection(name=collection_name, schema=schema)
        logging.info(f"create {collection_name} successfully")

    collection_parkey_name = f"{collection_name}_parkey"
    schema = CollectionSchema(fields=[id_field, category_field, embedding_field],
                              auto_id=auto_id, primary_field=id_field.name,
                              partition_key_field=category_field.name,
                              description=f"{collection_parkey_name} partition key field: category")
    collection_parkey = Collection(name=collection_parkey_name, schema=schema,
                                   num_partitions=num_partitions)
    logging.info(f"create {collection_parkey_name} successfully")

    if pre_load:
        logging.info(f"pre_load: {pre_load}")
        if not parkey_collection_only:
            collection.flush()
            logging.info(f"{collection_name} entities: {collection.num_entities}")
            build(collection, index_type, metric_type)
            collection.load()
            logging.info(f"{collection_name} loaded")
        collection_parkey.flush()
        logging.info(f"{collection_parkey_name} entities: {collection_parkey.num_entities}")
        build(collection_parkey, index_type, metric_type)
        collection_parkey.load()
        logging.info(f"{collection_parkey_name} loaded")

    # insert data
    for i in range(insert_times):
        # prepare data
        categories = [(i+1) for _ in range(nb)]
        if parkey_values_evenly:
            categories = [random.randint(1, parkey_num) for _ in range(nb)]
        embeddings = [[random.random() for _ in range(dim)] for _ in range(nb)]
        data = [categories, embeddings]
        if not parkey_collection_only:
            t0 = time.time()
            collection.insert(data)
            tt = round(time.time() - t0, 3)
            logging.info(f"{collection_name} insert {i} costs {tt}")
        t0 = time.time()
        collection_parkey.insert(data)
        tt = round(time.time() - t0, 3)
        logging.info(f"{collection_parkey_name} insert {i} costs {tt}")

    if not parkey_collection_only:
        collection.flush()
        logging.info(f"{collection_name} entities: {collection.num_entities}")
        build(collection, index_type, metric_type)
        collection.load()
        logging.info(f"{collection_name} loaded")

    collection_parkey.flush()
    logging.info(f"{collection_parkey_name} entities: {collection_parkey.num_entities}")
    build(collection_parkey, index_type, metric_type)
    collection_parkey.load()
    logging.info(f"{collection_parkey_name} loaded")


if __name__ == '__main__':
    host = sys.argv[1]  # host address
    name = str(sys.argv[2])  # collection name
    dim = int(sys.argv[3])  # collection dimension
    nb = int(sys.argv[4])  # collection insert batch size
    insert_times = int(sys.argv[5])  # collection insert times
    index = str(sys.argv[6]).upper()    # index type
    metric = str(sys.argv[7]).upper()   # metric type, L2 or IP
    parkey_num = int(sys.argv[8])   # partition key number for evenly distributed partition keys
    parkey_collection_only = str(sys.argv[9]).upper()   # true if only create partition key collection
    parkey_values_evenly = str(sys.argv[10]).upper()   # true if partition key values are evenly distributed
    num_partitions = int(sys.argv[11])   # number of partitions
    pre_load = str(sys.argv[12]).upper()   # true if pre load data
    port = 19530
    log_name = f"prepare_parkey_{name}"

    parkey_collection_only = True if parkey_collection_only == "TRUE" else False
    parkey_values_evenly = True if parkey_values_evenly == "TRUE" else False
    pre_load = True if pre_load == "TRUE" else False
    if num_partitions <= 0 or num_partitions > 4096:
        num_partitions = 64
    file_handler = logging.FileHandler(filename=f"/tmp/{log_name}.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    logger = logging.getLogger('LOGGER_NAME')

    logging.info("start")
    connections.add_connection(default={"host": host, "port": 19530})
    connections.connect('default')

    create_n_insert_parkey(collection_name=name, dim=dim, nb=nb, insert_times=insert_times,
                           index_type=index, metric_type=metric,
                           parkey_num=parkey_num, parkey_collection_only=parkey_collection_only,
                           parkey_values_evenly=parkey_values_evenly, num_partitions=num_partitions,
                           pre_load=pre_load)

    logging.info("collections prepared completed")

