import time
import sys
import random
import sklearn.preprocessing
import numpy as np
import logging
from pymilvus import connections, DataType, \
    Collection, FieldSchema, CollectionSchema

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"

nb = 1000
dim = 256
auto_id = True
metric = "IP"
index_params_dict = {
    "HNSW": {"index_type": "HNSW", "metric_type": metric, "params": {"M": 8, "efConstruction": 96}},
    "DISKANN": {"index_type": "DISKANN", "metric_type": metric, "params": {}}
}


def normalize(metric_type, X):
    if metric_type == "IP":
        # logging.info("Set normalize for metric_type: %s" % metric_type)
        X = sklearn.preprocessing.normalize(X, axis=1, norm="l2")
        X = X.astype(np.float32)
    elif metric_type == "L2":
        X = X.astype(np.float32)
    return X


if __name__ == '__main__':
    host = "10.102.6.119"     # sys.argv[1]  # host address
    collection_name = "partition_key"   # str(sys.argv[2])  # collection name
    index_type = "hnsw"  #  str(sys.argv[3])  # index type
    insert_times = 2000   # insert times

    port = 19530
    log_name = f"prepare_{collection_name}"

    file_handler = logging.FileHandler(filename=f"/tmp/{log_name}.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    logger = logging.getLogger('LOGGER_NAME')

    logging.info("start")
    connections.add_connection(default={"host": host, "port": 19530})
    connections.connect('default')

    id_field = FieldSchema(name="id", dtype=DataType.INT64, description="auto primary id")
    category_field = FieldSchema(name="category", dtype=DataType.INT64,
                                 is_partition_key=True, partitions=1024,
                                 description="categories")
    group_field = FieldSchema(name="group", dtype=DataType.INT64,
                              description="groups")
    embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    schema = CollectionSchema(fields=[id_field, category_field, group_field, embedding_field],
                              auto_id=auto_id, primary_field=id_field.name,
                              description=f"{collection_name}")
    collection = Collection(name=collection_name, schema=schema)
    logging.info(f"create {collection_name} successfully")

    index_params = index_params_dict.get(index_type.upper(), None)
    if index_params is None:
        logging.error(f"index type {index_type} no supported")
        exit(1)

    for i in range(insert_times):
        # prepare data
        categories = [i for _ in range(nb)]     # same partition key for one inset batch
        groups = [i for _ in range(nb)]
        embeddings = normalize(metric, [[random.random() for _ in range(dim)] for _ in range(nb)])
        data = [categories, groups, embeddings]
        t0 = time.time()
        collection.insert(data)
        tt = round(time.time() - t0, 3)
        logging.info(f"insert {i} costs {tt}")

    collection.flush()
    logging.info(f"collection entities: {collection.num_entities}")

    if not collection.has_index():
        t0 = time.time()
        collection.create_index(field_name=embedding_field.name, index_params=index_params)
        tt = round(time.time() - t0, 3)
        logging.info(f"build index {index_params} costs {tt}")
    else:
        idx = collection.index()
        logging.info(f"index {idx.params} already exists")

    collection.load()
    logging.info("collection prepared completed")




