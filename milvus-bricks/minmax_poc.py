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


def normalize_data(similarity_metric_type, vectors):
    if similarity_metric_type == "IP":
        vectors = sklearn.preprocessing.normalize(vectors, axis=1, norm='l2')
        vectors = vectors.astype(np.float32)
    elif similarity_metric_type == "L2":
        vectors = vectors.astype(np.float32)
    return vectors


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


def create_n_insert_parkey(collection_name, dim, nb, index_type, metric_type="IP",
                           parkey_num=10000, tenants_startid=0, rows_per_tenant=100000, num_partitions=64, shards_num=1):
    pk_field = FieldSchema(name="id", dtype=DataType.INT64, description="auto primary id")
    index_name_field = FieldSchema(name="index_name", dtype=DataType.VARCHAR, max_length=255, description="user id")
    index_field = FieldSchema(name="index", dtype=DataType.FLOAT, description="index")
    document_field = FieldSchema(name="document", dtype=DataType.VARCHAR, max_length=255, description="document name")
    embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    collection_parkey_name = f"{collection_name}"
    schema = CollectionSchema(fields=[pk_field, index_name_field, index_field, document_field, embedding_field],
                              auto_id=auto_id, primary_field=pk_field.name,
                              partition_key_field=index_name_field.name,
                              description=f"{collection_parkey_name} partition key field: index_name")
    collection_parkey = Collection(name=collection_parkey_name, schema=schema, shards_num=shards_num,
                                   num_partitions=num_partitions)
    logging.info(f"create {collection_parkey_name} successfully")

    # insert data tenant by tenant
    insert_times = rows_per_tenant // nb
    for i in range(tenants_startid, tenants_startid+parkey_num):
        for j in range(insert_times):
            # prepare data
            index_names = [str(i) for _ in range(nb)]
            indexes = [float(i) for _ in range(nb)]
            document = ["doc_" + str(j) for _ in range(nb)]
            embeddings = [[random.random() for _ in range(dim)] for _ in range(nb)]
            embeddings = normalize_data(metric_type, embeddings)
            data = [index_names, indexes, document, embeddings]
            t0 = time.time()
            collection_parkey.insert(data)
            tt = round(time.time() - t0, 3)
            logging.info(f"{collection_parkey_name} tanant {i} insert {j} costs {tt}")
        if i % 5 == 0:      # flush every 5 tenants
            t0 = time.time()
            collection_parkey.flush()
            tt = round(time.time() - t0, 3)
            logging.info(f"flush every 5 tenants cost {tt}")

    collection_parkey.flush()
    logging.info(f"{collection_parkey_name} entities: {collection_parkey.num_entities}")
    build(collection_parkey, index_type, metric_type)
    collection_parkey.load()
    logging.info(f"{collection_parkey_name} loaded")


if __name__ == '__main__':
    host = sys.argv[1]                  # host address or uri
    name = str(sys.argv[2])             # collection name
    dim = int(sys.argv[3])              # collection dimension
    shards = int(sys.argv[4])           # shards num
    nb = int(sys.argv[5])               # collection insert batch size
    index = str(sys.argv[6]).upper()    # index type
    metric = str(sys.argv[7]).upper()   # metric type, L2 or IP
    tenants_num = int(sys.argv[8])      # tenants number means partition key number
    tenants_startid = int(sys.argv[9])  # start id for tenants
    rows_per_tenant = int(sys.argv[10])  # avg entities per tenant
    num_partitions = int(sys.argv[11])   # number of partitions
    api_key = str(sys.argv[12])          # api key to connect to milvus

    port = 19530
    log_name = f"prepare_parkey_{name}"

    if num_partitions <= 0 or num_partitions > 4096:
        num_partitions = 64
    file_handler = logging.FileHandler(filename=f"/tmp/{log_name}.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    logger = logging.getLogger('LOGGER_NAME')

    logging.info("start")
    if api_key is None or api_key == "" or api_key.upper() == "NONE":
        conn = connections.connect('default', host=host, port=port)
    else:
        conn = connections.connect('default', uri=host, token=api_key)

    create_n_insert_parkey(collection_name=name, dim=dim, nb=nb,
                           index_type=index, metric_type=metric, shards_num=shards,
                           parkey_num=tenants_num, tenants_startid=tenants_startid,
                           rows_per_tenant=rows_per_tenant,
                           num_partitions=num_partitions)

    logging.info("collections prepared completed")

