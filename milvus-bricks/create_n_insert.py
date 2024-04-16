import time
import sys
import random
import sklearn.preprocessing
import numpy as np
import logging
from pymilvus import connections, DataType, \
    Collection, FieldSchema, CollectionSchema, utility
from common import insert_entities, get_vector_field_name, get_default_params_by_index_type


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"

intpk_field = FieldSchema(name="id", dtype=DataType.INT64, description="primary id")
strpk_field = FieldSchema(name="id", dtype=DataType.VARCHAR, description="primary id", max_length=100)

category_field = FieldSchema(name="category", dtype=DataType.INT64, is_clustering_key=True,
                             description="category for partition key or clustering key")
groupid_field = FieldSchema(name="groupid", dtype=DataType.INT64, description="groupid")
device_field = FieldSchema(name="device", dtype=DataType.VARCHAR, max_length=500, description="device")
fname_field = FieldSchema(name="fname", dtype=DataType.VARCHAR, max_length=256, description="fname")
ext_field = FieldSchema(name="ext", dtype=DataType.VARCHAR, max_length=20, description="ext")
ver_field = FieldSchema(name="version", dtype=DataType.INT32, description="data version")
content_field = FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535, description="content")
flag_field = FieldSchema(name="flag", dtype=DataType.BOOL, description="flag")
json_field = FieldSchema(name="json_field", dtype=DataType.JSON, max_length=65535, description="json content")


def create_n_insert(collection_name, dim, nb, insert_times, index_type, metric_type="L2",
                    auto_id=True, use_str_pk=False, ttl=0, build_index=True, shards_num=1):
    if not utility.has_collection(collection_name=collection_name):
        embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
        id_field = strpk_field if use_str_pk else intpk_field
        # fields = [id_field, category_field, flag_field, ver_field, fname_field, embedding_field, json_field]
        fields = [id_field, category_field, embedding_field]
        schema = CollectionSchema(fields=fields, auto_id=auto_id, primary_field=id_field.name,
                                  description=f"{collection_name}")    # do not change the description
        collection = Collection(name=collection_name, schema=schema,
                                shards_num=shards_num, properties={"collection.ttl.seconds": ttl})
        # logging.info(f"create {collection_name} successfully, auto_id: {auto_id}, dim: {dim}, shards: {shards_num}")
    else:
        collection = Collection(name=collection_name)
        logging.info(f"{collection_name} already exists")

    logging.info(f"{collection_name} collection schema: {collection.schema}")
    if build_index:
        index_params = get_default_params_by_index_type(index_type.upper(), metric_type)

    # insert data
    insert_entities(collection=collection, nb=nb, rounds=insert_times)

    collection = Collection(name=collection_name)
    collection.flush()
    logging.info(f"collection entities: {collection.num_entities}")

    if build_index:
        if not collection.has_index():
            t0 = time.time()
            collection.create_index(field_name=get_vector_field_name(collection), index_params=index_params)
            tt = round(time.time() - t0, 3)
            logging.info(f"build index {index_params} costs {tt}")
        else:
            idx = collection.index()
            logging.info(f"index {idx.params} already exists")
    else:
        logging.info("skip build index")


if __name__ == '__main__':
    host = sys.argv[1]                              # host ip or uri
    name = str(sys.argv[2])                         # collection name
    dim = int(sys.argv[3])                          # collection dimension
    nb = int(sys.argv[4])                           # collection insert batch size
    shards = int(sys.argv[5])                       # collection shared number
    insert_times = int(sys.argv[6])                 # collection insert times
    index = str(sys.argv[7]).upper()                # index type
    metric = str(sys.argv[8]).upper()               # metric type, L2 or IP
    auto_id = str(sys.argv[9]).upper()              # auto id
    use_str_pk = str(sys.argv[10]).upper()          # use varchar as pk type or not
    ttl = int(sys.argv[11])                         # ttl for the collection property
    need_build_index = str(sys.argv[12]).upper()    # build index or not after insert
    need_load = str(sys.argv[13]).upper()           # load the collection or not at the end
    api_key = str(sys.argv[14])                     # api key to connect to milvus

    port = 19530
    log_name = f"prepare_{name}"

    auto_id = True if auto_id == "TRUE" else False
    need_load = True if need_load == "TRUE" else False
    need_build_index = True if need_build_index == "TRUE" else False
    use_str_pk = True if use_str_pk == "TRUE" else False

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

    create_n_insert(collection_name=name, dim=dim, nb=nb, insert_times=insert_times, index_type=index,
                    metric_type=metric, auto_id=auto_id, use_str_pk=use_str_pk, ttl=ttl,
                    build_index=need_build_index, shards_num=shards)

    # load the collection
    if need_load:
        c = Collection(name=name)
        c.load()

    logging.info(f"collection prepared completed, create index: {need_build_index}, load collection: {need_load}")

