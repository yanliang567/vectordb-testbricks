import time
import sys
import random
import sklearn.preprocessing
import numpy as np
import logging
from pymilvus import connections, DataType, \
    Collection, FieldSchema, CollectionSchema, utility
from common import insert_entities, get_vector_field_name, get_default_params_by_index_type, gen_data_by_collection


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"

id_field = FieldSchema(name="id", dtype=DataType.INT64, description="primary id")
# id_field = FieldSchema(name="id", dtype=DataType.VARCHAR, description="primary id", max_length=100)

age_field = FieldSchema(name="age", dtype=DataType.INT64, description="age")
groupid_field = FieldSchema(name="groupid", dtype=DataType.INT64, description="groupid")
device_field = FieldSchema(name="device", dtype=DataType.VARCHAR, max_length=500, description="device")
fname_field = FieldSchema(name="fname", dtype=DataType.VARCHAR, max_length=256, description="fname")
ext_field = FieldSchema(name="ext", dtype=DataType.VARCHAR, max_length=20, description="ext")
mtime_field = FieldSchema(name="mtime", dtype=DataType.INT64, description="mtime")
content_field = FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535, description="content")
flag_field = FieldSchema(name="flag", dtype=DataType.BOOL, description="flag")
json_field = FieldSchema(name="json_field", dtype=DataType.JSON, max_length=65535, description="json content")


def create_n_insert(collection_name, dim, nb, insert_times, index_type, metric_type="L2",
                    auto_id=True, ttl=0, build_index=True, shards_num=1):
    if not utility.has_collection(collection_name=collection_name):
        embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
        schema = CollectionSchema(fields=[id_field, age_field, flag_field, ext_field, fname_field,
                                          embedding_field, json_field],
                                  auto_id=auto_id, primary_field=id_field.name,
                                  description=f"{collection_name}")    # do not change the description
        collection = Collection(name=collection_name, schema=schema,
                                shards_num=shards_num, properties={"collection.ttl.seconds": ttl})
        logging.info(f"create {collection_name} successfully, auto_id: {auto_id}, dim: {dim}, shards: {shards_num}")
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
    host = sys.argv[1]                      # host address, ip or uri
    name = str(sys.argv[2])                 # collection name
    dim = int(sys.argv[3])                  # collection dimension
    nb = int(sys.argv[4])                   # collection insert batch size
    shards = int(sys.argv[5])               # collection shard number
    index = str(sys.argv[6]).upper()        # index type
    metric = str(sys.argv[7]).upper()       # metric type, L2 or IP
    auto_id = str(sys.argv[8]).upper()      # auto id
    api_key = str(sys.argv[9])              # api key for uri connections on cloud instances
    port = 19530
    log_name = f"prepare_{name}"

    auto_id = True if auto_id == "TRUE" else False

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

    create_n_insert(collection_name=name, dim=dim, nb=nb, insert_times=0, index_type=index,
                    metric_type=metric, auto_id=auto_id, build_index=True, shards_num=shards)

    # load the collection
    c = Collection(name=name)
    c.load()

    # insert data
    deny_times = 0
    r = 0
    while True and deny_times < 3:
        data = gen_data_by_collection(collection=c, nb=nb, r=r)
        try:
            t1 = time.time()
            c.insert(data)
            t2 = round(time.time() - t1, 3)
            logging.info(f"{c.name} insert {r} costs {t2}")
        except Exception as e:
            if "limitWriting.forceDeny" in str(e):
                logging.error(f"insert expected error: {e}")
                deny_times += 1
                logging.error(f"wait for 15 minutes and try again, deny times: {deny_times}")
                time.sleep(900)
            else:
                logging.error(f"insert error: {e}")
                deny_times += 1
                time.sleep(900)
        r += 1

    logging.info(f"collection prepared completed, collection entities: {c.num_entities}")

