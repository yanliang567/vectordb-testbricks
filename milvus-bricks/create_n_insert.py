import time
import sys
import logging
from pymilvus import connections, DataType, \
    Collection, FieldSchema, CollectionSchema, utility
from common import insert_entities, get_float_vec_field_names, get_default_params_by_index_type


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"

intpk_field = FieldSchema(name="id", dtype=DataType.INT64, description="primary id")
strpk_field = FieldSchema(name="id", dtype=DataType.VARCHAR, description="primary id", max_length=65535)

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


def create_n_insert(collection_name, dims, nb, insert_times, index_types, vector_types=[DataType.FLOAT_VECTOR],
                    metric_types=["L2"], auto_id=True, use_str_pk=False, ttl=0,
                    build_index=True, shards_num=1, is_flush=True, use_insert=True):
    id_field = strpk_field if use_str_pk else intpk_field
    fields = [id_field, category_field, ver_field]
    # vec_field_names = []
    if not utility.has_collection(collection_name=collection_name):
        for i in range(len(dims)):
            embedding_field = FieldSchema(name=f"embedding_{i}", dtype=vector_types[i], dim=int(dims[i]))
            fields.append(embedding_field)
            # vec_field_names.append(embedding_field.name)
        schema = CollectionSchema(fields=fields, auto_id=auto_id, primary_field=id_field.name,
                                  description=f"{collection_name}")    # do not change the description
        collection = Collection(name=collection_name, schema=schema,
                                shards_num=shards_num, properties={"collection.ttl.seconds": ttl})
        # collection.set_properties({'mmap.enabled': True})
    else:
        collection = Collection(name=collection_name)
        logging.info(f"{collection_name} already exists")

    logging.info(f"{collection_name} collection schema: {collection.schema}")
    # insert data
    insert_entities(collection=collection, nb=nb, rounds=insert_times, use_insert=use_insert)
    collection = Collection(name=collection_name)
    if is_flush:
        collection.flush()
    logging.info(f"collection entities: {collection.num_entities}")

    if build_index:
        vec_field_names = get_float_vec_field_names(collection)
        logging.info(f"build index for {vec_field_names}")
        for i in range(len(dims)):
            index_type = str(index_types[i]).upper()
            metric_type = str(metric_types[i]).upper()
            index_params = get_default_params_by_index_type(index_type.upper(), metric_type)
            vec_field_name = vec_field_names[i]
            if not collection.has_index(index_name=vec_field_name):
                t0 = time.time()
                collection.create_index(field_name=vec_field_name, index_params=index_params)
                tt = round(time.time() - t0, 3)
                logging.info(f"build {vec_field_name} index {index_params} costs {tt}")
            else:
                idx = collection.index(index_name=vec_field_name)
                logging.info(f"{vec_field_name} index already exists: {idx.params}")
    else:
        logging.info("skip build index")


if __name__ == '__main__':
    host = sys.argv[1]                              # host ip or uri
    name = str(sys.argv[2])                         # collection name
    dims = sys.argv[3]                              # collection dimensions for all the vector fields
    vec_types = sys.argv[4]                         # vector types for all the vector fields
    indexes = sys.argv[5]                           # index types for all the vector fields
    metrics = sys.argv[6]                           # metric types for all the vector fields
    nb = int(sys.argv[7])                           # collection insert batch size
    shards = int(sys.argv[8])                       # collection shared number
    insert_times = int(sys.argv[9])                 # collection insert times
    auto_id = str(sys.argv[10]).upper()              # auto id
    use_str_pk = str(sys.argv[11]).upper()          # use varchar as pk type or not
    ttl = int(sys.argv[12])                         # ttl for the collection property
    need_build_index = str(sys.argv[13]).upper()    # build index or not after insert
    need_load = str(sys.argv[14]).upper()           # load the collection or not at the end
    use_insert = str(sys.argv[15]).upper()          # use insert or upsert
    api_key = str(sys.argv[16])                     # api key to connect to milvus

    port = 19530
    log_name = f"prepare_{name}"

    auto_id = True if auto_id == "TRUE" else False
    need_load = True if need_load == "TRUE" else False
    need_build_index = True if need_build_index == "TRUE" else False
    use_str_pk = True if use_str_pk == "TRUE" else False
    use_insert = True if use_insert == "TRUE" else False

    file_handler = logging.FileHandler(filename=f"/tmp/{log_name}.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    logger = logging.getLogger('LOGGER_NAME')

    dims = dims.split(",")
    vec_types = vec_types.split(",")
    indexes = indexes.split(",")
    metrics = metrics.split(",")
    if len(dims) != len(indexes) or len(dims) != len(metrics) or len(dims) != len(vec_types):
        logging.error("dimensions, vec_types, indexes and metrics should have the same length")
        sys.exit(1)

    vector_types_in_enum = []
    for vec_type in vec_types:
        if str(vec_type).upper() not in ["FLOAT", "FLOAT32", "F", "101"]:
            logging.error("only support DataType.FLOAT_VECTOR(101) for now")
            sys.exit(1)
        else:
            vector_types_in_enum.append(DataType.FLOAT_VECTOR)

    logging.info(f"host: {host}, collection: {name}, dims: {dims}, vec_types:{vector_types_in_enum}, indexes: {indexes}, "
                 f"metrics: {metrics},  nb: {nb}, shards: {shards}, insert_times: {insert_times},  "
                 f"auto_id: {auto_id}, use_str_pk: {use_str_pk}, ttl: {ttl}, "
                 f"build_index: {need_build_index}, load: {need_load}, api_key: {api_key}")
    logging.info("start")
    if api_key is None or api_key == "" or api_key.upper() == "NONE":
        conn = connections.connect('default', host=host, port=port)
    else:
        conn = connections.connect('default', uri=host, token=api_key)

    create_n_insert(collection_name=name, dims=dims, vector_types=vector_types_in_enum, nb=nb, insert_times=insert_times,
                    index_types=indexes, metric_types=metrics, auto_id=auto_id, use_str_pk=use_str_pk, ttl=ttl,
                    build_index=need_build_index, shards_num=shards, use_insert=use_insert)

    # load the collection
    if need_load:
        c = Collection(name=name)
        c.load()

    logging.info(f"collection prepared completed, create index: {need_build_index}, load collection: {need_load}")

