import time
import sys
import logging
from pymilvus import MilvusClient, DataType
from common import insert_entities, get_float_vec_field_names, get_default_params_by_index_type, create_collection_schema, create_n_insert


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"



if __name__ == '__main__':
    hosts = sys.argv[1]                              # host ips or uris separated by comma, only 2 hosts max are supported for comparision tests
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
    need_build_index = str(sys.argv[13]).upper()    # build index for vector fields or not after insert
    need_load = str(sys.argv[14]).upper()           # load the collection or not at the end
    use_insert = str(sys.argv[15]).upper()          # use insert or upsert
    api_key = str(sys.argv[16])                     # api key to connect to milvus, should be same for both hosts

    # hosts = '10.104.33.208'                              # host ip or uri
    # name = 'tesdaa'                       # collection name
    # dims = '32,43'                              # collection dimensions for all the vector fields
    # vec_types = 'FLOAT,F'                         # vector types for all the vector fields
    # indexes = 'AUTOINDEX,HNSW'                           # index types for all the vector fields
    # metrics = 'L2,COSINE'                           # metric types for all the vector fields
    # nb = 1000                           # collection insert batch size
    # shards = 2                       # collection shared number
    # insert_times =3                 # collection insert times
    # auto_id = 'false'              # auto id
    # use_str_pk = 'FALSE'          # use varchar as pk type or not
    # ttl = 0                         # ttl for the collection property
    # need_build_index = 'TRUE'    # build index for vector fields or not after insert
    # need_load = 'TRUE'           # load the collection or not at the end
    # use_insert = 'TRUE'          # use insert or upsert
    # api_key = 'None'                     # api key to connect to milvus, should be same for both hosts

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

    hosts = hosts.split(",")
    dims = dims.split(",")
    vec_types = vec_types.split(",")
    indexes = indexes.split(",")
    metrics = metrics.split(",")
    if len(dims) != len(indexes) or len(dims) != len(metrics) or len(dims) != len(vec_types):
        logging.error("dimensions, vec_types, indexes and metrics should have the same length")
        sys.exit(1)
    
    if len(hosts) > 2:
        logging.error("only support 2 hosts max for now")
        sys.exit(1)

    vector_types_in_enum = []
    for vec_type in vec_types:
        if str(vec_type).upper() in ["FLOAT", "FLOAT32", "F", "101"]:
            vector_types_in_enum.append(DataType.FLOAT_VECTOR)
        elif str(vec_type).upper() in ["FLOAT16", "FLOAT16_VECTOR", "102"]:
            vector_types_in_enum.append(DataType.FLOAT16_VECTOR)
        elif str(vec_type).upper() in ["BFLOAT16", "BFLOAT16_VECTOR", "103"]:
            vector_types_in_enum.append(DataType.BFLOAT16_VECTOR)
        elif str(vec_type).upper() in ["BINARY", "BINARY_VECTOR", "100"]:
            vector_types_in_enum.append(DataType.BINARY_VECTOR)
        elif str(vec_type).upper() in ["SPARSE", "SPARSE_FLOAT_VECTOR", "104"]:
            vector_types_in_enum.append(DataType.SPARSE_FLOAT_VECTOR)
        elif str(vec_type).upper() in ["INT8", "INT8_VECTOR", "105"]:
            vector_types_in_enum.append(DataType.INT8_VECTOR)
        else:
            logging.error(f"only support vector types: FLOAT, FLOAT16, BFLOAT16, BINARY, SPARSE and INT8 for now, but got {vec_type}")
            sys.exit(1)

    logging.info(f"hosts: {hosts}, collection: {name}, dims: {dims}, vec_types:{vector_types_in_enum}, indexes: {indexes}, "
                 f"metrics: {metrics},  nb: {nb}, shards: {shards}, insert_times: {insert_times},  "
                 f"auto_id: {auto_id}, use_str_pk: {use_str_pk}, ttl: {ttl}, "
                 f"build_index for vector fields: {need_build_index}, load: {need_load}, api_key: {api_key}")
    
    # Create MilvusClient
    client_2 = None
    if api_key is None or api_key == "" or api_key.upper() == "NONE":
        client_1 = MilvusClient(uri=f"http://{hosts[0]}:{port}")
        if len(hosts) > 1:
            client_2 = MilvusClient(uri=f"http://{hosts[1]}:{port}")
    else:
        client_1 = MilvusClient(uri=f"http://{hosts[0]}:{port}", token=api_key)
        if len(hosts) > 1:
            client_2 = MilvusClient(uri=f"http://{hosts[1]}:{port}", token=api_key)

    logging.info(f"client_1: {client_1}, client_2: {client_2}")

    # Create collection schema in main
    schema = create_collection_schema(dims=dims, vector_types=vector_types_in_enum, 
                                    auto_id=auto_id, use_str_pk=use_str_pk)
    logging.info(f"Created schema: {schema}")

    # Create collection and insert data
    create_n_insert(collection_name=name, schema=schema, nb=nb, insert_times=insert_times, 
                    index_types=indexes, dims=dims, metric_types=metrics, ttl=ttl, 
                    is_flush=True, build_index=need_build_index, shards_num=shards, 
                    use_insert=use_insert, build_scalar_index=False, clients=[client_1, client_2])

    # Load the collection
    if need_load:
        client_1.load_collection(collection_name=name)
        if client_2 is not None:
            client_2.load_collection(collection_name=name)
        logging.info(f"Loaded collection: {name}")

    logging.info(f"collection prepared completed, create index: {need_build_index}, load collection: {need_load}")
