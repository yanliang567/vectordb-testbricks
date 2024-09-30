import sys
import os
import logging
from pymilvus import utility, DataType, connections, \
    Collection
import create_n_insert
import pymilvus.exceptions

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"


if __name__ == '__main__':
    host = sys.argv[1]
    c_name = sys.argv[2]                                    # collection name is <c_name>_aa or <c_name>_bb
    dim = sys.argv[3]                                       # vector dimension
    nb = sys.argv[4]                                        # number of entities per insert round
    insert_times = sys.argv[5]                              # number of insert rounds
    index_type = str(sys.argv[6]).upper()                   # index type is hnsw or diskann
    metric_type = str(sys.argv[7]).upper()                  # metric type is L2 or IP
    is_flush = str(sys.argv[8]).upper()                     # flush data or not
    load_replicas = int(sys.argv[9])                        # load replicas number
    drop_after_alter_alias = str(sys.argv[10]).upper()      # drop collection after alter alias or not
    api_key = str(sys.argv[11])                             # api key to connect to milvus

    alias_name = f"{c_name}_alias"                          # alias mame
    port = 19530

    file_handler = logging.FileHandler(filename=f"/tmp/alter_alias_{alias_name}.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    logger = logging.getLogger('LOGGER_NAME')

    drop_after_alter_alias = True if drop_after_alter_alias == "TRUE" else False
    is_flush = True if is_flush == "TRUE" else False
    load_replicas = 1 if load_replicas < 1 else load_replicas
    logging.info(f"collection name: {c_name}, dimension: {dim}, nb: {nb}, insert_times: {insert_times}, "
                 f"metric_type: {metric_type}, index_type: {index_type}, is_flush: {is_flush}, "
                 f"load_replicas: {load_replicas}, drop_after_alter_alias: {drop_after_alter_alias},"
                 f"api_key: {api_key}")

    # connect to milvus
    if api_key is None or api_key == "" or api_key.upper() == "NONE":
        conn = connections.connect('default', host=host, port=port)
    else:
        conn = connections.connect('default', uri=host, token=api_key)

    # check and get the collection info
    exists_aa = utility.has_collection(collection_name=f'{c_name}_aa')
    exists_bb = utility.has_collection(collection_name=f'{c_name}_bb')
    if not exists_aa:
        logging.info(f"creating collection {c_name}_aa")
        create_n_insert.create_n_insert(collection_name=f'{c_name}_aa', vector_types=[DataType.FLOAT_VECTOR],
                                        dims=[dim], nb=nb, insert_times=insert_times, is_flush=is_flush,
                                        index_types=[index_type], metric_types=[metric_type], auto_id=False)
    if not exists_bb:
        logging.info(f"creating collection {c_name}_bb with nb+1")
        create_n_insert.create_n_insert(collection_name=f'{c_name}_bb', vector_types=[DataType.FLOAT_VECTOR],
                                        dims=[dim], nb=nb+1, insert_times=insert_times, is_flush=is_flush,
                                        index_types=[index_type], metric_types=[metric_type], auto_id=False)

    # alter alias
    try:
        aliases = utility.list_aliases(collection_name=alias_name)
        if alias_name in aliases:
            current_collection = Collection(alias_name)
            description = current_collection.description
            logging.info(f"collection alias before altered: {description}, num_entities: {current_collection.num_entities}")
            next_name = f"{c_name}_bb" if description == f"{c_name}_aa" else f"{c_name}_aa"
            next_collection = Collection(next_name)
            next_collection.load(replica_number=load_replicas)
            utility.alter_alias(collection_name=next_name, alias=alias_name)
            current_collection = Collection(description)
            current_collection.release()
            if drop_after_alter_alias:
                current_collection.drop()
        else:
            next_name = f"{c_name}_aa"
            next_collection = Collection(next_name)
            next_collection.load(replica_number=load_replicas)
            utility.create_alias(collection_name=next_name, alias=alias_name)
    except pymilvus.exceptions.DescribeCollectionException as e:
        next_name = f"{c_name}_aa"
        next_collection = Collection(next_name)
        next_collection.load(replica_number=load_replicas)
        utility.create_alias(collection_name=next_name, alias=alias_name)
    collection = Collection(alias_name)
    logging.info(f"collection alias after altered: {collection.description}, num_entities: {collection.num_entities}")

    logging.info(f"alter alias completed")
