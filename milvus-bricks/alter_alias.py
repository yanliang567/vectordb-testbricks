import sys
import os
import logging
from pymilvus import utility, connections, \
    Collection
import create_n_insert
import pymilvus.exceptions

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"


if __name__ == '__main__':
    host = sys.argv[1]
    c_name = sys.argv[2]                                    # collection name is <c_name>_aa or <c_name>_bb
    index_type = str(sys.argv[3]).upper()                   # index type is hnsw or diskann
    metric_type = str(sys.argv[4]).upper()                  # metric type is L2 or IP
    need_insert = str(sys.argv[5]).upper()                  # need insert data or not
    is_flush = str(sys.argv[6]).upper()                     # flush data or not
    load_replicas = int(sys.argv[7])                        # load replicas number
    drop_after_alter_alias = str(sys.argv[8]).upper()       # drop collection after alter alias or not

    alias_name = f"{c_name}_alias"                  # alias mame
    port = 19530
    conn = connections.connect('default', host=host, port=port)

    file_handler = logging.FileHandler(filename=f"/tmp/alter_alias_{alias_name}.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    logger = logging.getLogger('LOGGER_NAME')

    dim = 128
    nb = 2000
    need_insert = True if need_insert == "TRUE" else False
    insert_times = 50 if need_insert else 0
    drop_after_alter_alias = True if drop_after_alter_alias == "TRUE" else False
    is_flush = True if is_flush == "TRUE" else False
    load_replicas = 1 if load_replicas < 1 else load_replicas
    # check and get the collection info
    exists_aa = utility.has_collection(collection_name=f'{c_name}_aa')
    exists_bb = utility.has_collection(collection_name=f'{c_name}_bb')
    if not exists_aa:
        logging.info(f"creating collection {c_name}_aa")
        create_n_insert.create_n_insert(collection_name=f'{c_name}_aa',
                                        dim=dim, nb=nb, insert_times=insert_times, is_flush=is_flush,
                                        index_type=index_type, metric_type=metric_type, auto_id=False)
    if not exists_bb:
        logging.info(f"creating collection {c_name}_bb")
        create_n_insert.create_n_insert(collection_name=f'{c_name}_bb',
                                        dim=dim, nb=nb, insert_times=insert_times, is_flush=is_flush,
                                        index_type=index_type, metric_type=metric_type, auto_id=False)

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
