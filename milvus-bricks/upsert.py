import time
import sys
import random
import logging
from pymilvus import utility, connections, DataType, \
    Collection, FieldSchema, CollectionSchema

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"


def get_dim(collection):
    fields = collection.schema.fields
    for field in fields:
        if field.dtype == DataType.FLOAT_VECTOR:
            dim = field.params.get("dim")
    return dim


def get_vector_field_name(collection):
    fields = collection.schema.fields
    for field in fields:
        if field.dtype == DataType.FLOAT_VECTOR:
            vector_field_name = field.name
    return vector_field_name


def delete_entities(collection, nb, search_params, rounds):
    dim = get_dim(collection=collection)
    auto_id = collection.schema.auto_id
    primary_field_name = collection.primary_field.name
    vector_field_name = get_vector_field_name(collection=collection)
    if auto_id:
        for r in range(rounds):
            search_vector = [[random.random() for _ in range(dim)] for _ in range(1)]
            results = c.search(data=search_vector, anns_field=vector_field_name,
                               param=search_params, limit=nb)
            for hits in results:
                ids = hits.ids
                c.delete(expr=f"{primary_field_name} in {ids}")
                logging.info(f"deleted {len(ids)} entities")
    else:
        for r in range(rounds):
            start_uid = r * nb
            end_uid = start_uid + nb
            c.delete(expr=f"{primary_field_name} in [{start_uid}, {end_uid}]")
            logging.info(f"deleted entities {start_uid}-{end_uid}")


def gen_data_by_collection(collection, nb, r):
    data = []
    start_uid = r * nb
    fields = collection.schema.fields
    auto_id = collection.schema.auto_id
    for field in fields:
        field_values = []
        if field.dtype == DataType.FLOAT_VECTOR:
            dim = field.params.get("dim")
            field_values = [[random.random() for _ in range(dim)] for _ in range(nb)]
        if field.dtype in [DataType.INT64, DataType.INT32, DataType.INT16, DataType.INT8]:
            if field.is_primary:
                if not auto_id:
                    field_values = [_ for _ in range(start_uid, start_uid + nb)]
                else:
                    continue
            else:
                field_values = [_ for _ in range(nb)]
        if field.dtype == DataType.VARCHAR:
            field_values = [str(random.random()) for _ in range(nb)]
        if field.dtype == DataType.FLOAT:
            field_values = [random.random() for _ in range(nb)]
        if field.dtype == DataType.BOOL:
            field_values = [False for _ in range(nb)]
        data.append(field_values)
    return data


def insert_entities(collection, nb, rounds):
    for r in range(rounds):
        data = gen_data_by_collection(collection=collection, nb=nb, r=r)
        t1 = time.time()
        collection.insert(data)
        t2 = round(time.time() - t1, 3)
        logging.info(f"{collection.name} {r} upsert in {t2}")


def get_search_params(collection):
    idx = collection.index()
    metric_type = idx.params.get("metric_type")
    index_type = idx.params.get("index_type")
    if index_type == "HNSW":
        search_params = {"metric_type": metric_type, "params": {"ef": 64}}
    elif index_type in ["IVF_SQ8", "IVF_FLAT"]:
        search_params = {"metric_type": metric_type, "params": {"nprobe": 32}}
    elif index_type == "DISKANN":
        search_params = {"metric_type": metric_type, "params": {"search_list": 100}}
    else:
        logging.error(f"index: {index_type} does not support yet")
        exit(0)
    return search_params


if __name__ == '__main__':
    host = sys.argv[1]
    collection_name = sys.argv[2]            # collection mame
    delete_percent = int(sys.argv[3])        # percent of entities to be deleted
    insert_percent = int(sys.argv[4])        # percent of entities to be inserted
    port = 19530

    file_handler = logging.FileHandler(filename=f"/tmp/upsert_{collection_name}.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    logger = logging.getLogger('LOGGER_NAME')

    conn = connections.connect('default', host=host, port=port)

    skip_delete = False
    skip_insert = False
    if delete_percent <= 0 or delete_percent > 100:
        logging.error(f"delete percent{delete_percent} shall be 0-100, skip delete")
        skip_delete = True
    if insert_percent <= 0 or insert_percent > 100:
        logging.error(f"insert percent{insert_percent} shall be 1-100, skip insert")
        skip_insert = True

    # check and get the collection info
    if not utility.has_collection(collection_name=collection_name):
        logging.error(f"collection: {collection_name} does not exit")
        exit(0)

    c = Collection(name=collection_name)
    nb = 10000
    num_entities = c.num_entities
    if num_entities <= 0:
        logging.error(f"collection: {collection_name} num_entities is empty")
        exit(0)
    logging.info(f"{collection_name} num_entities {num_entities}")

    if not c.has_index():
        logging.error(f"collection: {collection_name} has no index")
        exit(0)

    search_params = get_search_params(collection=c)

    # load collection
    t1 = time.time()
    c.load()
    t2 = round(time.time() - t1, 3)
    logging.info(f"load {collection_name}: {t2}")

    if not skip_delete:
        delete_num = num_entities * delete_percent // 100
        # insert nb if less than nb
        delete_rounds = int(delete_num // nb + 1)
        logging.info(f"{delete_rounds * nb} entities to be deleted in {delete_rounds} rounds")
        # delete xx% entities
        delete_entities(collection=c, nb=nb, search_params=search_params,
                        rounds=delete_rounds)

    if not skip_insert:
        insert_num = num_entities * insert_percent // 100
        # insert nb if less than nb
        insert_rounds = int(insert_num // nb + 1)
        logging.info(f"{insert_rounds * nb} entities to be insert in {insert_rounds} rounds")
        # insert xx% entities
        insert_entities(collection=c, nb=nb, rounds=insert_rounds)

    logging.info(f"{collection_name} upsert completed")
