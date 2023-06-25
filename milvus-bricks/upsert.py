import time
import sys
import random
import logging
from pymilvus import utility, connections, DataType, \
    Collection, FieldSchema, CollectionSchema
from common import delete_entities, insert_entities, get_search_params
from create_n_insert import create_n_insert


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"


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
        logging.error(f"collection: {collection_name} does not exit, create 10m-128d as default")
        create_n_insert(collection_name=collection_name, dim=128, nb=20000, insert_times=50,
                        index_type="HNSW", metric_type="L2")

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

    search_params = get_search_params(collection=c, topk=nb)

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
