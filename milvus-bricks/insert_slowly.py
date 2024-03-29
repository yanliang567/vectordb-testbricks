import time
import sys
import random
import logging
from pymilvus import utility, connections, DataType, \
    Collection, FieldSchema, CollectionSchema
from create_n_insert import create_n_insert
from common import insert_entities, delete_entities, get_search_params


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"


if __name__ == '__main__':
    host = sys.argv[1]
    collection_name = sys.argv[2]              # collection mame
    nb = int(sys.argv[3])                      # batch size of one insert request
    insert_interval = int(sys.argv[4])         # frequency of insert in seconds
    timeout = int(sys.argv[5])                 # insert timeout

    port = 19530

    file_handler = logging.FileHandler(filename=f"/tmp/insert_slowly_{collection_name}.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    logger = logging.getLogger('LOGGER_NAME')

    conn = connections.connect('default', host=host, port=port)
    if insert_interval <= 0:
        insert_interval = 10
        logging.info(f"set insert_interval to default: 10s")
    if nb <= 0:
        nb = 10
        logging.info(f"set insert_ni to default: 10")

    # check and get the collection info
    if not utility.has_collection(collection_name=collection_name):
        dim = 768
        logging.info(f"collection {collection_name} not exists, create a HNSW-L2 collection with dim={dim}")
        create_n_insert(collection_name=collection_name,
                        dim=dim, nb=1000, insert_times=10, auto_id=False,
                        index_type="AUTOINDEX", metric_type="L2")
        collection = Collection(name=collection_name)
        logging.info(f"create {collection_name}  successfully")

    c = Collection(name=collection_name)
    num_entities = c.num_entities
    logging.info(f"{collection_name} num_entities {num_entities}")

    # t1 = time.time()
    # c.load()
    # t2 = round(time.time() - t1, 3)
    # logging.info(f"{collection_name} load in {t2}")

    logging.info(f"start to insert slowly with nb={nb}, interval={insert_interval}, timeout={timeout}")
    start_time = time.time()
    r = 0
    while time.time() < start_time + timeout:
        # insert and delete entities
        insert_entities(collection=c, nb=nb, rounds=1)
        r += 1
        time.sleep(insert_interval)

    logging.info(f"{collection_name} insert slowly completed")
