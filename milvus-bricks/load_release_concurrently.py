import time
import sys
import random
import logging
from pymilvus import utility, connections, DataType, \
    Collection, FieldSchema, CollectionSchema
from common import get_float_vec_dim, get_float_vec_field_name, get_search_params, get_index_params
from create_n_insert import create_n_insert

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"

if __name__ == '__main__':
    host = sys.argv[1]
    name = str(sys.argv[2])  # collection mame/alias
    reload_times = int(sys.argv[3])  # reload times
    api_key = str(sys.argv[4])  # api key for cloud instances
    port = 19530

    file_handler = logging.FileHandler(filename=f"/tmp/load_release_concurrently.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    logger = logging.getLogger('LOGGER_NAME')

    logging.info(f"collection={name}, host={host}, reload_times={reload_times} api_key={api_key}")

    if api_key is None or api_key == "" or api_key.upper() == "NONE":
        conn = connections.connect('default', host=host, port=port)
    else:
        conn = connections.connect('default', uri=host, token=api_key)

    if name.upper() == "NONE":
        logging.info(f"collection name is not specified, try to load and release for all the collections")

    names = utility.list_collections() if name.upper() == "NONE" else [name]
    collections_tb_load = []
    for name in names:
        if not utility.has_collection(collection_name=name):
            logging.error(f"collection:{name} does not exit, skip")
            continue
        collection = Collection(name=name)
        if len(collection.indexes) == 0:
            logging.error(f"collection:{name} has no index, skip")
            continue
        index_params = get_index_params(collection)
        logging.info(f"collection:{name} index param: {index_params}")

        # flush before indexing
        num = collection.num_entities
        logging.info(f"collection:{name} num_entities: {num}")
        collections_tb_load.append(collection)

    for i in range(reload_times):
        for c in collections_tb_load:
            # release collection
            c.release()
            logging.info(f"collection:{name} released")

        time.sleep(200)
        for c in collections_tb_load:
            # load collection
            c.load(_async=True)
            logging.info(f"collection:{name} loaded")

        for c in collections_tb_load:
            utility.wait_for_loading_complete(c.name)
            logging.info(f"collection:{name} loaded completed")

    logging.info(f"load release for {reload_times} times completed")


