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
    name = sys.argv[2]                              # collection mame/alias
    reload_times = int(sys.argv[3])                 # reload times
    api_key = str(sys.argv[4])                      # api key for cloud instances
    port = 19530

    file_handler = logging.FileHandler(filename=f"/tmp/load_release_{name}.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    logger = logging.getLogger('LOGGER_NAME')

    logging.info(f"searching collection={name}, host={host}, reload_times={reload_times} api_key={api_key}")

    if api_key is None or api_key == "" or api_key.upper() == "NONE":
        conn = connections.connect('default', host=host, port=port)
    else:
        conn = connections.connect('default', uri=host, token=api_key)

    if name.upper() == "NONE" or name == "":
        name = utility.list_collections()[0]
        logging.info(f"collection name is not specified, use the first one: {name}")

    # check and get the collection info
    if not utility.has_collection(collection_name=name):
        logging.error(f"collection: {name} does not exit, create 10m-128d as default")
        create_n_insert(collection_name=name, dim=128, nb=20000, insert_times=50,
                        index_type="HNSW", metric_type="L2")

    collection = Collection(name=name)
    if not collection.has_index():
        logging.error(f"collection: {name} has no index")
        exit(-1)

    index_params = get_index_params(collection)
    logging.info(f"index param: {index_params}")

    # flush before indexing
    num = collection.num_entities
    logging.info(f"{name} num_entities: {num}")
    logging.info(f"{name} index progress: {utility.index_building_progress(name)}")

    for i in range(reload_times):
        # release collection
        t1 = time.time()
        collection.release()
        t2 = round(time.time() - t1, 3)
        logging.info(f"assert release {name}: {t2}")

        time.sleep(120)
        # load collection
        t1 = time.time()
        collection.load()
        t2 = round(time.time() - t1, 3)
        logging.info(f"assert re-load {name}: {t2}")

        dim = get_float_vec_dim(collection)
        vector_field_name = get_float_vec_field_name(collection)
        nq = 1
        topk = 10
        search_vectors = [[random.random() for _ in range(dim)] for _ in range(nq)]
        search_params = get_search_params(collection, topk)
        t1 = time.time()
        try:
            collection.search(data=search_vectors, anns_field=vector_field_name, param=search_params, limit=topk)
        except Exception as e:
            logging.error(e)
        t2 = round(time.time() - t1, 4)
        logging.info(f"collection {collection.description} search: cost {t2}")

    logging.info(f"load release for {reload_times} times completed")
