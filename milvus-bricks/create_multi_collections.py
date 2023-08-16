import time
import sys
import random
import logging
from pymilvus import utility, connections, DataType, \
    Collection, FieldSchema, CollectionSchema
from create_n_insert import create_n_insert


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"


if __name__ == '__main__':
    host = sys.argv[1]
    collection_prefix = sys.argv[2]                 # collection mame prefix
    numbers = int(sys.argv[3])                      # how many collections to create
    need_build_index = str(sys.argv[4]).upper()     # build index or not after insert
    need_load = str(sys.argv[5]).upper()            # load the collection or not at the end

    port = 19530

    file_handler = logging.FileHandler(filename=f"/tmp/create_{numbers}_collections.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    logger = logging.getLogger('LOGGER_NAME')

    conn = connections.connect('default', host=host, port=port)

    need_build_index = True if need_build_index == "TRUE" else False
    need_load = True if need_load == "TRUE" else False

    # check and get the collection info
    logging.info(f"start to create {numbers} collections")
    for i in range(numbers):
        collection_name = f"{collection_prefix}_{i}"
        if not utility.has_collection(collection_name=collection_name):
            dim = random.randint(100, 1000)
            insert_times = random.randint(5, 20)
            auto_id = random.choice([True, False])
            create_n_insert(collection_name=collection_name,
                            dim=dim, nb=1000, insert_times=insert_times, auto_id=auto_id,
                            index_type="HNSW", metric_type="L2", need_build_index=need_build_index)
            logging.info(f"create {collection_name}  successfully")
        else:
            logging.info(f"{collection_name} already exists")

        if need_load:
            c = Collection(name=collection_name)
            t1 = time.time()
            c.load()
            t2 = round(time.time() - t1, 3)
            logging.info(f"{collection_name} load in {t2}")

    logging.info(f"create multi collections completed")
