import time
import sys
import random
import logging
from pymilvus import utility, connections, DataType, \
    Collection, FieldSchema, CollectionSchema
from pymilvus.orm.types import CONSISTENCY_STRONG
from common import upsert_entities
from create_n_insert import create_n_insert
import os


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"


if __name__ == '__main__':
    """
    if there already exists entities in the collection, this script will do upsert with existing entities' PK
    if there is no entities in the collection, this script will do upsert as insert(unqiue PKs in one request)
    1. upsert M entities from versionN to versionN+1
    2. delete all the entities with versionN
    3. check the number of versionN+1 is M
    """
    host = sys.argv[1]
    collection_name = sys.argv[2]                   # collection mame
    upsert_rounds = int(sys.argv[3])                # upsert time
    entities_per_round = int(sys.argv[4])           # entities to be upsert per round
    new_version = int(sys.argv[5])                  # the new value for version field in upsert requests
    unique_in_requests = str(sys.argv[6]).upper()   # if gen unique pk in all upsert requests
    interval = int(sys.argv[7])                     # interval between upsert rounds
    port = 19530

    file_handler = logging.FileHandler(filename=f"/tmp/upsert_n_deletebyexpr_{collection_name}.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    logger = logging.getLogger('LOGGER_NAME')

    conn = connections.connect('default', host=host, port=port)
    if not utility.has_collection(collection_name):
        logging.error(f"collection: {collection_name} not found")
        exit(-1)

    c = Collection(name=collection_name)
    if not c.has_index():
        logging.error(f"collection: {collection_name} has no index")
        exit(-1)

    t1 = time.time()
    c.load()
    t2 = round(time.time() - t1, 3)
    logging.info(f"load {collection_name}: {t2}")
    old_version = c.query(expr="", limit=1, output_fields=["version"])[0].get("version")

    # doing upsert: update the version to new value
    os.system(f"python3.8 upsert2.py {host} {collection_name} {upsert_rounds} {entities_per_round} "
              f"{new_version} {unique_in_requests} {interval} false")

    # delete all the old version data
    c.delete(expr=f"version=={old_version}")
    count_after_delete = c.query(expr="", output_fields=["count(*)"],
                                 consistency_level=CONSISTENCY_STRONG)[0].get("count(*)")
    logging.info(f"{collection_name} delete by expr completed, count(*) after delete: {count_after_delete}")
    c.flush()
    count_after_flush = c.query(expr="", output_fields=["count(*)"],
                                consistency_level=CONSISTENCY_STRONG)[0].get("count(*)")
    logging.info(f"{collection_name} flush, count(*) after flush: {count_after_flush}")

    logging.info("upsert and delete by expr completed")
