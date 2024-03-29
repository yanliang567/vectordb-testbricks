import time
import sys
import random
import logging
from pymilvus import utility, connections, DataType, \
    Collection, FieldSchema, CollectionSchema
from pymilvus.orm.types import CONSISTENCY_STRONG
from common import upsert_entities
from create_n_insert import create_n_insert


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"


if __name__ == '__main__':
    """
    if there already exists entities in the collection, this script will do upsert with existing entities' PK
    if there is no entities in the collection, this script will do upsert as insert(unique PKs in one request)
    1. upsert M entities from versionN to new version
    """
    host = sys.argv[1]
    collection_name = sys.argv[2]                       # collection mame
    upsert_rounds = int(sys.argv[3])                    # upsert time
    entities_per_round = int(sys.argv[4])               # entities to be upsert per round
    new_version = int(sys.argv[5])                      # the new value for version field in upsert requests
    unique_in_all_requests = str(sys.argv[6]).upper()   # if gen unique pk in all upsert requests
    interval = int(sys.argv[7])                         # interval between upsert rounds
    check_diff = str(sys.argv[8]).upper()               # if check dup entity
    port = 19530

    file_handler = logging.FileHandler(filename=f"/tmp/upsert2_{collection_name}.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    logger = logging.getLogger('LOGGER_NAME')

    conn = connections.connect('default', host=host, port=port)
    unique_in_requests = True if unique_in_all_requests == "TRUE" else False
    check_diff = True if check_diff == "TRUE" else False

    # check and get the collection info
    if not utility.has_collection(collection_name=collection_name):
        logging.error(f"collection: {collection_name} does not exit, create an empty collection as default")
        create_n_insert(collection_name=collection_name, dim=768, nb=2000, insert_times=0, auto_id=False,
                        use_str_pk=True, index_type="AUTOINDEX", metric_type="L2")

    c = Collection(name=collection_name)
    if not c.has_index():
        logging.error(f"collection: {collection_name} has no index")
        exit(0)
    auto_id = c.schema.auto_id
    primary_field_type = c.primary_field.dtype
    if auto_id:
        logging.error(f"{collection_name} has auto_id=True, which is not supported")
        exit(0)

    # load collection
    t1 = time.time()
    c.load()
    t2 = round(time.time() - t1, 3)
    logging.info(f"load {collection_name}: {t2}")
    max_id = c.query(expr="", output_fields=["count(*)"],
                     consistency_level=CONSISTENCY_STRONG)[0].get("count(*)")
    if max_id == 0:
        max_id = upsert_rounds * entities_per_round
        logging.info(f"{collection_name} is empty, set max_id=upsert_rounds * entities_per_round")
    # start upsert
    logging.info(f"{collection_name} max_id={max_id}, upsert2 start: nb={entities_per_round}, rounds={upsert_rounds}")
    upsert_entities(collection=c, nb=entities_per_round, rounds=upsert_rounds, maxid=max_id, new_version=new_version,
                    unique_in_requests=unique_in_requests, interval=interval)
    c.flush()
    new_max_id = c.query(expr="", output_fields=["count(*)"],
                         consistency_level=CONSISTENCY_STRONG)[0].get("count(*)")

    logging.info(f"{collection_name} upsert2 completed, max_id: {max_id}, new_max_id: {new_max_id}")

    if max_id != new_max_id and check_diff:
        dup_count = 0
        logging.info(f"start checking the difference between max_id and new_max_id...")
        for i in range(max_id):
            res = c.query(expr=f"id=={i}", output_fields=["count(*)"], consistency_level=CONSISTENCY_STRONG)
            count = res[0]["count(*)"]
            if count == 1:
                pass
            else:
                dup_count += 1
                logging.error(f"id {i} found {count} entities")
                break

        logging.info(f"check difference completed, dup_count: {dup_count}")

