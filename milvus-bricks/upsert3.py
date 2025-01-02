import time
import sys
import random
import logging
from pymilvus import utility, connections, DataType, \
    Collection, FieldSchema, CollectionSchema
from pymilvus.orm.types import CONSISTENCY_STRONG
from common import insert_entities
from create_n_insert import create_n_insert


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"


if __name__ == '__main__':
    """
    upsert the entities in the collection with the same id, the version field will be updated as input
    only works for int64 primary key, staring from 0
    """
    host = sys.argv[1]
    collection_name = sys.argv[2]                       # collection mame
    upsert_rounds = int(sys.argv[3])                    # upsert time
    entities_per_round = int(sys.argv[4])               # entities to be upsert per round
    new_version = str(sys.argv[5]).upper()                      # the new value for version field in upsert requests
    interval = int(sys.argv[6])                         # interval between upsert rounds
    check_diff = str(sys.argv[7]).upper()               # if check dup entity
    port = 19530

    file_handler = logging.FileHandler(filename=f"/tmp/upsert3_{collection_name}.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    logger = logging.getLogger('LOGGER_NAME')

    logging.info(f"upsert3: host={host}, collection_name={collection_name}, upsert_rounds={upsert_rounds}, "
                 f"entities_per_round={entities_per_round}, new_version={new_version}, interval={interval}, "
                 f"check_diff={check_diff}")

    conn = connections.connect('default', host=host, port=port)
    check_diff = True if check_diff == "TRUE" else False
    new_version = time.asctime() if new_version == "NONE" else new_version

    # check and get the collection info
    if not utility.has_collection(collection_name=collection_name):
        logging.error(f"collection: {collection_name} does not exit, create an empty collection as default")
        dim = 128
        intpk_field = FieldSchema(name="id", dtype=DataType.INT64, description="primary id")
        fname_field = FieldSchema(name="fname", dtype=DataType.VARCHAR, max_length=256, description="fname")
        version_field = FieldSchema(name="version", dtype=DataType.VARCHAR, max_length=256, description="data version")
        embedding_field = FieldSchema(name=f"embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim)
        fields = [intpk_field, fname_field, version_field, embedding_field]
        schema = CollectionSchema(fields=fields, auto_id=False, primary_field=intpk_field.name,
                                  description=f"{collection_name}")
        create_n_insert(collection_name=collection_name, dims=[dim], nb=2000, insert_times=0, index_types=["HNSW"],
                        auto_id=False, vector_types=[DataType.FLOAT_VECTOR], metric_types=["L2"],
                        build_index=True, schema=schema)

    c = Collection(name=collection_name)
    if len(c.indexes) == 0:
        logging.error(f"collection: {collection_name} has no index")
        exit(-1)
    auto_id = c.schema.auto_id
    primary_field_type = c.primary_field.dtype
    if auto_id:
        logging.error(f"{collection_name} has auto_id=True, which is not supported")
        exit(-1)

    # load collection
    t1 = time.time()
    c.load()
    t2 = round(time.time() - t1, 3)
    logging.info(f"load {collection_name}: {t2}")

    max_id = upsert_rounds * entities_per_round
    logging.info(f"{collection_name} is going to upsert {max_id} entities, starting from id 0")
    # start upsert
    logging.info(f"{collection_name} upsert3 start: nb={entities_per_round}, rounds={upsert_rounds}")
    insert_entities(collection=c, nb=entities_per_round, rounds=upsert_rounds,
                    use_insert=False, interval=interval, new_version=new_version)
    c.flush()
    new_max_id = c.query(expr="", output_fields=["count(*)"],
                         consistency_level=CONSISTENCY_STRONG)[0].get("count(*)")

    logging.info(f"{collection_name} upsert3 completed, max_id: {max_id}, new_query_count*: {new_max_id}")

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

