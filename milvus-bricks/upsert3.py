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
    new_version = sys.argv[5]                           # the new value for version field in upsert requests
    interval = int(sys.argv[6])                         # interval between upsert rounds
    check_diff = str(sys.argv[7]).upper()               # if check dup entity
    port = 19530

    file_handler = logging.FileHandler(filename=f"/tmp/upsert3_{collection_name}.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    logger = logging.getLogger('LOGGER_NAME')

    check_diff = True if check_diff == "TRUE" else False
    rand_c = True if collection_name.upper() == "RAND" or collection_name.upper() == "RANDOM" else False

    logging.info(f"upsert3: host={host}, collection_name={collection_name}, upsert_rounds={upsert_rounds}, "
                 f"entities_per_round={entities_per_round}, new_version={new_version}, interval={interval}, "
                 f"check_diff={check_diff}")

    conn = connections.connect('default', host=host, port=port)

    collection_names = None
    # check and get the collection info
    if rand_c:
        collection_names = random.sample(utility.list_collections(), 100)
    else:
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
            collection_names = [collection_name]

    for collection_name in collection_names:
        c = Collection(name=collection_name)
        if len(c.indexes) == 0:
            logging.error(f"collection: {collection_name} has no index")
            if rand_c:
                continue
            else:
                exit(-1)
        auto_id = c.schema.auto_id
        primary_field_type = c.primary_field.dtype
        if auto_id:
            logging.error(f"{collection_name} has auto_id=True, which is not supported")
            if rand_c:
                continue
            else:
                exit(-1)

        # load collection
        t1 = time.time()
        c.load()
        t2 = round(time.time() - t1, 3)
        logging.info(f"load {collection_name}: {t2}")

        if c.num_entities > 0:
            res = c.query(expr="", limit=1, output_fields=["version"])
            old_version = res[0].get("version")
            logging.info(f"old_version: {old_version}")
        else:
            old_version = "NONE"
            logging.info(f"collection is empty, old_version: {old_version}")

        # get version field type
        fields = c.schema.fields
        for field in fields:
            logging.info(f"field: {field.name}, {field.dtype}")
            if field.name == "version":
                version_field_type = field.dtype
                break
        if new_version == "NONE":
            logging.info(f"version field type: {version_field_type}")
            if version_field_type == DataType.VARCHAR:
                new_version = time.asctime()
            elif version_field_type == DataType.INT32:
                new_version = int(time.time())
        else:
            pass

        max_id = upsert_rounds * entities_per_round
        logging.info(f"{collection_name} is going to upsert {max_id} entities, "
                     f"starting from id 0, new_version: {new_version}")
        # start upsert
        logging.info(f"{collection_name} upsert3 start: nb={entities_per_round}, rounds={upsert_rounds}")
        insert_entities(collection=c, nb=entities_per_round, rounds=upsert_rounds,
                        use_insert=False, interval=interval, new_version=new_version)
        # c.flush()
        new_max_id = c.query(expr="", output_fields=["count(*)"], consistency_level=CONSISTENCY_STRONG)[0].get("count(*)")

        logging.info(f"{collection_name} upsert3 completed, max_id: {max_id}, new_query_count*: {new_max_id}")

        expr = f"version=='{old_version}'" if version_field_type == DataType.VARCHAR else f"version=={old_version}"
        res = c.query(expr=expr, output_fields=["count(*)"], consistency_level=CONSISTENCY_STRONG)
        count = res[0]["count(*)"]
        if count > 0:
            logging.error(f"{collection_name} old_version {old_version} found {count} entities")
        else:
            logging.info(f"{collection_name} old_version {old_version} not found in the collection")

        if check_diff:
            dup_count = 0
            logging.info(f"start checking {collection_name} the difference between max_id and new_max_id...")
            for i in range(max_id):
                res = c.query(expr=f"id=={i}", output_fields=["count(*)"], consistency_level=CONSISTENCY_STRONG)
                count = res[0]["count(*)"]
                if count == 1:
                    pass
                else:
                    dup_count += 1
                    logging.error(f"id {i} found {count} entities")
                    break

            logging.info(f"check {collection_name} difference completed, dup_count: {dup_count}")

