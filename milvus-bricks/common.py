import time
import sys
import string
import random
import logging
import json
import numpy as np
from pymilvus import utility, connections, DataType, \
    Collection, FieldSchema, CollectionSchema

pk_prefix = "iamprimarykey_"
FLOAT_VECTOR_TYPES = [DataType.FLOAT_VECTOR, DataType.BFLOAT16_VECTOR, DataType.FLOAT16_VECTOR]
ALL_VECTOR_TYPES = [DataType.FLOAT_VECTOR, DataType.BFLOAT16_VECTOR, DataType.FLOAT16_VECTOR,
                    DataType.BINARY_VECTOR, DataType.SPARSE_FLOAT_VECTOR]


def get_float_vec_dim(collection):
    """
    return the dim of float32 vector field in collection
    if there are multiple float32 vector fields, return the first one
    :param collection: Collection object
    :return:
    """
    fields = collection.schema.fields
    for field in fields:
        if field.dtype == DataType.FLOAT_VECTOR:
            dim = field.params.get("dim")
            break
    return dim


def get_dim_by_field_name(collection, name):
    """
    return the dim of vector field in collection by field name
    :param collection : Collection object
    :param name : str, field name
    :return: dim int
    """
    fields = collection.schema.fields
    for field in fields:
        if field.name == name:
            dim = field.params.get("dim", None)
            break
    return dim


def get_dims(collection):
    """
    return the dims of all vector fields in dict format
    :param collection:
    :return: dict e.g. {"field1": dim1, "field2": dim2}
    """
    dims = {}
    fields = collection.schema.fields
    for field in fields:
        if field.dtype in FLOAT_VECTOR_TYPES:
            dim = field.params.get("dim")
            dims.update({field.name: dim})
            continue
    return dims


def get_float_vec_field_name(collection):
    """
    return the name of float32 vector field in collection,
    if there are mutli float32 vector fields, return the first one
    :param collection
    :return: str
    """
    fields = collection.schema.fields
    for field in fields:
        if field.dtype == DataType.FLOAT_VECTOR:
            vector_field_name = field.name
            break
    return vector_field_name


def get_float_vec_field_names(collection):
    """
    return the names of all the float vector fields in collection
    :param collection
    :return: list
    """
    vector_field_names = [field.name for field in collection.schema.fields
                          if field.dtype in ALL_VECTOR_TYPES]
    return vector_field_names


def delete_entities(collection, nb, search_params, rounds):
    dim = get_float_vec_dim(collection=collection)
    auto_id = collection.schema.auto_id
    primary_field_name = collection.primary_field.name
    vector_field_name = get_float_vec_field_name(collection=collection)
    if auto_id:
        for r in range(rounds):
            search_vector = [[random.random() for _ in range(dim)] for _ in range(1)]
            results = collection.search(data=search_vector, anns_field=vector_field_name,
                                        param=search_params, limit=nb)
            for hits in results:
                ids = hits.ids
                collection.delete(expr=f"{primary_field_name} in {ids}")
                logging.info(f"deleted {len(ids)} entities")
    else:
        for r in range(rounds):
            start_uid = r * nb
            end_uid = start_uid + nb
            collection.delete(expr=f"{primary_field_name} in [{start_uid}, {end_uid}]")
            logging.info(f"deleted entities {start_uid}-{end_uid}")


def gen_data_by_collection(collection, nb, r, new_version=0):
    data = []
    s = '{"glossary": {"title": "example glossary", "GlossDiv": {"title": "S", "GlossList": ' \
        '{"GlossEntry": {"ID": "SGML","SortAs": "SGML","GlossTerm": ' \
        '"Standard Generalized Markup Language","GlossDef": ' \
        '{"para": "A meta-markup language, used to create markup languages such as DocBook.",' \
        '"GlossSeeAlso": ["GML", "XML"]},"GlossSee": "markup"}}}}}'
    start_uid = r * nb
    fields = collection.schema.fields
    auto_id = collection.schema.auto_id
    for field in fields:
        nullable = field.nullable
        if field.dtype == DataType.FLOAT_VECTOR:
            dim = field.params.get("dim")
            data.append([[random.random() for _ in range(dim)] for _ in range(nb)])
            continue
        if field.dtype == DataType.INT64:
            if nullable:
                seed = random.randint(0, 100)
                if seed % 5 == 0:
                    data.append([None for _ in range(start_uid, start_uid + nb)])
                    continue
            if field.is_primary:
                if not auto_id:
                    pks = [_ for _ in range(start_uid, start_uid + nb)]
                    random.shuffle(pks)
                    data.append(pks)
                    continue
                else:
                    continue
            else:
                if field.name == "category":
                    data.append([random.randint(1, 100) for _ in range(nb)])
                else:
                    data.append([_ for _ in range(start_uid, start_uid + nb)])
                continue
        if field.dtype == DataType.INT8:
            if nullable:
                seed = random.randint(0, 100)
                if seed % 5 == 0:
                    data.append([None for _ in range(start_uid, start_uid + nb)])
                    continue
            data.append([random.randint(-128, 127) for _ in range(nb)])
            continue
        if field.dtype == DataType.INT16:
            if nullable:
                seed = random.randint(0, 100)
                if seed % 5 == 0:
                    data.append([None for _ in range(start_uid, start_uid + nb)])
                    continue
            data.append([random.randint(-32768, 32767) for _ in range(nb)])
            continue
        if field.dtype == DataType.INT32:
            if nullable:
                seed = random.randint(0, 100)
                if seed % 5 == 0:
                    data.append([None for _ in range(start_uid, start_uid + nb)])
                    continue
            if field.name == "version":
                data.append([new_version for _ in range(nb)])
            else:
                data.append([random.randint(-2147483648, 2147483647) for _ in range(nb)])
            continue
        if field.dtype == DataType.VARCHAR:
            if nullable:
                seed = random.randint(0, 100)
                if seed % 5 == 0:
                    data.append([None for _ in range(start_uid, start_uid + nb)])
                    continue
            if field.is_primary:
                if not auto_id:
                    pks = [pk_prefix + str(j) for j in range(start_uid, start_uid + nb)]
                    random.shuffle(pks)
                    data.append(pks)
                    continue
                else:
                    logging.error(f"varchar pk shall not be auto_id.")
                    return None
            else:
                if field.name == "version":
                    data.append([str(new_version) for _ in range(nb)])
                else:
                    max_length = field.params.get("max_length")
                    data.append(["bb_" + gen_str_by_length(max_length//10) for _ in range(nb)])
                    # data.append([json.dumps(s) for _ in range(start_uid, start_uid + nb)])
                continue
        if field.dtype == DataType.JSON:
            if nullable:
                seed = random.randint(0, 100)
                if seed % 5 == 0:
                    data.append([None for _ in range(start_uid, start_uid + nb)])
                    continue
            # data.append([{"number": i, "float": i * 1.0} for i in range(start_uid, start_uid + nb)])
            data.append([json.loads(s) for _ in range(start_uid, start_uid + nb)])
            continue
        if field.dtype in [DataType.FLOAT, DataType.DOUBLE]:
            if nullable:
                seed = random.randint(0, 100)
                if seed % 5 == 0:
                    data.append([None for _ in range(start_uid, start_uid + nb)])
                    continue
            data.append([random.random() for _ in range(nb)])
            continue
        if field.dtype == DataType.BOOL:
            if nullable:
                seed = random.randint(0, 100)
                if seed % 5 == 0:
                    data.append([None for _ in range(start_uid, start_uid + nb)])
                    continue
            data.append([False for _ in range(nb)])
            continue
        else:
            logging.error(f"found undefined datatype: {field.dtype} in field {field.name} in collection {collection.name}")
            exit(-1)
    return data


def gen_upsert_data_by_pk_collection(collection, nb, start=0, end=0, new_version=0):
    data = []
    s = '{"glossary": {"title": "example glossary", "GlossDiv": {"title": "S", "GlossList": ' \
        '{"GlossEntry": {"ID": "SGML","SortAs": "SGML","GlossTerm": ' \
        '"Standard Generalized Markup Language","GlossDef": ' \
        '{"para": "A meta-markup language, used to create markup languages such as DocBook.",' \
        '"GlossSeeAlso": ["GML", "XML"]},"GlossSee": "markup"}}}}}'
    fields = collection.schema.fields
    auto_id = collection.schema.auto_id
    for field in fields:
        if field.dtype == DataType.FLOAT_VECTOR:
            dim = field.params.get("dim")
            data.append([[random.random() for _ in range(dim)] for _ in range(nb)])
            continue
        if field.dtype in [DataType.INT64]:
            if field.is_primary:
                if not auto_id:
                    pop = list(range(start, end))
                    ids = random.sample(pop, nb)
                    data.append(ids)
                    # logging.info(f"ids to be upsert: {ids}")
                    continue
                else:
                    continue
            else:
                data.append([_ for _ in range(nb)])
                continue
        if field.dtype == DataType.INT8:
            data.append([random.randint(-128, 127) for _ in range(nb)])
            continue
        if field.dtype == DataType.INT16:
            data.append([random.randint(-32768, 32767) for _ in range(nb)])
            continue
        if field.dtype == DataType.INT32:
            if field.name == "version":
                data.append([new_version for _ in range(nb)])
            else:
                data.append([random.randint(-2147483648, 2147483647) for _ in range(nb)])
                continue
        if field.dtype == DataType.VARCHAR:
            if not field.is_primary:
                max_length = field.params.get("max_length")
                data.append([gen_str_by_length(max_length // 10) for _ in range(nb)])
                continue
            else:
                if not auto_id:
                    pop = list(range(start, end))
                    ids = random.sample(pop, nb)
                    data.append([pk_prefix + str(j) for j in ids])
                else:
                    continue
        if field.dtype == DataType.JSON:
            data.append([json.loads(s) for _ in range(nb)])
            continue
        if field.dtype in [DataType.FLOAT, DataType.DOUBLE]:
            data.append([random.random() for _ in range(nb)])
            continue
        if field.dtype == DataType.BOOL:
            data.append([True for _ in range(nb)])       # update to true in upsert
            continue
    return data


def insert_entities(collection, nb, rounds, use_insert=True, interval=0, new_version="0"):
    auto_id = collection.schema.auto_id
    for r in range(int(rounds)):
        data = gen_data_by_collection(collection=collection, nb=nb, r=r, new_version=new_version)
        t1 = time.time()
        if not use_insert and auto_id is False:
            collection.upsert(data)
        else:
            collection.insert(data)
        t2 = round(time.time() - t1, 3)
        logging.info(f"{collection.name} insert {r} costs {t2}")
        time.sleep(interval)


def upsert_entities(collection, nb, rounds, maxid, new_version="0", unique_in_requests=False, interval=0):
    start = 0
    sample_n = maxid // rounds
    for r in range(rounds):
        if unique_in_requests:
            end = start+sample_n
        else:
            start = 0
            end = maxid
        data = gen_upsert_data_by_pk_collection(collection=collection, nb=nb, start=start, end=end, new_version=new_version)
        t1 = time.time()
        collection.upsert(data)
        t2 = round(time.time() - t1, 3)
        logging.info(f"{collection.name} upsert2 {r} costs {t2}")
        time.sleep(interval)
        start += sample_n


def get_search_params(collection, topk, index_name=None):
    idx = collection.index(index_name=index_name)
    metric_type = idx.params.get("metric_type")
    index_type = idx.params.get("index_type")
    if index_type == "HNSW":
        ef = max(64, topk)
        search_params = {"metric_type": metric_type, "params": {"ef": ef}}
    elif index_type in ["IVF_SQ8", "IVF_FLAT"]:
        search_params = {"metric_type": metric_type, "params": {"nprobe": 32}}
    elif index_type == "DISKANN":
        search_params = {"metric_type": metric_type, "params": {"search_list": 100}}
    elif index_type == "AUTOINDEX":
        search_params = {"metric_type": metric_type, "params": {}}
    else:
        logging.error(f"index: {index_type} does not support yet")
        exit(-1)
    return search_params


def get_default_params_by_index_type(index_type, metric_type):
    index_params_dict = {
            "HNSW": {"index_type": "HNSW", "metric_type": metric_type, "params": {"M": 30, "efConstruction": 360}},
            "FLAT": {"index_type": "FLAT", "metric_type": metric_type, "params": {}},
            "IVF_FLAT": {"index_type": "IVF_FLAT", "metric_type": metric_type, "params": {"nlist": 1024}},
            "IVF_SQ8": {"index_type": "IVF_SQ8", "metric_type": metric_type, "params": {"nlist": 1024}},
            "DISKANN": {"index_type": "DISKANN", "metric_type": metric_type, "params": {}},
            "AUTOINDEX": {"index_type": "AUTOINDEX", "metric_type": metric_type, "params": {}},
    }
    index_params = index_params_dict.get(index_type.upper(), None)
    if index_params is None:
        logging.error(f"index type {index_type} no supported")
        exit(-1)
    return index_params


def get_index_params(collection, index_name=None):
    idx = collection.index(index_name=index_name)
    return idx.params


def get_index_by_field_name(collection, field_name):
    for idx in collection.indexes:
        if idx.field_name == field_name:
            return idx
    return None


def is_vector_field(collection, field_name):
    fields = collection.schema.fields
    for field in fields:
        if field.name == field_name and field.dtype in ALL_VECTOR_TYPES:
            return True
    return False


def gen_str_by_length(length=8, letters_only=False):
    if letters_only:
        return "".join(random.choice(string.ascii_letters) for _ in range(length))
    return "".join(random.choice(string.ascii_letters + string.digits) for _ in range(length))

