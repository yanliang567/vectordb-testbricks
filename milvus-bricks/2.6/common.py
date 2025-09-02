import time
import sys
import string
import random
import logging
from faker import Faker
from sklearn import preprocessing
import json
import numpy as np
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema

pk_prefix = "iamprimarykey_"
FLOAT_VECTOR_TYPES = [DataType.FLOAT_VECTOR, DataType.BFLOAT16_VECTOR, DataType.FLOAT16_VECTOR]
ALL_VECTOR_TYPES = [
        DataType.FLOAT_VECTOR,
        DataType.FLOAT16_VECTOR,
        DataType.BFLOAT16_VECTOR,
        DataType.SPARSE_FLOAT_VECTOR,
        DataType.INT8_VECTOR,
        DataType.BINARY_VECTOR,
    ]

fake = Faker()
default_dim = 128


def get_float_vec_dim(client, collection_name):
    """
    return the dim of float32 vector field in collection
    if there are multiple float32 vector fields, return the first one
    :param client: MilvusClient object
    :param collection_name: str, collection name
    :return: int, dimension
    """
    schema = client.describe_collection(collection_name)
    fields = schema.get('fields', [])
    for field in fields:
        if field.get('type') == DataType.FLOAT_VECTOR:
            dim = field.get('params', {}).get('dim')
            if dim:
                return dim
    return None


def get_dim_by_field_name(client, collection_name, field_name):
    """
    return the dim of vector field in collection by field name
    :param client: MilvusClient object
    :param collection_name: str, collection name
    :param field_name: str, field name
    :return: dim int
    """
    schema = client.describe_collection(collection_name)
    fields = schema.get('fields', [])
    for field in fields:
        if field.get('name') == field_name:
            dim = field.get('params', {}).get('dim', None)
            return dim
    return None


def get_dims(client, collection_name):
    """
    return the dims of all vector fields in dict format
    :param client: MilvusClient object
    :param collection_name: str, collection name
    :return: dict e.g. {"field1": dim1, "field2": dim2}
    """
    dims = {}
    schema = client.describe_collection(collection_name)
    fields = schema.get('fields', [])
    for field in fields:
        if field.get('type') in FLOAT_VECTOR_TYPES:
            dim = field.get('params', {}).get('dim')
            if dim:
                dims.update({field.get('name'): dim})
    return dims


def get_float_vec_field_name(client, collection_name):
    """
    return the name of float32 vector field in collection,
    if there are multi float32 vector fields, return the first one
    :param client: MilvusClient object
    :param collection_name: str, collection name
    :return: str
    """
    schema = client.describe_collection(collection_name)
    fields = schema.get('fields', [])
    for field in fields:
        if field.get('type') == DataType.FLOAT_VECTOR:
            return field.get('name')
    return None


def get_float_vec_field_names(client, collection_name):
    """
    return the names of all the float vector fields in collection
    :param client: MilvusClient object
    :param collection_name: str, collection name
    :return: list
    """
    schema = client.describe_collection(collection_name)
    fields = schema.get('fields', [])
    vector_field_names = [field.get('name') for field in fields
                          if field.get('type') in ALL_VECTOR_TYPES]
    return vector_field_names


def get_primary_field_name(client, collection_name):
    """
    return the name of primary field in collection
    :param client: MilvusClient object
    :param collection_name: str, collection name
    :return: str
    """
    schema = client.describe_collection(collection_name)
    fields = schema.get('fields', [])
    for field in fields:
        if field.get('is_primary', False):
            return field.get('name')
    return None


def delete_entities(client, collection_name, nb, search_params, rounds):
    """
    Delete entities from collection
    :param client: MilvusClient object
    :param collection_name: str, collection name
    :param nb: int, number of entities to delete each round
    :param search_params: dict, search parameters
    :param rounds: int, number of rounds to delete
    """
    dim = get_float_vec_dim(client, collection_name)
    schema = client.describe_collection(collection_name)
    auto_id = schema.get('auto_id', False)
    primary_field_name = get_primary_field_name(client, collection_name)
    vector_field_name = get_float_vec_field_name(client, collection_name)
    
    if auto_id:
        for r in range(rounds):
            search_vector = [[random.random() for _ in range(dim)] for _ in range(1)]
            results = client.search(
                collection_name=collection_name,
                data=search_vector,
                anns_field=vector_field_name,
                search_params=search_params,
                limit=nb
            )
            if results and len(results[0]) > 0:
                ids = [hit['id'] for hit in results[0]]
                expr = f"{primary_field_name} in {ids}"
                client.delete(collection_name=collection_name, filter=expr)
                logging.info(f"deleted {len(ids)} entities")
    else:
        for r in range(rounds):
            start_uid = r * nb
            end_uid = start_uid + nb
            expr = f"{primary_field_name} >= {start_uid} and {primary_field_name} < {end_uid}"
            client.delete(collection_name=collection_name, filter=expr)
            logging.info(f"deleted entities {start_uid}-{end_uid}")


def gen_bf16_vectors(num, dim):
    """
    generate brain float16 vector data
    raw_vectors : the vectors
    bf16_vectors: the bytes used for insert
    return: raw_vectors and bf16_vectors
    """
    raw_vectors = []
    bf16_vectors = []
    for _ in range(num):
        raw_vector = [random.random() for _ in range(dim)]
        raw_vectors.append(raw_vector)
        # bf16_vector = np.array(raw_vector, dtype=bfloat16)  # bfloat16 is not standard numpy type
        bf16_vector = np.array(raw_vector, dtype=np.float32)  # fallback to float32
        bf16_vectors.append(bf16_vector)

    return raw_vectors, bf16_vectors


def gen_fp16_vectors(num, dim):
    """
    generate float16 vector data
    raw_vectors : the vectors
    fp16_vectors: the bytes used for insert
    return: raw_vectors and fp16_vectors
    """
    raw_vectors = []
    fp16_vectors = []
    for _ in range(num):
        raw_vector = [random.random() for _ in range(dim)]
        raw_vectors.append(raw_vector)
        fp16_vector = np.array(raw_vector, dtype=np.float16)
        fp16_vectors.append(fp16_vector)

    return raw_vectors, fp16_vectors


def gen_sparse_vectors(nb, dim=1000, sparse_format="dok", empty_percentage=0):
    # default sparse format is dok, dict of keys
    # another option is coo, coordinate List

    rng = np.random.default_rng()
    vectors = [{
        d: rng.random() for d in list(set(random.sample(range(dim), random.randint(20, 30)) + [0, 1]))
    } for _ in range(nb)]
    if empty_percentage > 0:
        empty_nb = int(nb * empty_percentage / 100)
        empty_ids = random.sample(range(nb), empty_nb)
        for i in empty_ids:
            vectors[i] = {}
    if sparse_format == "coo":
        vectors = [
            {"indices": list(x.keys()), "values": list(x.values())} for x in vectors
        ]
    return vectors


def gen_text_vectors(nb, language="en"):
    fake = Faker("en_US")
    if language == "zh":
        fake = Faker("zh_CN")
    vectors = [" milvus " + fake.text() for _ in range(nb)]
    return vectors


def gen_int8_vectors(num, dim):
    raw_vectors = []
    int8_vectors = []
    for _ in range(num):
        raw_vector = [random.randint(-128, 127) for _ in range(dim)]
        raw_vectors.append(raw_vector)
        int8_vector = np.array(raw_vector, dtype=np.int8)
        int8_vectors.append(int8_vector)
    return raw_vectors, int8_vectors


def gen_binary_vectors(num, dim):
    raw_vectors = []
    binary_vectors = []
    for _ in range(num):
        raw_vector = [random.randint(0, 1) for _ in range(dim)]
        raw_vectors.append(raw_vector)
        # packs a binary-valued array into bits in a unit8 array, and bytes array_of_ints
        binary_vectors.append(bytes(np.packbits(raw_vector, axis=-1).tolist()))
    return raw_vectors, binary_vectors


def gen_vectors(nb, dim, vector_data_type=DataType.FLOAT_VECTOR):
    vectors = []
    if vector_data_type == DataType.FLOAT_VECTOR:
        vectors = [[random.uniform(-1, 1) for _ in range(dim)] for _ in range(nb)]
    elif vector_data_type == DataType.FLOAT16_VECTOR:
        vectors = gen_fp16_vectors(nb, dim)[1]
    elif vector_data_type == DataType.BFLOAT16_VECTOR:
        vectors = gen_bf16_vectors(nb, dim)[1]
    elif vector_data_type == DataType.SPARSE_FLOAT_VECTOR:
        vectors = gen_sparse_vectors(nb, dim)
    elif vector_data_type == "TEXT_SPARSE_VECTOR":
        vectors = gen_text_vectors(nb)    # for Full Text Search
    elif vector_data_type == DataType.BINARY_VECTOR:
        vectors = gen_binary_vectors(nb, dim)[1]
    elif vector_data_type == DataType.INT8_VECTOR:
        vectors = gen_int8_vectors(nb, dim)[1]
    else:
        logging.error(f"Invalid vector data type: {vector_data_type}")
        raise Exception(f"Invalid vector data type: {vector_data_type}")
    if dim > 1:
        if vector_data_type == DataType.FLOAT_VECTOR:
            vectors = preprocessing.normalize(vectors, axis=1, norm='l2')
            vectors = vectors.tolist()
    return vectors


def gen_str_by_length(length=8, letters_only=False, contain_numbers=False):
    if letters_only:
        return "".join(random.choice(string.ascii_letters) for _ in range(length))
    if contain_numbers:
        return "".join(random.choice(string.ascii_letters) for _ in range(length-1)) + \
            "".join(random.choice(string.digits))
    return "".join(random.choice(string.ascii_letters + string.digits) for _ in range(length))


def gen_varchar_data(length: int, nb: int, text_mode=False):
    if text_mode:
        return [fake.text() for _ in range(nb)]
    else:
        return ["".join([chr(random.randint(97, 122)) for _ in range(length)]) for _ in range(nb)]


def gen_data_by_collection_field(field, nb=None, start=0, random_pk=False):
    """
    Generates test data for a given collection field based on its data type and properties.

    Args:
        field (dict or Field): Field information, either as a dictionary (v2 client) or Field object (ORM client)
        nb (int, optional): Bumber of data batch to generate. If None, returns a single value which usually used by row data generation
        start (int, optional): Starting value for primary key fields (default: 0)
        random_pk (bool, optional): Whether to generate random primary key values (default: False)
    Returns:
        Single value if nb is None, otherwise returns a list of generated values

    Notes:
        - Handles various data types including primitive types, vectors, arrays and JSON
        - For nullable fields, generates None values approximately 20% of the time
        - Special handling for primary key fields (sequential values)
        - For varchar field, use min(20, max_length) to gen data
        - For vector fields, generates random vectors of specified dimension
        - For array fields, generates arrays filled with random values of element type
    """

    if isinstance(field, dict):
        # for v2 client, it accepts a dict of field info
        nullable = field.get('nullable', False)
        data_type = field.get('type', None)
        enable_analyzer = field.get('params').get("enable_analyzer", False)
        is_primary = field.get('is_primary', False)
    else:
        # for ORM client, it accepts a field object
        nullable = field.nullable
        data_type = field.dtype
        enable_analyzer = field.params.get("enable_analyzer", False)
        is_primary = field.is_primary

    # generate data according to the data type
    if data_type == DataType.BOOL:
        if nb is None:
            return random.choice([True, False]) if random.random() < 0.8 or nullable is False else None
        if nullable is False:
            return [random.choice([True, False]) for _ in range(nb)]
        else:
            # gen 20% none data for nullable field
            return [None if i % 2 == 0 and random.random() < 0.4 else random.choice([True, False]) for i in range(nb)]
    elif data_type == DataType.INT8:
        if nb is None:
            return random.randint(-128, 127) if random.random() < 0.8 or nullable is False else None
        if nullable is False:
            return [random.randint(-128, 127) for _ in range(nb)]
        else:
            # gen 20% none data for nullable field
            return [None if i % 2 == 0 and random.random() < 0.4 else random.randint(-128, 127) for i in range(nb)]
    elif data_type == DataType.INT16:
        if nb is None:
            return random.randint(-32768, 32767) if random.random() < 0.8 or nullable is False else None
        if nullable is False:
            return [random.randint(-32768, 32767) for _ in range(nb)]
        else:
            # gen 20% none data for nullable field
            return [None if i % 2 == 0 and random.random() < 0.4 else random.randint(-32768, 32767) for i in range(nb)]
    elif data_type == DataType.INT32:
        if nb is None:
            return random.randint(-2147483648, 2147483647) if random.random() < 0.8 or nullable is False else None
        if nullable is False:
            return [random.randint(-2147483648, 2147483647) for _ in range(nb)]
        else:
            # gen 20% none data for nullable field
            return [None if i % 2 == 0 and random.random() < 0.4 else random.randint(-2147483648, 2147483647) for i in
                    range(nb)]
    elif data_type == DataType.INT64:
        if nb is None:
            return random.randint(-9223372036854775808,
                                  9223372036854775807) if random.random() < 0.8 or nullable is False else None
        if nullable is False:
            if is_primary is True and random_pk is False:
                return [i for i in range(start, start + nb)]
            else:
                return [random.randint(-9223372036854775808, 9223372036854775807) for _ in range(nb)]
        else:
            # gen 20% none data for nullable field
            return [None if i % 2 == 0 and random.random() < 0.4 else random.randint(-9223372036854775808,
                                                                                     9223372036854775807) for i in
                    range(nb)]
    elif data_type == DataType.FLOAT:
        if nb is None:
            return np.float32(random.random()) if random.random() < 0.8 or nullable is False else None
        if nullable is False:
            return [np.float32(random.random()) for _ in range(nb)]
        else:
            # gen 20% none data for nullable field
            return [None if i % 2 == 0 and random.random() < 0.4 else np.float32(random.random()) for i in range(nb)]
    elif data_type == DataType.DOUBLE:
        if nb is None:
            return np.float64(random.random()) if random.random() < 0.8 or nullable is False else None
        if nullable is False:
            return [np.float64(random.random()) for _ in range(nb)]
        else:
            # gen 20% none data for nullable field
            return [None if i % 2 == 0 and random.random() < 0.4 else np.float64(random.random()) for i in range(nb)]
    elif data_type == DataType.VARCHAR:
        if isinstance(field, dict):
            max_length = field.get('params')['max_length']
        else:
            max_length = field.params['max_length']
        max_length = min(20, max_length - 1)
        length = random.randint(0, max_length)
        if nb is None:
            return gen_varchar_data(length=length, nb=1, text_mode=enable_analyzer)[
                0] if random.random() < 0.8 or nullable is False else None
        if nullable is False:
            if is_primary is True and random_pk is False:
                return [str(i) for i in range(start, start + nb)]
            else:
                return gen_varchar_data(length=length, nb=nb, text_mode=enable_analyzer)
        else:
            # gen 20% none data for nullable field
            return [None if i % 2 == 0 and random.random() < 0.4 else
                    gen_varchar_data(length=length, nb=1, text_mode=enable_analyzer)[0] for i in range(nb)]
    elif data_type == DataType.JSON:
        if nb is None:
            return {"name": fake.name(), "address": fake.address(),
                    "count": random.randint(0, 100)} if random.random() < 0.8 or nullable is False else None
        if nullable is False:
            return [{"name": str(i), "address": i, "count": random.randint(0, 100)} for i in range(nb)]
        else:
            # gen 20% none data for nullable field
            return [None if i % 2 == 0 and random.random() < 0.4 else {"name": str(i), "address": i,
                                                                       "count": random.randint(0, 100)} for i in
                    range(nb)]
    elif data_type in ALL_VECTOR_TYPES:
        if isinstance(field, dict):
            dim = default_dim if data_type == DataType.SPARSE_FLOAT_VECTOR else field.get('params')['dim']
        else:
            dim = default_dim if data_type == DataType.SPARSE_FLOAT_VECTOR else field.params['dim']
        if nb is None:
            return gen_vectors(1, dim, vector_data_type=data_type)[0]
        if nullable is False:
            return gen_vectors(nb, dim, vector_data_type=data_type)
        else:
            raise Exception(f"gen data failed, vector field does not support nullable")
    elif data_type == DataType.ARRAY:
        if isinstance(field, dict):
            max_capacity = field.get('params')['max_capacity']
        else:
            max_capacity = field.params['max_capacity']
        element_type = field.element_type
        if element_type == DataType.INT8:
            if nb is None:
                return [random.randint(-128, 127) for _ in
                        range(max_capacity)] if random.random() < 0.8 or nullable is False else None
            if nullable is False:
                return [[random.randint(-128, 127) for _ in range(max_capacity)] for _ in range(nb)]
            else:
                # gen 20% none data for nullable field
                return [None if i % 2 == 0 and random.random() < 0.4 else random.randint(-128, 127) for i in range(nb)]
        if element_type == DataType.INT16:
            if nb is None:
                return [random.randint(-32768, 32767) for _ in
                        range(max_capacity)] if random.random() < 0.8 or nullable is False else None
            if nullable is False:
                return [[random.randint(-32768, 32767) for _ in range(max_capacity)] for _ in range(nb)]
            else:
                # gen 20% none data for nullable field
                return [None if i % 2 == 0 and random.random() < 0.4 else random.randint(-32768, 32767) for i in
                        range(nb)]
        if element_type == DataType.INT32:
            if nb is None:
                return [random.randint(-2147483648, 2147483647) for _ in
                        range(max_capacity)] if random.random() < 0.8 or nullable is False else None
            if nullable is False:
                return [[random.randint(-2147483648, 2147483647) for _ in range(max_capacity)] for _ in range(nb)]
            else:
                # gen 20% none data for nullable field
                return [None if i % 2 == 0 and random.random() < 0.4 else random.randint(-2147483648, 2147483647) for i
                        in range(nb)]
        if element_type == DataType.INT64:
            if nb is None:
                return [random.randint(-9223372036854775808, 9223372036854775807) for _ in
                        range(max_capacity)] if random.random() < 0.8 or nullable is False else None
            if nullable is False:
                return [[random.randint(-9223372036854775808, 9223372036854775807) for _ in range(max_capacity)] for _
                        in range(nb)]
            else:
                # gen 20% none data for nullable field
                return [None if i % 2 == 0 and random.random() < 0.4 else random.randint(-9223372036854775808,
                                                                                         9223372036854775807) for i in
                        range(nb)]
        if element_type == DataType.BOOL:
            if nb is None:
                return [random.choice([True, False]) for _ in
                        range(max_capacity)] if random.random() < 0.8 or nullable is False else None
            if nullable is False:
                return [[random.choice([True, False]) for _ in range(max_capacity)] for _ in range(nb)]
            else:
                # gen 20% none data for nullable field
                return [None if i % 2 == 0 and random.random() < 0.4 else random.choice([True, False]) for i in
                        range(nb)]
        if element_type == DataType.FLOAT:
            if nb is None:
                return [np.float32(random.random()) for _ in
                        range(max_capacity)] if random.random() < 0.8 or nullable is False else None
            if nullable is False:
                return [[np.float32(random.random()) for _ in range(max_capacity)] for _ in range(nb)]
            else:
                # gen 20% none data for nullable field
                return [None if i % 2 == 0 and random.random() < 0.4 else np.float32(random.random()) for i in
                        range(nb)]
        if element_type == DataType.DOUBLE:
            if nb is None:
                return [np.float64(random.random()) for _ in
                        range(max_capacity)] if random.random() < 0.8 or nullable is False else None
            if nullable is False:
                return [[np.float64(random.random()) for _ in range(max_capacity)] for _ in range(nb)]
            else:
                # gen 20% none data for nullable field
                return [None if i % 2 == 0 and random.random() < 0.4 else np.float64(random.random()) for i in
                        range(nb)]
        if element_type == DataType.VARCHAR:
            if isinstance(field, dict):
                max_length = field.get('params')['max_length']
            else:
                max_length = field.params['max_length']
            max_length = min(20, max_length - 1)
            length = random.randint(0, max_length)
            if nb is None:
                return ["".join([chr(random.randint(97, 122)) for _ in range(length)]) for _ in
                        range(max_capacity)] if random.random() < 0.8 or nullable is False else None
            if nullable is False:
                return [["".join([chr(random.randint(97, 122)) for _ in range(length)]) for _ in range(max_capacity)]
                        for _ in range(nb)]
            else:
                # gen 20% none data for nullable field
                return [None if i % 2 == 0 and random.random() < 0.4 else "".join(
                    [chr(random.randint(97, 122)) for _ in range(length)]) for i in range(nb)]
    else:
        raise Exception(f"gen data failed, data type {data_type} not implemented")
    return None


def gen_row_data_by_schema(nb=2000, schema=None, start=0, random_pk=False, skip_field_names=[], new_version=0):
    """
    Generates row data based on the given schema.

    Args:
        nb (int): Number of rows to generate. Defaults to ct.default_nb.
        schema (Schema): Collection schema or collection info. If None, uses default schema.
        start (int): Starting value for primary key fields. Defaults to 0.
        random_pk (bool, optional): Whether to generate random primary key values (default: False)
        skip_field_names(list, optional): whether to skip some field to gen data manually (default: [])
        new_version (str): used for recording the timestamp for insert/upsert, or customize a value

    Returns:
        list[dict]: List of dictionaries where each dictionary represents a row,
                    with field names as keys and generated data as values.

    Notes:
        - Skips auto_id fields and function output fields.
        - For primary key fields, generates sequential values starting from 'start'.
        - For non-primary fields, generates random data based on field type.
    """
    if schema is None:
        return None

    # ignore auto id field and the fields in function output
    func_output_fields = []
    if isinstance(schema, dict):
        # a dict of collection schema info is usually from client.describe_collection()
        fields = schema.get('fields', [])
        functions = schema.get('functions', [])
        for func in functions:
            output_field_names = func.get('output_field_names', [])
            func_output_fields.extend(output_field_names)
        func_output_fields = list(set(func_output_fields))

        fields_needs_data = []
        for field in fields:
            field_name = field.get('name', None)
            if field.get('auto_id', False):
                continue
            if field_name in func_output_fields or field_name in skip_field_names:
                continue
            fields_needs_data.append(field)
        data = []
        for i in range(nb):
            tmp = {}
            for field in fields_needs_data:
                tmp[field.get('name', None)] = gen_data_by_collection_field(field, random_pk=random_pk)
                if field.get('is_primary', False) is True and field.get('type', None) == DataType.INT64:
                    tmp[field.get('name', None)] = start
                    start += 1
                if field.get('is_primary', False) is True and field.get('type', None) == DataType.VARCHAR:
                    tmp[field.get('name', None)] = str(start)
                    start += 1
                if field.get('name', None) == 'version':
                    tmp[field.get('name')] = str(new_version)
            data.append(tmp)
    else:
        # a schema object is usually form orm schema object
        fields = schema.fields
        if hasattr(schema, "functions"):
            functions = schema.functions
            for func in functions:
                output_field_names = func.output_field_names
                func_output_fields.extend(output_field_names)
        func_output_fields = list(set(func_output_fields))

        fields_needs_data = []
        for field in fields:
            if field.auto_id:
                continue
            if field.name in func_output_fields or field.name in skip_field_names:
                continue
            fields_needs_data.append(field)
        data = []
        for i in range(nb):
            tmp = {}
            for field in fields_needs_data:
                tmp[field.name] = gen_data_by_collection_field(field, random_pk=random_pk)
                if field.is_primary is True and field.dtype == DataType.INT64:
                    tmp[field.name] = start
                    start += 1
                if field.is_primary is True and field.dtype == DataType.VARCHAR:
                    tmp[field.name] = str(start)
                    start += 1
                if field.get('name', None) == 'version':
                    tmp[field.get('name')] = str(new_version)
            data.append(tmp)
    return data


def gen_upsert_data_by_pk_collection(client, collection_name, nb, start=0, end=0, new_version=0):
    """
    Generate upsert data for collection
    """
    data = []
    s = '{"glossary": {"title": "example glossary", "GlossDiv": {"title": "S", "GlossList": ' \
        '{"GlossEntry": {"ID": "SGML","SortAs": "SGML","GlossTerm": ' \
        '"Standard Generalized Markup Language","GlossDef": ' \
        '{"para": "A meta-markup language, used to create markup languages such as DocBook.",' \
        '"GlossSeeAlso": ["GML", "XML"]},"GlossSee": "markup"}}}}}'
    schema = client.describe_collection(collection_name)
    fields = schema.get('fields', [])
    auto_id = schema.get('auto_id', False)
    
    for field in fields:
        field_name = field.get('name')
        field_type = field.get('type')
        is_primary = field.get('is_primary', False)
        
        if field_type == DataType.FLOAT_VECTOR:
            dim = field.get('params', {}).get('dim')
            data.append([[random.random() for _ in range(dim)] for _ in range(nb)])
            continue
            
        if field_type in [DataType.INT64]:
            if is_primary:
                if not auto_id:
                    pop = list(range(start, end))
                    ids = random.sample(pop, nb)
                    data.append(ids)
                    continue
                else:
                    continue
            else:
                data.append([_ for _ in range(nb)])
                continue
                
        if field_type == DataType.INT8:
            data.append([random.randint(-128, 127) for _ in range(nb)])
            continue
            
        if field_type == DataType.INT16:
            data.append([random.randint(-32768, 32767) for _ in range(nb)])
            continue
            
        if field_type == DataType.INT32:
            if field_name == "version":
                data.append([new_version for _ in range(nb)])
            else:
                data.append([random.randint(-2147483648, 2147483647) for _ in range(nb)])
                continue
                
        if field_type == DataType.VARCHAR:
            if not is_primary:
                max_length = field.get('params', {}).get('max_length', 100)
                data.append([gen_str_by_length(max_length // 10) for _ in range(nb)])
                continue
            else:
                if not auto_id:
                    pop = list(range(start, end))
                    ids = random.sample(pop, nb)
                    data.append([pk_prefix + str(j) for j in ids])
                else:
                    continue
                    
        if field_type == DataType.JSON:
            data.append([json.loads(s) for _ in range(nb)])
            continue
            
        if field_type in [DataType.FLOAT, DataType.DOUBLE]:
            data.append([random.random() for _ in range(nb)])
            continue
            
        if field_type == DataType.BOOL:
            data.append([True for _ in range(nb)])       # update to true in upsert
            continue
    return data


def insert_entities(clients, collection_name, nb, rounds, use_insert=True, interval=0, new_version="0"):
    """
    Insert entities into collection
    :param clients: MilvusClient objects
    :param collection_name: str, collection name
    :param nb: int, number of entities per round
    :param rounds: int, number of rounds
    :param use_insert: bool, use insert or upsert
    :param interval: int, sleep interval between rounds
    :param new_version: str, version info
    """
    schema = clients[0].describe_collection(collection_name)
    auto_id = schema.get('auto_id', False)
    
    for r in range(int(rounds)):
        data = gen_row_data_by_schema(schema=schema, nb=nb, start=r * nb, new_version=new_version)
        t1 = time.time()
        if not use_insert and auto_id is False:
            clients[0].upsert(collection_name=collection_name, data=data)
            if clients[1] is not None:
                clients[1].upsert(collection_name=collection_name, data=data)
        else:
            clients[0].insert(collection_name=collection_name, data=data)
            if clients[1] is not None:
                clients[1].insert(collection_name=collection_name, data=data)
        t2 = round(time.time() - t1, 3)
        logging.info(f"{collection_name} insert {r} costs {t2}")
        time.sleep(interval)


def upsert_entities(client, collection_name, nb, rounds, maxid, new_version="0", unique_in_requests=False, interval=0):
    """
    Upsert entities in collection
    """
    start = 0
    sample_n = maxid // rounds
    for r in range(rounds):
        if unique_in_requests:
            end = start+sample_n
        else:
            start = 0
            end = maxid
        data = gen_upsert_data_by_pk_collection(client=client, collection_name=collection_name, 
                                              nb=nb, start=start, end=end, new_version=new_version)
        t1 = time.time()
        client.upsert(collection_name=collection_name, data=data)
        t2 = round(time.time() - t1, 3)
        logging.info(f"{collection_name} upsert2 {r} costs {t2}")
        time.sleep(interval)
        start += sample_n


def get_search_params(index_type, metric_type, topk):
    """
    Get search params based on index type
    :param index_type: str, index type
    :param metric_type: str, metric type
    :param topk: int, topk value
    :return: dict, search params
    """
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
    """
    Get default index params by index type
    """
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


def get_index_params(client, collection_name, field_name):
    """
    Get index params for a specific field
    :param client: MilvusClient object
    :param collection_name: str, collection name
    :param field_name: str, field name
    :return: dict, index params
    """
    indexes = client.list_indexes(collection_name=collection_name)
    for index in indexes:
        if index.get('field_name') == field_name:
            return index
    return None


def is_vector_field(client, collection_name, field_name):
    """
    Check if field is vector field
    :param client: MilvusClient object
    :param collection_name: str, collection name
    :param field_name: str, field name
    :return: bool
    """
    schema = client.describe_collection(collection_name)
    fields = schema.get('fields', [])
    for field in fields:
        if field.get('name') == field_name and field.get('type') in ALL_VECTOR_TYPES:
            return True
    return False


def create_collection_schema(dims, vector_types, auto_id=True, use_str_pk=False):
    """
    Create collection schema for MilvusClient using FieldSchema and CollectionSchema
    :param dims: list of dimensions for vector fields
    :param vector_types: list of vector types
    :param auto_id: bool, whether to use auto id
    :param use_str_pk: bool, whether to use string primary key
    :return: CollectionSchema object
    """
    fields = []
    
    # Primary key field
    if use_str_pk:
        id_field = FieldSchema(
            name="id",
            dtype=DataType.VARCHAR,
            is_primary=True,
            max_length=65535
        )
    else:
        id_field = FieldSchema(
            name="id", 
            dtype=DataType.INT64,
            is_primary=True
        )
    fields.append(id_field)
    
    # Scalar fields
    fields.extend([
        FieldSchema(
            name="category",
            dtype=DataType.INT64,
            description="category for partition key or clustering key"
        ),
        FieldSchema(
            name="groupid",
            dtype=DataType.INT64,
            description="groupid",
            nullable=True
        ),
        FieldSchema(
            name="device",
            dtype=DataType.VARCHAR,
            max_length=500,
            description="device",
            nullable=True
        ),
        FieldSchema(
            name="fname",
            dtype=DataType.VARCHAR,
            max_length=256,
            description="fname",
            nullable=True
        ),
        FieldSchema(
            name="flag",
            dtype=DataType.BOOL,
            description="flag",
            nullable=True
        ),
        FieldSchema(
            name="version",
            dtype=DataType.VARCHAR,
            max_length=200,
            description="data version",
            nullable=True
        )
    ])
    
    # Vector fields
    for i in range(len(dims)):
        embedding_field = FieldSchema(
            name=f"embedding_{i}",
            dtype=vector_types[i],
            dim=int(dims[i])
        )
        fields.append(embedding_field)
    
    # Create CollectionSchema object
    schema = CollectionSchema(
        fields=fields,
        description="Collection created by MilvusClient",
        auto_id=auto_id
    )
    
    return schema


def create_n_insert(collection_name, schema, nb, insert_times, index_types, dims, 
                    metric_types=["L2"], ttl=0, build_index=True, shards_num=1, 
                    is_flush=True, use_insert=True, pre_load=False, new_version="0", 
                    build_scalar_index=False, clients=[]):
    """
    Create collection and insert data using MilvusClient
    :param collection_name: str, collection name
    :param schema: CollectionSchema, collection schema definition (required)
    :param nb: int, number of entities per batch
    :param insert_times: int, number of insert rounds
    :param index_types: list, index types for vector fields
    :param dims: list, dimensions for vector fields (for index building)
    :param metric_types: list, metric types
    :param ttl: int, collection TTL in seconds
    :param build_index: bool, whether to build index
    :param shards_num: int, number of shards
    :param is_flush: bool, whether to flush after insert
    :param use_insert: bool, use insert or upsert
    :param pre_load: bool, whether to load before insert
    :param new_version: str, version info
    :param build_scalar_index: bool, whether to build scalar index
    :param clients: list, MilvusClient objects
    """
    
    # Check if collection exists
    for client in clients:
        if client is None:
            continue
        if not client.has_collection(collection_name=collection_name):
            # Collection properties
            properties = {}
            if ttl > 0:
                properties["collection.ttl.seconds"] = ttl
            if shards_num > 1:
                properties["shards_num"] = shards_num
                
            # Create collection
            client.create_collection(
                collection_name=collection_name,
                schema=schema,
                **properties
            )
            logging.info(f"Created collection: {collection_name}")
        else:
            logging.info(f"{collection_name} already exists")

        logging.info(f"{collection_name} collection schema: {client.describe_collection(collection_name)}")

        # Build index
        if build_index:
            vec_field_names = get_float_vec_field_names(client, collection_name)
            logging.info(f"build index for {vec_field_names}")
            
            for i in range(len(dims)):
                index_type = str(index_types[i]).upper()
                metric_type = str(metric_types[i]).upper()
                index_params = get_default_params_by_index_type(index_type.upper(), metric_type)
                vec_field_name = vec_field_names[i]
                
                # Check if index exists
                index_exists = client.list_indexes(collection_name=collection_name, field_name=vec_field_name)
                
                if not index_exists:
                    t0 = time.time()
                    # Prepare index params for MilvusClient
                    index_params_for_client = client.prepare_index_params()
                    index_params_for_client.add_index(
                        field_name=vec_field_name,
                        index_type=index_type,
                        metric_type=metric_type,
                        params=index_params.get("params", {})
                    )
                    client.create_index(
                        collection_name=collection_name,
                        index_params=index_params_for_client
                    )
                    tt = round(time.time() - t0, 3)
                    logging.info(f"build {vec_field_name} index {index_params} costs {tt}")
                else:
                    logging.info(f"{vec_field_name} index already exists")
                    
            # Build index for scalar fields
            if build_scalar_index:
                schema_desc = client.describe_collection(collection_name)
                fields = schema_desc.get('fields', [])
                for field in fields:
                    field_name = field.get('name')
                    field_type = field.get('type')
                    if field_name not in vec_field_names and field_type != DataType.JSON:
                        try:
                            scalar_index_params = client.prepare_index_params()
                            scalar_index_params.add_index(field_name=field_name)
                            client.create_index(
                                collection_name=collection_name,
                                index_params=scalar_index_params
                            )
                            logging.info(f"build index for scalar field: {field_name}")
                        except Exception as e:
                            logging.warning(f"Failed to build index for {field_name}: {e}")
        else:
            logging.info("skip build index")

        # Pre load before insert/upsert
        if pre_load:
            client.load_collection(collection_name=collection_name)
            logging.info(f"Pre-loaded collection: {collection_name}")

    # Insert data
    insert_entities(clients=clients, collection_name=collection_name, nb=nb, rounds=insert_times,
                    use_insert=use_insert, new_version=new_version)
    
    if is_flush:
        for client in clients:
            if client is None:
                continue
            client.flush(collection_name=collection_name)
            logging.info(f"Flushed collection: {collection_name} by {client}")
    
    # Get entity count (optional, can be enabled if needed)
    # stats = client.get_collection_stats(collection_name=collection_name)
    # entity_count = stats.get('row_count', 0)
    # logging.info(f"collection entities: {entity_count}")
