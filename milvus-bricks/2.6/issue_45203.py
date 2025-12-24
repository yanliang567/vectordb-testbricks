from pymilvus import Collection, CollectionSchema, FieldSchema, Function, FunctionType, DataType, \
    AnnSearchRequest, RRFRanker, WeightedRanker, connections, utility
from common import gen_row_data_by_schema, gen_vectors, gen_varchar_data, gen_str_by_length

# Create a collection

uri = "http://localhost:19530"
token = "root:Milvus"

connections.connect(uri=uri, token=token)

collection_name = "issue_45203"
dim = 128
analyzer_params = {
    "type": "standard",  # Specifies the standard analyzer type
}

# if utility.has_collection(collection_name):
#     utility.drop_collection(collection_name)

fields = [
    FieldSchema(name='id', dtype=DataType.VARCHAR, description='Business unique identifier', max_length=128,
                is_primary=True, auto_id=True),
    FieldSchema(name='class', dtype=DataType.VARCHAR, description='class name', max_length=512),
    FieldSchema(name='tag', dtype=DataType.VARCHAR, description='tag name', max_length=512),
    FieldSchema(name='text', dtype=DataType.VARCHAR, description='Raw information', analyzer_params=analyzer_params,
                enable_analyzer=True, enable_match=True, max_length=65535),
    FieldSchema(name='text2', dtype=DataType.VARCHAR, description='Raw information', analyzer_params=analyzer_params,
                enable_analyzer=True, enable_match=True, max_length=65535),
    FieldSchema(name='sparse_vector', dtype=DataType.SPARSE_FLOAT_VECTOR, description='Sparse vector'),
    FieldSchema(name='sparse_vector2', dtype=DataType.SPARSE_FLOAT_VECTOR, description='Sparse vector'),
    FieldSchema(name='dense_vector1', dtype=DataType.FLOAT_VECTOR, description='Dense vector', dim=dim),
    FieldSchema(name='dense_vector2', dtype=DataType.FLOAT_VECTOR, description='Dense vector', dim=dim)
]




schema = CollectionSchema(fields=fields, description='AI Search', enable_dynamic_field=False)

bm25_function = Function(
    name="text_bm25_emb",  # Function name
    input_field_names=["text"],  # Name of the VARCHAR field containing raw text data
    output_field_names=["sparse_vector"],
    function_type=FunctionType.BM25,
)
schema.add_function(bm25_function)
bm25_function2 = Function(
    name="text_bm25_emb2",  # Function name
    input_field_names=["text2"],  # Name of the VARCHAR field containing raw text data
    output_field_names=["sparse_vector2"],
    function_type=FunctionType.BM25,
)
schema.add_function(bm25_function2)

collection = Collection(name=collection_name, schema=schema, consistency_level="Strong")

text = gen_varchar_data(10, 1, True)
float_vector1 = gen_vectors(1, dim, DataType.FLOAT_VECTOR)
float_vector2 = gen_vectors(1, dim, DataType.FLOAT_VECTOR)
nb=2000
data = [{
    'class': gen_str_by_length(10) + "_" + str(i),
    'tag': gen_str_by_length(10) + "_" + str(i),
    'text': text[0],
    'text2': text[0],
    'dense_vector1': gen_vectors(1, dim, DataType.FLOAT_VECTOR)[0],
    'dense_vector2': gen_vectors(1, dim, DataType.FLOAT_VECTOR)[0]} for i in range(nb)]

# collection.insert(data)
collection.flush()

sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "BM25"}
collection.create_index("sparse_vector", sparse_index)
sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "BM25"}
collection.create_index("sparse_vector2", sparse_index)

dense_index = {"index_type": "AUTOINDEX", "metric_type": "COSINE"}
collection.create_index("dense_vector1", dense_index)
collection.create_index("dense_vector2", dense_index)
collection.load()

expr = ""
query_dense_embeddings = gen_vectors(nb=2, dim=dim)
dense_search_params = {"metric_type": "COSINE"}
limit = 100
offset = 0

dense_req1 = AnnSearchRequest(
    expr="",
    data=query_dense_embeddings,
    anns_field="dense_vector1",
    param=dense_search_params,
    limit=limit
)
dense_req2 = AnnSearchRequest(
    expr="",
    data=query_dense_embeddings,
    anns_field="dense_vector2",
    param=dense_search_params,
    limit=limit
)

ranker = RRFRanker()
output_fields = ["id", "tag"]
# print(f"query embedding 1: {query_dense_embeddings[0]}")
# print(f"query embedding 2: {query_dense_embeddings[1]}")
res_nq2 = collection.hybrid_search(
    [dense_req1, dense_req2],
    rerank=ranker,
    offset=offset,
    limit=20,
    output_fields=output_fields
)

dense_req1 = AnnSearchRequest(
    expr="",
    data=[query_dense_embeddings[0]],
    anns_field="dense_vector1",
    param=dense_search_params,
    limit=limit
)
dense_req2 = AnnSearchRequest(
    expr="",
    data=[query_dense_embeddings[0]],
    anns_field="dense_vector2",
    param=dense_search_params,
    limit=limit
)

res_1 = collection.hybrid_search(
    [dense_req1, dense_req2],
    rerank=ranker,
    offset=offset,
    limit=20,
    output_fields=output_fields
)

dense_req1 = AnnSearchRequest(
    expr="",
    data=[query_dense_embeddings[1]],
    anns_field="dense_vector1",
    param=dense_search_params,
    limit=limit
)
dense_req2 = AnnSearchRequest(
    expr="",
    data=[query_dense_embeddings[1]],
    anns_field="dense_vector2",
    param=dense_search_params,
    limit=limit
)

res_2 = collection.hybrid_search(
    [dense_req1, dense_req2],
    rerank=ranker,
    offset=offset,
    limit=20,
    output_fields=output_fields
)

print(f"resnq2-1: {res_nq2[0]}")
print(f"res_1: {res_1}")

print(f"resnq2-2: {res_nq2[1]}")
print(f"res_2: {res_2}")

assert res_nq2[0] == res_1[0], "issue reproduced: the 1st nq results"
assert res_nq2[1] == res_2[0], "issue reproduced: the 2nd nq results"

print("completed")