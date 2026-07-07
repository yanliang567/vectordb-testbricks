import time
from pymilvus import MilvusClient
import random
import os
import pymilvus

random.seed(468468)

client = MilvusClient(
    uri="http://10.104.30.203:19530"
)
collection_name = 'demo_collection'
schema = MilvusClient.create_schema(
    description="Test collection",
    enable_dynamic_field=False,
)

schema.add_field(
    field_name="id",
    datatype=pymilvus.DataType.INT64,
    is_primary=True,
    auto_id=False,
)
schema.add_field(
    field_name="vector",
    datatype=pymilvus.DataType.FLOAT_VECTOR,
    dim=768,
)
schema.add_field(
    field_name="docId",
    datatype=pymilvus.DataType.INT64,
)

index_params = client.prepare_index_params()

index_params.add_index(
    field_name="vector",
    index_type="FLAT", # Same results for other indexes (IVF_FLAT, HNSW_SQ)
    metric_type="L2",
)


if client.has_collection(collection_name=collection_name):
    client.drop_collection(collection_name=collection_name)
client.create_collection(
    collection_name=collection_name,
    schema=schema,
)
# client.create_index("demo_collection", index_params=index_params)

# Use fake representation with random vectors (768 dimension).
vectors = [[random.uniform(-1, 1) for _ in range(768)] for _ in range(1000)]
data = [{"id": i, "vector": vectors[i], "docId": i % 10} for i in range(len(vectors))]

print("Data has", len(data), "entities, each with fields: ", data[0].keys())
print("Vector dim:", len(data[0]["vector"]))

res = client.insert(collection_name=collection_name, data=data)
client.flush(collection_name=collection_name)
client.create_index(collection_name, index_params=index_params)
print(res)

client.load_collection(collection_name)

time.sleep(1)

query_vectors = [[random.uniform(-1, 1) for _ in range(768)]]

res = client.search(
    collection_name=collection_name,  # target collection
    data=query_vectors,  # query vectors
    limit=5,  # number of returned entities
    output_fields=["docId"],  # specifies fields to be returned
    search_params={"metric_type": "L2"},
)

print("RESULT OF NORMAL SEARCH", res)

res = client.search(
    collection_name=collection_name,  # target collection
    data=query_vectors,  # query vectors
    limit=5,  # number of returned entities
    group_by_field="docId",
    output_fields=["docId"],  # specifies fields to be returned
    search_params={"metric_type": "L2"},
)

print("RESULT OF GROUP SEARCH", res)
print('completed')