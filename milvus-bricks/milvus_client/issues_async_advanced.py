from pymilvus import MilvusClient, AsyncMilvusClient, FieldSchema, CollectionSchema, DataType
import numpy as np
import random
import uuid
import time
import asyncio
from common import gen_row_data_by_schema, gen_vectors, gen_varchar_data

# 1. Connect to Milvus server (Sync client for setup operations)
token = 'root:Milvus'
uri = f'http://10.104.6.127:19530'
client = MilvusClient(uri=uri, token=token)

# 2. Create collection
collection_name = 'codehh44'
dim = 1024

schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
schema.add_field(field_name='id', datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name='vector', datatype=DataType.FLOAT_VECTOR, dim=dim)
schema.add_field(field_name='filename', datatype=DataType.VARCHAR, max_length=300)
# schema.add_field(field_name='source_type', datatype=DataType.VARCHAR, max_length=300)
# schema.add_field(field_name='end_pos', datatype=DataType.INT64)
client.create_collection(collection_name=collection_name, schema=schema)

# Create index and load collection
index_params = client.prepare_index_params()
index_params.add_index(field_name="vector", index_type="HNSW", metric_type="COSINE", M=16, efConstruction=200)
client.create_index(collection_name=collection_name, index_params=index_params)

# client.load_collection(collection_name=collection_name)

# 3. Insert data
insert_times = 50
# batch_size = 20

# insert data in batches
for j in range(insert_times):
    filenames = ['test.txt', 'demo.txt', 'sample.txt', 'poc.txt', 'text.json', 'vecto.json']
    nb = 30
    entities = [
        {
            "id": j * nb + i,  # generate unique id
            "vector": gen_vectors(dim=dim, nb=1)[0],
            "filename": random.choice(filenames),
            "sentence": gen_varchar_data(length=10, nb=1, text_mode=True),
        }
        for i in range(nb)
    ]
    client.insert(collection_name=collection_name, data=entities)
    
# client.flush(collection_name=collection_name)
print(f"insert/upsert completed")

client.load_collection(collection_name=collection_name)
print(f"load completed, start to search")


# Define async search function
async def async_search_test():
    """Test search using AsyncMilvusClient"""
    # Create async client inside async function to avoid event loop issues
    async_client = AsyncMilvusClient(uri=uri, token=token)
    
    try:
        # 5. Search vectors
        search_params = {"ef": 200}
        filter_expr = "filename == 'test.txt'"
        
        search_vector = gen_vectors(dim=dim, nb=1)
        
        # First search with filter: filename == 'test.txt'
        print("Performing first async search with filter: filename == 'test.txt'")
        result = await async_client.search(
            collection_name=collection_name,
            data=search_vector,
            anns_field="vector", 
            search_params=search_params,
            limit=5,
            output_fields=["id", "filename"],
            filter=filter_expr,
            consistency_level="Strong"
        )
        
        # Second search with filter: filename in ['test.txt']
        filter_expr2 = "filename in ['test.txt']"
        print("Performing second async search with filter: filename in ['test.txt']")
        result2 = await async_client.search(
            collection_name=collection_name,
            data=search_vector,
            anns_field="vector",
            search_params=search_params,
            limit=5,
            output_fields=["id", "filename"],
            filter=filter_expr2,
            consistency_level="Strong"
        )
        
        print("\nSearch Result 1 (filename == 'test.txt'):")
        print(result)
        print("\nSearch Result 2 (filename in ['test.txt']):")
        print(result2)
    finally:
        # Always close async client
        await async_client.close()


async def async_concurrent_search_test():
    """Test concurrent searches using AsyncMilvusClient"""
    # Create async client inside async function
    async_client = AsyncMilvusClient(uri=uri, token=token)
    
    try:
        print("\n" + "="*60)
        print("Starting Concurrent Async Search Tests")
        print("="*60)
        
        search_params = {"ef": 200}
        filters = [
            "filename == 'test.txt'",
            "filename in ['test.txt']",
            "filename == 'demo.txt'",
            "filename in ['demo.txt', 'sample.txt']"
        ]
        
        # Generate search vectors for each filter
        search_vectors = [gen_vectors(dim=dim, nb=1) for _ in filters]
        
        # Create search tasks for concurrent execution
        search_tasks = []
        for i, (search_vector, filter_expr) in enumerate(zip(search_vectors, filters)):
            task = async_client.search(
                collection_name=collection_name,
                data=search_vector,
                anns_field="vector",
                search_params=search_params,
                limit=5,
                output_fields=["id", "filename"],
                filter=filter_expr,
                consistency_level="Strong"
            )
            search_tasks.append(task)
        
        # Execute all searches concurrently
        start_time = time.time()
        results = await asyncio.gather(*search_tasks)
        elapsed_time = time.time() - start_time
        
        # Display results
        for i, (result, filter_expr) in enumerate(zip(results, filters)):
            print(f"\nSearch {i+1} with filter: {filter_expr}")
            print(f"  Result count: {len(result[0]) if result and result[0] else 0}")
            if result and result[0]:
                print(f"  Top result: {result[0][0]}")
        
        print(f"\nTotal elapsed time for {len(filters)} concurrent searches: {elapsed_time:.4f}s")
        print("="*60)
    finally:
        await async_client.close()


async def async_stress_test(num_searches=100):
    """Stress test with multiple concurrent searches"""
    # Create async client inside async function
    async_client = AsyncMilvusClient(uri=uri, token=token)
    
    try:
        print("\n" + "="*60)
        print(f"Starting Async Stress Test ({num_searches} searches)")
        print("="*60)
        
        search_params = {"ef": 200}
        filenames = ['test.txt', 'demo.txt', 'sample.txt', 'poc.txt', 'text.json', 'vecto.json']
        
        # Create search tasks
        search_tasks = []
        for i in range(num_searches):
            search_vector = gen_vectors(dim=dim, nb=1)
            filter_expr = f"filename == '{random.choice(filenames)}'"
            
            task = async_client.search(
                collection_name=collection_name,
                data=search_vector,
                anns_field="vector",
                search_params=search_params,
                limit=5,
                output_fields=["id", "filename"],
                filter=filter_expr,
                consistency_level="Strong"
            )
            search_tasks.append(task)
        
        # Execute all searches concurrently and measure time
        start_time = time.time()
        results = await asyncio.gather(*search_tasks, return_exceptions=True)
        elapsed_time = time.time() - start_time
        
        # Count successful and failed searches
        successful = sum(1 for r in results if not isinstance(r, Exception))
        failed = sum(1 for r in results if isinstance(r, Exception))
        
        print(f"\nStress Test Results:")
        print(f"  Total searches: {num_searches}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Total time: {elapsed_time:.4f}s")
        print(f"  QPS: {num_searches / elapsed_time:.2f}")
        print(f"  Avg latency: {elapsed_time / num_searches * 1000:.2f}ms")
        print("="*60)
    finally:
        await async_client.close()


# Run async search tests
print("\n" + "="*60)
print("Test 1: Basic Async Search")
print("="*60)
asyncio.run(async_search_test())

# Run concurrent search test
print("\n" + "="*60)
print("Test 2: Concurrent Async Searches")
print("="*60)
asyncio.run(async_concurrent_search_test())

# Run stress test
print("\n" + "="*60)
print("Test 3: Async Stress Test")
print("="*60)
asyncio.run(async_stress_test(num_searches=50))

print("\nAll tests completed!")
print("done")

