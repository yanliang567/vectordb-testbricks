import asyncio

import numpy as np

from pymilvus import AsyncMilvusClient

# Replace with your Milvus server
MILVUS_URI = "./milvus.db"
COLLECTION_NAME = "async_example_collection"

rng = np.random.default_rng()

async def insert_parallel(client: AsyncMilvusClient, collection_name: str, num_tasks: int = 10, entities_per_task: int = 100) -> None:
    async def insert_batch(batch_id: int) -> None:
        vectors = rng.random((entities_per_task, 8)).astype("float32")
        data = [{"vector": v.tolist()} for v in vectors]
        result = await client.insert(collection_name, data)
        print(f"Batch {batch_id}: Inserted {len(result['ids'])} entities")

    tasks = [insert_batch(i) for i in range(num_tasks)]
    await asyncio.gather(*tasks)
    print(f"Total parallel inserts completed: {num_tasks * entities_per_task} entities")

async def search_parallel(client: AsyncMilvusClient, collection_name: str, num_tasks: int = 10) -> None:
    async def search_once(task_id: int) -> None:
        query_vec = rng.random((1, 8)).astype("float32").tolist()
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}
        result = await client.search(
            collection_name,
            data=query_vec,
            anns_field="vector",
            search_params=search_params,
            limit=5
        )
        print(f"Task {task_id}: Found {len(result[0])} results")

    tasks = [search_once(i) for i in range(num_tasks)]
    await asyncio.gather(*tasks)
    print(f"Total parallel searches completed: {num_tasks}")

async def main():
    client = AsyncMilvusClient(uri=MILVUS_URI)

    if await client.has_collection(COLLECTION_NAME):
        await client.drop_collection(COLLECTION_NAME)
    await client.create_collection(COLLECTION_NAME, dimension=8, auto_id=True)

    vectors = rng.random((1000, 8)).astype("float32")
    data = [{"vector": v.tolist()} for v in vectors]

    insert_result = await client.insert(COLLECTION_NAME, data)
    print(f"Inserted {len(insert_result['ids'])} entities")

    query_vec = np.random.random((1, 8)).astype("float32").tolist()
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}

    result = await client.search(
        COLLECTION_NAME,
        data=query_vec,
        anns_field="vector",
        search_params=search_params,
        limit=5
    )
    print("Search result IDs:", [hit.id for hit in result[0]])

    print("\n--- Testing parallel inserts ---")
    await insert_parallel(client, COLLECTION_NAME, num_tasks=5, entities_per_task=50)

    print("\n--- Testing parallel searches ---")
    await search_parallel(client, COLLECTION_NAME, num_tasks=10)

    await client.release_collection(COLLECTION_NAME)
    await client.drop_collection(COLLECTION_NAME)
    await client.close()

# Run async main
if __name__ == "__main__":
    asyncio.run(main())
