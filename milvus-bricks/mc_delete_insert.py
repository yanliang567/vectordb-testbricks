import sys
import logging
import argparse
from common import gen_vectors
import os
from pymilvus import MilvusClient, DataType
import time

def setup_logging():
    # Create log directory if it doesn't exist
    log_dir = "/tmp"
    log_file = os.path.join(log_dir, "delete_insert.log")
    
    # Configure logging format
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Logging initialized. Log file: %s", log_file)

def parse_args():
    parser = argparse.ArgumentParser(description='Delete and insert data in Milvus collection')
    parser.add_argument('--uri', type=str, default="http://localhost:19530",
                      help='Milvus server URI')
    parser.add_argument('--token', type=str, default="root:Milvus",
                      help='Milvus authentication token')
    parser.add_argument('--collection-name', type=str, default="test_delete_insert",
                      help='Name of the collection to create')
    parser.add_argument('--dim', type=int, default=32,
                      help='Dimension of the vector field')
    parser.add_argument('--batch-size', type=int, default=300,
                      help='Number of entities to insert in each batch')
    parser.add_argument('--num-batches', type=int, default=5,
                      help='Number of batches to insert')
    parser.add_argument('--sleep-time', type=int, default=10,
                      help='Number of seconds to sleep between operations')
    return parser.parse_args()

def main():
    args = parse_args()
    setup_logging()
    
    logging.info("Starting Milvus collection creation with parameters: %s", args)
    
    client = MilvusClient(
        uri=args.uri,
        token=args.token
    )
    logging.info("Connected to Milvus server at %s", args.uri)

    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=False,
    )

    collection_name = args.collection_name
    dim = args.dim

    logging.info("Creating collection '%s' with dimension %d", collection_name, dim)

    def create_index(client, collection_name, field_names, index_type="AUTOINDEX"):
        """Create index for specified fields"""
        logging.info(f"Creating index for fields: {field_names}")
        index_params = client.prepare_index_params()
        for field_name in field_names:
            index_params.add_index(field_name=field_name, index_type=index_type)
        client.create_index(collection_name=collection_name, index_params=index_params)
        logging.info(f"Index created successfully for fields: {field_names}")

    # 只添加 id 和向量列
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dim)

    # 创建集合
    client.create_collection(collection_name=collection_name, schema=schema)
    
    def gen_rows(batch_size, dim, start_id):
        vectors = gen_vectors(batch_size, dim)
        rows = []
        for i in range(batch_size):
            id = start_id + i
            row = {
                "id": id,
                "vector": list(vectors[i])
            }
            rows.append(row)
        return rows

    # 插入数据
    logging.info("Insert data with batch size %d and %d batches", args.batch_size, args.num_batches)
    for batch in range(args.num_batches):
        start_id = batch * args.batch_size
        rows = gen_rows(args.batch_size, dim, start_id)
        client.insert(collection_name=collection_name, data=rows)
        logging.info("Inserted %d entities for batch %d", args.batch_size, batch)

    # 创建向量索引
    # client.flush(collection_name=collection_name)
    create_index(client, collection_name, ["vector"])
    
    # 加载集合
    client.load_collection(collection_name=collection_name)
    logging.info("Collection loaded successfully")

    # query count(*)
    res = client.query(collection_name=collection_name, filter="", output_fields=["count(*)"])
    logging.info("Collection '%s' has %d entities", collection_name, res[0]["count(*)"])

    # sleep 5 mins
    logging.info("Sleeping %d seconds...", args.sleep_time)
    time.sleep(args.sleep_time) 

    # 分批删除和插入数据
    logging.info("Deleting and re-inserting data batch by batch")
    max_id = args.batch_size * args.num_batches - 1
    for batch in range(args.num_batches):
        start_id = batch * args.batch_size
        end_id = start_id + args.batch_size - 1
        
        # 删除当前batch
        filter = f"id in {list(range(start_id, end_id + 1))}"
        client.delete(collection_name=collection_name, filter=filter)
        logging.info("Deleted batch %d (ids %d to %d)", batch, start_id, end_id)
        
        # 插入新的batch,从当前最大id开始
        max_id += 1
        rows = gen_rows(args.batch_size, dim, max_id)
        client.insert(collection_name=collection_name, data=rows)
        logging.info("Inserted new batch %d starting from id %d", batch, max_id)
        max_id += args.batch_size - 1

    # query count(*)
    res2 = client.query(collection_name=collection_name, filter="", output_fields=["count(*)"])
    logging.info("Collection '%s' has %d entities", collection_name, res2[0]["count(*)"])
    if res2[0]["count(*)"] != res[0]["count(*)"]:
        logging.error("Collection count mismatch, expected %d, got %d after delete and insert", res[0]["count(*)"], res2[0]["count(*)"])
        client.flush(collection_name=collection_name)
        res3 = client.query(collection_name=collection_name, filter="", output_fields=["count(*)"])
        logging.info("Collection '%s' has %d entities after flush", collection_name, res3[0]["count(*)"])
    else:
        logging.info("Collection count equal after delete and insert")

if __name__ == "__main__":
    main() 