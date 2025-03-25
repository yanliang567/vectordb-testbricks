import sys
import logging
import random
import argparse
from common import gen_vectors, gen_str_by_length
import os
import time
from pymilvus import MilvusClient, DataType


def setup_logging():
    # Create log directory if it doesn't exist
    log_dir = "/tmp"
    log_file = os.path.join(log_dir, "rabitq_mmap.log")
    
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
    parser = argparse.ArgumentParser(description='Create Milvus collection with mmap support')
    parser.add_argument('--uri', type=str, default="http://localhost:19530",
                      help='Milvus server URI')
    parser.add_argument('--token', type=str, default="root:Milvus",
                      help='Milvus authentication token')
    parser.add_argument('--create-scalar-index', type=str, default="false",
                      help='Whether to create scalar indexes')
    parser.add_argument('--start-id', type=int, default=0,
                      help='Start ID for primary key')
    parser.add_argument('--collection-name', type=str, default="test_mmap",
                      help='Name of the collection to create')
    parser.add_argument('--dim', type=int, default=32,
                      help='Dimension of the vector field')
    parser.add_argument('--batch-size', type=int, default=300,
                      help='Number of entities to insert in each batch')
    parser.add_argument('--num-batches', type=int, default=5,
                      help='Number of batches to insert if --max-deny-times is 0')
    parser.add_argument('--pre-load', type=str, default="false",
                      help='Whether to load the collection before insert')
    parser.add_argument('--max-deny-times', type=int, default=1,
                      help='Maximum number of times to retry insert due to denied errors')
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
    create_scalar_index = True if str(args.create_scalar_index).upper() == "TRUE" else False
    pre_load = True if str(args.pre_load).upper() == "TRUE" else False
    max_deny_times = 1 if args.max_deny_times < 1 else args.max_deny_times

    logging.info("Creating collection '%s' with dimension %d", collection_name, dim)

    def create_index(client, collection_name, field_names, index_type="AUTOINDEX"):
        """Create index for specified fields
        Args:
            client: Milvus client instance
            collection_name: Name of the collection
            field_names: List of field names to create index on
            index_type: Index type, defaults to AUTOINDEX
        """
        logging.info(f"Creating index for fields: {field_names}")
        index_params = client.prepare_index_params()
        for field_name in field_names:
            index_params.add_index(field_name=field_name, index_type=index_type)
        client.create_index(collection_name=collection_name, index_params=index_params)
        logging.info(f"Index created successfully for fields: {field_names}")

    
    scalar_fields = [
            "id", "bool_1", "int32_1",
            "int64_1", "float_1", "double_1",
            "string_1", "string_2",
            "int64_array", "string_array"
        ]

    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
    # schema.add_field(field_name="padding_string", datatype=DataType.VARCHAR, max_length=1500, mmap_enabled=False)
    # Other Mmap columns:
    #     Numeric: (bool) (int32), (int64) (float) (double) two columns each
    #     String: (string1 short column, len 10), (string2 long column, len 100) one column each
    #     Array:  int64_array(len = 10),  string_array(len = 20, string_len = 10) 
    #     Json: (json, 20keys) 
    schema.add_field(field_name="bool_1", datatype=DataType.BOOL)
    # schema.add_field(field_name="bool_2", datatype=DataType.BOOL)
    schema.add_field(field_name="int32_1", datatype=DataType.INT32)
    # schema.add_field(field_name="int32_2", datatype=DataType.INT32)
    schema.add_field(field_name="int64_1", datatype=DataType.INT64)
    # schema.add_field(field_name="int64_2", datatype=DataType.INT64)
    schema.add_field(field_name="float_1", datatype=DataType.FLOAT)
    # schema.add_field(field_name="float_2", datatype=DataType.FLOAT)
    schema.add_field(field_name="double_1", datatype=DataType.DOUBLE)
    # schema.add_field(field_name="double_2", datatype=DataType.DOUBLE)
    schema.add_field(field_name="string_1", datatype=DataType.VARCHAR, max_length=10)
    schema.add_field(field_name="string_2", datatype=DataType.VARCHAR, max_length=100)
    schema.add_field(field_name="int64_array", datatype=DataType.ARRAY, max_capacity=10, element_type=DataType.INT64)
    schema.add_field(field_name="string_array", datatype=DataType.ARRAY, max_capacity=20,
                     element_type=DataType.VARCHAR, max_length=10)
    schema.add_field(field_name="json", datatype=DataType.JSON)

    client.create_collection(collection_name=collection_name, schema=schema)
    if pre_load is True:
        logging.info("Pre-loading collection")
         # Create vector index
        create_index(client, collection_name, ["vector"])
        # Create scalar indexes if specified
        if create_scalar_index:
            create_index(client, collection_name, scalar_fields)
        client.load_collection(collection_name=collection_name)
        logging.info("Collection loaded successfully")

    
    def gen_rows(batch_size, dim, start_id):
        vectors = gen_vectors(batch_size, dim)
        rows = []
        for i in range(batch_size):
            id = start_id + i
            row = {
                "id": id,
                "vector": list(vectors[i]),
                # "padding_string": gen_str_by_length(length=500),
                "bool_1": True,
                # "bool_2": False,
                "int32_1": 1,
                # "int32_2": 2,
                "int64_1": id,
                # "int64_2": id + id,
                "float_1": id * 1.0,
                # "float_2": id * 2.0,
                "double_1": id * 1.0,
                # "double_2": id * 2.0,
                "string_1": gen_str_by_length(length=10),
                "string_2": gen_str_by_length(length=100),
                "int64_array": [id, id+1, id+2, id+3, id+4, id+5, id+6, id+7, id+8, id+9],
                "string_array": [gen_str_by_length(length=10) for _ in range(20)],
                "json": {
                    "key1": id,
                    "key2": True if id % 2 == 0 else False,
                    "key3": id * 1.0,
                    "key4": [id, id+1, id+2, id+3, id+4, id+5, id+6, id+7, id+8, id+9],
                    "key5": "value5",
                    "key6": "value6",
                    "key7": "value7",
                    "key8": "value8",
                    "key9": "value9",
                    "key10": "value10",
                    "key11": "value11",
                    "key12": "value12",
                    "key13": "value13",
                    "key14": "value14",
                    "key15": "value15",
                    "key16": "value16",
                    "key17": "value17",
                    "key18": "value18",
                    "key19": "value19",
                    "key20": "value20"
                }
            }
            rows.append(row)
        return rows

    # insert data
    logging.info("Insert data with batch size %d and %d batches", args.batch_size, args.num_batches)
    deny_times = 0
    batch = 0
    msg = "memory quota exceeded"
    msg_cloud = "cu quota exhausted"
    
    while batch < args.num_batches and deny_times < max_deny_times:
        start_id = args.start_id + batch * args.batch_size
        rows = gen_rows(args.batch_size, dim, start_id)
        try:
            client.insert(collection_name=collection_name, data=rows)
            logging.info("Inserted %d entities for batch %d", args.batch_size, batch)
            batch += 1
        except Exception as e:
            if msg in str(e) or msg_cloud in str(e):
                logging.error(f"insert expected error: {e}")
                deny_times += 1
                if deny_times >= max_deny_times:
                    logging.error(f"Reached max deny times {max_deny_times}, stopping insertion")
                    break
                logging.error(f"wait for 15 minutes and retry, deny times: {deny_times}")
                time.sleep(900)
            else:
                logging.error(f"insert error: {e}")
                break

    if pre_load is False:
        # Create vector index
        create_index(client, collection_name, ["vector"])
        # Create scalar indexes if specified
        if create_scalar_index:
            create_index(client, collection_name, scalar_fields)

        # load the collection
        client.load_collection(collection_name=collection_name)
        logging.info("Collection loaded successfully")

    logging.info("Collection creation completed successfully")

if __name__ == "__main__":
    main()

