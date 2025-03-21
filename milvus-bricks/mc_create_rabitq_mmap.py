import sys
import logging
import random
import argparse
from pymilvus import MilvusClient
from common import gen_vectors, gen_str_by_length
import os

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
    parser.add_argument('--create-scalar-index', type=bool, default=False,
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
                      help='Number of batches to insert')
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

    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field(field_name="padding_string", datatype=DataType.VARCHAR, max_length=1500, mmap_enabled=False)
    # Other Mmap columns:
    #     Numeric: (bool) (int32), (int64) (float) (double) two columns each
    #     String: (string1 short column, len 10), (string2 long column, len 100) one column each
    #     Array:  int64_array(len = 10),  string_array(len = 20, string_len = 10) 
    #     Json: (json, 20keys) 
    schema.add_field(field_name="bool_1", datatype=DataType.BOOL)
    schema.add_field(field_name="bool_2", datatype=DataType.BOOL)
    schema.add_field(field_name="int32_1", datatype=DataType.INT32)
    schema.add_field(field_name="int32_2", datatype=DataType.INT32)
    schema.add_field(field_name="int64_1", datatype=DataType.INT64)
    schema.add_field(field_name="int64_2", datatype=DataType.INT64)
    schema.add_field(field_name="float_1", datatype=DataType.FLOAT)
    schema.add_field(field_name="float_2", datatype=DataType.FLOAT)
    schema.add_field(field_name="double_1", datatype=DataType.DOUBLE)
    schema.add_field(field_name="double_2", datatype=DataType.DOUBLE)
    schema.add_field(field_name="string_1", datatype=DataType.VARCHAR, max_length=10)
    schema.add_field(field_name="string_2", datatype=DataType.VARCHAR, max_length=100)
    schema.add_field(field_name="int64_array", datatype=DataType.ARRAY, max_capacity=10, element_type=DataType.INT64)
    schema.add_field(field_name="string_array", datatype=DataType.ARRAY, max_capacity=20,
                     element_type=DataType.VARCHAR, max_length=10)
    schema.add_field(field_name="json", datatype=DataType.JSON)

    client.create_collection(collection_name=collection_name, schema=schema)

    # insert data for n times, each time insert nb entities
    logging.info("Starting data insertion with batch size %d and %d batches", args.batch_size, args.num_batches)
    for batch in range(args.num_batches):
        vectors = gen_vectors(args.batch_size, dim)
        rows = []
        start_id = args.start_id + batch * args.batch_size  # Start from args.start_id to avoid duplicate IDs

        for i in range(args.batch_size):
            id = start_id + i
            row = {
                "id": id,  # Use incremental IDs starting from args.start_id to ensure uniqueness
                "vector": list(vectors[i]),
                "padding_string": gen_str_by_length(length=1000),
                "bool_1": True,
                "bool_2": False,
                "int32_1": 1,
                "int32_2": 2,
                "int64_1": id,
                "int64_2": id + id,
                "float_1": id * 1.0,
                "float_2": id * 2.0,
                "double_1": id * 1.0,
                "double_2": id * 2.0,
                "string_1": gen_str_by_length(length=10),
                "string_2": gen_str_by_length(length=100),
                "int64_array": [id, id+1, id+2, id+3, id+4, id+5, id+6, id+7, id+8, id+9],
                "string_array": [gen_str_by_length(length=10) for _ in range(20)],
                "json": {
                    "key1": id,
                    "key2": True if id % 2 == 0 else False,  # if odd, True, else False
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
        client.insert(collection_name=collection_name, data=rows)
        logging.info("Inserted %d entities for batch %d", args.batch_size, batch)

    # create vector index for the collection
    logging.info("Creating vector index")
    index_params = client.prepare_index_params()  # Prepare IndexParams object
    index_params.add_index(field_name="vector", index_type="AUTOINDEX")
    client.create_index(collection_name=collection_name, index_params=index_params)
    logging.info("Vector index created successfully")

    # create scalar index for the collection if specified
    if args.create_scalar_index:
        logging.info("Creating scalar indexes")
        index_params = client.prepare_index_params()
        index_params.add_index(field_name="id", index_type="AUTOINDEX")
        index_params.add_index(field_name="bool_1", index_type="AUTOINDEX")
        index_params.add_index(field_name="bool_2", index_type="AUTOINDEX")
        index_params.add_index(field_name="int32_1", index_type="AUTOINDEX")
        index_params.add_index(field_name="int32_2", index_type="AUTOINDEX") 
        index_params.add_index(field_name="int64_1", index_type="AUTOINDEX")
        index_params.add_index(field_name="int64_2", index_type="AUTOINDEX")
        index_params.add_index(field_name="float_1", index_type="AUTOINDEX")
        index_params.add_index(field_name="float_2", index_type="AUTOINDEX")
        index_params.add_index(field_name="double_1", index_type="AUTOINDEX")
        index_params.add_index(field_name="double_2", index_type="AUTOINDEX")
        index_params.add_index(field_name="string_1", index_type="AUTOINDEX")
        index_params.add_index(field_name="string_2", index_type="AUTOINDEX")
        index_params.add_index(field_name="int64_array", index_type="AUTOINDEX")
        index_params.add_index(field_name="string_array", index_type="AUTOINDEX")

        client.create_index(collection_name=collection_name, index_params=index_params)
        logging.info("Scalar indexes created successfully")

    # load the collection
    logging.info("Loading collection")
    client.load_collection(collection_name=collection_name)
    logging.info("Collection loaded successfully")
    logging.info("Collection creation completed successfully")

if __name__ == "__main__":
    main()

