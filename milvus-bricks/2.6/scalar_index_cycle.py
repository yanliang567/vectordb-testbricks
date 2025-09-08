#!/usr/bin/env python3
"""
Scalar Index Cycling Test - Repeatedly drop and recreate scalar field indexes

This script performs the following operations in a loop until timeout:
1. Get all scalar indexes from the collection
2. Drop all scalar indexes (keep vector indexes)
3. Wait for specified seconds
4. Recreate the previously dropped scalar indexes
5. Repeat until timeout is reached

Usage:
    python3 scalar_index_cycle.py <host> <collection_name> <wait_seconds> <timeout> [api_key]

Examples:
    # Local Milvus, cycle every 30 seconds for 5 minutes
    python3 scalar_index_cycle.py localhost my_collection 30 300 None
    
    # Cloud Milvus with API key, cycle every 10 seconds for 2 minutes
    python3 scalar_index_cycle.py your-cloud-host.com my_collection 10 120 your_api_key
"""

import time
import sys
import logging
from pymilvus import MilvusClient, DataType

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"

# Scalar field types that can have indexes
SCALAR_TYPES = {
    DataType.BOOL,
    DataType.INT8, DataType.INT16, DataType.INT32, DataType.INT64,
    DataType.FLOAT, DataType.DOUBLE,
    DataType.VARCHAR, DataType.STRING
}

# Vector field types (should not be touched)
VECTOR_TYPES = {
    DataType.BINARY_VECTOR,
    DataType.FLOAT_VECTOR,
    DataType.FLOAT16_VECTOR,
    DataType.BFLOAT16_VECTOR,
    DataType.SPARSE_FLOAT_VECTOR
}

class ScalarIndexManager:
    """Manager for scalar index operations"""
    
    def __init__(self, client, collection_name):
        self.client = client
        self.collection_name = collection_name
        self.scalar_indexes = []  # Store dropped scalar indexes for recreation
        
    def get_collection_schema(self):
        """Get collection schema information"""
        try:
            schema = self.client.describe_collection(self.collection_name)
            return schema
        except Exception as e:
            logging.error(f"Failed to get collection schema: {e}")
            return None
    
    def get_all_indexes(self):
        """Get all indexes from the collection"""
        try:
            indexes = self.client.list_indexes(self.collection_name)
            logging.info(f"Found {len(indexes)} indexes in collection '{self.collection_name}': {indexes}")
            return indexes
        except Exception as e:
            logging.error(f"Failed to get indexes: {e}")
            return []
    
    def get_field_type(self, field_name, schema):
        """Get the data type of a specific field"""
        fields = schema.get('fields', [])
        for field in fields:
            if field.get('name') == field_name:
                return field.get('type')
        return None
    
    def identify_scalar_indexes(self, indexes, schema):
        """Identify which indexes are for scalar fields"""
        scalar_indexes = []
        vector_indexes = []
        
        for index_name in indexes:
            try:
                # Get index information
                index_info = self.client.describe_index(self.collection_name, index_name)
                field_name = index_info.get('field_name', '')
                
                if not field_name:
                    logging.warning(f"Could not determine field name for index '{index_name}', skipping")
                    continue
                
                field_type = self.get_field_type(field_name, schema)
                
                if field_type in SCALAR_TYPES:
                    scalar_indexes.append({
                        'index_name': index_name,
                        'field_name': field_name,
                        'field_type': field_type,
                        'index_info': index_info
                    })
                    logging.info(f"Identified scalar index: {index_name} on field '{field_name}' (type: {field_type})")
                elif field_type in VECTOR_TYPES:
                    vector_indexes.append({
                        'index_name': index_name,
                        'field_name': field_name,
                        'field_type': field_type
                    })
                    logging.info(f"Identified vector index: {index_name} on field '{field_name}' (type: {field_type}) - will keep")
                else:
                    logging.warning(f"Unknown field type for index '{index_name}' on field '{field_name}': {field_type}")
                    
            except Exception as e:
                logging.error(f"Failed to get info for index '{index_name}': {e}")
                continue
        
        return scalar_indexes, vector_indexes
    
    def drop_scalar_indexes(self, scalar_indexes):
        """Drop all scalar indexes and store their info for recreation"""
        dropped_count = 0
        self.scalar_indexes = []  # Reset stored indexes
        
        for index_info in scalar_indexes:
            index_name = index_info['index_name']
            field_name = index_info['field_name']
            
            try:
                logging.info(f"Dropping scalar index '{index_name}' on field '{field_name}'...")
                self.client.drop_index(self.collection_name, index_name)
                
                # Store index info for recreation
                self.scalar_indexes.append(index_info)
                dropped_count += 1
                logging.info(f"✅ Successfully dropped scalar index '{index_name}'")
                
            except Exception as e:
                logging.error(f"❌ Failed to drop scalar index '{index_name}': {e}")
        
        return dropped_count
    
    def recreate_scalar_indexes(self):
        """Recreate previously dropped scalar indexes"""
        recreated_count = 0
        
        hard_code_keys = ['index_name', 'index_type', 'field_name', 'total_rows', 'pending_index_rows', 'indexed_rows', 'state']
        
        index_params = self.client.prepare_index_params()
        for index_info in self.scalar_indexes:
            index_name = index_info['index_name']
            field_name = index_info['field_name']
            # Extract index parameters from original info
            index_type = index_info.get('index_type', 'AUTOINDEX')
            params = index_info.get('params', {})

            for key, value in index_info.items():
                if key not in hard_code_keys:
                    params[key] = value
            
            # Create index params
            index_params.add_index(
                field_name=field_name, 
                index_type=index_type, 
                index_name=index_name,
                params=params)
            recreated_count += 1

        self.client.create_index(self.collection_name, index_params)
        
        logging.info(f"✅ Successfully recreated scalar index '{index_name}'")
        
        return recreated_count


def scalar_index_cycle_test(client, collection_name, wait_seconds, timeout):
    """
    Main function to perform scalar index cycling test
    
    :param client: MilvusClient instance
    :param collection_name: Name of the collection
    :param wait_seconds: Seconds to wait between drop and recreate
    :param timeout: Total test timeout in seconds
    """
    manager = ScalarIndexManager(client, collection_name)
    start_time = time.time()
    end_time = start_time + timeout
    cycle_count = 0
  
    # Initial setup - get schema and identify indexes
    schema = manager.get_collection_schema()
    if not schema:
        logging.error("Failed to get collection schema, aborting test")
        return
    
    while time.time() < end_time:
        cycle_start = time.time()
        
        cycle_count += 1
        
        try:
            # Step 1: Get all current indexes
            all_indexes = manager.get_all_indexes()
            if not all_indexes:
                raise Exception("No indexes found in collection, waiting before next cycle...")
            
            # Step 2: Identify scalar vs vector indexes
            scalar_indexes, vector_indexes = manager.identify_scalar_indexes(all_indexes, schema)
            
            if not scalar_indexes:
                raise Exception("No scalar indexes found to cycle")
                
            logging.info(f"Found {len(scalar_indexes)} scalar indexes and {len(vector_indexes)} vector indexes")
            
            # Step 3: Drop scalar indexes
            dropped_count = manager.drop_scalar_indexes(scalar_indexes)
            logging.info(f"✅ Phase 1: Dropped {dropped_count} scalar indexes")
            
            # Step 4: Wait
            logging.info(f"⏳ Phase 2: Waiting {wait_seconds:.1f}s...")
            time.sleep(wait_seconds)
            
            # Step 5: Recreate scalar indexes
            recreated_count = manager.recreate_scalar_indexes()
            logging.info(f"✅ Phase 3: Recreated {recreated_count} scalar indexes")

            logging.info(f"⏳ Phase 4: Waiting {wait_seconds:.1f}s...")
            time.sleep(wait_seconds)
            
            cycle_duration = time.time() - cycle_start
            logging.info(f"🏁 Cycle #{cycle_count} completed in {cycle_duration:.2f}s")

        except Exception as e:
            logging.error(f"❌ Error during cycle #{cycle_count}: {e}")
            time.sleep(1)  # Brief pause before retrying
    
    total_duration = time.time() - start_time
    logging.info("=" * 80)
    logging.info("🎯 SCALAR INDEX CYCLING TEST COMPLETED")
    logging.info(f"  Total Cycles: {cycle_count}")
    logging.info(f"  Total Duration: {total_duration:.2f}s")
    logging.info(f"  Average Cycle Time: {total_duration / max(cycle_count, 1):.2f}s")
    logging.info(f"  Current Index Status: {client.list_indexes(collection_name)}")
    logging.info("=" * 80)


def verify_collection(client, collection_name):
    """Verify that the collection exists and is loaded"""
    if not client.has_collection(collection_name):
        logging.error(f"Collection '{collection_name}' does not exist")
        return False
    
    # Check if collection is loaded
    load_state = client.get_load_state(collection_name)
    if load_state.get('state').name != 'Loaded':
        logging.info(f"Collection '{collection_name}' is not loaded")
        return False
    
    return True
 

if __name__ == '__main__':
    # Parse command line arguments
    try:
        host = sys.argv[1]
        collection_name = sys.argv[2]
        wait_seconds = int(sys.argv[3])
        timeout = int(sys.argv[4])
        api_key = sys.argv[5] 
    except (IndexError, ValueError) as e:
        print("Usage: will drop and recreate scalar indexes in a loop until timeout")
        print("python3 scalar_index_cycle.py <host> <collection_name> <wait_seconds> <timeout> [api_key]")
        print("\nParameters:")
        print("  host           : Milvus server host")
        print("  collection_name: Collection name")
        print("  wait_seconds   : Seconds to wait between drop and recreate")
        print("  timeout        : Total test timeout in seconds")
        print("  api_key        : API key (optional, use 'None' for local)")
        print("\nExamples:")
        print("  # Local Milvus, cycle every 30 seconds for 5 minutes")
        print("  python3 scalar_index_cycle.py localhost test_aa 30 300 None")
        print()
        print("  # Cloud Milvus, cycle every 10 seconds for 2 minutes")
        print("  python3 scalar_index_cycle.py host.com test_aa 10 120 api_key")
        sys.exit(1)
    
    port = 19530
    
    # Parameter validation
    if wait_seconds <= 0:
        print("Error: wait_seconds must be positive")
        sys.exit(1)
    
    if timeout <= 0:
        print("Error: timeout must be positive")
        sys.exit(1)
    
    if wait_seconds >= timeout:
        print("Warning: wait_seconds should be less than timeout for multiple cycles")
    
    # Setup logging
    log_filename = f"/tmp/scalar_index_cycle_{collection_name}_{int(time.time())}.log"
    file_handler = logging.FileHandler(filename=log_filename)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    
    logging.info("🚀 Starting Scalar Index Cycling Test")
    logging.info(f"  Host: {host}")
    logging.info(f"  Collection: {collection_name}")
    logging.info(f"  Wait Seconds: {wait_seconds}")
    logging.info(f"  Timeout: {timeout}s")
    logging.info(f"  API Key: {'***' if api_key and api_key.upper() != 'NONE' else 'None (local)'}")
    
    # Create MilvusClient
    if api_key is None or api_key == "" or api_key.upper() == "NONE":
        client = MilvusClient(uri=f"http://{host}:{port}")
    else:
        client = MilvusClient(uri=host, token=api_key)
    
    logging.info(f"✅ Connected to MilvusClient at {host}")
    
    # Verify collection
    if not verify_collection(client, collection_name):
        logging.error("Collection verification failed")
        sys.exit(1)
    
    # Run the test
    scalar_index_cycle_test(client, collection_name, wait_seconds, timeout)
    
    logging.info(f"📁 Log file: {log_filename}")
        
