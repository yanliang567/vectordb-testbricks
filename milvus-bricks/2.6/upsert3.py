#!/usr/bin/env python3
"""
Upsert Testing Script - Update entities with same ID and new version field

This script performs upsert operations on Milvus collections:
1. Updates entities in collection with the same ID
2. Updates version field with new value
3. Only works for INT64 primary key starting from 0
4. Can randomly select 100 collections if collection_name is "rand" or "random"
5. Validates collection existence and structure before operations

Key Features:
- MilvusClient API (Milvus 2.6 compatible)
- Strict collection validation (fail fast if invalid)
- Support for both single collection and random collection testing
- Version field handling for VARCHAR and INT32 types
- Optional duplicate entity checking
- Comprehensive logging and error handling
- Uses common.py utility functions for optimized performance:
  * gen_upsert_data_by_pk_collection: Smart data generation based on collection schema
  * upsert_entities: Efficient round-by-round upsert operations
  * get_primary_field_name: Reliable primary field detection
  * create_collection_schema: Standardized schema creation
  * create_n_insert: Complete collection setup with indexing

Usage:
    python3 upsert3.py <hosts> <collection_name> <upsert_rounds> <entities_per_round> <new_version> <interval> <check_diff> [api_key]

Examples:
    # Single host upsert test
    python3 upsert3.py localhost my_collection 10 100 "v2.0" 5 TRUE None
    
    # Multi-host upsert test (comparison mode)
    python3 upsert3.py "host1,host2" my_collection 10 100 "v2.0" 5 TRUE None
    
    # Random collections test (will pick 100 random collections)
    python3 upsert3.py localhost random 5 50 NONE 2 FALSE None
"""

import time
import sys
import random
import logging
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema

# Import common utility functions from 2.6 directory
from common import (
    create_collection_schema, 
    create_n_insert,
    gen_upsert_data_by_pk_collection,
    insert_entities,
    gen_row_data_by_schema,
    get_primary_field_name,
    get_float_vec_field_name
)

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"


class UpsertTester:
    """
    Upsert testing manager using MilvusClient API
    
    This class leverages common.py utility functions for optimal performance:
    - Uses upsert_entities() for efficient batch upsert operations
    - Uses get_primary_field_name() for reliable schema analysis
    - Integrates with gen_upsert_data_by_pk_collection() for smart data generation
    """
    
    def __init__(self, clients, collection_name):
        self.clients = clients if isinstance(clients, list) else [clients]
        self.client = self.clients[0]  # Primary client for operations
        self.collection_name = collection_name
    
    def validate_collection_for_upsert(self):
        """
        Validate collection for upsert operations
        Returns: (is_valid, error_message)
        """
        # Check if collection exists
        if not self.client.has_collection(self.collection_name):
            return False, f"Collection '{self.collection_name}' does not exist"
        
        # Get collection schema
        schema = self.client.describe_collection(self.collection_name)
        
        # Check if collection has indexes
        indexes = self.client.list_indexes(self.collection_name)
        if not indexes:
            return False, f"Collection '{self.collection_name}' has no indexes"
        
        # Get primary field using common utility function
        primary_field_name = get_primary_field_name(schema)
        if not primary_field_name:
            return False, f"Collection '{self.collection_name}' has no primary field"
        
        # Find primary field and version field details
        fields = schema.get('fields', [])
        primary_field = None
        version_field = None
        
        for field in fields:
            if field.get('name') == primary_field_name:
                primary_field = field
            if field.get('name') == 'version':
                version_field = field
        
        # Check if auto_id is disabled (required for upsert with specific IDs)
        if schema.get('auto_id', True):
            return False, f"Collection '{self.collection_name}' has auto_id=True, which is not supported for upsert"
        
        # Check primary field type (must be INT64)
        if primary_field.get('type') != DataType.INT64:
            return False, f"Collection '{self.collection_name}' primary field type is not INT64"
        
        # # Check if version field exists
        # if not version_field:
        #     return False, f"Collection '{self.collection_name}' does not have 'version' field"
        
        return True, "Collection validation passed"
            
    
    def ensure_collection_loaded(self):
        """Ensure collection is loaded"""
        load_state = self.client.get_load_state(self.collection_name)
        if load_state.get('state').name != 'Loaded':
            logging.info(f"Loading collection '{self.collection_name}'...")
            start_time = time.time()
            self.client.load_collection(self.collection_name)
            load_duration = round(time.time() - start_time, 3)
            logging.info(f"✅ Collection '{self.collection_name}' loaded in {load_duration}s")
        else:
            logging.info(f"Collection '{self.collection_name}' is already loaded")
        
        return True
    
    
    def get_current_version(self):
        """Get current version from collection"""
        try:
            # Get collection info
            result = self.client.query(
                collection_name=self.collection_name,
                filter="",  # No filter to get any entity
                output_fields=["version"],
                limit=1
            )
            
            if result and len(result) > 0:
                old_version = result[0].get("version")
                logging.info(f"Current version in collection: {old_version}")
                return old_version
            else:
                logging.info("Collection is empty, no current version found")
                return "NONE"
                
        except Exception as e:
            logging.warning(f"Failed to get current version: {e}, assuming empty collection")
            return "NONE"
    
    def get_version_field_type(self):
        """Get version field data type"""
        try:
            schema = self.client.describe_collection(self.collection_name)
            fields = schema.get('fields', [])
            
            for field in fields:
                if field.get('name') == 'version':
                    return field.get('type')
            
            return None
        except Exception as e:
            logging.error(f"Failed to get version field type: {e}")
            return None
    
    def generate_new_version_value(self, new_version, version_field_type):
        """Generate new version value based on field type"""
        if new_version.upper() == "NONE":
            if version_field_type == DataType.VARCHAR:
                return time.asctime()
            elif version_field_type == DataType.INT32:
                return int(time.time())
            else:
                return str(int(time.time()))  # Default to string representation
        else:
            return new_version
    
    def perform_upsert_operations(self, upsert_rounds, entities_per_round, new_version, interval):
        """Perform upsert operations using common utility functions"""
        try:
            # Get version field type
            version_field_type = self.get_version_field_type()
            if not version_field_type:
                raise Exception("Cannot determine version field type")
            
            # Get current version
            old_version = self.get_current_version()
            
            # Generate new version value
            final_new_version = self.generate_new_version_value(new_version, version_field_type)
          
            # Use common utility function for upsert operations
            # This function handles round-by-round upsert with proper data generation
            insert_entities(
                clients=self.clients,
                collection_name=self.collection_name,
                nb=entities_per_round,
                rounds=upsert_rounds,
                use_insert=False,
                new_version=final_new_version,
                interval=interval
            )
            
            # Get final count
            final_count = self.get_entity_count()
            logging.info(f"🎯 Upsert completed - Final entity count: {final_count}")
            
            # Verify old version is replaced
            self.verify_version_replacement(old_version, version_field_type)
              
            max_id = upsert_rounds * entities_per_round
            
            return True, max_id, final_count
            
        except Exception as e:
            logging.error(f"Failed to perform upsert operations: {e}")
            return False, 0, 0
    
    # Note: generate_upsert_entities method removed - now using common.py's gen_upsert_data_by_pk_collection
    
    def get_entity_count(self):
        """Get total entity count in collection"""
        try:
            result = self.client.query(
                collection_name=self.collection_name,
                filter="",
                output_fields=["count(*)"]
            )
            return result[0].get("count(*)", 0)
        except Exception as e:
            logging.error(f"Failed to get entity count: {e}")
            return -1
    
    def verify_version_replacement(self, old_version, version_field_type):
        """Verify that old version entities are replaced"""
        try:
            if old_version == "NONE":
                logging.info("No old version to verify (collection was empty)")
                return
            
            # Build query expression based on field type
            if version_field_type == DataType.VARCHAR:
                expr = f"version=='{old_version}'"
            else:
                expr = f"version=={old_version}"
            
            result = self.client.query(
                collection_name=self.collection_name,
                filter=expr,
                output_fields=["count(*)"]
            )
            
            old_count = result[0].get("count(*)", 0)
            
            if old_count > 0:
                logging.error(f"❌ Found {old_count} entities with old version '{old_version}' - upsert may have failed")
            else:
                logging.info(f"✅ No entities with old version '{old_version}' found - upsert successful")
                
        except Exception as e:
            logging.error(f"Failed to verify version replacement: {e}")
    
    def check_duplicate_entities(self, max_id):
        """Check for duplicate entities by ID"""
        try:
            logging.info(f"🔍 Checking for duplicate entities (IDs 0 to {max_id - 1})...")
            dup_count = 0
            
            # Sample check for efficiency (check every 10th ID for large datasets)
            check_interval = max(1, max_id // 1000)  # Check at most 1000 IDs
            
            for i in range(0, max_id, check_interval):
                result = self.client.query(
                    collection_name=self.collection_name,
                    filter=f"id=={i}",
                    output_fields=["count(*)"]
                )
                
                count = result[0].get("count(*)", 0)
                if count != 1:
                    dup_count += 1
                    logging.error(f"❌ ID {i} has {count} entities (should be 1)")
                    if dup_count >= 10:  # Stop after finding 10 errors
                        logging.error("Too many duplicate entities found, stopping check")
                        break
            
            if dup_count == 0:
                logging.info(f"✅ No duplicate entities found in sample check")
            else:
                logging.error(f"❌ Found {dup_count} IDs with duplicate/missing entities")
                
            return dup_count
            
        except Exception as e:
            logging.error(f"Failed to check duplicate entities: {e}")
            return -1

def get_random_collections(client, count=100):
    """Get random collections for testing"""
    try:
        all_collections = client.list_collections()
        if len(all_collections) < count:
            logging.warning(f"Only {len(all_collections)} collections available, using all of them")
            return all_collections
        else:
            selected = random.sample(all_collections, count)
            logging.info(f"Selected {len(selected)} random collections for testing")
            return selected
    except Exception as e:
        logging.error(f"Failed to get random collections: {e}")
        return []


def main():
    """Main upsert testing function"""
    # Parse command line arguments
    try:
        hosts = sys.argv[1]  # hosts ips or uris separated by comma, only 2 hosts max are supported for comparison tests
        collection_name = sys.argv[2]
        upsert_rounds = int(sys.argv[3])
        entities_per_round = int(sys.argv[4])
        new_version = sys.argv[5]
        interval = int(sys.argv[6])
        check_diff = str(sys.argv[7]).upper()
        api_key = sys.argv[8]
        
    except (IndexError, ValueError) as e:
        print("Usage: python3 upsert3.py <hosts> <collection_name> <upsert_rounds> <entities_per_round> <new_version> <interval> <check_diff> [api_key]")
        print("\nDescription:")
        print("  Upsert entities with same ID and new version field. Only works for INT64 primary key starting from 0.")
        print("  Will randomly select 100 collections if collection_name is 'rand' or 'random'.")
        print("  Supports up to 2 hosts for comparison testing.")
        print("\nParameters:")
        print("  hosts              : Milvus server hosts (comma-separated, max 2 hosts)")
        print("  collection_name    : Collection name (or 'rand'/'random' for random selection)")
        print("  upsert_rounds      : Number of upsert rounds")
        print("  entities_per_round : Number of entities to upsert per round")
        print("  new_version        : New value for version field ('NONE' for auto-generated)")
        print("  interval           : Interval between upsert rounds (seconds)")
        print("  check_diff         : Check for duplicate entities (TRUE/FALSE)")
        print("  api_key            : API key (optional, use 'None' for local)")
        print("\nExamples:")
        print("  # Single host upsert test")
        print("  python3 upsert3.py localhost my_collection 10 100 'v2.0' 5 TRUE None")
        print()
        print("  # Multi-host comparison test")
        print("  python3 upsert3.py 'host1,host2' my_collection 10 100 'v2.0' 5 TRUE None")
        print()
        print("  # Random collections test")
        print("  python3 upsert3.py localhost random 5 50 NONE 2 FALSE None")
        sys.exit(1)
    
    port = 19530
    
    # Parse and validate hosts
    hosts = hosts.split(",")
    if len(hosts) > 2:
        logging.error("Only support 2 hosts max for now")
        sys.exit(1)
    
    # Setup logging
    log_filename = f"/tmp/upsert3_{collection_name}_{int(time.time())}.log"
    file_handler = logging.FileHandler(filename=log_filename)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    
    # Convert parameters
    check_diff = True if check_diff == "TRUE" else False
    is_random_collections = True if collection_name.upper() in ["RAND", "RANDOM"] else False
    
    logging.info("🚀 Starting Upsert3 Testing")
    logging.info(f"  Hosts: {hosts}")
    logging.info(f"  Collection: {collection_name}")
    logging.info(f"  Random Collections: {is_random_collections}")
    logging.info(f"  Upsert Rounds: {upsert_rounds}")
    logging.info(f"  Entities Per Round: {entities_per_round}")
    logging.info(f"  New Version: {new_version}")
    logging.info(f"  Interval: {interval}s")
    logging.info(f"  Check Duplicates: {check_diff}")
    logging.info(f"  API Key: {'***' if api_key and api_key.upper() != 'NONE' else 'None (local)'}")
    
    # Create MilvusClients (similar to create_n_insert.py)
    client_2 = None
    if api_key is None or api_key == "" or api_key.upper() == "NONE":
        client_1 = MilvusClient(uri=f"http://{hosts[0]}:{port}")
        if len(hosts) > 1:
            client_2 = MilvusClient(uri=f"http://{hosts[1]}:{port}")
    else:
        client_1 = MilvusClient(uri=f"http://{hosts[0]}:{port}", token=api_key)
        if len(hosts) > 1:
            client_2 = MilvusClient(uri=f"http://{hosts[1]}:{port}", token=api_key)
    
    # Create client list for UpsertTester
    clients = [client_1]
    if client_2 is not None:
        clients.append(client_2)
    
    logging.info(f"✅ Connected to MilvusClients: client_1={client_1}, client_2={client_2}")

    
    # Determine collections to test
    collections_to_test = []
    if is_random_collections:
        collections_to_test = get_random_collections(client_1, 100)
        if not collections_to_test:
            logging.error("Failed to get random collections")
            sys.exit(1)
    else:
        # Single collection mode
        if not client_1.has_collection(collection_name):
            logging.error(f"Collection '{collection_name}' does not exist")
            sys.exit(1) 
        collections_to_test = [collection_name]
    
    # Process each collection
    total_success = 0
    total_failed = 0
    
    for test_collection in collections_to_test:
        logging.info("=" * 80)
        logging.info(f"🧪 Processing collection: {test_collection}")
        
        try:
            # Create upsert tester with multiple clients
            tester = UpsertTester(clients, test_collection)
            
            # Validate collection
            is_valid, error_message = tester.validate_collection_for_upsert()
            if not is_valid:
                logging.error(f"❌ Collection validation failed: {error_message}")
                
                if is_random_collections:
                    total_failed += 1
                    continue  # Skip this collection in random mode
                else:
                    logging.error("Collection validation failed in single collection mode, exiting")
                    sys.exit(1)
            
            logging.info(f"✅ Collection '{test_collection}' validation passed")
            
            # Ensure collection is loaded
            if not tester.ensure_collection_loaded():
                logging.error(f"Failed to load collection '{test_collection}'")
                if is_random_collections:
                    total_failed += 1
                    continue
                else:
                    sys.exit(1)
            
            # Perform upsert operations
            success, max_id, final_count = tester.perform_upsert_operations(
                upsert_rounds, entities_per_round, new_version, interval
            )
            
            if not success:
                logging.error(f"❌ Upsert operations failed for collection '{test_collection}'")
                total_failed += 1
                if not is_random_collections:
                    sys.exit(1)
                continue
            
            # Check for duplicates if requested
            if check_diff:
                dup_count = tester.check_duplicate_entities(max_id)
                if dup_count > 0:
                    logging.warning(f"Found {dup_count} duplicate/missing entities in '{test_collection}'")
            
            total_success += 1
            logging.info(f"✅ Collection '{test_collection}' processing completed successfully")
            
        except Exception as e:
            logging.error(f"❌ Unexpected error processing collection '{test_collection}': {e}")
            total_failed += 1
            if not is_random_collections:
                sys.exit(1)
    
    # Final summary
    logging.info("=" * 80)
    logging.info("🎯 UPSERT3 TESTING SUMMARY")
    logging.info(f"  Total Collections Processed: {len(collections_to_test)}")
    logging.info(f"  Successful: {total_success}")
    logging.info(f"  Failed: {total_failed}")
    logging.info(f"  Success Rate: {total_success / len(collections_to_test) * 100:.1f}%")
    logging.info(f"📁 Log file: {log_filename}")
    logging.info("=" * 80)


if __name__ == '__main__':
    main()
