#!/usr/bin/env python3
"""
Create Horizon Test Collection based on create_horizon.p configuration

This script creates a Milvus collection based on the configuration specified 
in create_horizon.p file, using the MilvusClient API (Milvus 2.6 compatible).

Collection Info:
- Collection Name: horizon_test_collection  
- Primary Key: id (VARCHAR, auto_id=True)
- Vector Field: feature (FLOAT_VECTOR, dim=768) - provided by Horizon
- Scalar Fields: timestamp, url, device_id, longitude, latitude
- Dynamic Field: Enabled
- Shards: 1, Partitions: 16

Usage:
    python3 create_horizon_collection.py <host> [api_key] [drop_if_exists]

Examples:
    # Create on local Milvus
    python3 create_horizon_collection.py localhost
    
    # Create on cloud with API key
    python3 create_horizon_collection.py https://your-cluster.zillizcloud.com your_api_key
    
    # Drop existing collection first
    python3 create_horizon_collection.py localhost None TRUE
"""

import sys
import time
import logging
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema

# Setup logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"

def create_horizon_collection_schema():
    """
    Create collection schema based on create_horizon.p configuration
    
    Returns:
        CollectionSchema: The complete schema for horizon_test_collection
    """
    
    # Define all fields according to create_horizon.p
    fields = [
        # Primary key field (VARCHAR with auto_id)
        FieldSchema(
            name="id",
            dtype=DataType.VARCHAR,
            max_length=64,
            is_primary=True,
            auto_id=False,
            description="Primary key field with auto-generated values"
        ),
        
        # Vector field (provided by Horizon)
        FieldSchema(
            name="feature",
            dtype=DataType.FLOAT_VECTOR,
            dim=768,
            description="Feature vector provided by Horizon (768 dimensions)"
        ),
        
        # Timestamp field (INT64)
        FieldSchema(
            name="timestamp",
            dtype=DataType.INT64,
            description="Timestamp field (example: 1751961530077, within past 16 months)"
        ),
        
        # URL field (VARCHAR)
        FieldSchema(
            name="url",
            dtype=DataType.VARCHAR,
            max_length=1024,
            description="URL information field"
        ),
        
        # Device ID field (VARCHAR)
        FieldSchema(
            name="device_id", 
            dtype=DataType.VARCHAR,
            max_length=32,
            description="Device ID field (example: 'DV345', ~1000 unique values, length 10-20)"
        ),
        
        # Location field (GEOMETRY)
        FieldSchema(
            name="location",
            dtype=DataType.GEOMETRY,
            description="Location coordinate (from NYC-Taxi dataset, adapted for location)"
        )
        
    ]
    
    # Create collection schema
    schema = CollectionSchema(
        fields=fields,
        description="Horizon test collection for feature vector storage and retrieval",
        enable_dynamic_field=True  # As specified in create_horizon.p
    )
    
    return schema

def create_collection_indexes(client, collection_name):
    """
    Create appropriate indexes for the horizon collection
    
    Args:
        client: MilvusClient instance
        collection_name: Name of the collection
    """
    
    logging.info("🔧 Creating indexes for collection...")
    
    # Prepare index parameters
    index_params = client.prepare_index_params()
    
    # Vector index for feature field (HNSW for high performance)
    index_params.add_index(
        field_name="feature",
        index_type="AUTOINDEX",
        metric_type="L2"
    )
    
    # Scalar indexes for common query fields
    
    # Index for device_id (for device-specific queries)
    index_params.add_index(
        field_name="device_id", 
        index_type="INVERTED"
    )

    # INdex for location
    index_params.add_index(
        field_name="location",
        index_type="RTREE"
    )
    
    # Build all indexes
    client.create_index(
        collection_name=collection_name,
        index_params=index_params
    )
    
    logging.info("✅ All indexes created successfully")

def verify_collection_creation(client, collection_name):
    """
    Verify that the collection was created correctly
    
    Args:
        client: MilvusClient instance  
        collection_name: Name of the collection
    """
    
    logging.info("🔍 Verifying collection creation...")
    
    # Check if collection exists
    if not client.has_collection(collection_name):
        raise Exception(f"Collection '{collection_name}' was not created successfully")
    
    # Get collection info
    collection_info = client.describe_collection(collection_name)
    
    logging.info(f"📋 Collection Info:")
    logging.info(f"   Name: {collection_info.get('collection_name')}")
    logging.info(f"   Description: {collection_info.get('description')}")
    logging.info(f"   Auto ID: {collection_info.get('auto_id')}")
    logging.info(f"   Fields Count: {len(collection_info.get('fields', []))}")
    
    # Verify fields
    fields = collection_info.get('fields', [])
    expected_fields = ['id', 'feature', 'timestamp', 'url', 'device_id', 'location']
    
    actual_fields = [field.get('name') for field in fields]
    missing_fields = set(expected_fields) - set(actual_fields)
    extra_fields = set(actual_fields) - set(expected_fields)
    if missing_fields:
        raise Exception(f"Missing expected fields: {missing_fields}")
    
    if extra_fields:
        raise Exception(f"Extra fields: {extra_fields}")
    
    logging.info("✅ Collection structure verification passed")
    
    # List indexes
    indexes = client.list_indexes(collection_name)
    logging.info(f"📊 Indexes created: {len(indexes)}")
    for idx_name in indexes:
        logging.info(f"   - {idx_name}")
    
    return True

def main():
    """Main function to create horizon collection"""
    
    # Parse command line arguments
    try:
        host = sys.argv[1]
        api_key = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2].upper() != "NONE" else None
        drop_if_exists = len(sys.argv) > 3 and sys.argv[3].upper() == "TRUE"
        
    except IndexError:
        print("Usage: python3 create_horizon_collection.py <host> [api_key] [drop_if_exists]")
        print("\nParameters:")
        print("  host            : Milvus server host or URI")  
        print("  api_key         : API key for cloud (optional, use 'None' for local)")
        print("  drop_if_exists  : Drop existing collection first (TRUE/FALSE, default: FALSE)")
        print("\nExamples:")
        print("  # Local Milvus")
        print("  python3 create_horizon_collection.py localhost")
        print()
        print("  # Cloud Milvus with API key")
        print("  python3 create_horizon_collection.py https://your-cluster.zillizcloud.com your_api_key")
        print()
        print("  # Drop existing first")
        print("  python3 create_horizon_collection.py localhost None TRUE")
        sys.exit(1)
    
    # Setup logging
    log_filename = f"/tmp/create_horizon_collection_{int(time.time())}.log"
    file_handler = logging.FileHandler(filename=log_filename)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    
    collection_name = "horizon_test_collection"
    port = 19530
    
    logging.info("🚀 Starting Horizon Collection Creation")
    logging.info(f"  Host: {host}")
    logging.info(f"  Collection Name: {collection_name}")
    logging.info(f"  API Key: {'***' if api_key else 'None (local)'}")
    logging.info(f"  Drop if Exists: {drop_if_exists}")
    
    try:
        # Create MilvusClient
        if api_key:
            client = MilvusClient(uri=host, token=api_key)
        else:
            # For local deployment
            if not host.startswith('http'):
                host = f"http://{host}:{port}"
            client = MilvusClient(uri=host)
        
        logging.info(f"✅ Connected to Milvus at {host}")
        
        # Drop existing collection if requested
        if drop_if_exists and client.has_collection(collection_name):
            logging.info(f"🗑️ Dropping existing collection '{collection_name}'...")
            client.drop_collection(collection_name)
            logging.info("✅ Existing collection dropped")
        
        # Check if collection already exists
        if client.has_collection(collection_name):
            logging.warning(f"⚠️ Collection '{collection_name}' already exists")
            choice = input("Do you want to continue anyway? (y/N): ").lower()
            if choice != 'y':
                logging.info("Operation cancelled by user")
                sys.exit(0)
        
        # Create collection schema
        logging.info("📋 Creating collection schema...")
        schema = create_horizon_collection_schema()
        logging.info("✅ Schema created successfully")
        
        # Create collection with specified configuration 
        logging.info(f"🏗️ Creating collection '{collection_name}'...")
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            # Collection configuration from create_horizon.p
        )
        logging.info("✅ Collection created successfully")
        
        # Create indexes
        create_collection_indexes(client, collection_name)
        
        # Verify creation
        verify_collection_creation(client, collection_name)
        
        # Final summary
        logging.info("=" * 80)
        logging.info("🎯 HORIZON COLLECTION CREATION SUMMARY")
        logging.info(f"  ✅ Collection: {collection_name}")
        logging.info(f"  ✅ Schema: 7 fields (1 vector, 6 scalar)")
        logging.info(f"  ✅ Vector Dimension: 768")
        logging.info(f"  ✅ Indexes: Vector + Scalar indexes created")
        logging.info(f"  ✅ Configuration: 1 shard, 16 partitions, dynamic fields enabled")
        logging.info(f"  📁 Log file: {log_filename}")
        logging.info("=" * 80)
        logging.info("🎉 Collection is ready for data insertion and querying!")
        
    except Exception as e:
        logging.error(f"❌ Failed to create collection: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
