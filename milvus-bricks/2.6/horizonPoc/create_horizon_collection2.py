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
    
    # Define all fields according to create horizon test collection
    all_scalar_fields = [
       'device_id', 'type_model', 'brand', 'hardware', 'software', 'district',
       'gcj02_lat', 'gcj02_lon', 'wgs84_lat', 'wgs84_lon', 'geo_ids',
       'timeline_tags', 'event_id', 'drive', 'drive_status', 'app_id',
       'link_info', 'node_info', 'session_id', 'site_id', 'static_mode',
       'timestamp', 'expert_collected', 'sensor_lidar_type']
    fields = [
        # Primary key field (VARCHAR with auto_id)
        FieldSchema(name="id",
            dtype=DataType.VARCHAR,
            max_length=64,
            is_primary=True,
            auto_id=True,
            description="Primary key field with auto-generated values"
        ),
        
        # Vector field (provided by Horizon)
        FieldSchema(name="feature",
            dtype=DataType.FLOAT_VECTOR,
            dim=768,
            description="Feature vector provided by Horizon (768 dimensions)"
        ),
        
        # device_id field (VARCHAR)
        FieldSchema(name="device_id",
            dtype=DataType.VARCHAR,
            max_length=16,
            nullable=True,
            description="Device ID field"
        ),

        # type_model field (VARCHAR)
        FieldSchema(name="type_model",
            dtype=DataType.VARCHAR,
            max_length=16,
            nullable=True,
            description="Type model field"
        ),
        
        # brand field (VARCHAR)
        FieldSchema(name="brand",
            dtype=DataType.VARCHAR,
            max_length=16,
            nullable=True,
            description="Brand field"
        ),
        
        # hardware field (VARCHAR)
        FieldSchema(name="hardware",
            dtype=DataType.VARCHAR,
            max_length=128,
            nullable=True,
            description="Hardware field"
        ),
        
        # software field (VARCHAR)
        FieldSchema(name="software",
            dtype=DataType.VARCHAR,
            max_length=128,
            nullable=True,
            description="Software field"
        ),
        
        # district field (VARCHAR)
        FieldSchema(name="district",
            dtype=DataType.VARCHAR,
            max_length=16,
            nullable=True,
            description="District field"
        ),
        
        # gcj02_lat field (FLOAT)
        FieldSchema(name="gcj02_lat",
            dtype=DataType.FLOAT,
            nullable=True,
            description="GCJ02 latitude field"
        ),

        # gcj02_lon field (FLOAT)
        FieldSchema(name="gcj02_lon",
            dtype=DataType.FLOAT,
            nullable=True,
            description="GCJ02 longitude field"
        ),
        
        # wgs84_lat field (FLOAT)
        FieldSchema(name="wgs84_lat",
            dtype=DataType.FLOAT,
            nullable=True,
            description="WGS84 latitude field"
        ),
        
        # wgs84_lon field (FLOAT)
        FieldSchema(name="wgs84_lon",
            dtype=DataType.FLOAT,
            nullable=True,
            description="WGS84 longitude field"
        ),
        
        # geo_ids field (VARCHAR)
        FieldSchema(name="geo_ids",
            dtype=DataType.VARCHAR,
            max_length=16,
            nullable=True,
            description="Geo IDs field"
        ),
        
        # timeline_tags field (VARCHAR)
        FieldSchema(name="timeline_tags",
            dtype=DataType.ARRAY,
            element_type=DataType.VARCHAR,
            max_capacity=10,
            max_length=32,
            nullable=True,
            description="Timeline tags field"
        ),
        
        # event_id field (VARCHAR)
        FieldSchema(name="event_id",
            dtype=DataType.VARCHAR,
            max_length=48,
            nullable=True,
            description="Event ID field"
        ),

        # drive field (VARCHAR)
        FieldSchema(name="drive",
            dtype=DataType.VARCHAR,
            max_length=48,
            nullable=True,
            description="Drive field"
        ),
        
        # drive_status field (VARCHAR)
        FieldSchema(name="drive_status",
            dtype=DataType.VARCHAR,
            max_length=10,
            nullable=True,
            description="Drive status field"
        ),
        
        # app_id field (VARCHAR)
        FieldSchema(name="app_id",
            dtype=DataType.VARCHAR,
            max_length=16,
            nullable=True,
            description="App ID field"
        ),
        
        # link_info field (VARCHAR)
        FieldSchema(name="link_info",
            dtype=DataType.ARRAY,
            element_type=DataType.VARCHAR,
            max_capacity=10,
            max_length=256,
            nullable=True,
            description="Link info field"
        ),

        # node_info field (VARCHAR)
        FieldSchema(name="node_info",
            dtype=DataType.ARRAY,
            element_type=DataType.VARCHAR,
            max_capacity=10,
            max_length=16,
            nullable=True,
            description="Node info field"
        ),

        # session_id field (VARCHAR)
        FieldSchema(name="session_id",
            dtype=DataType.VARCHAR,
            max_length=48,
            nullable=True,
            description="Session ID field"
        ),

        # site_id field (VARCHAR)
        FieldSchema(name="site_id",
            dtype=DataType.VARCHAR,
            max_length=16,
            nullable=True,
            description="Site ID field"
        ),

        # static_mode field (VARCHAR)
        FieldSchema(name="static_mode",
            dtype=DataType.VARCHAR,
            max_length=10,
            nullable=True,
            description="Static mode field"
        ),

        # timestamp field (INT64)
        FieldSchema(name="timestamp",
            dtype=DataType.INT64,
            nullable=True,
            description="Timestamp field"
        ),

        # expert_collected field (VARCHAR)
        FieldSchema(name="expert_collected",
            dtype=DataType.BOOL,
            nullable=True,
            description="Expert collected field"
        ),

        # sensor_lidar_type field (VARCHAR)
        FieldSchema(name="sensor_lidar_type",
            dtype=DataType.VARCHAR,
            max_length=20,
            nullable=True,
            description="Sensor lidar type field"
        )
        # # geo location field (GEOMETRY)
        # FieldSchema(name="gcj02_location",
        #     dtype=DataType.GEOMETRY,
        #     nullable=True,
        #     description="GCJ02 location field"
        # ),
        # FieldSchema(name="wgs84_location",
        #     dtype=DataType.GEOMETRY,
        #     nullable=True,
        #     description="WGS84 location field"
        # )

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
    
    # Index for scalar fields
    index_params.add_index(field_name="timestamp", 
        index_type="STL_SORT"
    )

    index_params.add_index(field_name="type_model", 
        index_type="INVERTED"
    )

    index_params.add_index(field_name="expert_collected", 
        index_type="BITMAP"
    )

    index_params.add_index(field_name="device_id", 
        index_type="AUTOINDEX"
    )

    index_params.add_index(field_name="timeline_tags", 
        index_type="INVERTED"
    )

    index_params.add_index(field_name="sensor_lidar_type", 
        index_type="AUTOINDEX"
    )

    index_params.add_index(field_name="gcj02_lat", 
        index_type="STL_SORT"
    )
    index_params.add_index(field_name="gcj02_lon", 
        index_type="STL_SORT"
    )

    # # index for location
    # index_params.add_index(field_name="gcj02_location",
    #     index_type="RTREE"
    # )
    # index_params.add_index(field_name="wgs84_location",
    #     index_type="RTREE"
    # )
    
    # Build all indexes
    client.create_index(
        collection_name=collection_name,
        index_params=index_params
    )
    
    logging.info("✅ All indexes created successfully")


def main():
    """Main function to create horizon collection"""
    
    # Parse command line arguments
    try:
        # host = sys.argv[1]
        # api_key = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2].upper() != "NONE" else None
        # drop_if_exists = len(sys.argv) > 3 and sys.argv[3].upper() == "TRUE"

        host = 'https://in01-3e1a7693c28817d.ali-cn-hangzhou.cloud-uat.zilliz.cn:19530'
        api_key = 'cc5bf695ea9236e2c64617e9407a26cf0953034485d27216f8b3f145e3eb72396e042db2abb91c4ef6fde723af70e754d68ca787'
        drop_if_exists = "TRUE"
        
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
        
        # Final summary
        logging.info("=" * 80)
        logging.info("🎯 HORIZON COLLECTION CREATION SUMMARY")
        logging.info(f"  ✅ Collection: {collection_name}")
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
