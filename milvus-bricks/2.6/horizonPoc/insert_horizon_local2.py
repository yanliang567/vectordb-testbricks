#!/usr/bin/env python3
"""
Local Horizon Data Insertion with Concurrent Batch Processing

This script reads parquet files from a local directory and inserts them into Milvus collection
with concurrent batch processing to handle large datasets efficiently.

Directory structure expected:
- data/: Contains parquet files with all required fields (id, feature, location, timestamp, url, 
         device_id, expert_collected, sensor_lidar_type, p_url)

Features:
- Concurrent batch processing for large datasets (process one file at a time)
- Configurable concurrency (default: 2 threads)
- Memory-efficient processing with configurable batch sizes
- Strict validation - no mock data generation
- Thread-safe processing - each file processed exactly once
- Error handling with detailed logging

Usage:
    python3 insert_horizon_local2.py [data_dir] [host] [collection_name] [batch_size] [concurrency]

Examples:
    # Use default settings (concurrency=2)
    python3 insert_horizon_local2.py
    
    # Custom configuration for large datasets
    python3 insert_horizon_local2.py ./data localhost horizon_test_collection 2500 4
    
    # Adjust concurrency based on system resources
    # concurrency=1: Sequential processing (no concurrency)
    # concurrency=2: Default, balanced performance (recommended)
    # concurrency=4: Higher concurrency for powerful systems
"""

import sys
import time
import logging
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import random
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# PyMilvus imports
from pymilvus import MilvusClient, DataType

# Setup logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"


class LocalParquetReader:
    """Local parquet file reader"""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        logging.info(f"📁 Initialized local reader for directory: {data_dir}")

    def list_parquet_files(self) -> List[str]:
        """List all parquet files in the data directory"""
        pattern = os.path.join(self.data_dir, "*.parquet")
        files = glob.glob(pattern)
        logging.info(f"📋 Found {len(files)} parquet files in {self.data_dir}")
        return sorted(files)

    def read_parquet_file(self, file_path: str) -> pd.DataFrame:
        """Read a single parquet file"""
        try:
            df = pd.read_parquet(file_path)
            logging.info(f"📄 Read {file_path}: {len(df)} records, {len(df.columns)} columns")
            return df
        except Exception as e:
            logging.error(f"❌ Failed to read {file_path}: {e}")
            raise


class BatchDataProcessor:
    """Simplified batch data processor - reads one file at a time"""

    def __init__(self, parquet_reader: LocalParquetReader):
        self.parquet_reader = parquet_reader

        # Initialize file list
        self.parquet_files = []

        # Required columns
        self.required_columns = ['device_id', 'type_model', 'brand', 'hardware', 'software', 'district',
                                 'gcj02_lat', 'gcj02_lon', 'wgs84_lat', 'wgs84_lon', 'geo_ids',
                                 'timeline_tags', 'event_id', 'drive', 'drive_status', 'app_id',
                                 'link_info', 'node_info', 'session_id', 'site_id', 'static_mode',
                                 'timestamp', 'expert_collected', 'sensor_lidar_type', 'vector']

        self._initialize_file_list()

    def _initialize_file_list(self):
        """Initialize file list and validate"""
        logging.info("📋 Initializing batch processing...")

        # Get all parquet files from the data directory
        self.parquet_files = self.parquet_reader.list_parquet_files()

        # Calculate total batches (1 file = 1 batch)
        self.total_batches = len(self.parquet_files)

        logging.info(f"📊 Batch processing plan:")
        logging.info(f"  - Total files: {len(self.parquet_files)}")
        logging.info(f"  - Total batches: {self.total_batches} (1 file per batch)")
        logging.info(f"  - Required columns: {', '.join(self.required_columns)}")

        # Validation
        if not self.parquet_files:
            logging.error("❌ No parquet files found - cannot proceed")
            raise Exception("No parquet files found")

    def read_batch_data(self, batch_idx: int) -> pd.DataFrame:
        """Read data for a specific batch (one file at a time)"""

        if batch_idx >= self.total_batches:
            raise Exception(f"Batch index {batch_idx} out of range (max: {self.total_batches - 1})")

        logging.info(f"📖 Reading data for batch {batch_idx + 1}/{self.total_batches}")

        # Get the file for this batch
        file_path = self.parquet_files[batch_idx]

        try:
            # Read the parquet file
            df = self.parquet_reader.read_parquet_file(file_path)

            # Validate required columns
            missing_columns = [col for col in self.required_columns if col not in df.columns]
            if missing_columns:
                logging.error(f"❌ Missing required columns in {file_path}: {missing_columns}")
                logging.info(f"📋 Available columns: {list(df.columns)}")
                logging.info(f"📋 Required columns: {self.required_columns}")
                raise Exception(f"Missing required columns in {file_path}: {missing_columns}")

            # Select only required columns
            df = df[self.required_columns].copy()

            logging.info(f"✅ Read batch {batch_idx + 1}: {len(df)} records, {len(df.columns)} columns")

            return df

        except Exception as e:
            logging.error(f"❌ Failed to read file {file_path}: {e}")
            raise Exception(f"Failed to read file {file_path}: {e}")

    def convert_to_wkt_point(self, lon: float, lat: float) -> str:
        """
        Convert longitude and latitude to WKT Point format
        
        Args:
            lon: Longitude value
            lat: Latitude value
            
        Returns:
            WKT Point string in format: POINT(longitude latitude)
        """
        try:
            # Validate that lon and lat are numeric
            lon_val = float(lon)
            lat_val = float(lat)
            
            # Create WKT Point format: POINT(longitude latitude)
            wkt_point = f"POINT({lon_val} {lat_val})"
            
            return wkt_point
            
        except (ValueError, TypeError) as e:
            logging.warning(f"⚠️ Invalid coordinate values - lon: {lon}, lat: {lat}, error: {e}")
            # Return a default or raise exception based on your requirements
            raise Exception(f"Invalid coordinate values - lon: {lon}, lat: {lat}: {e}")

    def convert_to_records(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert DataFrame to records ready for Milvus insertion"""

        if df is None or df.empty:
            raise Exception("No data available")

        logging.info(f"🔗 Converting {len(df)} records to Milvus format")

        records = []

        for i in range(len(df)):
            try:
                row = df.iloc[i]

                # # Convert gcj02_lat and gcj02_lon to WKT Point format for location field
                # gcj02_location_wkt = self.convert_to_wkt_point(row['gcj02_lon'], row['gcj02_lat'])
                # wgs84_location_wkt = self.convert_to_wkt_point(row['wgs84_lon'], row['wgs84_lat'])

                # Create record with all required fields
                record = {
                    #'id': str(row['id']),
                    'feature': row['vector'],
                    # 'gcj02_location': gcj02_location_wkt,  # WKT Point format from gcj02_lon and gcj02_lat
                    # 'wgs84_location': wgs84_location_wkt,  # WKT Point format from wgs84_lon and wgs84_lat
                    'device_id': row['device_id'],
                    'type_model': row['type_model'],
                    'brand': row['brand'],
                    'hardware': row['hardware'],
                    'software': row['software'],
                    'district': row['district'],
                    'gcj02_lat': row['gcj02_lat'],
                    'gcj02_lon': row['gcj02_lon'],
                    'wgs84_lat': row['wgs84_lat'],
                    'wgs84_lon': row['wgs84_lon'],
                    'geo_ids': row['geo_ids'],
                    'timeline_tags': row['timeline_tags'],
                    'event_id': row['event_id'],
                    'drive': row['drive'],
                    'drive_status': row['drive_status'],
                    'app_id': row['app_id'],
                    'link_info': row['link_info'],
                    'node_info': row['node_info'],
                    'session_id': row['session_id'],
                    'site_id': row['site_id'],
                    'static_mode': row['static_mode'],
                    'timestamp': row['timestamp'],
                    'expert_collected': row['expert_collected'],
                    'sensor_lidar_type': row['sensor_lidar_type'],
                }

                records.append(record)

            except Exception as e:
                # Fail immediately - no tolerance for data issues
                raise Exception(f"Error processing record {i}: {e}")

        logging.info(f"✅ Converted {len(records)} records for insertion")
        return records


class SimplifiedInserter:
    """Simplified data inserter for Milvus"""

    def __init__(self, client: MilvusClient, collection_name: str):
        self.client = client
        self.collection_name = collection_name

    def verify_collection_exists(self) -> bool:
        """Verify that the target collection exists"""
        try:
            if not self.client.has_collection(self.collection_name):
                logging.error(f"❌ Collection '{self.collection_name}' does not exist")
                return False

            return True

        except Exception as e:
            logging.error(f"❌ Error verifying collection: {e}")
            return False

    def insert_data(self, data: List[Dict[str, Any]], batch_size: int = 1000) -> Tuple[bool, int]:
        """Insert data into Milvus collection"""
        if not data:
            logging.warning("⚠️ No data to insert")
            return True, 0

        total_records = len(data)
        total_inserted = 0

        logging.info(f"🚀 Inserting {total_records} records in batches of {batch_size}")

        # Split data into batches
        for batch_num in range(0, total_records, batch_size):
            batch_data = data[batch_num:batch_num + batch_size]

            try:
                # Insert batch
                self.client.insert(
                    collection_name=self.collection_name,
                    data=batch_data
                )

                total_inserted += len(batch_data)
                logging.info(f"✅ Batch {batch_num // batch_size + 1} inserted: {len(batch_data)} records")

                # Brief pause between batches
                # time.sleep(0.1)

            except Exception as e:
                logging.error(f"❌ Critical error: Failed to insert batch {batch_num // batch_size + 1}: {e}")
                logging.error(f"❌ Data insertion failure, stopping processing")
                raise Exception(f"Failed to insert batch {batch_num // batch_size + 1}: {e}")

        success_rate = (total_inserted / total_records) * 100 if total_records > 0 else 0
        logging.info(f"🎯 Insertion complete: {total_inserted}/{total_records} records ({success_rate:.1f}%)")

        return total_inserted == total_records, total_inserted

    def flush_collection(self):
        """Flush the collection to ensure data persistence"""
        try:
            logging.info("💾 Flushing collection...")
            self.client.flush(self.collection_name)
            logging.info("✅ Collection flushed successfully")
        except Exception as e:
            logging.error(f"❌ Failed to flush collection: {e}")
            raise Exception(f"Failed to flush collection: {e}")


def process_single_batch(batch_idx: int, processor: BatchDataProcessor, 
                         inserter: SimplifiedInserter, batch_size: int,
                         total_batches: int) -> Tuple[int, int]:
    """
    Process a single batch (one file) with thread safety
    
    Args:
        batch_idx: Index of the batch to process
        processor: BatchDataProcessor instance
        inserter: SimplifiedInserter instance
        batch_size: Size of each insertion batch
        total_batches: Total number of batches
        
    Returns:
        Tuple of (batch_idx, number of records inserted)
    """
    try:
        logging.info("=" * 60)
        logging.info(f"📖 [Thread] Processing batch {batch_idx + 1}/{total_batches}")
        
        # Read batch data (one file)
        df = processor.read_batch_data(batch_idx)
        
        if df is None or df.empty:
            logging.error(f"❌ No data available for batch {batch_idx + 1}")
            raise Exception(f"No data available for batch {batch_idx + 1}")
        
        # Convert DataFrame to records
        logging.info(f"🔗 [Thread] Converting batch {batch_idx + 1} data... {len(df)} records")
        records = processor.convert_to_records(df)
        
        if not records:
            logging.error(f"❌ No records available for batch {batch_idx + 1}")
            raise Exception(f"No records available for batch {batch_idx + 1}")
        
        # Insert data
        logging.info(f"💾 [Thread] Inserting batch {batch_idx + 1} data... {len(records)} records")
        success, batch_inserted = inserter.insert_data(records, batch_size)
        
        if not success:
            logging.error(f"❌ Batch {batch_idx + 1} insertion failed")
            raise Exception(f"Batch {batch_idx + 1} insertion failed")
        
        # Memory cleanup after each batch
        logging.info(f"🧹 [Thread] Cleaning up memory for batch {batch_idx + 1}")
        del df, records
        gc.collect()
        
        logging.info(f"✅ [Thread] Batch {batch_idx + 1} complete: {batch_inserted} records inserted")
        
        return batch_idx, batch_inserted
        
    except Exception as e:
        logging.error(f"❌ Critical error in batch {batch_idx + 1}: {e}")
        raise Exception(f"Critical error in batch {batch_idx + 1}: {e}")


def main():
    """Main function for simplified horizon data insertion"""

    # Setup logging
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)

    # Parse command line arguments with defaults
    data_dir = '/root/horizon/horizonPoc/data/expanded_dataset_100M'  # sys.argv[1] if len(sys.argv) > 1 else "./data"
    # data_dir = '/Users/yanliang/Downloads/horizonPoc/data'
    # host = 'http://10.104.19.128:19530'
    host = 'https://in01-3e1axxxx28817d.ali-cn-hangzhou.cloud-uat.zilliz.cn:19530'  # sys.argv[2] if len(sys.argv) > 2 else "localhost"

    collection_name = 'horizon_test_collection'  # sys.argv[3] if len(sys.argv) > 3 else "horizon_test_collection"
    batch_size = 2500  # int(sys.argv[4]) if len(sys.argv) > 4 else 2500
    concurrency = 2  # int(sys.argv[5]) if len(sys.argv) > 5 else 2  # Default concurrency: 2

    # Use hardcoded API key from user's settings
    api_key = 'token'

    logging.info("🚀 Starting Local Horizon Data Insertion with Concurrent Batch Processing")
    logging.info(f"  Data Directory: {data_dir}")
    logging.info(f"  Milvus Host: {host}")
    logging.info(f"  Collection: {collection_name}")
    logging.info(f"  Insertion Batch Size: {batch_size}")
    logging.info(f"  Concurrency: {concurrency} threads")

    try:
        # 1. Connect to Milvus
        logging.info("🔌 Connecting to Milvus...")
        if api_key:
            client = MilvusClient(uri=host, token=api_key)
        else:
            if not host.startswith('http'):
                host = f"http://{host}:19530"
            client = MilvusClient(uri=host)

        logging.info(f"✅ Connected to Milvus at {host}")

        # 2. Initialize local data processor
        logging.info("📁 Initializing batch data processor...")
        parquet_reader = LocalParquetReader(data_dir)
        processor = BatchDataProcessor(parquet_reader)

        # 3. Initialize inserter and verify collection
        inserter = SimplifiedInserter(client, collection_name)

        if not inserter.verify_collection_exists():
            logging.error("❌ Target collection verification failed")
            raise Exception("Target collection verification failed")

        # 4. Concurrent batch processing
        start_time = time.time()
        total_inserted = 0
        completed_batches = 0
        failed_batches = []

        logging.info("=" * 80)
        logging.info(f"🔄 Starting concurrent batch processing: {processor.total_batches} batches (1 file per batch)")
        logging.info(f"⚡ Using {concurrency} concurrent threads")

        # Use ThreadPoolExecutor for concurrent processing
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            # Submit all batch processing tasks
            future_to_batch = {
                executor.submit(
                    process_single_batch, 
                    batch_idx, 
                    processor, 
                    inserter, 
                    batch_size, 
                    processor.total_batches
                ): batch_idx 
                for batch_idx in range(processor.total_batches)
            }
            
            # Process completed tasks as they finish
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    # Get result from completed task
                    _, batch_inserted = future.result()
                    total_inserted += batch_inserted
                    completed_batches += 1
                    
                    logging.info(f"📊 Progress: {completed_batches}/{processor.total_batches} batches completed, "
                               f"{total_inserted} total records inserted")
                    
                except Exception as e:
                    failed_batches.append(batch_idx)
                    logging.error(f"❌ Batch {batch_idx + 1} failed: {e}")
                    # Continue processing other batches even if one fails
        
        # Check if any batches failed
        if failed_batches:
            logging.error(f"❌ {len(failed_batches)} batch(es) failed: {[idx + 1 for idx in failed_batches]}")
            raise Exception(f"{len(failed_batches)} batch(es) failed during processing")

        # 5. Final flush
        logging.info("💾 Final collection flush...")
        inserter.flush_collection()

        # 6. Summary
        elapsed_time = time.time() - start_time

        logging.info("=" * 80)
        logging.info("🎉 Batch processing completed successfully!")
        logging.info(f"📊 Summary:")
        logging.info(f"  - Total batches processed: {processor.total_batches}")
        logging.info(f"  - Total records inserted: {total_inserted}")
        logging.info(f"  - Total time: {elapsed_time:.2f}s")
        logging.info(f"  - Average speed: {total_inserted / elapsed_time:.2f} records/s")
        logging.info(f"  - Average records per batch: {total_inserted / processor.total_batches:.0f}")

        # Final memory cleanup
        gc.collect()

    except Exception as e:
        logging.error(f"❌ Local horizon data insertion failed: {e}")
        raise Exception(f"Local horizon data insertion failed: {e}")


if __name__ == '__main__':
    main()
