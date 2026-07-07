#!/usr/bin/env python3
"""
Optimized Horizon Data Insertion for Large Datasets

This script is optimized for large-scale data processing with balanced data density:
- Feature files: 500 files, 200K records/file (~100M total records)
- Scalars files: 100+ files, 1M records/file (high-density data)  
- Location files: single 'location' column with WKT Point format
- Optimized batch ratios: 10:2:2 (feature:scalars:location files per batch)
- Data volume balance: ~2M records per batch from each source
- Memory usage: ~1100MB per batch (900MB feature + 200MB scalars)
- Default insertion batch size: 2500 records

Usage:
    python3 insert_horizon_data_optimized.py <host> <oss_endpoint> <oss_access_key_id> <oss_access_key_secret> <bucket_name> [api_key] [batch_size] [max_batches]

Examples:
    # Process all data with default settings
    python3 insert_horizon_data_optimized.py localhost oss-cn-beijing.aliyuncs.com ACCESS_KEY SECRET bucket_name
    
    # Process with custom batch size and limited batches for testing
    python3 insert_horizon_data_optimized.py localhost oss-cn-beijing.aliyuncs.com ACCESS_KEY SECRET bucket_name None 2500 10
"""

import sys
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import json
import random
import gc  # For garbage collection

# Alibaba OSS SDK
import oss2

# PyMilvus imports
from pymilvus import MilvusClient, DataType

# Setup logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"

class OptimizedOSSReader:
    """Optimized OSS client for batch reading of large parquet files"""
    
    def __init__(self, endpoint: str, access_key_id: str, access_key_secret: str, bucket_name: str):
        self.endpoint = endpoint
        self.access_key_id = access_key_id
        self.access_key_secret = access_key_secret
        self.bucket_name = bucket_name
        
        # Create OSS client
        auth = oss2.Auth(access_key_id, access_key_secret)
        self.bucket = oss2.Bucket(auth, endpoint, bucket_name)
        
        logging.info(f"✅ Connected to OSS bucket: {bucket_name}")
    
    def list_parquet_files(self, prefix: str) -> List[str]:
        """List all parquet files under a prefix"""
        parquet_files = []
        
        for obj in oss2.ObjectIterator(self.bucket, prefix=prefix):
            if obj.key.endswith('.parquet'):
                parquet_files.append(obj.key)
        
        logging.info(f"📁 Found {len(parquet_files)} parquet files in {prefix}")
        return parquet_files
    
    def read_parquet_file(self, file_key: str) -> pd.DataFrame:
        """Read a parquet file from OSS into pandas DataFrame"""
        try:
            # Download file to memory
            obj = self.bucket.get_object(file_key)
            data = obj.read()
            
            # Create temporary file-like object
            import io
            parquet_buffer = io.BytesIO(data)
            
            # Read parquet
            df = pd.read_parquet(parquet_buffer)
            
            logging.info(f"📄 Read {file_key}: {len(df)} records, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logging.error(f"❌ Failed to read {file_key}: {e}")
            raise

class BatchDataProcessor:
    """Optimized data processor for large-scale batch processing"""
    
    def __init__(self, oss_reader: OptimizedOSSReader):
        self.oss_reader = oss_reader
        
        # Optimized batch sizes based on actual data density
        # Feature data: 200K records per file → 10 files = 2M records per batch
        self.feature_files_per_batch = 5  
        
        # Scalars data: 1M records per file → 2 files = 2M records per batch (matching feature data)
        self.scalars_files_per_batch = 1  
        
        # Location data: assume similar to feature data, use 2 files per batch
        self.location_files_per_batch = 1
        
        # Cache file lists
        self.feature_files = []
        self.location_files = []
        self.scalars_files = []
        
        self._initialize_file_lists()
    
    def _initialize_file_lists(self):
        """Initialize and cache file lists for all data sources"""
        logging.info("📋 Initializing file lists for batch processing...")
        
        self.feature_files = self.oss_reader.list_parquet_files("test/")
        self.location_files = self.oss_reader.list_parquet_files("nyc-taxi/")
        self.scalars_files = self.oss_reader.list_parquet_files("other_scalars/")
        
        # Calculate total batches
        self.total_feature_batches = (len(self.feature_files) + self.feature_files_per_batch - 1) // self.feature_files_per_batch
        
        logging.info(f"📊 Optimized batch planning (data-density aware):")
        logging.info(f"  - Feature files: {len(self.feature_files)} files → {self.total_feature_batches} batches (10 files/batch = ~2M records)")
        logging.info(f"  - Scalars files: {len(self.scalars_files)} files → {self.scalars_files_per_batch} files/batch (2 files/batch = ~2M records)")
        logging.info(f"  - Location files: {len(self.location_files)} files → {self.location_files_per_batch} files/batch")
    
    def get_batch_info(self) -> Dict[str, Any]:
        """Get batch processing information"""
        return {
            'feature_files': len(self.feature_files),
            'feature_batches': self.total_feature_batches,
            'location_files': len(self.location_files),
            'scalars_files': len(self.scalars_files),
            'feature_files_per_batch': self.feature_files_per_batch,
            'scalars_files_per_batch': self.scalars_files_per_batch,
            'location_files_per_batch': self.location_files_per_batch
        }
    
    def parse_wkt_point(self, wkt_string: str) -> Optional[str]:
        """Parse WKT Point string and validate format"""
        if not isinstance(wkt_string, str):
            return None
        
        wkt_clean = wkt_string.strip().upper()
        
        if wkt_clean.startswith('POINT') and '(' in wkt_clean and ')' in wkt_clean:
            try:
                coords_part = wkt_clean.split('(')[1].split(')')[0].strip()
                coords = coords_part.split()
                
                if len(coords) >= 2:
                    float(coords[0])  # longitude/x
                    float(coords[1])  # latitude/y
                    return wkt_string.strip()
                
            except (ValueError, IndexError):
                logging.warning(f"⚠️ Invalid WKT Point format: {wkt_string}")
                return None
        
        return None
    
    def read_batch_data(self, batch_idx: int) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Read data for a specific batch from all sources
        
        Returns:
            Tuple of (feature_df, location_df, scalars_df) for the batch
        """
        logging.info(f"📖 Reading data for batch {batch_idx + 1}")
        
        # Read feature data batch
        feature_df = self._read_feature_batch(batch_idx)
        
        # Read location data batch (cycle through files if needed)
        location_df = self._read_location_batch(batch_idx)
        
        # Read scalars data batch (cycle through files if needed)
        scalars_df = self._read_scalars_batch(batch_idx)
        
        return feature_df, location_df, scalars_df
    
    def _read_feature_batch(self, batch_idx: int) -> Optional[pd.DataFrame]:
        """Read feature data for a specific batch"""
        if batch_idx >= self.total_feature_batches:
            return None
        
        start_idx = batch_idx * self.feature_files_per_batch
        end_idx = min((batch_idx + 1) * self.feature_files_per_batch, len(self.feature_files))
        batch_files = self.feature_files[start_idx:end_idx]
        
        logging.info(f"📊 Reading feature batch {batch_idx + 1}: {len(batch_files)} files")
        
        dataframes = []
        for file_key in batch_files:
            try:
                df = self.oss_reader.read_parquet_file(file_key)
                dataframes.append(df)
            except Exception as e:
                logging.error(f"❌ Failed to read feature file {file_key}: {e}")
                raise Exception(f"Failed to read feature file {file_key}: {e}")
        
        if not dataframes:
            logging.error(f"❌ No feature files could be read for batch {batch_idx + 1}")
            raise Exception(f"No feature files could be read for batch {batch_idx + 1}")
        
        # Combine and validate
        batch_df = pd.concat(dataframes, ignore_index=True)
        
        # Validate required columns
        if 'id' not in batch_df.columns or 'feature' not in batch_df.columns:
            logging.warning(f"⚠️ Missing required columns id and feature in feature batch {batch_idx + 1}")
            logging.info(f"📋 Available columns: {list(batch_df.columns)}")
            raise Exception(f"Missing required columns id and feature in feature batch {batch_idx + 1}")
        
        logging.info(f"✅ Feature batch {batch_idx + 1}: {len(batch_df)} records")
        
        # Free memory
        del dataframes
        gc.collect()
        
        return batch_df
    
    def _read_location_batch(self, batch_idx: int) -> Optional[pd.DataFrame]:
        """Read location data for a specific batch"""
        if not self.location_files:
            return None
        
        # Calculate optimal batch files using the new location_files_per_batch setting
        if len(self.location_files) <= self.total_feature_batches:
            # Cycle through location files if we have fewer files than feature batches
            file_idx = batch_idx % len(self.location_files)
            batch_files = [self.location_files[file_idx]]
        else:
            # Use configured files per batch
            start_idx = batch_idx * self.location_files_per_batch
            end_idx = min(start_idx + self.location_files_per_batch, len(self.location_files))
            if start_idx >= len(self.location_files):
                # Cycle back if we exceed available files
                start_idx = batch_idx % len(self.location_files)
                batch_files = [self.location_files[start_idx]]
            else:
                batch_files = self.location_files[start_idx:end_idx]
        
        logging.info(f"📍 Reading location batch {batch_idx + 1}: {len(batch_files)} files (optimized for balanced data volume)")
        
        dataframes = []
        for file_key in batch_files:
            try:
                df = self.oss_reader.read_parquet_file(file_key)
                dataframes.append(df)
            except Exception as e:
                logging.error(f"❌ Failed to read location file {file_key}: {e}")
                raise Exception(f"Failed to read location file {file_key}: {e}")
        
        if not dataframes:
            logging.error(f"❌ No location files could be read for batch {batch_idx + 1}")
            raise Exception(f"No location files could be read for batch {batch_idx + 1}")
        
        batch_df = pd.concat(dataframes, ignore_index=True)
        
        # Check for location column (simplified - only check 'location' column)
        if 'location' in batch_df.columns:
            logging.info(f"✅ Location batch {batch_idx + 1}: {len(batch_df)} records, location column found")
        else:
            logging.error(f"❌ No 'location' column found in location batch {batch_idx + 1}")
            logging.info(f"📋 Available columns: {list(batch_df.columns)}")
        
        # Free memory
        del dataframes
        gc.collect()
        
        return batch_df
    
    def _read_scalars_batch(self, batch_idx: int) -> Optional[pd.DataFrame]:
        """Read scalars data for a specific batch"""
        if not self.scalars_files:
            return None
        
        # Calculate optimal batch files using the new scalars_files_per_batch setting
        # Each scalars file has ~1M records, so 2 files per batch = ~2M records (matching feature data)
        if len(self.scalars_files) <= self.total_feature_batches:
            # Cycle through scalars files if we have fewer files than feature batches
            file_idx = batch_idx % len(self.scalars_files)
            batch_files = [self.scalars_files[file_idx]]
        else:
            # Use optimized files per batch (2 files = ~2M records)
            start_idx = batch_idx * self.scalars_files_per_batch
            end_idx = min(start_idx + self.scalars_files_per_batch, len(self.scalars_files))
            if start_idx >= len(self.scalars_files):
                # Cycle back if we exceed available files
                start_idx = batch_idx % len(self.scalars_files)
                batch_files = [self.scalars_files[start_idx]]
            else:
                batch_files = self.scalars_files[start_idx:end_idx]
        
        logging.info(f"📊 Reading scalars batch {batch_idx + 1}: {len(batch_files)} files (~{len(batch_files)}M records - density optimized)")
        
        dataframes = []
        for file_key in batch_files:
            try:
                df = self.oss_reader.read_parquet_file(file_key)
                dataframes.append(df)
            except Exception as e:
                logging.error(f"❌ Failed to read scalars file {file_key}: {e}")
                raise Exception(f"Failed to read scalars file {file_key}: {e}")
        
        if not dataframes:
            logging.error(f"❌ No scalars files could be read for batch {batch_idx + 1}")
            raise Exception(f"No scalars files could be read for batch {batch_idx + 1}")
        
        batch_df = pd.concat(dataframes, ignore_index=True)
        logging.info(f"✅ Scalars batch {batch_idx + 1}: {len(batch_df)} records")
        
        # Free memory
        del dataframes
        gc.collect()
        
        return batch_df
    
    def merge_batch_data(self, feature_df: Optional[pd.DataFrame], location_df: Optional[pd.DataFrame], 
                        scalars_df: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
        """Merge batch data from all sources into records ready for Milvus insertion"""
        
        if feature_df is None or feature_df.empty:
            logging.error("❌ No feature data for this batch")
            raise Exception("No feature data for this batch")
            return []
        
        logging.info(f"🔗 Merging batch data: {len(feature_df)} feature records")
        
        merged_data = []
        failed_records = 0
        
        for i in range(len(feature_df)):
            try:
                # Get feature data
                feature_row = feature_df.iloc[i]
                
                # Create base record
                record = {
                    'id': str(feature_row.get('id', f'id_{i}')),
                    'feature': feature_row.get('feature', self._generate_mock_vector()),
                    'timestamp': int(time.time() * 1000),
                    'url': f'https://example.com/data/{i}',
                    'device_id': f'DV{random.randint(100, 999)}'
                }
                
                # Add location from location data (simplified - only check 'location' column)
                if location_df is not None and not location_df.empty:
                    location_row = location_df.iloc[i % len(location_df)]
                    
                    # Directly use 'location' column (WKT Point format)
                    if 'location' in location_row and pd.notna(location_row['location']):
                        wkt_point = self.parse_wkt_point(str(location_row['location']))
                        if wkt_point:
                            record['location'] = wkt_point
                        else:
                            # Invalid WKT, generate mock location
                            mock_lon = -74.0 + random.uniform(-0.2, 0.2)
                            mock_lat = 40.7 + random.uniform(-0.2, 0.2)
                            record['location'] = f"POINT ({mock_lon} {mock_lat})"
                    else:
                        # No location column or null value, generate mock location
                        mock_lon = -74.0 + random.uniform(-0.2, 0.2)
                        mock_lat = 40.7 + random.uniform(-0.2, 0.2)
                        record['location'] = f"POINT ({mock_lon} {mock_lat})"
                else:
                    # No location data, generate mock location
                    mock_lon = -74.0 + random.uniform(-0.2, 0.2)
                    mock_lat = 40.7 + random.uniform(-0.2, 0.2)
                    record['location'] = f"POINT ({mock_lon} {mock_lat})"
                
                # Add scalars fields (exclude 'location' column, include all others)
                if scalars_df is not None and not scalars_df.empty:
                    scalars_row = scalars_df.iloc[i % len(scalars_df)]
                    
                    for col, value in scalars_row.items():
                        # Skip 'location' column and already existing fields, include all other scalar fields
                        if col != 'location' and col not in record and pd.notna(value):
                            # Convert numpy types to Python types
                            if isinstance(value, np.integer):
                                record[col] = int(value)
                            elif isinstance(value, np.floating):
                                record[col] = float(value)
                            elif isinstance(value, np.bool_):
                                record[col] = bool(value)
                            else:
                                record[col] = str(value)
                
                merged_data.append(record)
                
            except Exception as e:
                logging.error(f"❌ Error processing record {i}: {e}")
                failed_records += 1
                raise Exception(f"Error processing record {i}: {e}")
        
        # Check failure rate
        total_records = len(feature_df)
        if failed_records > 0:
            failure_rate = (failed_records / total_records) * 100
            logging.info(f"📊 Record processing: {failed_records}/{total_records} failed ({failure_rate:.1f}%)")
            
            # Exit if failure rate is too high (>50%)
            if failure_rate > 50:
                logging.error(f"❌ Critical error: Record failure rate too high ({failure_rate:.1f}%)")
                logging.error(f"❌ Data quality issues, stopping processing")
                raise Exception(f"Record failure rate too high ({failure_rate:.1f}%)")
        
        logging.info(f"✅ Merged {len(merged_data)} records for insertion")
        return merged_data
    
    def _generate_mock_vector(self, dim: int = 768) -> List[float]:
        """Generate a mock feature vector"""
        return np.random.random(dim).astype(np.float32).tolist()

class OptimizedInserter:
    """Optimized data inserter for large-scale batch processing"""
    
    def __init__(self, client: MilvusClient, collection_name: str = "horizon_test_collection"):
        self.client = client
        self.collection_name = collection_name
    
    def verify_collection_exists(self) -> bool:
        """Verify that the target collection exists"""
        if not self.client.has_collection(self.collection_name):
            logging.error(f"❌ Collection '{self.collection_name}' does not exist")
            return False
        
        # collection_info = self.client.describe_collection(self.collection_name)
        # logging.info(f"✅ Target collection: {self.collection_name}")
        # logging.info(f"📋 Fields: {len(collection_info.get('fields', []))}")
        
        return True
    
    def insert_batch_data(self, data: List[Dict[str, Any]], batch_size: int = 2500) -> Tuple[bool, int]:
        """Insert data with optimized batch processing"""
        if not data:
            return True, 0
        
        total_records = len(data)
        total_inserted = 0
        
        logging.info(f"🚀 Inserting {total_records} records in batches of {batch_size}")
        
        for i in range(0, total_records, batch_size):
            batch_data = data[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_records + batch_size - 1) // batch_size
            
            try:
                logging.info(f"📦 Inserting batch {batch_num}/{total_batches} ({len(batch_data)} records)...")
                
                self.client.insert(
                    collection_name=self.collection_name,
                    data=batch_data
                )
                
                total_inserted += len(batch_data)
                logging.info(f"✅ Batch {batch_num} inserted successfully")
                
                # Brief pause between batches
                time.sleep(0.1)
                
            except Exception as e:
                logging.error(f"❌ Critical error: Failed to insert batch {batch_num}: {e}")
                logging.error(f"❌ Data insertion failure, stopping processing")
                raise Exception(f"Failed to insert batch {batch_num}: {e}")
        
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

def main():
    """Main function for optimized large-scale data insertion"""
    
    # Parse command line arguments
    try:
        host = 'https://in01-mock.zilliz.cn:19530'       # sys.argv[1]
        oss_endpoint = 'oss-cn-hangzhou-internal.aliyuncs.com'       # sys.argv[2]
        oss_access_key_id = 'mock-tke-access-key-id'       # sys.argv[3]
        oss_access_key_secret = 'mock-tke-access-key-secret'       # sys.argv[4]
        bucket_name = 'qa-mock-bucket-tmp'       # sys.argv[5]
        api_key = 'mock-tke-api-key' # sys.argv[6] if len(sys.argv) > 6 and sys.argv[6].upper() != "NONE" else None
        batch_size = 2500 # int(sys.argv[7]) if len(sys.argv) > 7 else 2500
        max_batches = 2 # int(sys.argv[8]) if len(sys.argv) > 8 else None
        
    except (IndexError, ValueError):
        print("Usage: python3 insert_horizon_data_optimized.py <host> <oss_endpoint> <oss_access_key_id> <oss_access_key_secret> <bucket_name> [api_key] [batch_size] [max_batches]")
        print("\nParameters:")
        print("  host                 : Milvus server host or URI")
        print("  oss_endpoint         : OSS endpoint (e.g., oss-cn-beijing.aliyuncs.com)")
        print("  oss_access_key_id    : OSS access key ID")
        print("  oss_access_key_secret: OSS access key secret")
        print("  bucket_name          : OSS bucket name")
        print("  api_key              : Milvus API key (optional, use 'None' for local)")
        print("  batch_size           : Insert batch size (default: 2500)")
        print("  max_batches          : Maximum batches to process (optional, for testing)")
        print("\nOptimized for large datasets:")
        print("  - 500 test parquet files (~100M records)")
        print("  - Memory-efficient batch processing")
        print("  - Automatic file cycling for smaller datasets")
        sys.exit(1)
    
    # Setup logging
    log_filename = f"/tmp/insert_horizon_optimized_{int(time.time())}.log"
    file_handler = logging.FileHandler(filename=log_filename)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    
    collection_name = "horizon_test_collection"
    port = 19530
    
    logging.info("🚀 Starting Optimized Horizon Data Insertion")
    logging.info(f"  Milvus Host: {host}")
    logging.info(f"  OSS Endpoint: {oss_endpoint}")
    logging.info(f"  OSS Bucket: {bucket_name}")
    logging.info(f"  Collection: {collection_name}")
    logging.info(f"  Batch Size: {batch_size}")
    logging.info(f"  Max Batches: {max_batches or 'No limit'}")
    logging.info(f"  API Key: {'***' if api_key else 'None (local)'}")
    
    try:
        # 1. Connect to Milvus
        logging.info("🔌 Connecting to Milvus...")
        if api_key:
            client = MilvusClient(uri=host, token=api_key)
        else:
            if not host.startswith('http'):
                host = f"http://{host}:{port}"
            client = MilvusClient(uri=host)
        
        logging.info(f"✅ Connected to Milvus at {host}")
        
        # 2. Connect to OSS and initialize processor
        logging.info("🔌 Connecting to Alibaba OSS...")
        oss_reader = OptimizedOSSReader(oss_endpoint, oss_access_key_id, oss_access_key_secret, bucket_name)
        processor = BatchDataProcessor(oss_reader)
        
        # 3. Get batch information
        batch_info = processor.get_batch_info()
        logging.info(f"📊 Batch processing plan:")
        logging.info(f"  - Total feature files: {batch_info['feature_files']}")
        logging.info(f"  - Total feature batches: {batch_info['feature_batches']}")
        logging.info(f"  - Files per batch: Feature={batch_info['feature_files_per_batch']}, Scalars={batch_info['scalars_files_per_batch']}, Location={batch_info['location_files_per_batch']}")
        logging.info(f"  - Memory per batch: ~{batch_info['feature_files_per_batch'] * 900}MB (feature) + ~{batch_info['scalars_files_per_batch'] * 200}MB (scalars)")
        
        # 4. Initialize inserter and verify collection
        inserter = OptimizedInserter(client, collection_name)
        
        if not inserter.verify_collection_exists():
            logging.error("❌ Target collection verification failed")
            raise Exception("Target collection verification failed")
        
        # 5. Determine batches to process
        total_batches = batch_info['feature_batches']
        if max_batches:
            total_batches = min(total_batches, max_batches)
            logging.info(f"📝 Limited to {total_batches} batches for testing")
        
        # 6. Process batches
        total_inserted = 0
        successful_batches = 0
        
        start_time = time.time()
        
        for batch_idx in range(total_batches):
            logging.info("=" * 80)
            logging.info(f"📦 Processing batch {batch_idx + 1}/{total_batches}")
            batch_start_time = time.time()
            
            try:
                # Read batch data
                feature_df, location_df, scalars_df = processor.read_batch_data(batch_idx)
                
                if feature_df is None or feature_df.empty:
                    logging.warning(f"⚠️ No feature data for batch {batch_idx + 1}, Stop insertion")
                    raise Exception(f"No feature data for batch {batch_idx + 1}")
                
                if location_df is None or location_df.empty:
                    logging.warning(f"⚠️ No location data for batch {batch_idx + 1}, Stop insertion")
                    raise Exception(f"No location data for batch {batch_idx + 1}")
                
                if scalars_df is None or scalars_df.empty:
                    logging.warning(f"⚠️ No scalars data for batch {batch_idx + 1}, Stop insertion")
                    raise Exception(f"No scalars data for batch {batch_idx + 1}")
                
                # Merge data sources
                merged_data = processor.merge_batch_data(feature_df, location_df, scalars_df)
                
                if not merged_data:
                    logging.warning(f"⚠️ No merged data for batch {batch_idx + 1}, Stop insertion")
                    raise Exception(f"No merged data for batch {batch_idx + 1}")
                
                # Insert batch
                success, batch_inserted = inserter.insert_batch_data(merged_data, batch_size)
                total_inserted += batch_inserted
                successful_batches += 1
                
                batch_duration = time.time() - batch_start_time
                logging.info(f"✅ Batch {batch_idx + 1} completed in {batch_duration:.1f}s: {batch_inserted} records")
                
                # Clear memory
                del feature_df, location_df, scalars_df, merged_data
                gc.collect()
                
            except Exception as e:
                logging.error(f"❌ Error processing batch {batch_idx + 1}: {e}")
                raise Exception(f"Error processing batch {batch_idx + 1}: {e}")
        
        # 7. Flush collection
        logging.info("💾 Flushing collection after all batches...")
        inserter.flush_collection()
        
        # 8. Final summary
        total_duration = time.time() - start_time
        
        logging.info("=" * 80)
        logging.info("🎯 OPTIMIZED HORIZON DATA INSERTION SUMMARY")
        logging.info(f"  ✅ Total Duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
        logging.info(f"  ✅ Batches Processed: {successful_batches}/{total_batches}")
        logging.info(f"  ✅ Total Records Inserted: {total_inserted}")
        logging.info(f"  ✅ Average Records/Second: {total_inserted/total_duration:.1f}")
        logging.info(f"  ✅ Batch Success Rate: {successful_batches/total_batches*100:.1f}%")
        logging.info(f"  ✅ Memory Optimization: Processed ~{successful_batches * batch_info['files_per_batch'] * 900}MB in batches")
        logging.info(f"  📁 Log file: {log_filename}")
        logging.info("=" * 80)
        
        if successful_batches == total_batches:
            logging.info("🎉 Optimized data insertion completed successfully!")
        else:
            logging.warning(f"⚠️ Completed with some batch failures: {successful_batches}/{total_batches}")
        
    except Exception as e:
        logging.error(f"❌ Optimized data insertion failed: {e}")
        raise Exception(f"Optimized data insertion failed: {e}")

if __name__ == '__main__':
    main()
