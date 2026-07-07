#!/usr/bin/env python3
"""
Local Horizon Data Insertion with Batch Processing

This script reads parquet files from local directories and inserts them into Milvus collection
with batch processing to handle large datasets efficiently.

Directory structure expected:
- feature/: Contains parquet files with 'id' and 'feature' columns (500 files, ~900MB each)
- location/: Contains parquet files with 'location' column (WKT Point format)
- scalars/: Contains parquet files with additional scalar fields

Features:
- Batch processing for large datasets (500+ files, 100M+ records)
- Memory-efficient processing with configurable batch sizes
- Strict validation - no mock data generation
- Fail-fast error handling

Usage:
    python3 insert_horizon_local.py [data_dir] [host] [collection_name] [batch_size] [feature_files_per_batch]

Examples:
    # Use default settings
    python3 insert_horizon_local.py
    
    # Custom configuration for large datasets
    python3 insert_horizon_local.py ./data localhost horizon_test_collection 2500 5
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
import json
import random
import gc

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
    
    def list_parquet_files(self, subdir: str) -> List[str]:
        """List all parquet files in a subdirectory"""
        pattern = os.path.join(self.data_dir, subdir, "*.parquet")
        files = glob.glob(pattern)
        logging.info(f"📋 Found {len(files)} parquet files in {subdir}/")
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
    """Coordinated batch data processor for large-scale local file processing"""
    
    def __init__(self, parquet_reader: LocalParquetReader, feature_files_per_batch: int = 5):
        self.parquet_reader = parquet_reader
        
        # Coordinated batch processing configuration
        # Target: 1M records per batch from each data source
        # Feature files: 5 files × 200K records = 1M records per batch
        # Location files: 1 file × 1M records = 1M records per batch  
        # Scalars files: 1 file × 1M records = 1M records per batch
        self.feature_files_per_batch = feature_files_per_batch
        self.location_files_per_batch = 1
        self.scalars_files_per_batch = 1
        
        # Initialize file lists
        self.feature_files = []
        self.location_files = []
        self.scalars_files = []
        
        self._initialize_file_lists()
    
    def _initialize_file_lists(self):
        """Initialize file lists and calculate coordinated batch processing plan"""
        logging.info("📋 Initializing coordinated batch processing...")
        
        # Get all files (no limits for production datasets)
        self.feature_files = self.parquet_reader.list_parquet_files("feature")
        self.location_files = self.parquet_reader.list_parquet_files("location") 
        self.scalars_files = self.parquet_reader.list_parquet_files("scalars")
        
        # Calculate total batches based on feature files (primary driver)
        self.total_batches = (len(self.feature_files) + self.feature_files_per_batch - 1) // self.feature_files_per_batch
        
        # Calculate required files for each data source
        total_location_files_needed = self.total_batches * self.location_files_per_batch
        total_scalars_files_needed = self.total_batches * self.scalars_files_per_batch
        
        logging.info(f"📊 Coordinated batch processing plan:")
        logging.info(f"  - Total batches: {self.total_batches}")
        logging.info(f"  - Feature: {len(self.feature_files)} files → {self.feature_files_per_batch} files/batch")
        logging.info(f"  - Location: {len(self.location_files)} files → {self.location_files_per_batch} file/batch (need {total_location_files_needed} total)")
        logging.info(f"  - Scalars: {len(self.scalars_files)} files → {self.scalars_files_per_batch} file/batch (need {total_scalars_files_needed} total)")
        logging.info(f"  - Target records per batch: ~1M from each source")
        logging.info(f"  - Memory per batch: ~{self.feature_files_per_batch * 900 + 100 + 100}MB")
        
        # Validation - ensure we have sufficient files for coordinated processing
        if not self.feature_files:
            logging.error("❌ No feature files found - cannot proceed")
            raise Exception("No feature files found")
            
        if len(self.location_files) < total_location_files_needed:
            logging.warning(f"⚠️ Location files: have {len(self.location_files)}, need {total_location_files_needed} - will cycle through available files")
            
        if len(self.scalars_files) < total_scalars_files_needed:
            logging.warning(f"⚠️ Scalars files: have {len(self.scalars_files)}, need {total_scalars_files_needed} - will cycle through available files")
            
        if not self.location_files:
            logging.error("❌ No location files found - cannot proceed")
            raise Exception("No location files found")
            
        if not self.scalars_files:
            logging.error("❌ No scalars files found - cannot proceed")  
            raise Exception("No scalars files found")
    
    def parse_wkt_point(self, wkt_string: str) -> Optional[str]:
        """Parse WKT Point string and validate format"""
        if not isinstance(wkt_string, str):
            return None
        
        wkt_clean = wkt_string.strip().upper()
        
        if wkt_clean.startswith('POINT') and '(' in wkt_clean and ')' in wkt_clean:
            try:
                coords_part = wkt_clean.split('(')[1].split(')')[0].strip()
                coords = coords_part.split()
                if len(coords) == 2:
                    float(coords[0])  # longitude/x
                    float(coords[1])  # latitude/y
                    return wkt_string.strip()
                
            except (ValueError, IndexError):
                logging.warning(f"⚠️ Invalid WKT Point format: {wkt_string}")
                return None
        
        return None
    
    def read_batch_data(self, batch_idx: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Read coordinated data for a specific batch"""
        
        logging.info(f"📖 Reading coordinated data for batch {batch_idx + 1}/{self.total_batches}")
        
        # Read feature data batch
        feature_df = self._read_feature_batch(batch_idx)
        
        # Read location data batch (coordinated)
        location_df = self._read_location_batch(batch_idx)
        
        # Read scalars data batch (coordinated)
        scalars_df = self._read_scalars_batch(batch_idx)
        
        return feature_df, location_df, scalars_df
    
    def _read_feature_batch(self, batch_idx: int) -> pd.DataFrame:
        """Read feature data for a specific batch (only id and feature fields)"""
        if batch_idx >= self.total_batches:
            raise Exception(f"Batch index {batch_idx} out of range (max: {self.total_batches - 1})")
        
        start_idx = batch_idx * self.feature_files_per_batch
        end_idx = min((batch_idx + 1) * self.feature_files_per_batch, len(self.feature_files))
        batch_files = self.feature_files[start_idx:end_idx]
        
        logging.info(f"📊 Reading feature batch {batch_idx + 1}: {len(batch_files)} files")
        
        dataframes = []
        for file_path in batch_files:
            try:
                df = self.parquet_reader.read_parquet_file(file_path)
                
                # Field validation and filtering - ONLY id and feature
                if 'id' not in df.columns or 'feature' not in df.columns:
                    logging.error(f"❌ Missing required columns id and feature in {file_path}")
                    logging.info(f"📋 Available columns: {list(df.columns)}")
                    raise Exception(f"Missing required columns id and feature in {file_path}")
                
                # Keep only id and feature fields
                filtered_df = df[['id', 'feature']].copy()
                dataframes.append(filtered_df)
                
            except Exception as e:
                logging.error(f"❌ Failed to read feature file {file_path}: {e}")
                raise Exception(f"Failed to read feature file {file_path}: {e}")
        
        if not dataframes:
            logging.error(f"❌ No feature files could be read for batch {batch_idx + 1}")
            raise Exception(f"No feature files could be read for batch {batch_idx + 1}")
        
        # Combine dataframes
        batch_df = pd.concat(dataframes, ignore_index=True)
        logging.info(f"✅ Feature batch {batch_idx + 1}: {len(batch_df)} records (id, feature fields only)")
        
        # Free memory
        del dataframes
        gc.collect()
        
        return batch_df
    
    def _read_location_batch(self, batch_idx: int) -> pd.DataFrame:
        """Read location data for a specific batch (only location field)"""
        if batch_idx >= self.total_batches:
            raise Exception(f"Batch index {batch_idx} out of range (max: {self.total_batches - 1})")
        
        # Calculate which location files to read (with cycling)
        start_idx = (batch_idx * self.location_files_per_batch) % len(self.location_files)
        batch_files = []
        for i in range(self.location_files_per_batch):
            file_idx = (start_idx + i) % len(self.location_files)
            batch_files.append(self.location_files[file_idx])
        
        logging.info(f"📍 Reading location batch {batch_idx + 1}: {len(batch_files)} files")
        
        dataframes = []
        for file_path in batch_files:
            try:
                df = self.parquet_reader.read_parquet_file(file_path)
                
                # Field validation and filtering - ONLY location
                if 'location' not in df.columns:
                    logging.error(f"❌ Missing required column 'location' in {file_path}")
                    logging.info(f"📋 Available columns: {list(df.columns)}")
                    raise Exception(f"Missing required column 'location' in {file_path}")
                
                # Keep only location field
                filtered_df = df[['location']].copy()
                dataframes.append(filtered_df)
                
            except Exception as e:
                logging.error(f"❌ Failed to read location file {file_path}: {e}")
                raise Exception(f"Failed to read location file {file_path}: {e}")
        
        if not dataframes:
            logging.error(f"❌ No location files could be read for batch {batch_idx + 1}")
            raise Exception(f"No location files could be read for batch {batch_idx + 1}")
        
        # Combine dataframes
        batch_df = pd.concat(dataframes, ignore_index=True)
        logging.info(f"✅ Location batch {batch_idx + 1}: {len(batch_df)} records (location field only)")
        
        # Free memory
        del dataframes
        gc.collect()
        
        return batch_df
    
    def _read_scalars_batch(self, batch_idx: int) -> pd.DataFrame:
        """Read scalars data for a specific batch (all fields except location)"""
        if batch_idx >= self.total_batches:
            raise Exception(f"Batch index {batch_idx} out of range (max: {self.total_batches - 1})")
        
        # Calculate which scalars files to read (with cycling)
        start_idx = (batch_idx * self.scalars_files_per_batch) % len(self.scalars_files)
        batch_files = []
        for i in range(self.scalars_files_per_batch):
            file_idx = (start_idx + i) % len(self.scalars_files)
            batch_files.append(self.scalars_files[file_idx])
        
        logging.info(f"📊 Reading scalars batch {batch_idx + 1}: {len(batch_files)} files")
        
        dataframes = []
        for file_path in batch_files:
            try:
                df = self.parquet_reader.read_parquet_file(file_path)
                
                # Field filtering - EXCLUDE location field, keep all others
                filtered_columns = [col for col in df.columns if col != 'location']
                if not filtered_columns:
                    logging.error(f"❌ No valid scalars columns found in {file_path} (only location column present)")
                    raise Exception(f"No valid scalars columns found in {file_path}")
                
                # Keep all fields except location
                filtered_df = df[filtered_columns].copy()
                dataframes.append(filtered_df)
                
                logging.info(f"📊 Scalars file {file_path}: {len(df)} records, {len(filtered_columns)} scalar fields")
                
            except Exception as e:
                logging.error(f"❌ Failed to read scalars file {file_path}: {e}")
                raise Exception(f"Failed to read scalars file {file_path}: {e}")
        
        if not dataframes:
            logging.error(f"❌ No scalars files could be read for batch {batch_idx + 1}")
            raise Exception(f"No scalars files could be read for batch {batch_idx + 1}")
        
        # Combine dataframes
        batch_df = pd.concat(dataframes, ignore_index=True)
        scalar_columns = [col for col in batch_df.columns if col != 'location']
        logging.info(f"✅ Scalars batch {batch_idx + 1}: {len(batch_df)} records ({len(scalar_columns)} scalar fields, location excluded)")
        
        # Free memory
        del dataframes
        gc.collect()
        
        return batch_df
    
    def merge_data_sources(self, feature_df: pd.DataFrame, location_df: pd.DataFrame, 
                          scalars_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Merge data from all sources into records ready for Milvus insertion"""
        
        if feature_df is None or feature_df.empty:
            raise Exception("No feature data available")
        
        logging.info(f"🔗 Merging data: {len(feature_df)} feature records")
        
        merged_data = []
        failed_records = 0
        
        for i in range(len(feature_df)):
            try:
                # Get feature data
                feature_row = feature_df.iloc[i]
                # logging.info(f"feature_row[i]: {feature_row}")
                
                # Create base record with only essential feature fields
                record = {
                    'id': str(feature_row['id']),
                    'feature': feature_row['feature']
                }
                
                # Add location from location data - STRICT: no mock data
                location_row = location_df.iloc[i % len(location_df)]
                # logging.info(f"location_row[i]: {location_row}")
                
                record['location'] = str(location_row['location'])
                
                # Add scalars fields (location already filtered out during batch read)
                scalars_row = scalars_df.iloc[i % len(scalars_df)]
                # logging.info(f"scalars_row[i]: {scalars_row}")
                
                record['timestamp'] = scalars_row['timestamp']
                record['url'] = scalars_row['url']
                record['device_id'] = scalars_row['device_id']
                record['expert_collected'] = scalars_row['expert_collected']
                record['sensor_lidar_type'] = scalars_row['sensor_lidar_type']
                record['p_url'] = scalars_row['p_url']
                
                merged_data.append(record)
                
            except Exception as e:
                failed_records += 1
                # Fail immediately - no tolerance for data issues
                raise Exception(f"Error processing record {i}: {e}")
        
        logging.info(f"✅ Merged {len(merged_data)} records for insertion")
        return merged_data
    

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

def main():
    """Main function for simplified horizon data insertion"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)
    
    # Parse command line arguments with defaults
    data_dir = '/root/test/data'                                                        #sys.argv[1] if len(sys.argv) > 1 else "./data"
    host = 'https://localhost:19530'     #sys.argv[2] if len(sys.argv) > 2 else "localhost"
    collection_name = 'horizon_test_collection'                                         #sys.argv[3] if len(sys.argv) > 3 else "horizon_test_collection"
    batch_size = 2500                                                                   #int(sys.argv[4]) if len(sys.argv) > 4 else 2500
    feature_files_per_batch = 5                                                         #int(sys.argv[5]) if len(sys.argv) > 5 else 5
    
    # Use hardcoded API key from user's settings
    api_key = 'mock-tke-api-key'
    
    logging.info("🚀 Starting Local Horizon Data Insertion with Batch Processing")
    logging.info(f"  Data Directory: {data_dir}")
    logging.info(f"  Milvus Host: {host}")
    logging.info(f"  Collection: {collection_name}")
    logging.info(f"  Insertion Batch Size: {batch_size}")
    logging.info(f"  Feature Files Per Batch: {feature_files_per_batch}")
    
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
        processor = BatchDataProcessor(parquet_reader, feature_files_per_batch)
        
        # 3. Initialize inserter and verify collection
        inserter = SimplifiedInserter(client, collection_name)
        
        if not inserter.verify_collection_exists():
            logging.error("❌ Target collection verification failed")
            raise Exception("Target collection verification failed")
        
        # 4. Batch processing loop
        start_time = time.time()
        total_inserted = 0
        
        logging.info("=" * 80)
        logging.info(f"🔄 Starting coordinated batch processing: {processor.total_batches} batches")
        
        for batch_idx in range(processor.total_batches):
            logging.info("=" * 60)
            logging.info(f"📖 Processing batch {batch_idx + 1}/{processor.total_batches}")
            
            try:
                # Read batch data
                feature_df, location_df, scalars_df = processor.read_batch_data(batch_idx)
                
                if feature_df is None or feature_df.empty:
                    logging.error(f"❌ No feature data available for batch {batch_idx + 1}")
                    raise Exception(f"No feature data available for batch {batch_idx + 1}")
                
                if location_df is None or location_df.empty:
                    logging.error(f"❌ No location data available for batch {batch_idx + 1}")
                    raise Exception(f"No location data available for batch {batch_idx + 1}")
                
                if scalars_df is None or scalars_df.empty:
                    logging.error(f"❌ No scalars data available for batch {batch_idx + 1}")
                    raise Exception(f"No scalars data available for batch {batch_idx + 1}")
                
                # Merge data sources
                logging.info(f"🔗 Merging batch {batch_idx + 1} data... {len(feature_df)} records")
                merged_data = processor.merge_data_sources(feature_df, location_df, scalars_df)
                
                if not merged_data:
                    logging.error(f"❌ No merged data available for batch {batch_idx + 1}")
                    raise Exception(f"No merged data available for batch {batch_idx + 1}")
                
                # Insert data
                logging.info(f"💾 Inserting batch {batch_idx + 1} data... {len(merged_data)} records")
                success, batch_inserted = inserter.insert_data(merged_data, batch_size)
                
                if not success:
                    logging.error(f"❌ Batch {batch_idx + 1} insertion failed")
                    raise Exception(f"Batch {batch_idx + 1} insertion failed")
                
                total_inserted += batch_inserted
                
                # Memory cleanup after each batch
                logging.info(f"🧹 Cleaning up memory for batch {batch_idx + 1}")
                del feature_df, merged_data
                gc.collect()
                
                logging.info(f"✅ Batch {batch_idx + 1} complete: {batch_inserted} records inserted")
                
            except Exception as e:
                logging.error(f"❌ Critical error in batch {batch_idx + 1}: {e}")
                raise Exception(f"Critical error in batch {batch_idx + 1}: {e}")
        
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
