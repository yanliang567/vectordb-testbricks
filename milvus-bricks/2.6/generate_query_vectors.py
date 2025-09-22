#!/usr/bin/env python3
"""
Generate query vectors from feature parquet files

This script:
1. Randomly selects 100 parquet files from the feature directory
2. Randomly samples 100 records from each selected file 
3. Generates 10,000 total vectors
4. Saves as /tmp/query.parquet with 'feature' column

Usage:
    python3 generate_query_vectors.py <feature_directory>

Example:
    python3 generate_query_vectors.py /path/to/feature/parquet/files/
"""

import os
import sys
import random
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p'
)


class QueryVectorGenerator:
    """Generate query vectors from feature parquet files"""
    
    def __init__(self, feature_directory, target_files=100, records_per_file=100):
        """
        Initialize the generator
        
        Args:
            feature_directory: Directory containing feature parquet files
            target_files: Number of files to randomly select (default: 100)
            records_per_file: Number of records to sample from each file (default: 100)
        """
        self.feature_directory = Path(feature_directory)
        self.target_files = target_files
        self.records_per_file = records_per_file
        self.total_target_records = target_files * records_per_file
        
        if not self.feature_directory.exists():
            raise ValueError(f"Feature directory does not exist: {feature_directory}")
        
        if not self.feature_directory.is_dir():
            raise ValueError(f"Path is not a directory: {feature_directory}")
    
    def get_parquet_files(self):
        """Get all parquet files from the feature directory"""
        parquet_files = list(self.feature_directory.glob("*.parquet"))
        
        if len(parquet_files) == 0:
            raise ValueError(f"No parquet files found in {self.feature_directory}")
        
        logging.info(f"📁 Found {len(parquet_files)} parquet files in {self.feature_directory}")
        return parquet_files
    
    def select_random_files(self, parquet_files):
        """Randomly select target number of files"""
        if len(parquet_files) < self.target_files:
            logging.warning(f"⚠️ Only {len(parquet_files)} files available, but {self.target_files} requested")
            selected_files = parquet_files
        else:
            selected_files = random.sample(parquet_files, self.target_files)
        
        logging.info(f"🎯 Selected {len(selected_files)} files for processing")
        return selected_files
    
    def sample_records_from_file(self, file_path):
        """Sample random records from a single parquet file"""
        try:
            # Read the parquet file
            df = pd.read_parquet(file_path)
            
            # Check if 'feature' column exists
            if 'feature' not in df.columns:
                logging.warning(f"⚠️ No 'feature' column in {file_path.name}, available columns: {list(df.columns)}")
                return []
            
            # Get available records count
            available_records = len(df)
            if available_records == 0:
                logging.warning(f"⚠️ Empty file: {file_path.name}")
                return []
            
            # Determine sample size
            sample_size = min(self.records_per_file, available_records)
            if sample_size < self.records_per_file:
                logging.warning(f"⚠️ Only {available_records} records in {file_path.name}, sampling {sample_size}")
            
            # Randomly sample records
            if sample_size == available_records:
                sampled_df = df
            else:
                sampled_indices = random.sample(range(available_records), sample_size)
                sampled_df = df.iloc[sampled_indices]
            
            # Extract feature vectors
            feature_vectors = sampled_df['feature'].tolist()
            
            logging.info(f"✅ Sampled {len(feature_vectors)} vectors from {file_path.name}")
            return feature_vectors
            
        except Exception as e:
            logging.error(f"❌ Failed to process {file_path.name}: {e}")
            return []
    
    def generate_query_vectors(self):
        """Generate query vectors from selected files"""
        logging.info(f"🚀 Starting query vector generation...")
        logging.info(f"📊 Target: {self.target_files} files × {self.records_per_file} records = {self.total_target_records} vectors")
        
        # Get all parquet files
        parquet_files = self.get_parquet_files()
        
        # Randomly select files
        selected_files = self.select_random_files(parquet_files)
        
        # Collect vectors from all selected files
        all_vectors = []
        processed_files = 0
        
        for file_path in selected_files:
            logging.info(f"📖 Processing file {processed_files + 1}/{len(selected_files)}: {file_path.name}")
            
            vectors = self.sample_records_from_file(file_path)
            if vectors:
                all_vectors.extend(vectors)
                processed_files += 1
            
            # Progress logging
            if processed_files % 10 == 0:
                logging.info(f"📈 Progress: {processed_files}/{len(selected_files)} files, {len(all_vectors)} vectors collected")
        
        logging.info(f"✅ Completed processing {processed_files} files")
        logging.info(f"📊 Total vectors collected: {len(all_vectors)}")
        
        if len(all_vectors) == 0:
            raise ValueError("No valid vectors were collected from any files")
        
        return all_vectors
    
    def save_query_parquet(self, vectors, output_path="/root/test/data/query.parquet"):
        """Save vectors to query parquet file"""
        try:
            # Create DataFrame with 'feature' column
            query_df = pd.DataFrame({
                'feature': vectors
            })
            
            # Ensure output directory exists
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to parquet
            query_df.to_parquet(output_path, index=False)
            
            logging.info(f"💾 Saved {len(vectors)} query vectors to {output_path}")
            logging.info(f"📁 File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
            
            return output_path
            
        except Exception as e:
            logging.error(f"❌ Failed to save query parquet: {e}")
            raise
    
    def verify_saved_file(self, file_path):
        """Verify the saved parquet file"""
        try:
            # Read the saved file
            df = pd.read_parquet(file_path)
            
            logging.info(f"🔍 Verification results:")
            logging.info(f"   Columns: {list(df.columns)}")
            logging.info(f"   Records: {len(df)}")
            
            if 'feature' in df.columns:
                # Check first few vectors
                sample_vectors = df['feature'].head(3).tolist()
                for i, vector in enumerate(sample_vectors):
                    if isinstance(vector, list):
                        logging.info(f"   Sample vector {i+1}: dimension {len(vector)}, first 3 values: {vector[:3]}")
                    else:
                        logging.info(f"   Sample vector {i+1}: type {type(vector)}")
            
            return True
            
        except Exception as e:
            logging.error(f"❌ Verification failed: {e}")
            return False


def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python3 generate_query_vectors.py <feature_directory>")
        print()
        print("Parameters:")
        print("  feature_directory  : Directory containing feature parquet files")
        print()
        print("Examples:")
        print("  python3 generate_query_vectors.py /path/to/feature/")
        print("  python3 generate_query_vectors.py ./data/feature/")
        sys.exit(1)
    
    feature_directory = sys.argv[1]
    
    try:
        # Create generator
        generator = QueryVectorGenerator(
            feature_directory=feature_directory,
            target_files=2,
            records_per_file=5000
        )
        
        # Generate vectors
        vectors = generator.generate_query_vectors()
        
        # Limit to exactly 10,000 if we have more
        if len(vectors) > 10000:
            logging.info(f"🔧 Limiting vectors from {len(vectors)} to 10,000")
            vectors = random.sample(vectors, 10000)
        
        # Save to file
        output_path = generator.save_query_parquet(vectors)
        
        # Verify saved file
        if generator.verify_saved_file(output_path):
            logging.info(f"🎉 Successfully generated query.parquet with {len(vectors)} vectors!")
        else:
            logging.error(f"❌ File verification failed")
            sys.exit(1)
            
    except Exception as e:
        logging.error(f"❌ Query vector generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
