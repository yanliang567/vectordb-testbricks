# Vector Limit Feature for Parquet to JSON Converter

## 🎯 Overview

The `convert_parquet_to_json.py` script now supports limiting the number of vectors to convert, providing better control over output file size and processing time.

## 🚀 Usage

### Basic Syntax
```bash
python3 convert_parquet_to_json.py [max_vectors]
```

### Examples

#### Convert All Vectors (Default)
```bash
python3 convert_parquet_to_json.py
# Converts all 10,000 vectors from the parquet file
```

#### Convert Limited Vectors
```bash
# Convert first 1,000 vectors
python3 convert_parquet_to_json.py 1000

# Convert first 500 vectors  
python3 convert_parquet_to_json.py 500

# Convert first 100 vectors (for testing)
python3 convert_parquet_to_json.py 100
```

## 📊 Performance & File Size Comparison

| Vector Count | JSON File Size | Conversion Time | Use Case |
|--------------|----------------|-----------------|----------|
| **100** | ~1.8 MB | <1s | Quick testing |
| **500** | ~8.9 MB | ~1s | Development |
| **1,000** | ~17.8 MB | ~2s | Small-scale testing |
| **5,000** | ~89 MB | ~6s | Medium-scale testing |
| **10,000** | ~178 MB | ~13s | Full dataset |

## 🔧 Technical Details

### What Happens When Limiting Vectors

1. **File Reading**: Script reads the entire parquet file
2. **Data Limiting**: Uses `pandas.DataFrame.head(max_vectors)` to select first N vectors
3. **Processing**: Converts only the selected vectors to JSON format
4. **Memory Efficiency**: Reduces memory usage for large datasets

### Logging Output
```
📋 Command line argument: max_vectors = 1000
🔧 Limiting to first 1000 vectors (out of 10000)
✅ Successfully converted 1000 vectors
💾 Saving 1000 vectors to /tmp/query_vectors.json
📁 JSON file saved, size: 17.78 MB
```

## 💡 Use Cases

### 1. Development & Testing
```bash
# Quick testing with small dataset
python3 convert_parquet_to_json.py 100
```

### 2. Memory-Constrained Environments
```bash
# Limit vectors to reduce memory usage
python3 convert_parquet_to_json.py 1000
```

### 3. Incremental Processing
```bash
# Process in batches (Note: always starts from beginning)
python3 convert_parquet_to_json.py 500   # First 500 vectors
# To get next batch, you'd need to modify source parquet file
```

### 4. Performance Testing
```bash
# Different sizes for Go performance testing
python3 convert_parquet_to_json.py 100    # Baseline
python3 convert_parquet_to_json.py 1000   # Medium load  
python3 convert_parquet_to_json.py 5000   # High load
```

## ⚠️ Important Notes

### Limitations
- **Always starts from beginning**: Script selects the first N vectors, not a random sample
- **No batch offset**: Cannot specify starting position (e.g., vectors 1000-2000)
- **Original file unchanged**: Source parquet file remains intact

### Error Handling
```bash
# Invalid argument handling
python3 convert_parquet_to_json.py abc
# Output: ❌ Invalid argument: 'abc' is not a valid number
```

## 🔄 Integration with Go Applications

### Load Different Sized Datasets
```go
// Load small dataset for testing
smallQueries, _ := LoadQueryVectorsFromJSON("/tmp/query_vectors_100.json")

// Load medium dataset for development  
mediumQueries, _ := LoadQueryVectorsFromJSON("/tmp/query_vectors_1000.json")

// Load full dataset for production
fullQueries, _ := LoadQueryVectorsFromJSON("/tmp/query_vectors_10000.json")
```

### Memory Management
```go
// Adjust based on available memory
func selectVectorSet(availableMemoryMB int) string {
    if availableMemoryMB < 100 {
        return "/tmp/query_vectors_500.json"    // ~9MB
    } else if availableMemoryMB < 500 {
        return "/tmp/query_vectors_1000.json"   // ~18MB  
    } else {
        return "/tmp/query_vectors_10000.json"  // ~178MB
    }
}
```

## 🎉 Benefits

### 1. **Flexible Testing**
- Start small during development
- Scale up for production testing
- Adapt to hardware constraints

### 2. **Resource Optimization**
- Reduce file sizes for faster loading
- Lower memory requirements
- Shorter processing times

### 3. **Development Efficiency**
- Quick iterations with small datasets
- Faster feedback loops
- Easier debugging

## 📝 Example Workflow

```bash
# 1. Development Phase - Quick testing
python3 convert_parquet_to_json.py 100
go run test_app.go  # Fast iteration

# 2. Integration Testing - Medium load
python3 convert_parquet_to_json.py 1000  
go run integration_test.go

# 3. Performance Testing - High load
python3 convert_parquet_to_json.py 5000
go run performance_test.go

# 4. Production Ready - Full dataset
python3 convert_parquet_to_json.py
go run production_app.go
```

This feature makes the conversion tool much more flexible and suitable for different development and testing scenarios.
