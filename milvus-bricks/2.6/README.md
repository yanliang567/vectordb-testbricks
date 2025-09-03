# Milvus 2.6 MilvusClient Test Scripts

这个目录包含已转换为 **MilvusClient API** 的 Milvus 2.6 测试脚本，提供更简化和现代化的 Milvus 交互接口。

## 🚀 主要特性

### 🔄 从 Collection API 到 MilvusClient API

**Before (Collection API):**
```python
from pymilvus import connections, Collection, FieldSchema, CollectionSchema
connections.connect('default', host=host, port=port)
collection = Collection(name=collection_name)
collection.insert(data)
```

**After (MilvusClient API):**
```python
from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType
from common import create_n_insert, create_collection_schema

client = MilvusClient(uri=f"http://{host}:{port}")
client.insert(collection_name=collection_name, data=data)
```

### 🏗️ 架构改进

1. **函数集中化**: `create_n_insert()` 和 `create_collection_schema()` 现在位于 `common.py`
2. **多客户端支持**: 支持最多2个Milvus实例的比较测试
3. **API标准合规**: 完全符合 [Milvus 2.6 CollectionSchema API](https://milvus.io/api-reference/pymilvus/v2.6.x/MilvusClient/CollectionSchema/CollectionSchema.md)
4. **增强的错误处理**: 更好的异常处理和日志记录

## 📁 文件结构

| 文件 | 功能 | 状态 |
|------|------|------|
| `common.py` | 核心工具函数库 | ✅ 主要模块 |
| `create_n_insert.py` | 集合创建和数据插入 | ✅ 入口脚本 |
| `query_permanently.py` | 持续查询测试 | ✅ 新转换 |
| `query_example.py` | 查询测试使用示例 | ✅ 文档示例 |
| `requirements.txt` | 项目依赖 | ✅ 已更新 |

## 🧪 测试脚本功能

### 1. **create_n_insert.py** - 集合创建和数据插入

**功能特性:**
- ✅ 支持多种向量类型 (FLOAT_VECTOR, FLOAT16_VECTOR, BFLOAT16_VECTOR, BINARY_VECTOR, SPARSE_FLOAT_VECTOR, INT8_VECTOR)
- ✅ 多客户端支持 (最多2个实例比较测试)
- ✅ 灵活的schema定义
- ✅ 自动索引构建
- ✅ 并发数据插入

**使用示例:**
```bash
python3 create_n_insert.py \
  "host1.example.com,host2.example.com" \
  "test_collection" \
  "128,256" \
  "FLOAT,FLOAT16" \
  "HNSW,AUTOINDEX" \
  "L2,COSINE" \
  1000 \
  2 \
  5 \
  TRUE \
  FALSE \
  0 \
  TRUE \
  TRUE \
  TRUE \
  "your-api-key"
```

### 2. **query_permanently.py** - 持续查询测试 🆕

**功能特性:**
- ⚡ **多线程并发查询**: 支持多线程性能测试
- 📊 **实时性能监控**: QPS、延迟、P99统计
- 🎲 **随机查询支持**: 多种查询模式
- 🛡️ **严格集合验证**: 检查集合存在性，不存在则直接退出
- 🔒 **单实例专用**: 专注于单个Milvus实例的性能测试

**查询模式:**
- `category >= 0`: 基础范围查询
- `random`: 随机范围查询
- `random_content`: 随机内容查询
- `content like "test%"`: 文本匹配查询
- 自定义表达式: 任何有效的Milvus过滤表达式

**使用示例:**
```bash
# 基础查询测试
python3 query_permanently.py \
  "localhost" \
  "test_collection" \
  4 \
  60 \
  "id,category,content" \
  "category >= 100" \
  "None"

# 高性能随机查询测试
python3 query_permanently.py \
  "your-server.example.com" \
  "large_collection" \
  8 \
  300 \
  "*" \
  "random" \
  "your-api-key"
```

## 🔧 Schema 定义 (Milvus 2.6 API)

```python
from pymilvus import CollectionSchema, FieldSchema, DataType
from common import create_collection_schema

# 使用辅助函数
schema = create_collection_schema(
    dims=[128],
    vector_types=[DataType.FLOAT_VECTOR],
    auto_id=True,
    use_str_pk=False
)

# 或手动创建
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="category", dtype=DataType.INT64, description="category field"),
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=500, nullable=True),
    FieldSchema(name="flag", dtype=DataType.BOOL, nullable=True),
    FieldSchema(name="embedding_0", dtype=DataType.FLOAT_VECTOR, dim=128)
]

schema = CollectionSchema(
    fields=fields,
    auto_id=True,
    description="Test collection"
)
```

## 📊 性能监控输出示例

```
2024-09-03 10:15:23 - INFO - Thread 0: query 100 times, failures: 0, cost: 2.45s, qps: 40.82, avg: 0.024s, p99: 0.089s
2024-09-03 10:15:26 - INFO - Thread 1: query 100 times, failures: 1, cost: 2.67s, qps: 37.45, avg: 0.027s, p99: 0.095s
2024-09-03 10:15:26 - INFO - Thread 2: query 100 times, failures: 0, cost: 2.51s, qps: 39.84, avg: 0.025s, p99: 0.087s
2024-09-03 10:15:29 - INFO - Query test completed in 62.3 seconds
```

## 🔍 错误处理和日志

- **详细日志记录**: 每个操作都有相应的日志输出
- **异常处理**: 全面的错误处理和恢复机制
- **性能统计**: 自动收集和报告性能指标
- **调试信息**: 详细的调试信息帮助问题排查

## 📦 依赖安装

```bash
# 安装项目依赖
pip install -r requirements.txt

# 主要依赖
pip install pymilvus==2.6.0
pip install scikit-learn==1.1.3
pip install Faker==19.2.0
pip install numpy
```

## 🚀 快速开始

### 1. 创建测试集合
```python
from pymilvus import MilvusClient, DataType
from common import create_n_insert, create_collection_schema

# 连接Milvus
client = MilvusClient(uri="http://localhost:19530")

# 创建schema
schema = create_collection_schema(
    dims=[128],
    vector_types=[DataType.FLOAT_VECTOR],
    auto_id=True,
    use_str_pk=False
)

# 创建集合并插入数据
create_n_insert(
    collection_name="my_test_collection",
    schema=schema,
    nb=1000,
    insert_times=5,
    index_types=["HNSW"],
    dims=[128],
    metric_types=["L2"],
    clients=[client]
)
```

### 2. 执行查询测试
```python
# 运行示例查看可用选项
python3 query_example.py

# 执行简单的查询测试
python3 query_permanently.py \
  "localhost" \
  "my_test_collection" \
  2 \
  30 \
  "id,category" \
  "category >= 0" \
  "None"
```

## ⚠️ 注意事项

1. **向量类型支持**: 当前支持 FLOAT, FLOAT16, BFLOAT16, BINARY, SPARSE, INT8 向量类型
2. **集合必须预存在**: `query_permanently.py` 要求目标集合必须已存在，否则直接退出
3. **内存使用**: 大量数据插入时注意内存使用情况
4. **索引构建**: 确保为查询字段构建适当的索引以获得最佳性能
5. **单实例测试**: 查询测试专注于单个Milvus实例，不支持多实例比较

## 🔧 故障排除

### 常见问题
1. **连接错误**: 确保Milvus服务器运行正常且可访问
2. **Schema错误**: 检查字段类型和参数是否符合Milvus 2.6规范
3. **索引错误**: 验证索引类型是否支持指定的向量字段类型
4. **查询错误**: 确保查询表达式语法正确且字段存在

### 调试技巧
- 启用DEBUG日志: `logging.basicConfig(level=logging.DEBUG)`
- 检查集合状态: `client.describe_collection(collection_name)`
- 验证数据格式: 确保数据格式与schema定义匹配

## 📈 性能优化建议

1. **索引选择**: 根据查询模式选择合适的索引类型
2. **批量大小**: 调整插入批量大小以优化写入性能
3. **线程数量**: 根据系统资源调整查询线程数
4. **内存配置**: 为大规模数据集合理配置Milvus内存参数

---

**版本**: 2.6.1  
**更新日期**: 2024-09-03  
**兼容性**: Milvus 2.6+, pymilvus 2.6+
