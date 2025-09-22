# Search Horizon Performance Tool - Usage Examples (单层循环简化版)

## 简化后的参数列表

```bash
./search_horizon_perf -h

Usage of ./search_horizon_perf:
  -expr-workers string
        Expression type with workers: 'expr1:workers1,expr2:workers2' 
        (e.g., 'equal:10,device_id_in:20') (default "equal:10")
  -search-timeout int
        Individual search timeout in seconds (default 10)
  -search-type string
        Search type: normal or hybrid (default "normal")
  -timeout int
        Test timeout in seconds (default 120)
  -vector-file string
        Path to JSON file containing query vectors 
        (default "/root/test/data/query_vectors_100.json")
```

## 使用示例

### 1. 默认配置
```bash
# 使用默认配置：equal表达式，10个workers
./search_horizon_perf_linux
```

### 2. 单个表达式类型
```bash
# equal表达式，20个workers
./search_horizon_perf_linux -expr-workers "equal:20"

# device_id_in表达式，5个workers
./search_horizon_perf_linux -expr-workers "device_id_in:5"
```

### 3. 多个表达式类型，一对一配置
```bash
# equal用10个workers，device_id_in用20个workers
./search_horizon_perf_linux -expr-workers "equal:10,device_id_in:20"

# 三种表达式，不同的并发配置
./search_horizon_perf_linux -expr-workers "equal:5,device_id_in:10,geo_contains:15"
```

### 4. 复杂配置示例
```bash
# 多种表达式类型的性能对比
./search_horizon_perf_linux \
  -expr-workers "equal:10,device_id_in:20,geo_contains:8,sensor_contains:5"

# 混合搜索测试
./search_horizon_perf_linux \
  -search-type hybrid \
  -expr-workers "equal:15,device_id_in:25" \
  -timeout 600 \
  -search-timeout 15
```

### 6. 自定义向量文件
```bash
# 使用自定义向量文件
./search_horizon_perf_linux \
  -vector-file /path/to/your/vectors.json \
  -expr-workers "equal:10,device_id_in:20"
```

## 实际执行示例

运行这个命令：
```bash
./search_horizon_perf_linux -expr-workers "equal:5,device_id_in:10;20"
```

会得到如下配置和执行序列：
```
📋 Test configurations:
   equal: workers [5]
   device_id_in: workers [10 20]

执行顺序：
1. equal表达式，5个workers
2. device_id_in表达式，10个workers
3. device_id_in表达式，20个workers
```

## 支持的表达式类型

- `equal`
- `equal_and_expert_collected`
- `equal_and_timestamp_week`
- `equal_and_timestamp_month`
- `geo_contains`
- `sensor_contains`
- `device_id_in`
- `sensor_json_contains`

## 优势

1. **简洁**: 只有一个核心参数 `-expr-workers`
2. **灵活**: 支持任意表达式类型和并发数组合
3. **直观**: 参数格式清晰易懂
4. **强大**: 支持复杂的性能测试场景
