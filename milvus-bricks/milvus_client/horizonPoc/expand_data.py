import json
import os
import random
import sys
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import h5py
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import math

# =============================
# 配置参数
# =============================
INPUT_PARQUET = "test_dataset_1M.parquet"
VECTOR_HDF5 = "cohere-768-euclidean.hdf5"
QUERY_DIR = Path("query_scenarios")

MERGE_QUERY_DIR = Path("query_scenarios_merged_vector")
MERGE_QUERY_DIR.mkdir(exist_ok=True)

OUTPUT_DIR = Path("expanded_dataset_100M")
OUTPUT_DIR.mkdir(exist_ok=True)

# 清空输出目录（可选）
# shutil.rmtree(OUTPUT_DIR)
# OUTPUT_DIR.mkdir()

NUM_TARGET_ROWS = 100_000_000  # 1亿条
# NUM_TARGET_ROWS = 1_500_000  # 2M条
CHUNK_SIZE_PER_FILE = 250_000  # 每个文件 ~25万行，约 800MB~1GB
NUM_WORKERS = os.cpu_count()  # 并发进程数（推荐 4~16）

VECTOR_DIM = 768
DTYPE_VECTOR = np.float32

# 噪声强度（可根据需要调整）
NOISE_LAT_LON = 1e-5  # 约 ±1 米
NOISE_VECTOR = 1e-3  # 向量微小扰动，保持语义相似


def parse_hdf5(h5_path, dataset_name='train'):
    with h5py.File(h5_path, 'r') as hf:
        if dataset_name not in hf:
            raise KeyError(f"HDF5 文件中未找到 {dataset_name} 数据集")
        vectors = hf[dataset_name]
        vector_count = vectors.shape[0]
        assert vectors.shape[1] == VECTOR_DIM, f"维度不匹配: {vectors.shape[1]}"
        print(f"✅ 数据集 {dataset_name}, 向量数据: {vector_count:,} 条, dim={VECTOR_DIM}")
        return vectors[:]


# =============================
# 主函数：生成一批分块数据
# =============================
def generate_chunk(args):
    chunk_id, start_idx, num_rows, schema, df, vectors = args

    vector_n = len(vectors)
    original_n = len(df)

    # 用于保存结果的列表
    batch_data = []

    np.random.seed(chunk_id)  # 不同 chunk 种子不同

    for i in range(num_rows):
        idx = (start_idx + i) % original_n
        vec_idx = idx % vector_n  # 循环使用向量
        row = df.iloc[idx].to_dict()
        vec = vectors[vec_idx].astype(DTYPE_VECTOR).copy()
        # 添加噪声（轻微扰动）
        if row.get('gcj02_lat') and not pd.isna(row['gcj02_lat']):
            row['gcj02_lat'] += np.random.normal(0, NOISE_LAT_LON)
            row['gcj02_lon'] += np.random.normal(0, NOISE_LAT_LON)
            row['wgs84_lat'] += np.random.normal(0, NOISE_LAT_LON)
            row['wgs84_lon'] += np.random.normal(0, NOISE_LAT_LON)
        vec += np.random.normal(0, NOISE_VECTOR, vec.shape)  # 向量加噪
        # 添加向量字段
        row['vector'] = vec.tolist()  # 存为 list，PyArrow 自动识别为 list<float>
        batch_data.append(row)

    # 构建 Arrow 表
    schema_with_vector = schema.append(
        pa.field('vector', pa.list_(pa.float32()), nullable=False)
    )

    batch_df = pd.DataFrame(batch_data)
    batch_table = pa.Table.from_pandas(batch_df, schema=schema_with_vector, safe=False)

    # 写出 Parquet 文件
    output_file = OUTPUT_DIR / f"part_{chunk_id:04d}.parquet"
    pq.write_table(
        batch_table,
        output_file,
        compression=None,
        use_dictionary=False,
        write_statistics=True
    )

    return len(batch_table)


# =============================
# 主流程：调度并发任务
# =============================
def expand_to_100m():

    # 计算总 chunk 数
    total_chunks = math.ceil(NUM_TARGET_ROWS / CHUNK_SIZE_PER_FILE)
    chunks = []

    for chunk_id in range(total_chunks):
        start_idx = chunk_id * CHUNK_SIZE_PER_FILE
        num_rows = min(CHUNK_SIZE_PER_FILE, NUM_TARGET_ROWS - start_idx)
        chunks.append((chunk_id, start_idx, num_rows, table.schema, df_original, train_vectors))

    print(f"🚀 开始并发生成 {total_chunks} 个文件，目标总量: {NUM_TARGET_ROWS:,}")
    print(f"⚙️  使用 {NUM_WORKERS} 个进程，并发写入...")

    completed = 0
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(generate_chunk, arg) for arg in chunks]

        # 使用 tqdm 显示整体进度
        with tqdm(total=len(futures), desc="📦 生成文件", unit="file") as pbar:
            for future in as_completed(futures):
                try:
                    count = future.result()
                    completed += count
                except Exception as e:
                    print(f"❌ 任务失败: {e}")
                finally:
                    pbar.update(1)
                    pbar.set_postfix({"累计行数": f"{completed:,}"})

    print(f"\n🎉 完成！已生成 {completed:,} 条数据，存储于 {OUTPUT_DIR}/")


def merge_queries(test_vectors):
    query_files = [QUERY_DIR / f for f in os.listdir(QUERY_DIR) if f.endswith(".jsonl")]
    if not query_files:
        print("❌ 未找到任何查询文件！")
        return
    # 外层：遍历文件
    for query_file in query_files:
        with open(query_file, 'r', encoding='utf-8') as f:
            queries = [json.loads(line.strip()) for line in f if line.strip()]
            query_vector_path = MERGE_QUERY_DIR / f"{query_file.stem}_vector.jsonl"
            query_double_vector_path = MERGE_QUERY_DIR / f"{query_file.stem}_double_vector.jsonl"
            with open(query_vector_path, 'w', encoding='utf-8') as fout, \
                 open(query_double_vector_path, 'w', encoding='utf-8') as fout2:
                for orig_q in queries:
                    record = orig_q.copy()
                    record['pic_embedding'] = random.choice(test_vectors).tolist()
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    record['text_embedding'] = random.choice(test_vectors).tolist()
                    fout2.write(json.dumps(record, ensure_ascii=False) + "\n")


# =============================
# 运行
# =============================
if __name__ == "__main__":
    if sys.argv[1] not in ['train', 'test']:
        print("❌ 参数错误，请输入 'train' 或 'test'")
        sys.exit(1)

    print("🔍 正在读取源数据...")
    # 读取一次获取 schema
    table = pq.read_table(INPUT_PARQUET)
    print(f"✅ 标量原始数据: {table.num_rows:,} 行")
    df_original: pd.DataFrame = table.to_pandas()

    if sys.argv[1] == 'train':
        train_vectors = parse_hdf5(VECTOR_HDF5, 'train')
        expand_to_100m()
    elif sys.argv[1] == 'test':
        test_vectors = parse_hdf5(VECTOR_HDF5, 'test')
        merge_queries(test_vectors)
