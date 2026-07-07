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
INPUT_PARQUET = "data/test_dataset_1M.parquet"
VECTOR_HDF5 = "data/cohere-768-euclidean.hdf5"
QUERY_DIR = Path("data/query_scenarios")

MERGE_QUERY_DIR = Path("data/query_scenarios_merged_vector")
MERGE_QUERY_DIR.mkdir(exist_ok=True)

OUTPUT_DIR = Path("data/expanded_dataset_100M")
OUTPUT_DIR.mkdir(exist_ok=True)

NUM_TARGET_ROWS = 100_000_000
# NUM_TARGET_ROWS = 1_500_000  # 测试用 150万条
CHUNK_SIZE_PER_FILE = 250_000
NUM_WORKERS = min(os.cpu_count(), 8)  # 控制并发数避免资源耗尽

VECTOR_DIM = 768
DTYPE_VECTOR = np.float32

NOISE_LAT_LON = 1e-5
NOISE_VECTOR = 1e-3

# ---------------------------------
# 全局变量（由主进程初始化）
# 子进程会在 generate_chunk 中重新加载（模拟“共享”）
# ---------------------------------
df_original = None
train_vectors = None
test_vectors = None
table_schema = None


def parse_hdf5(h5_path, dataset_name='train'):
    with h5py.File(h5_path, 'r') as hf:
        if dataset_name not in hf:
            raise KeyError(f"HDF5 文件中未找到 {dataset_name} 数据集")
        vectors = hf[dataset_name][:]
        vector_count = vectors.shape[0]
        assert vectors.shape[1] == VECTOR_DIM, f"维度不匹配: {vectors.shape[1]}"
        print(f"✅ 数据集 {dataset_name}, 向量数据: {vector_count:,} 条, dim={VECTOR_DIM}")
        return vectors.astype(DTYPE_VECTOR)


# =============================
# 主函数：生成一批分块数据（子进程调用）
# =============================
def generate_chunk(args):
    global df_original, train_vectors  # 确保已加载

    chunk_id, start_idx, num_rows = args

    # ⚠️ 子进程必须自己加载数据（或继承只读副本）
    # 但我们假设主进程已设置好全局变量
    if df_original is None or train_vectors is None:
        raise RuntimeError("子进程中未正确加载 df_original 或 train_vectors")

    original_n = len(df_original)
    vector_n = len(train_vectors)

    batch_data = []
    np.random.seed(chunk_id)

    for i in range(num_rows):
        idx = (start_idx + i) % original_n
        vec_idx = idx % vector_n

        row = df_original.iloc[idx].to_dict()
        vec = train_vectors[vec_idx].copy()

        # 坐标加噪
        if pd.notna(row.get('gcj02_lat')):
            row['gcj02_lat'] += np.random.normal(0, NOISE_LAT_LON)
            row['gcj02_lon'] += np.random.normal(0, NOISE_LAT_LON)
            row['wgs84_lat'] += np.random.normal(0, NOISE_LAT_LON)
            row['wgs84_lon'] += np.random.normal(0, NOISE_LAT_LON)

        # 向量加噪
        vec += np.random.normal(0, NOISE_VECTOR, vec.shape)

        row['vector'] = vec.tolist()
        batch_data.append(row)

    schema_with_vector = pa.schema([
        *table_schema,
        pa.field('vector', pa.list_(pa.float32()), nullable=False)
    ])
    batch_table = pa.Table.from_pandas(pd.DataFrame(batch_data), schema=schema_with_vector)

    output_file = OUTPUT_DIR / f"part_{chunk_id:04d}.parquet"
    pq.write_table(batch_table, output_file, compression=None, use_dictionary=False)
    return len(batch_table)


# =============================
# 初始化函数（供主进程调用）
# =============================
def init_worker():
    """可选：用于初始化日志、随机种子等"""
    pass


# =============================
# 主流程：调度并发任务
# =============================
def expand_to_100m():
    global df_original, train_vectors, table_schema

    print("🧠 正在主进程中加载数据到内存...")
    table = pq.read_table(INPUT_PARQUET)
    df_original = table.to_pandas()
    train_vectors = parse_hdf5(VECTOR_HDF5, 'train')
    table_schema = table.schema

    total_chunks = math.ceil(NUM_TARGET_ROWS / CHUNK_SIZE_PER_FILE)
    chunks = []

    for chunk_id in range(total_chunks):
        start_idx = chunk_id * CHUNK_SIZE_PER_FILE
        num_rows = min(CHUNK_SIZE_PER_FILE, NUM_TARGET_ROWS - start_idx)
        chunks.append((chunk_id, start_idx, num_rows))

    print(f"🚀 开始并发生成 {total_chunks} 个文件，目标总量: {NUM_TARGET_ROWS:,}")
    print(f"⚙️  使用 {NUM_WORKERS} 个进程，并发写入...")

    completed = 0
    with ProcessPoolExecutor(
        max_workers=NUM_WORKERS,
        initializer=init_worker
    ) as executor:
        futures = [executor.submit(generate_chunk, arg) for arg in chunks]

        with tqdm(total=len(futures), desc="📦 生成文件", unit="file") as pbar:
            for future in as_completed(futures):
                try:
                    count = future.result()
                    completed += count
                except Exception as e:
                    print(f"❌ 任务失败: {type(e).__name__}: {e}")
                finally:
                    pbar.update(1)
                    pbar.set_postfix({"累计行数": f"{completed:,}"})

    print(f"\n🎉 完成！已生成 {completed:,} 条数据，存储于 {OUTPUT_DIR}/")


def merge_queries(test_vectors_arg, target_query_num: int = 10000):
    query_files = [f for f in QUERY_DIR.iterdir() if f.suffix == '.jsonl']
    if not query_files:
        print("❌ 未找到任何查询文件！")
        return

    vectors_count = len(test_vectors_arg)
    for query_file in query_files:
        with open(query_file, 'r', encoding='utf-8') as f:
            queries = [json.loads(line.strip()) for line in f if line.strip()]

        # 添加 pic_embedding
        with open(MERGE_QUERY_DIR / f"{query_file.stem}_vector.jsonl", 'w', encoding='utf-8') as fout:
            for idx in tqdm(range(target_query_num), desc=f"📝 Generating {query_file.stem}_vector.jsonl", unit="query"):
                q_out = queries[idx % len(queries)].copy()
                vec = test_vectors_arg[idx % vectors_count].copy()
                vec += np.random.normal(0, NOISE_VECTOR, vec.shape)
                q_out['pic_embedding'] = vec.tolist()
                fout.write(json.dumps(q_out, ensure_ascii=False) + "\n")

        # 添加 pic_embedding + text_embedding
        with open(MERGE_QUERY_DIR / f"{query_file.stem}_double_vector.jsonl", 'w', encoding='utf-8') as fout:
            for idx in tqdm(range(target_query_num), desc=f"📝 Generating {query_file.stem}_double_vector.jsonl", unit="query"):
                q_out = queries[idx % len(queries)].copy()
                vec = test_vectors_arg[idx % vectors_count].copy()
                q_out['pic_embedding'] = vec.tolist()
                vec += np.random.normal(0, NOISE_VECTOR, vec.shape)
                q_out['text_embedding'] = vec.tolist()
                fout.write(json.dumps(q_out, ensure_ascii=False) + "\n")

        print(f"✅ 已处理: {query_file.name}")


# =============================
# 运行
# =============================
if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ['train', 'test']:
        print("❌ 用法: python script.py [train|test]")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == 'train':
        expand_to_100m()
    elif mode == 'test':
        print("🔍 加载测试向量用于合并查询...")
        test_vecs = parse_hdf5(VECTOR_HDF5, 'test')
        merge_queries(test_vecs)
