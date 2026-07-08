import json
from pathlib import Path

from milvus_client.requests import seed_data


ROOT = Path(__file__).resolve().parents[1]


class FailingInsertClient:
    def has_collection(self, name):
        del name
        return True

    def insert(self, collection_name, data):
        del collection_name, data
        raise RuntimeError("insert failed")


class AutoIdClient:
    def __init__(self):
        self.insert_calls = []

    def has_collection(self, name):
        del name
        return True

    def insert(self, **kwargs):
        self.insert_calls.append(kwargs)
        return {"ids": [1000 + len(self.insert_calls) * 10 + offset for offset, _ in enumerate(kwargs["data"])]}

    def flush(self, *args, **kwargs):
        del args, kwargs


def test_seed_data_writes_structured_failure(monkeypatch, tmp_path):
    output_json = tmp_path / "result.json"
    monkeypatch.setattr(seed_data, "create_client", lambda *args, **kwargs: FailingInsertClient())

    code = seed_data.main(
        [
            "--uri",
            "http://localhost:19530",
            "--collection-prefix",
            "qa_seed",
            "--schema-matrix",
            str(ROOT / "manifests" / "schema_matrix_2_6.yaml"),
            "--rows-per-collection",
            "1",
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--output-json",
            str(output_json),
        ]
    )

    result = json.loads(output_json.read_text())
    assert code == 1
    assert result["status"] == "failed"
    assert result["failures"][0]["type"] == "SEED_COLLECTION_FAILED"
    assert result["failures"][0]["error"] == "insert failed"


def test_seed_data_writes_structured_unexpected_failure(monkeypatch, tmp_path):
    output_json = tmp_path / "result.json"

    def fail_connect(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("connect failed")

    monkeypatch.setattr(seed_data, "create_client", fail_connect)

    code = seed_data.main(
        [
            "--uri",
            "http://localhost:19530",
            "--collection-prefix",
            "qa_seed",
            "--schema-matrix",
            str(ROOT / "manifests" / "schema_matrix_2_6.yaml"),
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--output-json",
            str(output_json),
        ]
    )

    result = json.loads(output_json.read_text())
    assert code == 4
    assert result["status"] == "failed"
    assert result["failures"][0]["type"] == "SEED_DATA_FAILED"
    assert result["failures"][0]["error"] == "connect failed"


def test_seed_data_captures_auto_id_checkpoint(monkeypatch, tmp_path):
    schema_matrix = tmp_path / "schema.yaml"
    schema_matrix.write_text(
        """
version: "test"
schemas:
  - name: auto
    fields:
      - {name: id, dtype: INT64, primary: true, auto_id: true}
      - {name: category, dtype: INT64}
      - {name: embedding, dtype: FLOAT_VECTOR, dim: 2}
    indexes:
      - {field: embedding, index_type: AUTOINDEX, metric_type: COSINE}
"""
    )
    output_json = tmp_path / "result.json"
    client = AutoIdClient()
    monkeypatch.setattr(seed_data, "create_client", lambda *args, **kwargs: client)

    code = seed_data.main(
        [
            "--uri",
            "http://localhost:19530",
            "--collection-prefix",
            "qa_seed",
            "--schema-matrix",
            str(schema_matrix),
            "--rows-per-collection",
            "2",
            "--batch-size",
            "2",
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--output-json",
            str(output_json),
        ]
    )

    checkpoint = json.loads((tmp_path / "checkpoints" / "seed_data.json").read_text())
    meta = checkpoint["collections"]["qa_seed_auto"]
    assert code == 0
    assert meta["primary_field"] == "id"
    assert meta["min_pk"] == 1010
    assert meta["max_pk"] == 1011
    assert meta["pk_samples"] == [1010, 1011]
    assert meta["pk_values"] == [1010, 1011]


def test_seed_data_writes_explicit_checkpoint_file(monkeypatch, tmp_path):
    output_json = tmp_path / "result.json"
    checkpoint_file = tmp_path / "named" / "baseline.json"
    client = AutoIdClient()
    monkeypatch.setattr(seed_data, "create_client", lambda *args, **kwargs: client)

    code = seed_data.main(
        [
            "--uri",
            "http://localhost:19530",
            "--collection-prefix",
            "qa_seed",
            "--schema-matrix",
            str(ROOT / "manifests" / "schema_matrix_2_6.yaml"),
            "--rows-per-collection",
            "1",
            "--batch-size",
            "1",
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--checkpoint-file",
            str(checkpoint_file),
            "--output-json",
            str(output_json),
        ]
    )

    result = json.loads(output_json.read_text())
    assert code == 0
    assert checkpoint_file.exists()
    assert result["checkpoint"]["path"] == str(checkpoint_file)
    assert not (tmp_path / "checkpoints" / "seed_data.json").exists()
