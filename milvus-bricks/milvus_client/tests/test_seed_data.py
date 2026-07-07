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
