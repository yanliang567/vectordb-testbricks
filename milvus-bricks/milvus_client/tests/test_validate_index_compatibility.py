import json

from milvus_client.common.schema import FieldSpec, IndexSpec, SchemaSpec
from milvus_client.requests import validate_index_compatibility


class IndexCompatibilityClient:
    def __init__(self, *, search_fails: bool = False):
        self.calls = []
        self.search_fails = search_fails

    def flush(self, **kwargs):
        self.calls.append(("flush", kwargs))

    def release_collection(self, **kwargs):
        self.calls.append(("release_collection", kwargs))

    def list_indexes(self, **kwargs):
        self.calls.append(("list_indexes", kwargs))
        return [f"{kwargs['field_name']}_idx"]

    def drop_index(self, **kwargs):
        self.calls.append(("drop_index", kwargs))

    def create_index(self, **kwargs):
        self.calls.append(("create_index", kwargs))

    def load_collection(self, **kwargs):
        self.calls.append(("load_collection", kwargs))

    def query(self, **kwargs):
        self.calls.append(("query", kwargs))
        if kwargs.get("output_fields") == ["count(*)"]:
            return [{"count(*)": 3}]
        return [{"id": 0}]

    def search(self, **kwargs):
        self.calls.append(("search", kwargs))
        if self.search_fails:
            raise RuntimeError("load index failed: missing SLICE_META")
        return [[{"id": 0, "distance": 0.1}]]


def _spec():
    return SchemaSpec(
        name="dense",
        version="test",
        fields=[
            FieldSpec(name="id", dtype="INT64", primary=True),
            FieldSpec(name="category", dtype="INT64"),
            FieldSpec(name="embedding", dtype="FLOAT_VECTOR", dim=4),
        ],
        indexes=[
            IndexSpec(
                field="embedding",
                index_type="HNSW",
                metric_type="COSINE",
                params={"M": 8, "efConstruction": 32},
            ),
            IndexSpec(field="category", index_type="INVERTED"),
        ],
    )


def _seed_checkpoint(tmp_path):
    checkpoint = tmp_path / "seed_data.json"
    checkpoint.write_text(
        json.dumps(
            {
                "collections": {
                    "qa_dense": {
                        "schema_name": "dense",
                        "expected_count": 3,
                        "primary_field": "id",
                        "min_pk": 0,
                        "max_pk": 2,
                        "pk_samples": [0, 1, 2],
                    }
                }
            }
        )
    )
    return checkpoint


def _args(tmp_path, seed_checkpoint, index_checkpoint, output_json, *, phase, rebuild):
    return [
        "--uri",
        "http://localhost:19530",
        "--collection-prefix",
        "qa",
        "--schema-matrix",
        "schema.yaml",
        "--checkpoint-file",
        str(seed_checkpoint),
        "--index-checkpoint-file",
        str(index_checkpoint),
        "--phase",
        phase,
        "--rebuild-index",
        "true" if rebuild else "false",
        "--checkpoint-dir",
        str(tmp_path),
        "--output-json",
        str(output_json),
    ]


def test_after_upgrade_rebuilds_indexes_and_writes_index_checkpoint(
    monkeypatch, tmp_path
):
    seed_checkpoint = _seed_checkpoint(tmp_path)
    index_checkpoint = tmp_path / "index_compatibility.json"
    output_json = tmp_path / "result.json"
    client = IndexCompatibilityClient()
    monkeypatch.setattr(
        validate_index_compatibility,
        "load_schema_matrix",
        lambda path: [_spec()],
    )
    monkeypatch.setattr(
        validate_index_compatibility,
        "create_client",
        lambda *args, **kwargs: client,
    )

    code = validate_index_compatibility.main(
        _args(
            tmp_path,
            seed_checkpoint,
            index_checkpoint,
            output_json,
            phase="after-upgrade",
            rebuild=True,
        )
    )

    result = json.loads(output_json.read_text())
    checkpoint = json.loads(index_checkpoint.read_text())
    call_names = [name for name, _ in client.calls]
    assert code == 0
    assert result["status"] == "passed"
    assert call_names.index("release_collection") < call_names.index("drop_index")
    assert call_names.index("drop_index") < call_names.index("create_index")
    assert call_names.index("create_index") < call_names.index("load_collection")
    assert "search" in call_names
    assert result["metrics"]["indexes_rebuilt"] == 2
    assert result["metrics"]["searches_total"] == 1
    assert checkpoint["collections"]["qa_dense"]["indexed_fields"] == [
        "embedding",
        "category",
    ]
    assert checkpoint["collections"]["qa_dense"]["indexed_vector_fields"] == [
        "embedding"
    ]


def test_after_rollback_validates_existing_rebuilt_indexes_without_recreate(
    monkeypatch,
    tmp_path,
):
    seed_checkpoint = _seed_checkpoint(tmp_path)
    index_checkpoint = tmp_path / "index_compatibility.json"
    index_checkpoint.write_text(
        json.dumps(
            {
                "collections": {
                    "qa_dense": {
                        "schema_name": "dense",
                        "indexed_fields": ["embedding", "category"],
                    }
                }
            }
        )
    )
    output_json = tmp_path / "result.json"
    client = IndexCompatibilityClient()
    monkeypatch.setattr(
        validate_index_compatibility,
        "load_schema_matrix",
        lambda path: [_spec()],
    )
    monkeypatch.setattr(
        validate_index_compatibility,
        "create_client",
        lambda *args, **kwargs: client,
    )

    code = validate_index_compatibility.main(
        _args(
            tmp_path,
            seed_checkpoint,
            index_checkpoint,
            output_json,
            phase="after-rollback",
            rebuild=False,
        )
    )

    result = json.loads(output_json.read_text())
    call_names = [name for name, _ in client.calls]
    assert code == 0
    assert result["status"] == "passed"
    assert "drop_index" not in call_names
    assert "create_index" not in call_names
    assert "load_collection" in call_names
    assert "search" in call_names


def test_after_rollback_rejects_empty_index_checkpoint(monkeypatch, tmp_path):
    seed_checkpoint = _seed_checkpoint(tmp_path)
    index_checkpoint = tmp_path / "index_compatibility.json"
    index_checkpoint.write_text(json.dumps({"collections": {}}))
    output_json = tmp_path / "result.json"
    monkeypatch.setattr(
        validate_index_compatibility,
        "load_schema_matrix",
        lambda path: [_spec()],
    )
    monkeypatch.setattr(
        validate_index_compatibility,
        "create_client",
        lambda *args, **kwargs: IndexCompatibilityClient(),
    )

    code = validate_index_compatibility.main(
        _args(
            tmp_path,
            seed_checkpoint,
            index_checkpoint,
            output_json,
            phase="after-rollback",
            rebuild=False,
        )
    )

    result = json.loads(output_json.read_text())
    assert code == 2
    assert result["status"] == "failed"
    assert result["failures"][0]["type"] == "INDEX_COMPATIBILITY_CHECKPOINT_EMPTY"


def test_search_failure_is_reported_as_index_search_failed(monkeypatch, tmp_path):
    seed_checkpoint = _seed_checkpoint(tmp_path)
    index_checkpoint = tmp_path / "index_compatibility.json"
    output_json = tmp_path / "result.json"
    client = IndexCompatibilityClient(search_fails=True)
    monkeypatch.setattr(
        validate_index_compatibility,
        "load_schema_matrix",
        lambda path: [_spec()],
    )
    monkeypatch.setattr(
        validate_index_compatibility,
        "create_client",
        lambda *args, **kwargs: client,
    )

    code = validate_index_compatibility.main(
        _args(
            tmp_path,
            seed_checkpoint,
            index_checkpoint,
            output_json,
            phase="after-upgrade",
            rebuild=True,
        )
    )

    result = json.loads(output_json.read_text())
    assert code == 1
    assert result["status"] == "failed"
    assert result["failures"][-1]["type"] == "INDEX_SEARCH_FAILED"
