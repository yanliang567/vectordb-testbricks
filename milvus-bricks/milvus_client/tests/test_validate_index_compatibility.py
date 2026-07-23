import json

from milvus_client.common.data import stable_vector_value
from milvus_client.common.schema import FieldSpec, IndexSpec, SchemaSpec
from milvus_client.requests import validate_index_compatibility


class IndexCompatibilityClient:
    def __init__(
        self,
        *,
        search_fails: bool = False,
        category_index_type: str = "INVERTED",
        scalar_query_pk=0,
        search_pk=0,
        search_distance=1.0,
    ):
        self.calls = []
        self.search_fails = search_fails
        self.scalar_query_pk = scalar_query_pk
        self.search_pk = search_pk
        self.search_distance = search_distance
        self.indexes = {
            "embedding": {
                "index_name": "embedding_idx",
                "field_name": "embedding",
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": 8, "efConstruction": 32},
            },
            "category": {
                "index_name": "category_idx",
                "field_name": "category",
                "index_type": category_index_type,
                "metric_type": None,
                "params": {},
            },
        }

    def flush(self, **kwargs):
        self.calls.append(("flush", kwargs))

    def release_collection(self, **kwargs):
        self.calls.append(("release_collection", kwargs))

    def list_indexes(self, **kwargs):
        self.calls.append(("list_indexes", kwargs))
        return [self.indexes[kwargs["field_name"]]["index_name"]]

    def describe_index(self, **kwargs):
        self.calls.append(("describe_index", kwargs))
        index_name = kwargs["index_name"]
        for index in self.indexes.values():
            if index["index_name"] == index_name:
                return dict(index)
        return {"index_name": index_name}

    def drop_index(self, **kwargs):
        self.calls.append(("drop_index", kwargs))
        index_name = kwargs["index_name"]
        for field_name, index in list(self.indexes.items()):
            if index["index_name"] == index_name:
                del self.indexes[field_name]

    def create_index(self, **kwargs):
        self.calls.append(("create_index", kwargs))
        self.indexes = {
            "embedding": {
                "index_name": "embedding_idx",
                "field_name": "embedding",
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": 8, "efConstruction": 32},
            },
            "category": {
                "index_name": "category_idx",
                "field_name": "category",
                "index_type": "INVERTED",
                "metric_type": None,
                "params": {},
            },
        }

    def load_collection(self, **kwargs):
        self.calls.append(("load_collection", kwargs))

    def query(self, **kwargs):
        self.calls.append(("query", kwargs))
        if kwargs.get("output_fields") == ["count(*)"]:
            return [{"count(*)": 3}]
        if self.scalar_query_pk is None:
            return []
        return [{"id": self.scalar_query_pk}]

    def search(self, **kwargs):
        self.calls.append(("search", kwargs))
        if self.search_fails:
            raise RuntimeError("load index failed: missing SLICE_META")
        if self.search_pk is None:
            return [[]]
        return [[{"id": self.search_pk, "distance": self.search_distance}]]


class AutoIdIndexCompatibilityClient(IndexCompatibilityClient):
    def __init__(self):
        super().__init__(scalar_query_pk=1010, search_pk=1010, search_distance=1.0)
        self.expected_query_vector = stable_vector_value(
            FieldSpec(name="embedding", dtype="FLOAT_VECTOR", dim=4),
            0,
            0,
        )

    def query(self, **kwargs):
        self.calls.append(("query", kwargs))
        if kwargs.get("output_fields") == ["count(*)"]:
            return [{"count(*)": 3}]
        if kwargs.get("filter") in {"id == 1010", "id == 1011", "id == 1012"}:
            pk = int(kwargs["filter"].rsplit(" ", 1)[-1])
            return [{"id": pk}]
        if kwargs.get("filter") == "category == 0":
            return [{"id": 1010}]
        return []

    def search(self, **kwargs):
        self.calls.append(("search", kwargs))
        if kwargs.get("data") == [self.expected_query_vector]:
            return [[{"id": 1010, "distance": 1.0}]]
        return [[{"id": 9999, "distance": 0.1}]]


class MissingScalarIndexClient(IndexCompatibilityClient):
    def __init__(self):
        super().__init__()
        del self.indexes["category"]

    def list_indexes(self, **kwargs):
        self.calls.append(("list_indexes", kwargs))
        index = self.indexes.get(kwargs["field_name"])
        return [index["index_name"]] if index else []


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


def _auto_id_spec():
    return SchemaSpec(
        name="auto",
        version="test",
        fields=[
            FieldSpec(name="id", dtype="INT64", primary=True, auto_id=True),
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


def _auto_id_seed_checkpoint(tmp_path):
    checkpoint = tmp_path / "seed_data.json"
    checkpoint.write_text(
        json.dumps(
            {
                "collections": {
                    "qa_auto": {
                        "schema_name": "auto",
                        "expected_count": 3,
                        "primary_field": "id",
                        "min_pk": 1010,
                        "max_pk": 1012,
                        "pk_samples": [1010, 1011, 1012],
                        "pk_values": [1010, 1011, 1012],
                        "data_min_pk": 0,
                        "data_max_pk": 2,
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
        "category",
        "embedding",
    ]
    assert checkpoint["collections"]["qa_dense"]["actual_indexes"] == [
        {
            "field_name": "category",
            "index_name": "category_idx",
            "index_type": "INVERTED",
            "metric_type": None,
            "params": {},
        },
        {
            "field_name": "embedding",
            "index_name": "embedding_idx",
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 8, "efConstruction": 32},
        },
    ]
    assert checkpoint["collections"]["qa_dense"]["indexed_vector_fields"] == [
        "embedding"
    ]


def test_cosine_self_search_accepts_high_similarity_score(monkeypatch, tmp_path):
    seed_checkpoint = _seed_checkpoint(tmp_path)
    index_checkpoint = tmp_path / "index_compatibility.json"
    output_json = tmp_path / "result.json"
    client = IndexCompatibilityClient(search_distance=1.0)
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
            rebuild=False,
        )
    )

    result = json.loads(output_json.read_text())
    assert code == 0
    assert result["status"] == "passed"


def test_auto_id_index_queries_use_generation_id_and_expect_actual_pk(
    monkeypatch, tmp_path
):
    seed_checkpoint = _auto_id_seed_checkpoint(tmp_path)
    index_checkpoint = tmp_path / "index_compatibility.json"
    output_json = tmp_path / "result.json"
    client = AutoIdIndexCompatibilityClient()
    monkeypatch.setattr(
        validate_index_compatibility,
        "load_schema_matrix",
        lambda path: [_auto_id_spec()],
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
            rebuild=False,
        )
    )

    result = json.loads(output_json.read_text())
    assert code == 0
    assert result["status"] == "passed"
    assert any(
        call[0] == "query" and call[1].get("filter") == "category == 0"
        for call in client.calls
    )


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
                        "actual_indexes": [
                            {
                                "field_name": "category",
                                "index_name": "category_idx",
                                "index_type": "INVERTED",
                                "metric_type": None,
                            },
                            {
                                "field_name": "embedding",
                                "index_name": "embedding_idx",
                                "index_type": "HNSW",
                                "metric_type": "COSINE",
                            },
                        ],
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
    assert "describe_index" in call_names
    assert "load_collection" in call_names
    assert "search" in call_names
    assert result["metrics"]["actual_indexes_total"] == 2
    assert result["metrics"]["scalar_index_queries_total"] == 1


def test_after_upgrade_fails_when_expected_scalar_index_is_missing(
    monkeypatch,
    tmp_path,
):
    seed_checkpoint = _seed_checkpoint(tmp_path)
    index_checkpoint = tmp_path / "index_compatibility.json"
    output_json = tmp_path / "result.json"
    client = MissingScalarIndexClient()
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
            rebuild=False,
        )
    )

    result = json.loads(output_json.read_text())
    assert code == 1
    assert result["status"] == "failed"
    assert any(
        failure["type"] == "INDEX_METADATA_MISMATCH"
        and failure["missing_fields"] == ["category"]
        for failure in result["failures"]
    )


def test_after_rollback_fails_when_actual_index_metadata_differs_from_checkpoint(
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
                        "actual_indexes": [
                            {
                                "field_name": "category",
                                "index_name": "category_idx",
                                "index_type": "BITMAP",
                                "metric_type": None,
                            },
                            {
                                "field_name": "embedding",
                                "index_name": "embedding_idx",
                                "index_type": "HNSW",
                                "metric_type": "COSINE",
                            },
                        ],
                    }
                }
            }
        )
    )
    output_json = tmp_path / "result.json"
    client = IndexCompatibilityClient(category_index_type="INVERTED")
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
    assert code == 1
    assert result["status"] == "failed"
    assert any(
        failure["type"] == "INDEX_METADATA_MISMATCH" for failure in result["failures"]
    )


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


def test_index_search_fails_when_expected_pk_is_not_returned(monkeypatch, tmp_path):
    seed_checkpoint = _seed_checkpoint(tmp_path)
    index_checkpoint = tmp_path / "index_compatibility.json"
    output_json = tmp_path / "result.json"
    client = IndexCompatibilityClient(search_pk=999)
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
            rebuild=False,
        )
    )

    result = json.loads(output_json.read_text())
    assert code == 1
    assert result["status"] == "failed"
    assert any(
        failure["type"] == "INDEX_SEARCH_FAILED" and failure["expected_pk"] == 0
        for failure in result["failures"]
    )


def test_index_search_fails_when_result_is_empty(monkeypatch, tmp_path):
    seed_checkpoint = _seed_checkpoint(tmp_path)
    index_checkpoint = tmp_path / "index_compatibility.json"
    output_json = tmp_path / "result.json"
    client = IndexCompatibilityClient(search_pk=None)
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
            rebuild=False,
        )
    )

    result = json.loads(output_json.read_text())
    assert code == 1
    assert result["status"] == "failed"
    assert any(
        failure["type"] == "INDEX_SEARCH_FAILED" for failure in result["failures"]
    )


def test_index_search_fails_when_self_search_distance_is_invalid(monkeypatch, tmp_path):
    seed_checkpoint = _seed_checkpoint(tmp_path)
    index_checkpoint = tmp_path / "index_compatibility.json"
    output_json = tmp_path / "result.json"
    client = IndexCompatibilityClient(search_pk=0, search_distance=0.5)
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
            rebuild=False,
        )
    )

    result = json.loads(output_json.read_text())
    assert code == 1
    assert result["status"] == "failed"
    assert any(
        failure["type"] == "INDEX_SEARCH_FAILED" and failure["distance"] == 0.5
        for failure in result["failures"]
    )


def test_scalar_index_query_fails_when_expected_pk_is_not_returned(
    monkeypatch, tmp_path
):
    seed_checkpoint = _seed_checkpoint(tmp_path)
    index_checkpoint = tmp_path / "index_compatibility.json"
    output_json = tmp_path / "result.json"
    client = IndexCompatibilityClient(scalar_query_pk=999)
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
            rebuild=False,
        )
    )

    result = json.loads(output_json.read_text())
    assert code == 1
    assert result["status"] == "failed"
    assert any(
        failure["type"] == "INDEX_SCALAR_QUERY_FAILED" and failure["expected_pk"] == 0
        for failure in result["failures"]
    )


def test_scalar_index_query_fails_when_result_is_empty(monkeypatch, tmp_path):
    seed_checkpoint = _seed_checkpoint(tmp_path)
    index_checkpoint = tmp_path / "index_compatibility.json"
    output_json = tmp_path / "result.json"
    client = IndexCompatibilityClient(scalar_query_pk=None)
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
            rebuild=False,
        )
    )

    result = json.loads(output_json.read_text())
    assert code == 1
    assert result["status"] == "failed"
    assert any(
        failure["type"] == "INDEX_SCALAR_QUERY_FAILED" for failure in result["failures"]
    )
