import json
from pathlib import Path

from milvus_client.common.data import stable_vector_value
from milvus_client.common.schema import (
    FieldSpec,
    FunctionSpec,
    IndexSpec,
    SchemaSpec,
    StructArraySpec,
)
from milvus_client.common.validators import ValidationReport
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
        if kwargs.get("filter") in {
            "category == 0",
            "(category == 0) && id == 1010",
        }:
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


class NonUniqueScalarIndexClient(IndexCompatibilityClient):
    def query(self, **kwargs):
        self.calls.append(("query", kwargs))
        if kwargs.get("output_fields") == ["count(*)"]:
            return [{"count(*)": 3}]
        if kwargs.get("filter") == "category == 0":
            return [{"id": pk} for pk in range(10, 15)]
        if kwargs.get("filter") == "(category == 0) && id == 0":
            return [{"id": 0}]
        if kwargs.get("filter") == "id == 0":
            return [{"id": 0}]
        if kwargs.get("filter") in {"id == 1", "id == 2"}:
            pk = int(kwargs["filter"].rsplit(" ", 1)[-1])
            return [{"id": pk}]
        return []


class NullableJsonIndexClient(IndexCompatibilityClient):
    def __init__(self):
        super().__init__(scalar_query_pk=1, search_pk=0, search_distance=1.0)
        self.indexes = {
            "embedding": {
                "index_name": "embedding_idx",
                "field_name": "embedding",
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": 8, "efConstruction": 32},
            },
            "json_profile": {
                "index_name": "json_profile_idx",
                "field_name": "json_profile",
                "index_type": "INVERTED",
                "metric_type": None,
                "params": {"json_cast_type": "double"},
            },
        }

    def query(self, **kwargs):
        self.calls.append(("query", kwargs))
        if kwargs.get("output_fields") == ["count(*)"]:
            return [{"count(*)": 3}]
        if kwargs.get("filter") in {"id == 0", "id == 1", "id == 2"}:
            pk = int(kwargs["filter"].rsplit(" ", 1)[-1])
            return [{"id": pk}]
        if kwargs.get("filter") == "json_profile['bucket'] == 1":
            return [{"id": 11}, {"id": 21}]
        if kwargs.get("filter") == "(json_profile['bucket'] == 1) && id == 1":
            return [{"id": 1}]
        return []


class NestedJsonPathIndexClient(IndexCompatibilityClient):
    def __init__(self):
        super().__init__(scalar_query_pk=1, search_pk=None)
        self.indexes = {
            "json_nested": {
                "index_name": "json_nested_idx",
                "field_name": "json_nested",
                "index_type": "INVERTED",
                "metric_type": None,
                "params": {
                    "json_cast_type": "double",
                    "json_path": "json_nested['nested']['score']",
                },
            }
        }

    def describe_index(self, **kwargs):
        self.calls.append(("describe_index", kwargs))
        index = self.indexes["json_nested"]
        return {
            "index_name": index["index_name"],
            "field_name": index["field_name"],
            "index_type": index["index_type"],
            "metric_type": index["metric_type"],
            "json_path": index["params"]["json_path"],
            "json_cast_type": index["params"]["json_cast_type"],
        }

    def query(self, **kwargs):
        self.calls.append(("query", kwargs))
        if kwargs.get("output_fields") == ["count(*)"]:
            return [{"count(*)": 3}]
        if kwargs.get("filter") in {"id == 0", "id == 1", "id == 2"}:
            pk = int(kwargs["filter"].rsplit(" ", 1)[-1])
            return [{"id": pk}]
        if kwargs.get("filter") == "json_nested['nested']['score'] == 0.0":
            return [{"id": 0}]
        if kwargs.get("filter") == "(json_nested['nested']['score'] == 0.0) && id == 0":
            return [{"id": 0}]
        return []


class NullableVectorIndexClient(IndexCompatibilityClient):
    def __init__(self):
        super().__init__(scalar_query_pk=None, search_pk=None, search_distance=1.0)
        self.indexes = {
            "embedding": {
                "index_name": "embedding_idx",
                "field_name": "embedding",
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": 8, "efConstruction": 32},
            }
        }
        self.expected_query_vector = stable_vector_value(
            FieldSpec(name="embedding", dtype="FLOAT_VECTOR", dim=4, nullable=True),
            1,
            0,
        )

    def query(self, **kwargs):
        self.calls.append(("query", kwargs))
        if kwargs.get("output_fields") == ["count(*)"]:
            return [{"count(*)": 3}]
        if kwargs.get("filter") in {"id == 0", "id == 1", "id == 2"}:
            pk = int(kwargs["filter"].rsplit(" ", 1)[-1])
            return [{"id": pk}]
        return []

    def search(self, **kwargs):
        self.calls.append(("search", kwargs))
        if (
            kwargs.get("data") == [self.expected_query_vector]
            and kwargs.get("filter") == "id == 1"
        ):
            return [[{"id": 1, "distance": 1.0}]]
        return [[{"id": 9999, "distance": 0.1}]]


class GeometryIndexClient(IndexCompatibilityClient):
    def __init__(self):
        super().__init__(scalar_query_pk=0, search_pk=0, search_distance=1.0)
        self.indexes = {
            "embedding": {
                "index_name": "embedding_idx",
                "field_name": "embedding",
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": 8, "efConstruction": 32},
            },
            "location": {
                "index_name": "location_idx",
                "field_name": "location",
                "index_type": "RTREE",
                "metric_type": None,
                "params": {},
            },
        }

    def query(self, **kwargs):
        self.calls.append(("query", kwargs))
        if kwargs.get("output_fields") == ["count(*)"]:
            return [{"count(*)": 3}]
        if kwargs.get("filter") in {"id == 0", "id == 1", "id == 2"}:
            pk = int(kwargs["filter"].rsplit(" ", 1)[-1])
            return [{"id": pk}]
        if kwargs.get("filter") == "ST_EQUALS(location, 'POINT (-122 37)')":
            return [{"id": 10}]
        if (
            kwargs.get("filter")
            == "(ST_EQUALS(location, 'POINT (-122 37)')) && id == 0"
        ):
            return [{"id": 0}]
        return []


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


def _geometry_spec():
    return SchemaSpec(
        name="geometry_rtree",
        version="test",
        fields=[
            FieldSpec(name="id", dtype="INT64", primary=True),
            FieldSpec(name="location", dtype="GEOMETRY"),
            FieldSpec(name="embedding", dtype="FLOAT_VECTOR", dim=4),
        ],
        indexes=[
            IndexSpec(field="location", index_type="RTREE"),
            IndexSpec(
                field="embedding",
                index_type="HNSW",
                metric_type="COSINE",
                params={"M": 8, "efConstruction": 32},
            ),
        ],
    )


def _nullable_vector_spec():
    return SchemaSpec(
        name="nullable_vector",
        version="test",
        fields=[
            FieldSpec(name="id", dtype="INT64", primary=True),
            FieldSpec(name="category", dtype="INT64"),
            FieldSpec(name="embedding", dtype="FLOAT_VECTOR", dim=4, nullable=True),
        ],
        indexes=[
            IndexSpec(
                field="embedding",
                index_type="HNSW",
                metric_type="COSINE",
                params={"M": 8, "efConstruction": 32},
            )
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


def _nullable_json_spec():
    return SchemaSpec(
        name="json_nullable",
        version="test",
        fields=[
            FieldSpec(name="id", dtype="INT64", primary=True),
            FieldSpec(name="json_profile", dtype="JSON", nullable=True),
            FieldSpec(name="embedding", dtype="FLOAT_VECTOR", dim=4),
        ],
        indexes=[
            IndexSpec(field="json_profile", index_type="INVERTED"),
            IndexSpec(
                field="embedding",
                index_type="HNSW",
                metric_type="COSINE",
                params={"M": 8, "efConstruction": 32},
            ),
        ],
    )


def _bm25_spec():
    return SchemaSpec(
        name="text_lob_storage_v3",
        version="3.0",
        fields=[
            FieldSpec(name="id", dtype="INT64", primary=True),
            FieldSpec(
                name="text",
                dtype="TEXT",
                nullable=True,
                value_profile="text_lob_boundary",
            ),
            FieldSpec(name="sparse_bm25", dtype="SPARSE_FLOAT_VECTOR"),
        ],
        indexes=[
            IndexSpec(
                field="sparse_bm25",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25",
            )
        ],
        functions=[
            FunctionSpec(
                name="text_bm25",
                function_type="BM25",
                input_fields=["text"],
                output_fields=["sparse_bm25"],
            )
        ],
    )


def _nested_json_path_spec():
    return SchemaSpec(
        name="json_nested",
        version="test",
        fields=[
            FieldSpec(name="id", dtype="INT64", primary=True),
            FieldSpec(name="json_nested", dtype="JSON"),
        ],
        indexes=[
            IndexSpec(
                field="json_nested",
                index_type="INVERTED",
                params={
                    "json_cast_type": "double",
                    "json_path": "json_nested['nested']['score']",
                },
            )
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


def _geometry_seed_checkpoint(tmp_path):
    checkpoint = tmp_path / "seed_data.json"
    checkpoint.write_text(
        json.dumps(
            {
                "collections": {
                    "qa_geometry_rtree": {
                        "schema_name": "geometry_rtree",
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


def _nullable_vector_seed_checkpoint(tmp_path):
    checkpoint = tmp_path / "seed_data.json"
    checkpoint.write_text(
        json.dumps(
            {
                "collections": {
                    "qa_nullable_vector": {
                        "schema_name": "nullable_vector",
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


def _json_seed_checkpoint(tmp_path):
    checkpoint = tmp_path / "seed_data.json"
    checkpoint.write_text(
        json.dumps(
            {
                "collections": {
                    "qa_json_nullable": {
                        "schema_name": "json_nullable",
                        "expected_count": 3,
                        "primary_field": "id",
                        "min_pk": 0,
                        "max_pk": 2,
                        "pk_samples": [0, 1, 2],
                        "data_min_pk": 0,
                        "data_max_pk": 2,
                    }
                }
            }
        )
    )
    return checkpoint


def _nested_json_seed_checkpoint(tmp_path):
    checkpoint = tmp_path / "seed_data.json"
    checkpoint.write_text(
        json.dumps(
            {
                "collections": {
                    "qa_json_nested": {
                        "schema_name": "json_nested",
                        "expected_count": 3,
                        "primary_field": "id",
                        "min_pk": 0,
                        "max_pk": 2,
                        "pk_samples": [0, 1, 2],
                        "data_min_pk": 0,
                        "data_max_pk": 2,
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


def test_index_validation_rechecks_query_and_search_after_release_reload(
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
            rebuild=False,
        )
    )

    result = json.loads(output_json.read_text())
    call_names = [name for name, _ in client.calls]
    release_position = call_names.index("release_collection")
    assert code == 0
    assert result["status"] == "passed"
    assert result["metrics"]["reload_cycles_total"] == 1
    assert result["metrics"]["reload_searches_total"] == 1
    assert result["metrics"]["reload_scalar_index_queries_total"] == 1
    assert result["metrics"]["qa_dense.actual_indexes_total"] == 2
    assert result["metrics"]["qa_dense.vector_searches_total"] == 1
    assert result["metrics"]["qa_dense.scalar_index_queries_total"] == 1
    assert result["metrics"]["qa_dense.reload_cycles_total"] == 1
    assert result["metrics"]["qa_dense.reload_vector_searches_total"] == 1
    assert result["metrics"]["qa_dense.reload_scalar_index_queries_total"] == 1
    assert result["metrics"]["qa_dense.declared_autoindexes_total"] == 0
    maintenance_windows = result["metrics"]["maintenance_windows"]
    assert len(maintenance_windows) == 1
    assert maintenance_windows[0]["kind"] == "collection-reload"
    assert maintenance_windows[0]["label"] == (
        "index-compatibility-reload-after-upgrade"
    )
    assert maintenance_windows[0]["source"] == "validate_index_compatibility"
    assert maintenance_windows[0]["collection"] == "qa_dense"
    assert maintenance_windows[0]["started_at"] <= maintenance_windows[0]["finished_at"]
    assert "search" in call_names[:release_position]
    assert "search" in call_names[release_position + 1 :]
    assert "query" in call_names[:release_position]
    assert "query" in call_names[release_position + 1 :]


def test_after_rollback_does_not_overwrite_after_upgrade_checkpoint(
    monkeypatch, tmp_path
):
    seed_checkpoint = _seed_checkpoint(tmp_path)
    index_checkpoint = tmp_path / "index_compatibility.json"
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

    assert (
        validate_index_compatibility.main(
            _args(
                tmp_path,
                seed_checkpoint,
                index_checkpoint,
                tmp_path / "after_upgrade.json",
                phase="after-upgrade",
                rebuild=False,
            )
        )
        == 0
    )
    expected_checkpoint = index_checkpoint.read_bytes()
    client.indexes["embedding"]["index_name"] = "embedding_idx_after_rollback"

    first_code = validate_index_compatibility.main(
        _args(
            tmp_path,
            seed_checkpoint,
            index_checkpoint,
            tmp_path / "after_rollback_1.json",
            phase="after-rollback",
            rebuild=False,
        )
    )
    second_code = validate_index_compatibility.main(
        _args(
            tmp_path,
            seed_checkpoint,
            index_checkpoint,
            tmp_path / "after_rollback_2.json",
            phase="after-rollback",
            rebuild=False,
        )
    )

    assert first_code == 1
    assert second_code == 1
    assert index_checkpoint.read_bytes() == expected_checkpoint


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
        call[0] == "query" and call[1].get("filter") == "(category == 0) && id == 1010"
        for call in client.calls
    )


def test_scalar_index_query_uses_pk_conjunction_for_non_unique_predicate(
    monkeypatch, tmp_path
):
    seed_checkpoint = _seed_checkpoint(tmp_path)
    index_checkpoint = tmp_path / "index_compatibility.json"
    output_json = tmp_path / "result.json"
    client = NonUniqueScalarIndexClient()
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
    assert any(
        call[0] == "query"
        and call[1].get("filter") == "(category == 0) && id == 0"
        and call[1].get("limit") == 1
        for call in client.calls
    )


def test_nullable_json_index_selects_non_null_probe_row(monkeypatch, tmp_path):
    seed_checkpoint = _json_seed_checkpoint(tmp_path)
    index_checkpoint = tmp_path / "index_compatibility.json"
    output_json = tmp_path / "result.json"
    client = NullableJsonIndexClient()
    monkeypatch.setattr(
        validate_index_compatibility,
        "load_schema_matrix",
        lambda path: [_nullable_json_spec()],
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
    assert result["metrics"]["scalar_index_queries_total"] == 1
    assert any(
        call[0] == "query"
        and call[1].get("filter") == "(json_profile['bucket'] == 1) && id == 1"
        for call in client.calls
    )


def test_nested_json_path_index_is_queried_after_upgrade_and_rollback(
    monkeypatch, tmp_path
):
    seed_checkpoint = _nested_json_seed_checkpoint(tmp_path)
    index_checkpoint = tmp_path / "index_compatibility.json"
    client = NestedJsonPathIndexClient()
    monkeypatch.setattr(
        validate_index_compatibility,
        "load_schema_matrix",
        lambda path: [_nested_json_path_spec()],
    )
    monkeypatch.setattr(
        validate_index_compatibility,
        "create_client",
        lambda *args, **kwargs: client,
    )

    upgrade_code = validate_index_compatibility.main(
        _args(
            tmp_path,
            seed_checkpoint,
            index_checkpoint,
            tmp_path / "after_upgrade.json",
            phase="after-upgrade",
            rebuild=False,
        )
    )
    rollback_code = validate_index_compatibility.main(
        _args(
            tmp_path,
            seed_checkpoint,
            index_checkpoint,
            tmp_path / "after_rollback.json",
            phase="after-rollback",
            rebuild=False,
        )
    )

    assert upgrade_code == 0
    assert rollback_code == 0
    checkpoint = json.loads(index_checkpoint.read_text())
    assert checkpoint["collections"]["qa_json_nested"]["actual_indexes"][0][
        "params"
    ] == {
        "json_path": "json_nested['nested']['score']",
        "json_cast_type": "double",
    }
    exact_filter = "(json_nested['nested']['score'] == 0.0) && id == 0"
    assert (
        sum(
            call[0] == "query" and call[1].get("filter") == exact_filter
            for call in client.calls
        )
        == 4
    )


def test_rollback_detects_top_level_json_index_parameter_change(monkeypatch, tmp_path):
    seed_checkpoint = _nested_json_seed_checkpoint(tmp_path)
    index_checkpoint = tmp_path / "index_compatibility.json"
    client = NestedJsonPathIndexClient()
    monkeypatch.setattr(
        validate_index_compatibility,
        "load_schema_matrix",
        lambda path: [_nested_json_path_spec()],
    )
    monkeypatch.setattr(
        validate_index_compatibility,
        "create_client",
        lambda *args, **kwargs: client,
    )

    assert (
        validate_index_compatibility.main(
            _args(
                tmp_path,
                seed_checkpoint,
                index_checkpoint,
                tmp_path / "after_upgrade.json",
                phase="after-upgrade",
                rebuild=False,
            )
        )
        == 0
    )
    client.indexes["json_nested"]["params"]["json_cast_type"] = "varchar"

    rollback_code = validate_index_compatibility.main(
        _args(
            tmp_path,
            seed_checkpoint,
            index_checkpoint,
            tmp_path / "after_rollback.json",
            phase="after-rollback",
            rebuild=False,
        )
    )

    result = json.loads((tmp_path / "after_rollback.json").read_text())
    assert rollback_code == 1
    assert any(
        failure["type"] == "INDEX_METADATA_MISMATCH" for failure in result["failures"]
    )


def test_nullable_vector_search_uses_non_null_probe_and_pk_filter(
    monkeypatch, tmp_path
):
    seed_checkpoint = _nullable_vector_seed_checkpoint(tmp_path)
    index_checkpoint = tmp_path / "index_compatibility.json"
    output_json = tmp_path / "result.json"
    client = NullableVectorIndexClient()
    monkeypatch.setattr(
        validate_index_compatibility,
        "load_schema_matrix",
        lambda path: [_nullable_vector_spec()],
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
        call[0] == "search"
        and call[1].get("data") == [client.expected_query_vector]
        and call[1].get("filter") == "id == 1"
        for call in client.calls
    )


def test_geometry_index_query_uses_spatial_predicate(monkeypatch, tmp_path):
    seed_checkpoint = _geometry_seed_checkpoint(tmp_path)
    index_checkpoint = tmp_path / "index_compatibility.json"
    output_json = tmp_path / "result.json"
    client = GeometryIndexClient()
    monkeypatch.setattr(
        validate_index_compatibility,
        "load_schema_matrix",
        lambda path: [_geometry_spec()],
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
        call[0] == "query"
        and call[1].get("filter")
        == "(ST_EQUALS(location, 'POINT (-122 37)')) && id == 0"
        for call in client.calls
    )
    assert not any(
        call[0] == "query" and call[1].get("filter") == 'location == "POINT (-122 37)"'
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


def _struct_index_spec(metric_type="MAX_SIM_COSINE"):
    return SchemaSpec(
        name="struct_indexes",
        version="3.0",
        fields=[FieldSpec(name="id", dtype="INT64", primary=True)],
        struct_arrays=[
            StructArraySpec(
                name="attributes",
                max_capacity=8,
                fields=[
                    FieldSpec(name="embedding", dtype="FLOAT_VECTOR", dim=4),
                    FieldSpec(name="score_sort", dtype="FLOAT"),
                    FieldSpec(name="category_inverted", dtype="VARCHAR"),
                ],
            )
        ],
        indexes=[
            IndexSpec(
                field="attributes[embedding]",
                index_type="HNSW",
                metric_type=metric_type,
                search_params={"ef": 48},
            ),
            IndexSpec(field="attributes[score_sort]", index_type="STL_SORT"),
            IndexSpec(field="attributes[category_inverted]", index_type="INVERTED"),
        ],
    )


def test_struct_scalar_index_filters_use_match_any():
    spec = _struct_index_spec()

    score_filter = validate_index_compatibility._scalar_index_filter(
        spec, spec.indexes[1], spec.struct_arrays[0].fields[1], 3, 7
    )
    category_filter = validate_index_compatibility._scalar_index_filter(
        spec, spec.indexes[2], spec.struct_arrays[0].fields[2], 3, 7
    )

    assert score_filter == "MATCH_ANY(attributes, $[score_sort] >= 3.0)"
    assert category_filter == (
        'MATCH_ANY(attributes, $[category_inverted] == "category_3")'
    )


def test_nested_scalar_index_queries_skip_on_2_6_runtime():
    class Client:
        def get_server_version(self):
            return "v2.6.18"

        def query(self, **kwargs):
            raise AssertionError("2.6 must not execute MATCH_ANY")

    report = ValidationReport()
    queries = validate_index_compatibility.validate_scalar_index_queries(
        Client(),
        "qa_struct",
        _struct_index_spec(),
        {"min_pk": 3, "max_pk": 3, "data_min_pk": 3, "data_max_pk": 3},
        7,
        report,
    )

    assert report.passed
    assert queries == 0
    assert (
        report.metrics[
            "qa_struct.struct_array_scalar_index_queries.skipped_unsupported_total"
        ]
        == 2
    )


def test_nested_scalar_index_queries_execute_on_3_0_runtime():
    class Client:
        calls = []

        def get_server_version(self):
            return "v3.0.0"

        def query(self, **kwargs):
            self.calls.append(kwargs)
            return [{"id": 3}]

    client = Client()
    report = ValidationReport()
    queries = validate_index_compatibility.validate_scalar_index_queries(
        client,
        "qa_struct",
        _struct_index_spec(),
        {"min_pk": 3, "max_pk": 3, "data_min_pk": 3, "data_max_pk": 3},
        7,
        report,
    )

    assert report.passed
    assert queries == 2
    assert len(client.calls) == 4


def test_rollback_safe_autoindex_matrix_builds_deterministic_scalar_filters():
    matrix = (
        Path(__file__).resolve().parents[1] / "manifests" / "schema_matrix_2_6.yaml"
    )
    specs = validate_index_compatibility.load_schema_matrix(matrix)
    autoindex_filters = {}
    for spec in specs:
        meta = {"min_pk": 3, "max_pk": 3, "data_min_pk": 3, "data_max_pk": 3}
        for index, field in validate_index_compatibility.indexed_scalar_indexes(spec):
            if index.index_type != "AUTOINDEX":
                continue
            probe = validate_index_compatibility._scalar_index_probe(
                spec, meta, index, field, seed=7
            )
            autoindex_filters[f"{spec.name}.{index.field}"] = (
                None if probe is None else probe[2]
            )

    assert autoindex_filters == {
        "scalar_dynamic_partition_key.int64_category": "int64_category == 3",
        "scalar_autoindex_formats_rollback_safe.int64_auto": "int64_auto == 3",
        "scalar_autoindex_formats_rollback_safe.float_auto": (
            "float_auto == 0.30000001192092896"
        ),
        "scalar_autoindex_formats_rollback_safe.bool_auto": "bool_auto == False",
        "scalar_autoindex_formats_rollback_safe.varchar_auto": (
            'varchar_auto == "varchar_auto_3"'
        ),
        "scalar_autoindex_formats_rollback_safe.json_auto": (
            "json_auto['bucket'] == 3"
        ),
        "scalar_autoindex_formats_rollback_safe.json_bool": (
            "json_bool['active'] == False"
        ),
        "scalar_autoindex_formats_rollback_safe.json_varchar": (
            "json_varchar['label'] == \"label_3\""
        ),
        "scalar_autoindex_formats_rollback_safe.arr_int64_auto": (
            "ARRAY_CONTAINS(arr_int64_auto, 3)"
        ),
        "scalar_autoindex_formats_rollback_safe.arr_float_auto": (
            "ARRAY_CONTAINS(arr_float_auto, 3.0)"
        ),
        "scalar_autoindex_formats_rollback_safe.arr_bool_auto": (
            "ARRAY_CONTAINS(arr_bool_auto, False)"
        ),
        "scalar_autoindex_formats_rollback_safe.arr_varchar_auto": (
            'ARRAY_CONTAINS(arr_varchar_auto, "tag_3")'
        ),
        "struct_array_varchar_autoindex_rollback_safe.items[category]": (
            'MATCH_ANY(items, $[category] == "category_3")'
        ),
        "struct_array_numeric_autoindex_rollback_safe.items[score]": (
            "MATCH_ANY(items, $[score] == 3.0)"
        ),
        "struct_array_numeric_autoindex_rollback_safe.items[rank]": (
            "MATCH_ANY(items, $[rank] == 30)"
        ),
        "struct_array_numeric_autoindex_rollback_safe.items[enabled]": (
            "MATCH_ANY(items, $[enabled] == False)"
        ),
    }


def test_scalar_index_filter_uses_like_for_varchar_ngram():
    spec = validate_index_compatibility.SchemaSpec(
        name="ngram",
        version="2.6",
        fields=[
            validate_index_compatibility.FieldSpec(
                name="id", dtype="INT64", primary=True
            ),
            validate_index_compatibility.FieldSpec(
                name="text", dtype="VARCHAR", max_length=128
            ),
        ],
        indexes=[
            validate_index_compatibility.IndexSpec(
                field="text",
                index_type="NGRAM",
                params={"min_gram": 2, "max_gram": 4},
            )
        ],
    )

    assert (
        validate_index_compatibility.scalar_index_filter_for_value(
            spec,
            spec.indexes[0],
            spec.fields[1],
            "text_7",
        )
        == 'text LIKE "%text_7%"'
    )


def test_timestamptz_scalar_index_filter_uses_iso_literal():
    field = FieldSpec(
        name="event_time",
        dtype="TIMESTAMPTZ",
        value_profile="future_timestamptz",
    )
    spec = SchemaSpec(
        name="timestamptz",
        version="3.0",
        fields=[
            FieldSpec(name="id", dtype="INT64", primary=True),
            field,
        ],
        indexes=[IndexSpec(field="event_time", index_type="STL_SORT")],
    )

    filter_expr = validate_index_compatibility._scalar_index_filter(
        spec, spec.indexes[0], field, 1, 7
    )

    assert filter_expr == "event_time == ISO '2100-01-01T00:00:01Z'"


def test_struct_max_sim_probe_uses_embedding_list_without_offset_requirement():
    spec = _struct_index_spec("MAX_SIM_COSINE")
    index = spec.indexes[0]
    field = spec.struct_arrays[0].fields[0]
    meta = {
        "primary_field": "id",
        "min_pk": 3,
        "max_pk": 3,
        "data_min_pk": 3,
        "data_max_pk": 3,
    }

    data_pk, expected_pk, query, offset = (
        validate_index_compatibility._vector_index_probe(
            spec, meta, index, field, seed=7
        )
    )
    report = ValidationReport()
    validate_index_compatibility._validate_vector_search_hit(
        [[{"id": expected_pk, "distance": 1.0}]],
        "qa_struct",
        index.field,
        "id",
        expected_pk,
        offset,
        "MAX_SIM_COSINE",
        report,
    )

    assert data_pk == 3
    assert offset is None
    assert type(query).__name__ == "EmbeddingList"
    assert len(query) == 1
    assert report.passed


def test_bm25_function_probe_skips_null_and_empty_source_values():
    spec = _bm25_spec()
    sparse = spec.fields[2]
    meta = {
        "primary_field": "id",
        "min_pk": 0,
        "max_pk": 9,
        "data_min_pk": 0,
        "data_max_pk": 9,
    }

    probe = validate_index_compatibility._vector_index_probe(
        spec, meta, spec.indexes[0], sparse, seed=7
    )

    assert probe == (
        2,
        2,
        "Milvus Unicode compatibility: 中文 日本語 한국어",
        None,
    )


def test_vector_search_fails_when_score_is_unobservable():
    report = ValidationReport()

    validate_index_compatibility._validate_vector_search_hit(
        [[{"id": 3}]],
        "qa",
        "embedding",
        "id",
        3,
        None,
        "COSINE",
        report,
        index_type="HNSW",
    )

    assert not report.passed
    assert report.failures[0]["type"] == "INDEX_SEARCH_FAILED"


def test_vector_search_fails_when_score_is_not_finite():
    report = ValidationReport()

    validate_index_compatibility._validate_vector_search_hit(
        [[{"id": 3, "distance": float("nan")}]],
        "qa",
        "embedding",
        "id",
        3,
        None,
        "COSINE",
        report,
        index_type="HNSW",
    )

    assert not report.passed
    assert report.failures[0]["type"] == "INDEX_SEARCH_FAILED"
    assert report.failures[0]["message"] == (
        "indexed vector self-search returned a non-finite distance or score"
    )


def test_bm25_index_search_requires_expected_primary_key():
    class Client:
        def search(self, **kwargs):
            return [[{"id": 1, "distance": 1.0}]]

    report = ValidationReport()
    searches = validate_index_compatibility._validate_index_searches(
        Client(),
        "qa_bm25",
        _bm25_spec(),
        {"primary_field": "id", "min_pk": 2, "max_pk": 2},
        7,
        report,
    )

    assert searches == 1
    assert not report.passed
    assert report.failures[0]["expected_pk"] == 2
    assert report.failures[0]["actual_pks"] == [1]


def test_vector_score_failure_records_actual_hits():
    report = ValidationReport()

    validate_index_compatibility._validate_vector_search_hit(
        [[{"id": 3, "distance": -0.99}, {"id": 4, "distance": -0.75}]],
        "qa_struct",
        "embeddings[vector]",
        "id",
        3,
        None,
        "MAX_SIM_COSINE",
        report,
        index_type="DISKANN",
    )

    assert not report.passed
    assert report.failures == [
        {
            "type": "INDEX_SEARCH_FAILED",
            "message": "indexed vector self-search score is lower than expected",
            "collection": "qa_struct",
            "field": "embeddings[vector]",
            "metric_type": "MAX_SIM_COSINE",
            "index_type": "DISKANN",
            "expected_pk": 3,
            "distance": -0.99,
            "min_score": 0.9,
            "actual_hits": [
                {"pk": 3, "offset": None, "distance": -0.99},
                {"pk": 4, "offset": None, "distance": -0.75},
            ],
        }
    ]


def test_struct_element_probe_and_hit_require_matching_offset():
    spec = _struct_index_spec("COSINE")
    index = spec.indexes[0]
    field = spec.struct_arrays[0].fields[0]
    meta = {
        "primary_field": "id",
        "min_pk": 3,
        "max_pk": 3,
        "data_min_pk": 3,
        "data_max_pk": 3,
    }

    data_pk, expected_pk, vector, offset = (
        validate_index_compatibility._vector_index_probe(
            spec, meta, index, field, seed=7
        )
    )
    report = ValidationReport()
    validate_index_compatibility._validate_vector_search_hit(
        [[{"id": expected_pk, "offset": offset, "distance": 1.0}]],
        "qa_struct",
        index.field,
        "id",
        expected_pk,
        offset,
        "COSINE",
        report,
    )

    assert data_pk == 3
    assert offset == 0
    assert vector == stable_vector_value(field, 3000, 7)
    assert report.passed

    mismatch = ValidationReport()
    validate_index_compatibility._validate_vector_search_hit(
        [[{"id": expected_pk, "offset": 1, "distance": 1.0}]],
        "qa_struct",
        index.field,
        "id",
        expected_pk,
        offset,
        "COSINE",
        mismatch,
    )
    assert not mismatch.passed
    assert mismatch.failures[0]["expected_offset"] == 0


def test_lossy_l2_index_allows_bounded_self_search_quantization_error():
    lossy = ValidationReport()
    validate_index_compatibility._validate_vector_search_hit(
        [[{"id": 0, "distance": 0.04343560338020325}]],
        "qa_lossy",
        "ivf_pq_vector",
        "id",
        0,
        None,
        "L2",
        lossy,
        index_type="IVF_PQ",
        lossy_index=True,
    )

    exact = ValidationReport()
    validate_index_compatibility._validate_vector_search_hit(
        [[{"id": 0, "distance": 0.04343560338020325}]],
        "qa_exact",
        "flat_vector",
        "id",
        0,
        None,
        "L2",
        exact,
        index_type="FLAT",
    )

    assert lossy.passed
    assert not exact.passed
    assert exact.failures[0]["max_distance"] == 1e-3


def test_diskann_max_sim_negative_score_skipped_on_v3_0_0_baseline_bug():
    known = ValidationReport()
    validate_index_compatibility._validate_vector_search_hit(
        [[{"id": 0, "distance": -0.999978244304657}]],
        "qa_diskann",
        "embeddings[vector]",
        "id",
        0,
        None,
        "MAX_SIM_COSINE",
        known,
        index_type="DISKANN",
        diskann_max_sim_bug=True,
    )

    regressed = ValidationReport()
    validate_index_compatibility._validate_vector_search_hit(
        [[{"id": 0, "distance": -0.999978244304657}]],
        "qa_diskann",
        "embeddings[vector]",
        "id",
        0,
        None,
        "MAX_SIM_COSINE",
        regressed,
        index_type="DISKANN",
        diskann_max_sim_bug=False,
    )

    assert known.passed
    assert known.metrics.get("diskann_max_sim_negative_score_known") is True
    assert not regressed.passed
    assert regressed.failures[0]["type"] == "INDEX_SEARCH_FAILED"


def test_describe_index_preserves_top_level_compatibility_params():
    class Client:
        def describe_index(self, **kwargs):
            return {
                "index_name": "faiss_idx",
                "field_name": "embedding",
                "index_type": "FAISS",
                "metric_type": "COSINE",
                "faiss_index_name": "OPQ16,IVF64,PQ16x4",
                "refine": True,
            }

    metadata = validate_index_compatibility._describe_index(
        Client(), "qa", "embedding", "faiss_idx"
    )

    assert metadata["params"] == {
        "faiss_index_name": "OPQ16,IVF64,PQ16x4",
        "refine": True,
    }
    assert (
        validate_index_compatibility._index_identity(metadata)["compatibility_params"]
        == metadata["params"]
    )


def test_resolved_autoindex_type_uses_server_resolved_params():
    spec = SchemaSpec(
        name="json_auto",
        version="3.0",
        fields=[
            FieldSpec(name="id", dtype="INT64", primary=True),
            FieldSpec(name="json_auto", dtype="JSON"),
        ],
        indexes=[
            IndexSpec(
                field="json_auto",
                index_type="AUTOINDEX",
                expected_resolved_index_type="HYBRID",
            )
        ],
    )
    report = ValidationReport()

    validate_index_compatibility._validate_resolved_index_types(
        "qa_json_auto",
        spec,
        [
            {
                "field_name": "json_auto",
                "index_type": "AUTOINDEX",
                "params": {"index_type": "HYBRID"},
            }
        ],
        report,
    )

    assert report.passed
    assert report.metrics == {
        "qa_json_auto.json_auto.resolved_index_type.expected": "HYBRID",
        "qa_json_auto.json_auto.resolved_index_type.observed": "HYBRID",
        "qa_json_auto.json_auto.resolved_index_type.source": "params.index_type",
    }


def test_resolved_autoindex_type_fails_when_real_sdk_metadata_is_unobservable():
    spec = SchemaSpec(
        name="json_auto",
        version="3.0",
        fields=[
            FieldSpec(name="id", dtype="INT64", primary=True),
            FieldSpec(name="json_auto", dtype="JSON"),
        ],
        indexes=[
            IndexSpec(
                field="json_auto",
                index_type="AUTOINDEX",
                expected_resolved_index_type="HYBRID",
            )
        ],
    )
    report = ValidationReport()

    validate_index_compatibility._validate_resolved_index_types(
        "qa_json_auto",
        spec,
        [
            {
                "field_name": "json_auto",
                "index_type": "AUTOINDEX",
                "params": {
                    "json_cast_type": "double",
                    "json_path": "json_auto['score']",
                },
            }
        ],
        report,
    )

    assert not report.passed
    assert report.metrics == {
        "qa_json_auto.json_auto.resolved_index_type.expected": "HYBRID",
        "qa_json_auto.json_auto.resolved_index_type.observed": "unavailable",
        "qa_json_auto.json_auto.resolved_index_type.source": "public_sdk_unavailable",
        "resolved_index_types_unobservable_total": 1,
    }
    assert report.failures == [
        {
            "type": validate_index_compatibility.INDEX_METADATA_MISMATCH,
            "message": "resolved index type is required but unavailable",
            "collection": "qa_json_auto",
            "field": "json_auto",
            "expected_resolved_index_type": "HYBRID",
            "actual_index_type": None,
            "resolved_index_type_source": "public_sdk_unavailable",
        }
    ]


def test_resolved_autoindex_type_fails_on_explicit_mismatch():
    spec = SchemaSpec(
        name="json_auto",
        version="3.0",
        fields=[
            FieldSpec(name="id", dtype="INT64", primary=True),
            FieldSpec(name="json_auto", dtype="JSON"),
        ],
        indexes=[
            IndexSpec(
                field="json_auto",
                index_type="AUTOINDEX",
                expected_resolved_index_type="HYBRID",
            )
        ],
    )
    report = ValidationReport()

    validate_index_compatibility._validate_resolved_index_types(
        "qa_json_auto",
        spec,
        [
            {
                "field_name": "json_auto",
                "index_type": "AUTOINDEX",
                "params": {"resolved_index_type": "INVERTED"},
            }
        ],
        report,
    )

    assert not report.passed
    assert report.failures == [
        {
            "type": validate_index_compatibility.INDEX_METADATA_MISMATCH,
            "message": "resolved index type differs from schema matrix expectation",
            "collection": "qa_json_auto",
            "field": "json_auto",
            "expected_resolved_index_type": "HYBRID",
            "actual_index_type": "INVERTED",
            "resolved_index_type_source": "params.resolved_index_type",
        }
    ]


def test_resolved_autoindex_type_is_accepted_from_top_level_metadata():
    spec = SchemaSpec(
        name="json_auto",
        version="3.0",
        fields=[
            FieldSpec(name="id", dtype="INT64", primary=True),
            FieldSpec(name="json_auto", dtype="JSON"),
        ],
        indexes=[
            IndexSpec(
                field="json_auto",
                index_type="AUTOINDEX",
                expected_resolved_index_type="HYBRID",
            )
        ],
    )
    actual = [
        {
            "field_name": "json_auto",
            "index_name": "json_auto",
            "index_type": "HYBRID",
            "metric_type": None,
            "params": {},
        }
    ]
    report = ValidationReport()

    validate_index_compatibility._validate_index_metadata_matches_spec(
        "qa_json_auto", spec, actual, report
    )
    validate_index_compatibility._validate_resolved_index_types(
        "qa_json_auto", spec, actual, report
    )

    assert report.passed


def test_actual_index_metadata_must_match_schema_matrix():
    spec = SchemaSpec(
        name="scalar_index",
        version="3.0",
        fields=[
            FieldSpec(name="id", dtype="INT64", primary=True),
            FieldSpec(name="category", dtype="INT64"),
        ],
        indexes=[IndexSpec(field="category", index_type="INVERTED")],
    )
    report = ValidationReport()

    validate_index_compatibility._validate_index_metadata_matches_spec(
        "qa_scalar",
        spec,
        [
            {
                "field_name": "category",
                "index_name": "category",
                "index_type": "BITMAP",
                "metric_type": None,
                "params": {},
            }
        ],
        report,
    )

    assert not report.passed
    assert report.failures[0]["type"] == "INDEX_METADATA_MISMATCH"
    assert report.failures[0]["expected_index_types"] == ["INVERTED"]
    assert report.failures[0]["actual_index_type"] == "BITMAP"
