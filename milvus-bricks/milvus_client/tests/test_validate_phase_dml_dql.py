import json
import math
import re

import pytest

from milvus_client.common.data import generate_rows
from milvus_client.common.schema import (
    FieldSpec,
    FunctionSpec,
    IndexSpec,
    SchemaSpec,
    StructArraySpec,
)
from milvus_client.common.validators import ValidationReport
from milvus_client.requests import validate_phase_dml_dql


class PhaseClient:
    def __init__(
        self,
        *,
        auto_id: bool = False,
        search_fails: bool = False,
        search_hit_id=None,
        search_hit_offset=None,
        search_distance: float = 1.0,
    ):
        self.auto_id = auto_id
        self.search_fails = search_fails
        self.search_hit_id = search_hit_id
        self.search_hit_offset = search_hit_offset
        self.search_distance = search_distance
        self.calls = []
        self.collections = {"qa_dense"}
        self.rows = {}
        self.next_id = 1000

    def has_collection(self, collection_name):
        self.calls.append(("has_collection", collection_name))
        return collection_name in self.collections

    def release_collection(self, *args, **kwargs):
        self.calls.append(("release_collection", {"args": args, **kwargs}))

    def drop_collection(self, collection_name):
        self.calls.append(("drop_collection", collection_name))
        self.collections.discard(collection_name)

    def create_collection(self, **kwargs):
        self.calls.append(("create_collection", kwargs))
        self.collections.add(kwargs["collection_name"])

    def create_index(self, **kwargs):
        self.calls.append(("create_index", kwargs))

    def load_collection(self, *args, **kwargs):
        self.calls.append(("load_collection", {"args": args, **kwargs}))

    def flush(self, *args, **kwargs):
        self.calls.append(("flush", {"args": args, **kwargs}))

    def insert(self, **kwargs):
        rows = kwargs["data"]
        self.calls.append(("insert", kwargs))
        if not self.auto_id:
            self._store_rows(kwargs["collection_name"], rows)
            return {"insert_count": len(rows)}
        ids = list(range(self.next_id, self.next_id + len(rows)))
        self.next_id += len(rows)
        self._store_rows(
            kwargs["collection_name"],
            [{**row, "id": pk} for row, pk in zip(rows, ids)],
        )
        return {"ids": ids}

    def upsert(self, **kwargs):
        self.calls.append(("upsert", kwargs))
        self._store_rows(kwargs["collection_name"], kwargs["data"])
        return {"upsert_count": len(kwargs["data"])}

    def delete(self, **kwargs):
        self.calls.append(("delete", kwargs))
        collection = kwargs["collection_name"]
        for value in re.findall(r"\d+", kwargs.get("filter", "")):
            self.rows.get(collection, {}).pop(int(value), None)
        return {"delete_count": 1}

    def drop_pk_range(self, collection_name, start_id, rows):
        for pk in range(start_id, start_id + rows):
            self.rows.get(collection_name, {}).pop(pk, None)

    def _store_rows(self, collection_name, rows):
        target = self.rows.setdefault(collection_name, {})
        for row in rows:
            if "id" in row:
                target[row["id"]] = dict(row)

    def _project_rows(self, collection_name, pks, output_fields):
        rows = []
        for pk in pks:
            row = self.rows.get(collection_name, {}).get(pk)
            if not row:
                continue
            rows.append({field: row.get(field) for field in output_fields})
        return rows

    def _rows_matching_filter(self, collection_name, filter_expr):
        rows_by_pk = self.rows.get(collection_name, {})
        if not filter_expr:
            return list(rows_by_pk.values())
        range_match = re.search(r"id\s*>=\s*(\d+)\s*&&\s*id\s*<=\s*(\d+)", filter_expr)
        if range_match:
            min_pk = int(range_match.group(1))
            max_pk = int(range_match.group(2))
            return [row for pk, row in rows_by_pk.items() if min_pk <= pk <= max_pk]
        equality = re.search(r"id\s*==\s*(\d+)", filter_expr)
        if equality:
            row = rows_by_pk.get(int(equality.group(1)))
            return [row] if row else []
        return list(rows_by_pk.values())

    def query(self, **kwargs):
        self.calls.append(("query", kwargs))
        collection_name = kwargs.get("collection_name", "")
        output_fields = kwargs.get("output_fields", [])
        filter_expr = kwargs.get("filter", "")
        if output_fields == ["count(*)"]:
            return [
                {
                    "count(*)": len(
                        self._rows_matching_filter(collection_name, filter_expr)
                    )
                }
            ]
        if " in [" in filter_expr:
            pks = [int(value) for value in re.findall(r"\d+", filter_expr)]
            rows = self._project_rows(collection_name, pks, output_fields)
            return rows
        if collection_name == "qa_dense" and (
            "== 50000000" in filter_expr or "== 70000000" in filter_expr
        ):
            return []
        if collection_name == "qa_after_upgrade_dense" and "== 80000000" in filter_expr:
            return []
        if "== 1000" in filter_expr:
            return []
        equalities = re.findall(r"==\s*(\d+)", filter_expr)
        if equalities:
            rows = self._project_rows(
                collection_name, [int(equalities[-1])], output_fields
            )
            if rows:
                return rows
        return [{"id": 1}]

    def search(self, **kwargs):
        self.calls.append(("search", kwargs))
        if self.search_fails:
            raise RuntimeError("search unavailable")
        equality = re.search(r"==\s*(\d+)", kwargs.get("filter", ""))
        hit_id = self.search_hit_id
        if hit_id is None:
            hit_id = int(equality.group(1)) if equality else 1
        hit = {"id": hit_id, "distance": self.search_distance}
        if self.search_hit_offset is not None:
            hit["offset"] = self.search_hit_offset
        return [[hit]]


class StoredVectorSearchPhaseClient(PhaseClient):
    def search(self, **kwargs):
        self.calls.append(("search", kwargs))
        equality = re.search(r"==\s*(\d+)", kwargs.get("filter", ""))
        hit_id = int(equality.group(1))
        query = kwargs["data"][0]
        stored = self.rows[kwargs["collection_name"]][hit_id][kwargs["anns_field"]]
        numerator = sum(
            float(left) * float(right) for left, right in zip(query, stored)
        )
        query_norm = math.sqrt(sum(float(value) ** 2 for value in query))
        stored_norm = math.sqrt(sum(float(value) ** 2 for value in stored))
        return [[{"id": hit_id, "distance": numerator / (query_norm * stored_norm)}]]


class NoopUpsertPhaseClient(PhaseClient):
    def upsert(self, **kwargs):
        self.calls.append(("upsert", kwargs))
        return {"upsert_count": len(kwargs["data"])}


class MissingAutoIdResponseClient(PhaseClient):
    def __init__(self):
        super().__init__(auto_id=True)

    def insert(self, **kwargs):
        self.calls.append(("insert", kwargs))
        return {}


class FailStrictReloadLoadPhaseClient(PhaseClient):
    def __init__(self):
        super().__init__()
        self.strict_reload_started = False

    def release_collection(self, *args, **kwargs):
        self.strict_reload_started = True
        super().release_collection(*args, **kwargs)

    def load_collection(self, *args, **kwargs):
        if self.strict_reload_started:
            raise RuntimeError("persisted index reload failed")
        super().load_collection(*args, **kwargs)


class CorruptAfterReloadPhaseClient(PhaseClient):
    def __init__(self, *, corrupt_vector: bool = False, corrupt_scalar: bool = False):
        super().__init__()
        self.corrupt_vector = corrupt_vector
        self.corrupt_scalar = corrupt_scalar
        self.strict_reload_started = False

    def release_collection(self, *args, **kwargs):
        self.strict_reload_started = True
        super().release_collection(*args, **kwargs)

    def query(self, **kwargs):
        if (
            self.strict_reload_started
            and self.corrupt_scalar
            and "category" in kwargs.get("filter", "")
        ):
            self.calls.append(("query", kwargs))
            return []
        return super().query(**kwargs)

    def search(self, **kwargs):
        if self.strict_reload_started and self.corrupt_vector:
            self.calls.append(("search", kwargs))
            return [[{"id": 1, "distance": 1.0}]]
        return super().search(**kwargs)


def _dense_spec(auto_id: bool = False, dim: int = 4) -> SchemaSpec:
    return SchemaSpec(
        name="dense",
        version="test",
        fields=[
            FieldSpec(name="id", dtype="INT64", primary=True, auto_id=auto_id),
            FieldSpec(name="category", dtype="INT64"),
            FieldSpec(name="embedding", dtype="FLOAT_VECTOR", dim=dim),
        ],
        indexes=[
            IndexSpec(field="category", index_type="INVERTED"),
            IndexSpec(field="embedding", index_type="HNSW", metric_type="COSINE"),
        ],
    )


def _explicit_partition_spec() -> SchemaSpec:
    return SchemaSpec(
        name="explicit_partition",
        version="test",
        partitions=["p0", "p1"],
        fields=[
            FieldSpec(
                name="pk",
                dtype="VARCHAR",
                primary=True,
                auto_id=False,
                max_length=64,
            ),
            FieldSpec(name="embedding", dtype="FLOAT_VECTOR", dim=4),
        ],
        indexes=[IndexSpec(field="embedding", index_type="HNSW", metric_type="COSINE")],
    )


def _auto_id_partition_spec() -> SchemaSpec:
    return SchemaSpec(
        name="auto_id_partition",
        version="test",
        partitions=["p0", "p1"],
        fields=[
            FieldSpec(name="id", dtype="INT64", primary=True, auto_id=True),
            FieldSpec(name="embedding", dtype="FLOAT_VECTOR", dim=4),
        ],
        indexes=[IndexSpec(field="embedding", index_type="HNSW", metric_type="COSINE")],
    )


def _minhash_spec() -> SchemaSpec:
    return SchemaSpec(
        name="minhash",
        version="3.0",
        fields=[
            FieldSpec(name="id", dtype="INT64", primary=True),
            FieldSpec(
                name="document",
                dtype="VARCHAR",
                max_length=65535,
                value_profile="minhash_documents",
            ),
            FieldSpec(name="minhash", dtype="BINARY_VECTOR", dim=4096),
        ],
        functions=[
            FunctionSpec(
                name="text_to_minhash",
                function_type="MINHASH",
                input_fields=["document"],
                output_fields=["minhash"],
            )
        ],
        indexes=[
            IndexSpec(
                field="minhash",
                index_type="MINHASH_LSH",
                metric_type="MHJACCARD",
            )
        ],
    )


def _struct_array_spec() -> SchemaSpec:
    return SchemaSpec(
        name="struct_array",
        version="3.0",
        fields=[FieldSpec(name="id", dtype="INT64", primary=True)],
        struct_arrays=[
            StructArraySpec(
                name="items",
                max_capacity=4,
                fields=[
                    FieldSpec(name="embedding", dtype="FLOAT_VECTOR", dim=4),
                ],
            )
        ],
        indexes=[
            IndexSpec(
                field="items[embedding]",
                index_type="HNSW",
                metric_type="COSINE",
            )
        ],
    )


def _checkpoint(tmp_path):
    path = tmp_path / "seed_data.json"
    path.write_text(
        json.dumps(
            {
                "collections": {
                    "qa_dense": {
                        "schema_name": "dense",
                        "expected_count": 4,
                        "primary_field": "id",
                        "min_pk": 0,
                        "max_pk": 3,
                    }
                }
            }
        )
    )
    return path


def _minimal_phase_checkpoint_payload(
    *,
    version=2,
    phase="after-upgrade",
    seed=7,
):
    spec = _dense_spec()
    primary = next(field for field in spec.fields if field.primary)
    return {
        "version": version,
        "phase": phase,
        "existing_start_id": 50_000_000,
        "new_start_id": 60_000_000,
        "existing_dml_rows": 4,
        "existing_delete_rows": 1,
        "new_collection_rows": 4,
        "existing_collections": {
            "qa_dense": {
                "collection": "qa_dense",
                "schema_name": "dense",
                "primary_field": "id",
                "start_id": 50_000_000,
                "rows": 4,
                "inserted": 4,
                "inserted_values": [],
                "upserted": 4,
                "deleted": 1,
                "deleted_values": [50_000_000],
                "remaining_count": 3,
                "remaining_min_pk": 50_000_001,
                "remaining_max_pk": 50_000_003,
                "remaining_values": [50_000_001, 50_000_003],
                "upsert_samples": validate_phase_dml_dql._upsert_sample_payload(
                    spec,
                    primary,
                    50_000_000,
                    [1, 3],
                    seed,
                ),
                "upsert_skipped_auto_id": False,
                "search_probe_data_pk": 50_000_003,
                "search_probe_pk": 50_000_003,
                "search_probe_seed": seed + 101,
                "searches": 1,
                "scalar_index_queries": 1,
            }
        },
        "new_collections": {
            "qa_after_upgrade_dense": {
                "collection": "qa_after_upgrade_dense",
                "schema_name": "dense",
                "primary_field": "id",
                "start_id": 60_000_000,
                "rows": 4,
                "inserted": 4,
                "inserted_values": [],
                "min_pk": 60_000_000,
                "max_pk": 60_000_003,
                "sample_values": [60_000_000, 60_000_003],
                "search_probe_data_pk": 60_000_003,
                "search_probe_pk": 60_000_003,
                "search_probe_seed": seed + 17,
                "searches": 1,
                "scalar_index_queries": 1,
            }
        },
    }


def _phase_checkpoint_expectations():
    return {
        "expected_existing_collections": {"qa_dense": "dense"},
        "expected_new_collection_prefix": "qa_after_upgrade",
        "expected_existing_dml_rows": 4,
        "expected_existing_delete_rows": 1,
        "expected_new_collection_rows": 4,
    }


def _auto_id_phase_checkpoint_payload(seed=7):
    payload = _minimal_phase_checkpoint_payload(seed=seed)
    existing = payload["existing_collections"]["qa_dense"]
    existing_ids = [1000, 1001, 1002, 1003]
    existing.update(
        {
            "inserted_values": existing_ids,
            "upserted": 0,
            "deleted_values": [1000],
            "remaining_min_pk": None,
            "remaining_max_pk": None,
            "remaining_values": [1001, 1002, 1003],
            "upsert_samples": {"field": None, "samples": []},
            "upsert_skipped_auto_id": True,
            "search_probe_pk": 1003,
            "search_probe_seed": seed,
        }
    )
    new = payload["new_collections"]["qa_after_upgrade_dense"]
    new_ids = [2000, 2001, 2002, 2003]
    new.update(
        {
            "inserted_values": new_ids,
            "min_pk": None,
            "max_pk": None,
            "sample_values": [2000, 2001, 2002],
            "search_probe_pk": 2003,
        }
    )
    return payload


def _args(tmp_path, checkpoint):
    return [
        "--uri",
        "http://localhost:19530",
        "--collection-prefix",
        "qa",
        "--schema-matrix",
        "schema.yaml",
        "--checkpoint-file",
        str(checkpoint),
        "--checkpoint-dir",
        str(tmp_path),
        "--output-json",
        str(tmp_path / "result.json"),
        "--phase",
        "after-upgrade",
        "--new-collection-prefix",
        "qa_after_upgrade",
        "--new-collection-rows",
        "4",
        "--existing-dml-rows",
        "4",
        "--existing-delete-rows",
        "1",
        "--batch-size",
        "2",
        "--visibility-timeout-sec",
        "0",
        "--visibility-interval-sec",
        "0",
    ]


def _patch_schema_helpers(monkeypatch, spec):
    monkeypatch.setattr(
        validate_phase_dml_dql,
        "load_schema_matrix",
        lambda path: [spec],
    )
    monkeypatch.setattr(
        validate_phase_dml_dql,
        "build_milvus_schema",
        lambda spec: {"schema": spec.name},
    )
    monkeypatch.setattr(
        validate_phase_dml_dql,
        "build_index_params",
        lambda spec: {"indexes": [index.field for index in spec.indexes]},
    )


def test_phase_dml_dql_mutates_existing_and_creates_new_collection(
    monkeypatch, tmp_path
):
    checkpoint = _checkpoint(tmp_path)
    client = PhaseClient()
    _patch_schema_helpers(monkeypatch, _dense_spec())
    monkeypatch.setattr(
        validate_phase_dml_dql,
        "create_client",
        lambda *args, **kwargs: client,
    )

    code = validate_phase_dml_dql.main(_args(tmp_path, checkpoint))

    result = json.loads((tmp_path / "result.json").read_text())
    call_names = [name for name, _ in client.calls]
    assert code == 0
    assert result["status"] == "passed"
    assert result["metrics"]["existing_collections_total"] == 1
    assert result["metrics"]["new_collections_total"] == 1
    assert (
        result["metrics"]["reload_timeout_sec"]
        == validate_phase_dml_dql.DEFAULT_RELOAD_TIMEOUT_SEC
    )
    assert result["metrics"]["existing_inserted_total"] == 4
    assert result["metrics"]["existing_upserted_total"] == 4
    assert result["metrics"]["existing_deleted_total"] == 1
    assert result["metrics"]["new_collection_inserted_total"] == 4
    assert result["metrics"]["scalar_index_queries_total"] == 2
    assert result["metrics"]["phase_reload_attempted_collections_total"] == 2
    assert result["metrics"]["phase_reload_collections_total"] == 2
    assert result["metrics"]["phase_reload_failures_total"] == 0
    assert result["metrics"]["phase_reload_vector_searches_total"] == 2
    assert result["metrics"]["phase_reload_scalar_index_queries_total"] == 2
    assert result["metrics"]["existing_collections"][0]["scalar_index_queries"] == 1
    assert result["metrics"]["new_collections"][0]["scalar_index_queries"] == 1
    assert result["metrics"]["existing_collections"][0]["reload_succeeded"]
    assert result["metrics"]["new_collections"][0]["reload_succeeded"]
    maintenance_windows = result["metrics"]["maintenance_windows"]
    assert len(maintenance_windows) == 2
    assert {window["collection"] for window in maintenance_windows} == {
        "qa_dense",
        "qa_after_upgrade_dense",
    }
    assert {window["kind"] for window in maintenance_windows} == {"collection-reload"}
    assert {window["label"] for window in maintenance_windows} == {
        "phase-dml-dql-reload-after-upgrade"
    }
    assert "create_collection" in call_names
    assert "upsert" in call_names
    assert "delete" in call_names
    assert "search" in call_names

    search_positions = [
        index for index, call in enumerate(client.calls) if call[0] == "search"
    ]
    strict_release_positions = [
        index
        for index, call in enumerate(client.calls)
        if call[0] == "release_collection"
    ]
    assert len(search_positions) == 4
    assert len(strict_release_positions) == 2
    assert search_positions[0] < strict_release_positions[0] < search_positions[1]
    assert search_positions[2] < strict_release_positions[1] < search_positions[3]


def test_new_phase_strict_reload_load_failure_fails_closed():
    client = FailStrictReloadLoadPhaseClient()
    report = ValidationReport()

    metrics = validate_phase_dml_dql._run_new_collection_dml_dql(
        client,
        _dense_spec(),
        "qa_after_upgrade_dense",
        rows=4,
        batch_size=2,
        start_id=60_000_000,
        seed=24,
        drop_if_exists=False,
        report=report,
    )

    assert not report.passed
    assert metrics["reload_attempted"]
    assert not metrics["reload_operations_succeeded"]
    assert not metrics["reload_succeeded"]
    maintenance_windows = report.metrics["maintenance_windows"]
    assert len(maintenance_windows) == 1
    assert maintenance_windows[0]["kind"] == "collection-reload"
    assert maintenance_windows[0]["collection"] == "qa_after_upgrade_dense"
    assert any(
        failure["type"] == validate_phase_dml_dql.PHASE_COLLECTION_RELOAD_FAILED
        and failure["operation"] == "load_collection"
        for failure in report.failures
    )


def test_existing_phase_reload_revalidates_vector_search():
    client = CorruptAfterReloadPhaseClient(corrupt_vector=True)
    report = ValidationReport()

    metrics = validate_phase_dml_dql._run_existing_collection_dml_dql(
        client,
        _dense_spec(),
        "qa_dense",
        rows=4,
        delete_rows=1,
        batch_size=2,
        start_id=50_000_000,
        seed=7,
        visibility_timeout_sec=0,
        visibility_interval_sec=0,
        report=report,
    )

    assert not report.passed
    assert metrics["reload_operations_succeeded"]
    assert not metrics["reload_succeeded"]
    assert metrics["reload_vector_searches"] == 1
    assert any(
        failure["type"] == validate_phase_dml_dql.PHASE_DQL_FAILED
        and failure.get("actual_pks") == [1]
        for failure in report.failures
    )


def test_existing_phase_reload_revalidates_scalar_index_queries():
    client = CorruptAfterReloadPhaseClient(corrupt_scalar=True)
    report = ValidationReport()

    metrics = validate_phase_dml_dql._run_existing_collection_dml_dql(
        client,
        _dense_spec(),
        "qa_dense",
        rows=4,
        delete_rows=1,
        batch_size=2,
        start_id=50_000_000,
        seed=7,
        visibility_timeout_sec=0,
        visibility_interval_sec=0,
        report=report,
    )

    assert not report.passed
    assert metrics["reload_operations_succeeded"]
    assert not metrics["reload_succeeded"]
    assert metrics["reload_scalar_index_queries"] == 1
    assert any(
        failure["type"] == "INDEX_SCALAR_QUERY_FAILED"
        and failure.get("field") == "category"
        for failure in report.failures
    )


def test_phase_scalar_index_validation_uses_effective_server_version(monkeypatch):
    observed = {}

    def record_scalar_queries(*args, server_version=None, **kwargs):
        observed["server_version"] = server_version
        return 1

    monkeypatch.setattr(
        validate_phase_dml_dql,
        "validate_scalar_index_queries",
        record_scalar_queries,
    )

    queries = validate_phase_dml_dql._validate_phase_checkpoint_scalar_indexes(
        object(),
        _dense_spec(),
        {
            "collection": "qa_dense",
            "primary_field": "id",
            "start_id": 50_000_000,
            "rows": 4,
            "deleted": 1,
        },
        7,
        ValidationReport(),
        existing=True,
        server_version="3.0.0",
    )

    assert queries == 1
    assert observed["server_version"] == "3.0.0"


def test_existing_phase_search_uses_upserted_vector_values():
    spec = _dense_spec(dim=128)
    client = StoredVectorSearchPhaseClient()
    report = ValidationReport()

    metrics = validate_phase_dml_dql._run_existing_collection_dml_dql(
        client,
        spec,
        "qa_dense",
        rows=4,
        delete_rows=1,
        batch_size=2,
        start_id=50_000_000,
        seed=7,
        visibility_timeout_sec=0,
        visibility_interval_sec=0,
        report=report,
    )

    search_call = next(payload for name, payload in client.calls if name == "search")
    expected = generate_rows(spec, 50_000_003, 1, seed=108)[0]["embedding"]
    assert report.passed
    assert metrics["search_probe_seed"] == 108
    assert search_call["data"] == [expected]


def test_existing_phase_search_uses_deterministically_updated_vector():
    spec = SchemaSpec(
        name="vector_only",
        version="2.6",
        fields=[
            FieldSpec(name="id", dtype="INT64", primary=True),
            FieldSpec(name="embedding", dtype="FLOAT_VECTOR", dim=4),
        ],
        indexes=[IndexSpec(field="embedding", index_type="HNSW", metric_type="COSINE")],
    )
    client = StoredVectorSearchPhaseClient()
    report = ValidationReport()

    metrics = validate_phase_dml_dql._run_existing_collection_dml_dql(
        client,
        spec,
        "qa_vector_only",
        rows=4,
        delete_rows=1,
        batch_size=2,
        start_id=50_000_000,
        seed=7,
        visibility_timeout_sec=0,
        visibility_interval_sec=0,
        report=report,
    )

    search_call = next(payload for name, payload in client.calls if name == "search")
    expected = generate_rows(spec, 50_000_003, 1, seed=108)[0]
    validate_phase_dml_dql.apply_deterministic_update(spec, expected, 50_000_003)
    assert report.passed
    assert metrics["search_probe_seed"] == 108
    assert search_call["data"] == [expected["embedding"]]


def test_phase_dml_dql_minhash_search_uses_function_input_text():
    client = PhaseClient()
    report = ValidationReport()

    searches = validate_phase_dml_dql._run_searches(
        client,
        _minhash_spec(),
        "qa_minhash",
        seed=7,
        pk=6_000_001,
        report=report,
    )

    search_call = next(call[1] for call in client.calls if call[0] == "search")
    assert report.passed
    assert searches == 1
    assert search_call["data"] == ["the quick brown fox jumps over a lazy dog"]
    assert search_call["filter"] == "id == 6000001"
    assert search_call["search_params"]["metric_type"] == "MHJACCARD"


def test_phase_dml_dql_minhash_search_uses_updated_input_text_after_upsert():
    client = PhaseClient()
    report = ValidationReport()

    searches = validate_phase_dml_dql._run_searches(
        client,
        _minhash_spec(),
        "qa_minhash",
        seed=7,
        pk=6_000_001,
        report=report,
        apply_update=True,
    )

    search_call = next(call[1] for call in client.calls if call[0] == "search")
    assert report.passed
    assert searches == 1
    assert search_call["data"] == ["phase_upsert_6000001_milvus_upgrade_rollback"]
    assert search_call["filter"] == "id == 6000001"
    assert search_call["search_params"]["metric_type"] == "MHJACCARD"


def test_phase_search_rejects_irrelevant_old_primary_key_hit():
    client = PhaseClient(search_hit_id=1)
    report = ValidationReport()

    searches = validate_phase_dml_dql._run_searches(
        client,
        _dense_spec(),
        "qa_dense",
        seed=7,
        pk=50_000_003,
        report=report,
    )

    assert searches == 1
    assert not report.passed
    assert report.failures[0]["type"] == validate_phase_dml_dql.PHASE_DQL_FAILED
    assert report.failures[0]["expected_pk"] == 50_000_003
    assert report.failures[0]["actual_pks"] == [1]


def test_phase_struct_array_search_requires_matching_element_offset():
    client = PhaseClient(search_hit_offset=1)
    report = ValidationReport()

    searches = validate_phase_dml_dql._run_searches(
        client,
        _struct_array_spec(),
        "qa_struct_array",
        seed=7,
        pk=50_000_003,
        report=report,
    )

    assert searches == 1
    assert not report.passed
    assert report.failures[0]["expected_pk"] == 50_000_003
    assert report.failures[0]["expected_offset"] == 0
    assert report.failures[0]["actual_offsets"] == [1]


def test_phase_search_rejects_invalid_self_search_score():
    client = PhaseClient(search_distance=0.1)
    report = ValidationReport()

    searches = validate_phase_dml_dql._run_searches(
        client,
        _dense_spec(),
        "qa_dense",
        seed=7,
        pk=50_000_003,
        report=report,
    )

    assert searches == 1
    assert not report.passed
    assert report.failures[0]["type"] == validate_phase_dml_dql.PHASE_DQL_FAILED
    assert report.failures[0]["distance"] == 0.1
    assert report.failures[0]["min_score"] == 0.9


def test_phase_search_rejects_non_finite_score():
    client = PhaseClient(search_distance=float("nan"))
    report = ValidationReport()

    searches = validate_phase_dml_dql._run_searches(
        client,
        _dense_spec(),
        "qa_dense",
        seed=7,
        pk=50_000_003,
        report=report,
    )

    assert searches == 1
    assert not report.passed
    assert report.failures[0]["message"] == (
        "phase vector self-search returned a non-finite distance or score"
    )


def test_phase_bm25_search_requires_observable_score():
    report = ValidationReport()

    validate_phase_dml_dql._validate_phase_search_hit(
        [[{"id": 3}]],
        "qa_bm25",
        "sparse_bm25",
        "id",
        3,
        None,
        "BM25",
        "SPARSE_INVERTED_INDEX",
        report,
    )

    assert not report.passed
    assert report.failures[0]["message"] == (
        "phase vector self-search did not expose a distance or score"
    )


def test_wait_for_validation_retries_until_dml_becomes_visible(monkeypatch):
    attempts = 0

    def validate(report):
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            report.fail("COUNT_DRIFT", "DML is not visible yet")

    times = iter([0.0, 0.1])
    monkeypatch.setattr(validate_phase_dml_dql, "monotonic", lambda: next(times))
    monkeypatch.setattr(validate_phase_dml_dql, "sleep", lambda _: None)

    report, actual_attempts = validate_phase_dml_dql._wait_for_validation(
        validate,
        timeout_sec=1,
        interval_sec=0,
    )

    assert report.passed
    assert actual_attempts == 2


def test_phase_dml_dql_upserts_explicit_partition_rows_in_original_partitions():
    client = PhaseClient()
    spec = _explicit_partition_spec()
    rows = [
        {"pk": "pk_00000000000000000000", "embedding": [0.1] * 4},
        {"pk": "pk_00000000000000000001", "embedding": [0.2] * 4},
        {"pk": "pk_00000000000000000002", "embedding": [0.3] * 4},
    ]

    validate_phase_dml_dql._upsert_rows(client, spec, "qa_partitioned", rows, 0)

    upsert_calls = [payload for name, payload in client.calls if name == "upsert"]
    assert [call["partition_name"] for call in upsert_calls] == ["p0", "p1"]
    assert [len(call["data"]) for call in upsert_calls] == [2, 1]


def test_auto_id_insert_preserves_data_row_to_generated_pk_mapping_across_partitions():
    client = PhaseClient(auto_id=True)
    spec = _auto_id_partition_spec()
    rows = [
        {"embedding": [0.1] * 4},
        {"embedding": [0.2] * 4},
        {"embedding": [0.3] * 4},
    ]

    ids = validate_phase_dml_dql._insert_rows(
        client,
        spec,
        "qa_auto_partitioned",
        rows,
        start_id=0,
    )

    assert ids == [1000, 1002, 1001]


def test_phase_dml_dql_writes_after_upgrade_phase_checkpoint(monkeypatch, tmp_path):
    checkpoint = _checkpoint(tmp_path)
    phase_checkpoint = tmp_path / "phase_dml_dql_after_upgrade.json"
    client = PhaseClient()
    _patch_schema_helpers(monkeypatch, _dense_spec())
    monkeypatch.setattr(
        validate_phase_dml_dql,
        "create_client",
        lambda *args, **kwargs: client,
    )

    code = validate_phase_dml_dql.main(
        [
            *_args(tmp_path, checkpoint),
            "--phase-checkpoint-file",
            str(phase_checkpoint),
        ]
    )

    result = json.loads((tmp_path / "result.json").read_text())
    checkpoint_payload = json.loads(phase_checkpoint.read_text())
    assert code == 0
    assert result["status"] == "passed"
    assert result["checkpoint"]["version"] == 2
    assert checkpoint_payload["version"] == 2
    assert checkpoint_payload["phase"] == "after-upgrade"
    assert (
        checkpoint_payload["existing_collections"]["qa_dense"]["start_id"] == 50000000
    )
    assert (
        checkpoint_payload["existing_collections"]["qa_dense"]["remaining_count"] == 3
    )
    assert checkpoint_payload["existing_collections"]["qa_dense"]["deleted_values"] == [
        50000000
    ]
    assert checkpoint_payload["existing_collections"]["qa_dense"]["upsert_samples"]
    assert (
        checkpoint_payload["existing_collections"]["qa_dense"]["search_probe_data_pk"]
        == 50000003
    )
    assert (
        checkpoint_payload["existing_collections"]["qa_dense"]["search_probe_pk"]
        == 50000003
    )
    assert (
        checkpoint_payload["existing_collections"]["qa_dense"]["search_probe_seed"]
        == 101
    )
    assert (
        checkpoint_payload["new_collections"]["qa_after_upgrade_dense"]["inserted"] == 4
    )
    assert checkpoint_payload["new_collections"]["qa_after_upgrade_dense"][
        "sample_values"
    ] == [60000000, 60000003]
    assert (
        checkpoint_payload["new_collections"]["qa_after_upgrade_dense"][
            "search_probe_seed"
        ]
        == 17
    )
    assert (
        checkpoint_payload["existing_collections"]["qa_dense"]["scalar_index_queries"]
        == 1
    )
    assert (
        checkpoint_payload["new_collections"]["qa_after_upgrade_dense"][
            "scalar_index_queries"
        ]
        == 1
    )


def test_phase_checkpoint_reuses_recorded_existing_search_probe_seed():
    spec = _dense_spec(dim=128)
    client = StoredVectorSearchPhaseClient()
    client._store_rows(
        "qa_dense",
        generate_rows(spec, start_id=50_000_003, count=1, seed=108),
    )
    report = ValidationReport()

    searches = validate_phase_dml_dql._validate_existing_phase_checkpoint_collection(
        client,
        spec,
        {
            "collection": "qa_dense",
            "primary_field": "id",
            "remaining_count": 1,
            "remaining_min_pk": 50_000_003,
            "remaining_max_pk": 50_000_003,
            "remaining_values": [50_000_003],
            "deleted_values": [],
            "upsert_samples": {"field": None, "samples": []},
            "rows": 1,
            "start_id": 50_000_003,
            "search_probe_data_pk": 50_000_003,
            "search_probe_pk": 50_000_003,
            "search_probe_seed": 108,
        },
        report,
        seed=7,
    )

    search_call = next(payload for name, payload in client.calls if name == "search")
    expected = generate_rows(spec, 50_000_003, 1, seed=108)[0]["embedding"]
    assert searches == 1
    assert report.passed
    assert search_call["data"] == [expected]


def test_phase_checkpoint_reloads_existing_and_target_written_collections(
    monkeypatch,
    tmp_path,
):
    checkpoint = tmp_path / "phase.json"
    checkpoint.write_text(json.dumps(_minimal_phase_checkpoint_payload()))
    client = PhaseClient()
    client.collections.add("qa_after_upgrade_dense")
    report = ValidationReport()
    monkeypatch.setattr(
        validate_phase_dml_dql,
        "_validate_existing_phase_checkpoint_collection",
        lambda *args, **kwargs: 1,
    )
    monkeypatch.setattr(
        validate_phase_dml_dql,
        "_validate_new_phase_checkpoint_collection",
        lambda *args, **kwargs: 1,
    )
    monkeypatch.setattr(
        validate_phase_dml_dql,
        "_validate_phase_checkpoint_scalar_indexes",
        lambda *args, **kwargs: 1,
    )

    metrics = validate_phase_dml_dql._validate_phase_checkpoint_before_rollback(
        client,
        {"dense": _dense_spec()},
        checkpoint,
        seed=7,
        report=report,
        **_phase_checkpoint_expectations(),
    )

    assert report.passed
    assert metrics["phase_checkpoint_reload_collections_total"] == 2
    assert metrics["phase_checkpoint_reload_failures_total"] == 0
    assert metrics["phase_checkpoint_searches_total"] == 2
    maintenance_windows = report.metrics["maintenance_windows"]
    assert len(maintenance_windows) == 2
    assert {window["collection"] for window in maintenance_windows} == {
        "qa_dense",
        "qa_after_upgrade_dense",
    }
    assert {window["label"] for window in maintenance_windows} == {
        "phase-checkpoint-reload-after-rollback"
    }
    reload_calls = [
        call
        for call in client.calls
        if call[0] in {"release_collection", "load_collection"}
    ]
    assert reload_calls == [
        (
            "release_collection",
            {
                "args": (),
                "collection_name": "qa_dense",
                "timeout": validate_phase_dml_dql.DEFAULT_RELOAD_TIMEOUT_SEC,
            },
        ),
        (
            "load_collection",
            {
                "args": (),
                "collection_name": "qa_dense",
                "timeout": validate_phase_dml_dql.DEFAULT_RELOAD_TIMEOUT_SEC,
            },
        ),
        (
            "release_collection",
            {
                "args": (),
                "collection_name": "qa_after_upgrade_dense",
                "timeout": validate_phase_dml_dql.DEFAULT_RELOAD_TIMEOUT_SEC,
            },
        ),
        (
            "load_collection",
            {
                "args": (),
                "collection_name": "qa_after_upgrade_dense",
                "timeout": validate_phase_dml_dql.DEFAULT_RELOAD_TIMEOUT_SEC,
            },
        ),
    ]


@pytest.mark.parametrize(
    ("payload", "expected_field"),
    [
        ({}, "actual_version"),
        (_minimal_phase_checkpoint_payload(version=1), "actual_version"),
        (_minimal_phase_checkpoint_payload(phase="after-rollback"), "actual_phase"),
    ],
    ids=["empty", "wrong-version", "wrong-phase"],
)
def test_phase_checkpoint_contract_rejects_invalid_metadata_before_reload(
    tmp_path,
    payload,
    expected_field,
):
    checkpoint = tmp_path / "phase.json"
    checkpoint.write_text(json.dumps(payload))
    client = PhaseClient()
    report = ValidationReport()

    metrics = validate_phase_dml_dql._validate_phase_checkpoint_before_rollback(
        client,
        {"dense": _dense_spec()},
        checkpoint,
        seed=7,
        report=report,
        **_phase_checkpoint_expectations(),
    )

    assert not report.passed
    assert not metrics["phase_checkpoint_validated"]
    assert metrics["phase_checkpoint_reload_collections_total"] == 0
    assert not any(
        call[0] in {"release_collection", "load_collection"} for call in client.calls
    )
    assert any(
        failure["type"] == validate_phase_dml_dql.PHASE_CHECKPOINT_INVALID
        and expected_field in failure
        for failure in report.failures
    )


def test_phase_checkpoint_contract_rejects_malformed_json_before_reload(tmp_path):
    checkpoint = tmp_path / "phase.json"
    checkpoint.write_text("{")
    client = PhaseClient()
    report = ValidationReport()

    metrics = validate_phase_dml_dql._validate_phase_checkpoint_before_rollback(
        client,
        {"dense": _dense_spec()},
        checkpoint,
        seed=7,
        report=report,
        **_phase_checkpoint_expectations(),
    )

    assert not report.passed
    assert not metrics["phase_checkpoint_validated"]
    assert not client.calls
    assert report.failures[0]["type"] == validate_phase_dml_dql.PHASE_CHECKPOINT_INVALID


def test_phase_checkpoint_contract_requires_every_schema_in_both_groups(tmp_path):
    checkpoint = tmp_path / "phase.json"
    checkpoint.write_text(json.dumps(_minimal_phase_checkpoint_payload()))
    client = PhaseClient()
    report = ValidationReport()

    expectations = _phase_checkpoint_expectations()
    expectations["expected_existing_collections"]["qa_missing"] = "missing"
    metrics = validate_phase_dml_dql._validate_phase_checkpoint_before_rollback(
        client,
        {"dense": _dense_spec(), "missing": _dense_spec()},
        checkpoint,
        seed=7,
        report=report,
        **expectations,
    )

    assert not report.passed
    assert not metrics["phase_checkpoint_validated"]
    assert metrics["phase_checkpoint_reload_collections_total"] == 0
    schema_failures = [
        failure
        for failure in report.failures
        if failure["type"] == validate_phase_dml_dql.PHASE_CHECKPOINT_INVALID
        and failure.get("missing_schemas") == ["missing"]
    ]
    assert {failure["group"] for failure in schema_failures} == {
        "existing_collections",
        "new_collections",
    }


def test_phase_checkpoint_contract_rejects_overlapping_collection_groups(tmp_path):
    payload = _minimal_phase_checkpoint_payload()
    new_entry = payload["new_collections"].pop("qa_after_upgrade_dense")
    new_entry["collection"] = "qa_dense"
    payload["new_collections"]["qa_dense"] = new_entry
    checkpoint = tmp_path / "phase.json"
    checkpoint.write_text(json.dumps(payload))
    client = PhaseClient()
    report = ValidationReport()

    metrics = validate_phase_dml_dql._validate_phase_checkpoint_before_rollback(
        client,
        {"dense": _dense_spec()},
        checkpoint,
        seed=7,
        report=report,
        **_phase_checkpoint_expectations(),
    )

    assert not report.passed
    assert not metrics["phase_checkpoint_validated"]
    assert metrics["phase_checkpoint_reload_collections_total"] == 0
    assert any(
        failure["type"] == validate_phase_dml_dql.PHASE_CHECKPOINT_INVALID
        and failure.get("overlapping_collections") == ["qa_dense"]
        for failure in report.failures
    )


@pytest.mark.parametrize(
    ("group_name", "original_name", "replacement_name"),
    [
        ("existing_collections", "qa_dense", "stale_dense"),
        (
            "new_collections",
            "qa_after_upgrade_dense",
            "stale_after_upgrade_dense",
        ),
    ],
    ids=["existing-seed-identity", "target-prefix-identity"],
)
def test_phase_checkpoint_contract_binds_collection_names_to_workflow(
    tmp_path,
    group_name,
    original_name,
    replacement_name,
):
    payload = _minimal_phase_checkpoint_payload()
    entry = payload[group_name].pop(original_name)
    entry["collection"] = replacement_name
    payload[group_name][replacement_name] = entry
    checkpoint = tmp_path / "phase.json"
    checkpoint.write_text(json.dumps(payload))
    client = PhaseClient()
    report = ValidationReport()

    metrics = validate_phase_dml_dql._validate_phase_checkpoint_before_rollback(
        client,
        {"dense": _dense_spec()},
        checkpoint,
        seed=7,
        report=report,
        **_phase_checkpoint_expectations(),
    )

    assert not report.passed
    assert not metrics["phase_checkpoint_validated"]
    assert metrics["phase_checkpoint_reload_collections_total"] == 0
    assert any(
        failure["type"] == validate_phase_dml_dql.PHASE_CHECKPOINT_INVALID
        and failure.get("group") == group_name
        and original_name in failure.get("missing_collections", [])
        and replacement_name in failure.get("unexpected_collections", [])
        for failure in report.failures
    )


def test_phase_checkpoint_contract_rejects_zero_row_payload(tmp_path):
    payload = _minimal_phase_checkpoint_payload()
    payload.update(
        {
            "existing_dml_rows": 0,
            "existing_delete_rows": 0,
            "new_collection_rows": 0,
        }
    )
    payload["existing_collections"]["qa_dense"].update(
        {
            "rows": 0,
            "inserted": 0,
            "deleted": 0,
            "remaining_count": 0,
            "searches": 0,
            "scalar_index_queries": 0,
        }
    )
    payload["new_collections"]["qa_after_upgrade_dense"].update(
        {
            "rows": 0,
            "inserted": 0,
            "searches": 0,
            "scalar_index_queries": 0,
        }
    )
    checkpoint = tmp_path / "phase.json"
    checkpoint.write_text(json.dumps(payload))
    client = PhaseClient()
    report = ValidationReport()

    metrics = validate_phase_dml_dql._validate_phase_checkpoint_before_rollback(
        client,
        {"dense": _dense_spec()},
        checkpoint,
        seed=7,
        report=report,
        **_phase_checkpoint_expectations(),
    )

    assert not report.passed
    assert not metrics["phase_checkpoint_validated"]
    assert metrics["phase_checkpoint_reload_collections_total"] == 0
    assert any(
        failure["type"] == validate_phase_dml_dql.PHASE_CHECKPOINT_INVALID
        and failure.get("field") == "new_collection_rows"
        and failure.get("actual") == 0
        for failure in report.failures
    )


@pytest.mark.parametrize(
    "field_name",
    [
        "remaining_min_pk",
        "remaining_max_pk",
        "remaining_values",
        "deleted_values",
        "upsert_samples",
    ],
)
def test_phase_checkpoint_contract_requires_explicit_pk_oracles_before_reload(
    tmp_path,
    field_name,
):
    payload = _minimal_phase_checkpoint_payload()
    payload["existing_collections"]["qa_dense"].pop(field_name)
    checkpoint = tmp_path / "phase.json"
    checkpoint.write_text(json.dumps(payload))
    client = PhaseClient()
    report = ValidationReport()

    metrics = validate_phase_dml_dql._validate_phase_checkpoint_before_rollback(
        client,
        {"dense": _dense_spec()},
        checkpoint,
        seed=7,
        report=report,
        **_phase_checkpoint_expectations(),
    )

    assert not report.passed
    assert not metrics["phase_checkpoint_validated"]
    assert metrics["phase_checkpoint_reload_collections_total"] == 0
    assert not client.calls
    assert any(
        failure["type"] == validate_phase_dml_dql.PHASE_CHECKPOINT_INVALID
        and failure.get("field") == field_name
        for failure in report.failures
    )


def test_phase_checkpoint_contract_accepts_complete_auto_id_oracles():
    report = ValidationReport()

    valid = validate_phase_dml_dql._validate_phase_checkpoint_contract(
        _auto_id_phase_checkpoint_payload(),
        {"dense": _dense_spec(auto_id=True)},
        report,
        **_phase_checkpoint_expectations(),
        expected_seed=7,
    )

    assert valid
    assert report.passed


def test_phase_checkpoint_contract_rejects_incomplete_auto_id_oracles():
    payload = _auto_id_phase_checkpoint_payload()
    payload["existing_collections"]["qa_dense"].pop("inserted_values")
    report = ValidationReport()

    valid = validate_phase_dml_dql._validate_phase_checkpoint_contract(
        payload,
        {"dense": _dense_spec(auto_id=True)},
        report,
        **_phase_checkpoint_expectations(),
        expected_seed=7,
    )

    assert not valid
    assert not report.passed
    assert any(
        failure["type"] == validate_phase_dml_dql.PHASE_CHECKPOINT_INVALID
        and failure.get("group") == "existing_collections"
        and failure.get("expected_count") == 4
        for failure in report.failures
    )


def test_phase_checkpoint_reload_failure_fails_closed(monkeypatch, tmp_path):
    checkpoint = tmp_path / "phase.json"
    checkpoint.write_text(json.dumps(_minimal_phase_checkpoint_payload()))
    client = PhaseClient()
    client.collections.add("qa_after_upgrade_dense")
    original_load = client.load_collection

    def fail_target_written_load(*args, **kwargs):
        collection = kwargs.get("collection_name") or args[0]
        if collection == "qa_after_upgrade_dense":
            raise RuntimeError("persisted index load failed")
        return original_load(*args, **kwargs)

    client.load_collection = fail_target_written_load
    report = ValidationReport()
    monkeypatch.setattr(
        validate_phase_dml_dql,
        "_validate_existing_phase_checkpoint_collection",
        lambda *args, **kwargs: 1,
    )
    monkeypatch.setattr(
        validate_phase_dml_dql,
        "_validate_new_phase_checkpoint_collection",
        lambda *args, **kwargs: 1,
    )
    monkeypatch.setattr(
        validate_phase_dml_dql,
        "_validate_phase_checkpoint_scalar_indexes",
        lambda *args, **kwargs: 1,
    )

    metrics = validate_phase_dml_dql._validate_phase_checkpoint_before_rollback(
        client,
        {"dense": _dense_spec()},
        checkpoint,
        seed=7,
        report=report,
        **_phase_checkpoint_expectations(),
    )

    assert not report.passed
    assert metrics["phase_checkpoint_validated"] is False
    assert metrics["phase_checkpoint_reload_collections_total"] == 1
    assert metrics["phase_checkpoint_reload_failures_total"] == 1
    assert metrics["phase_checkpoint_searches_total"] == 1
    assert any(
        failure["type"] == "PHASE_CHECKPOINT_RELOAD_FAILED"
        and failure["collection"] == "qa_after_upgrade_dense"
        and failure["operation"] == "load_collection"
        for failure in report.failures
    )


def test_phase_checkpoint_queries_scalar_indexes_after_reload(monkeypatch, tmp_path):
    checkpoint = tmp_path / "phase.json"
    payload = _minimal_phase_checkpoint_payload()
    spec = _dense_spec()
    primary = next(field for field in spec.fields if field.primary)
    payload["existing_start_id"] = 100
    payload["new_start_id"] = 200
    payload["existing_collections"]["qa_dense"].update(
        {
            "primary_field": "id",
            "start_id": 100,
            "deleted_values": [100],
            "remaining_min_pk": 101,
            "remaining_max_pk": 103,
            "remaining_values": [101, 103],
            "upsert_samples": validate_phase_dml_dql._upsert_sample_payload(
                spec,
                primary,
                100,
                [1, 3],
                7,
            ),
            "search_probe_data_pk": 103,
            "search_probe_pk": 103,
        }
    )
    payload["new_collections"]["qa_after_upgrade_dense"].update(
        {
            "primary_field": "id",
            "start_id": 200,
            "min_pk": 200,
            "max_pk": 203,
            "sample_values": [200, 203],
            "search_probe_data_pk": 203,
            "search_probe_pk": 203,
        }
    )
    checkpoint.write_text(json.dumps(payload))
    client = PhaseClient()
    client.collections.add("qa_after_upgrade_dense")
    report = ValidationReport()
    observed = []
    monkeypatch.setattr(
        validate_phase_dml_dql,
        "_validate_existing_phase_checkpoint_collection",
        lambda *args, **kwargs: 0,
    )
    monkeypatch.setattr(
        validate_phase_dml_dql,
        "_validate_new_phase_checkpoint_collection",
        lambda *args, **kwargs: 0,
    )

    def record_scalar_queries(
        client,
        collection,
        spec,
        meta,
        seed,
        report,
        probe_overrides=None,
        server_version=None,
    ):
        observed.append((collection, meta, seed, probe_overrides))
        return 2

    monkeypatch.setattr(
        validate_phase_dml_dql,
        "validate_scalar_index_queries",
        record_scalar_queries,
    )

    metrics = validate_phase_dml_dql._validate_phase_checkpoint_before_rollback(
        client,
        {"dense": _dense_spec()},
        checkpoint,
        seed=7,
        report=report,
        **_phase_checkpoint_expectations(),
    )

    assert report.passed
    assert metrics["phase_checkpoint_scalar_index_queries_total"] == 4
    assert metrics["phase_checkpoint.qa_dense.scalar_index_queries_total"] == 2
    assert (
        metrics["phase_checkpoint.qa_after_upgrade_dense.scalar_index_queries_total"]
        == 2
    )
    expected_overrides = validate_phase_dml_dql._phase_upsert_scalar_probe_overrides(
        spec,
        payload["existing_collections"]["qa_dense"],
    )
    assert observed == [
        (
            "qa_dense",
            {
                "primary_field": "id",
                "min_pk": 101,
                "max_pk": 103,
                "data_min_pk": 101,
                "data_max_pk": 103,
            },
            7,
            expected_overrides,
        ),
        (
            "qa_after_upgrade_dense",
            {
                "primary_field": "id",
                "min_pk": 200,
                "max_pk": 203,
                "data_min_pk": 200,
                "data_max_pk": 203,
            },
            24,
            {},
        ),
    ]


def test_phase_checkpoint_scalar_index_meta_maps_remaining_auto_ids():
    meta = validate_phase_dml_dql._phase_checkpoint_index_meta(
        _dense_spec(auto_id=True),
        {
            "primary_field": "id",
            "start_id": 100,
            "rows": 4,
            "deleted": 1,
            "remaining_values": [1001, 1002, 1003],
        },
        existing=True,
    )

    assert meta == {
        "primary_field": "id",
        "min_pk": 101,
        "max_pk": 103,
        "data_min_pk": 101,
        "data_max_pk": 103,
        "pk_values": [1001, 1002, 1003],
    }


def test_phase_upsert_scalar_probe_overrides_use_checkpoint_values():
    struct_spec = SchemaSpec(
        name="struct_scalar",
        version="2.6",
        fields=[FieldSpec(name="id", dtype="INT64", primary=True)],
        struct_arrays=[
            StructArraySpec(
                name="items",
                max_capacity=4,
                fields=[
                    FieldSpec(name="score", dtype="FLOAT"),
                    FieldSpec(name="category", dtype="VARCHAR"),
                ],
            )
        ],
        indexes=[
            IndexSpec(field="items[score]", index_type="AUTOINDEX"),
            IndexSpec(field="items[category]", index_type="AUTOINDEX"),
        ],
    )

    overrides = validate_phase_dml_dql._phase_upsert_scalar_probe_overrides(
        struct_spec,
        {
            "start_id": 100,
            "deleted": 1,
            "upsert_samples": {
                "field": "items",
                "samples": [
                    {
                        "pk": 101,
                        "expected": [{"score": 91.25, "category": "updated_category"}],
                    }
                ],
            },
        },
    )

    assert overrides == {
        "items[score]": (101, 101, "MATCH_ANY(items, $[score] == 91.25)"),
        "items[category]": (
            101,
            101,
            'MATCH_ANY(items, $[category] == "updated_category")',
        ),
    }


def test_after_rollback_validates_after_upgrade_phase_checkpoint_before_new_dml(
    monkeypatch, tmp_path
):
    checkpoint = _checkpoint(tmp_path)
    phase_checkpoint = tmp_path / "phase_dml_dql_after_upgrade.json"
    client = PhaseClient()
    _patch_schema_helpers(monkeypatch, _dense_spec())
    monkeypatch.setattr(
        validate_phase_dml_dql,
        "create_client",
        lambda *args, **kwargs: client,
    )
    assert (
        validate_phase_dml_dql.main(
            [
                *_args(tmp_path, checkpoint),
                "--phase-checkpoint-file",
                str(phase_checkpoint),
            ]
        )
        == 0
    )
    client.drop_pk_range("qa_dense", 50000000, 4)
    client.drop_pk_range("qa_after_upgrade_dense", 60000000, 4)
    calls_before_rollback = len(client.calls)

    code = validate_phase_dml_dql.main(
        [
            *_args(tmp_path, checkpoint),
            "--phase",
            "after-rollback",
            "--new-collection-prefix",
            "qa_after_rollback",
            "--carried-collection-prefix",
            "qa_after_upgrade",
            "--existing-start-id",
            "70000000",
            "--new-start-id",
            "80000000",
            "--phase-checkpoint-file",
            str(phase_checkpoint),
            "--validate-phase-checkpoint",
            "true",
        ]
    )

    result = json.loads((tmp_path / "result.json").read_text())
    rollback_calls = client.calls[calls_before_rollback:]
    assert code == 1
    assert result["status"] == "failed"
    assert any(
        failure["type"] in {"COUNT_DRIFT", "MISSING_PK", "PHASE_UPSERT_NOT_APPLIED"}
        for failure in result["failures"]
    )
    assert not any(call[0] == "insert" for call in rollback_calls)


def test_phase_dml_dql_fails_when_upsert_does_not_update_values(monkeypatch, tmp_path):
    checkpoint = _checkpoint(tmp_path)
    client = NoopUpsertPhaseClient()
    _patch_schema_helpers(monkeypatch, _dense_spec())
    monkeypatch.setattr(
        validate_phase_dml_dql,
        "create_client",
        lambda *args, **kwargs: client,
    )

    code = validate_phase_dml_dql.main(_args(tmp_path, checkpoint))

    result = json.loads((tmp_path / "result.json").read_text())
    assert code == 1
    assert result["status"] == "failed"
    assert any(
        failure["type"] == "PHASE_UPSERT_NOT_APPLIED" for failure in result["failures"]
    )


def test_phase_dml_dql_fails_when_minhash_upsert_is_a_noop():
    client = NoopUpsertPhaseClient()
    report = ValidationReport()

    validate_phase_dml_dql._run_existing_collection_dml_dql(
        client,
        _minhash_spec(),
        "qa_minhash",
        rows=4,
        delete_rows=1,
        batch_size=2,
        start_id=50_000_000,
        seed=7,
        visibility_timeout_sec=0,
        visibility_interval_sec=0,
        report=report,
    )

    assert not report.passed
    assert any(
        failure["type"] == "PHASE_UPSERT_NOT_APPLIED" for failure in report.failures
    )


def test_phase_dml_dql_fails_when_struct_array_upsert_is_a_noop():
    client = NoopUpsertPhaseClient()
    report = ValidationReport()

    validate_phase_dml_dql._run_existing_collection_dml_dql(
        client,
        _struct_array_spec(),
        "qa_struct_array",
        rows=4,
        delete_rows=1,
        batch_size=2,
        start_id=50_000_000,
        seed=7,
        visibility_timeout_sec=0,
        visibility_interval_sec=0,
        report=report,
    )

    assert not report.passed
    assert any(
        failure["type"] == "PHASE_UPSERT_NOT_APPLIED" for failure in report.failures
    )


def test_phase_dml_dql_deletes_auto_id_inserted_rows_and_skips_upsert(
    monkeypatch, tmp_path
):
    checkpoint = _checkpoint(tmp_path)
    client = PhaseClient(auto_id=True)
    _patch_schema_helpers(monkeypatch, _dense_spec(auto_id=True))
    monkeypatch.setattr(
        validate_phase_dml_dql,
        "create_client",
        lambda *args, **kwargs: client,
    )

    code = validate_phase_dml_dql.main(_args(tmp_path, checkpoint))

    result = json.loads((tmp_path / "result.json").read_text())
    call_names = [name for name, _ in client.calls]
    assert code == 0
    assert result["status"] == "passed"
    assert result["metrics"]["existing_upserted_total"] == 0
    assert result["metrics"]["existing_upsert_skipped_auto_id_total"] == 1
    assert result["metrics"]["scalar_index_queries_total"] == 2
    existing = result["metrics"]["existing_collections"][0]
    assert existing["inserted_values"] == [1000, 1001, 1002, 1003]
    assert existing["remaining_values"] == [1001, 1002, 1003]
    assert existing["search_probe_data_pk"] == 50000003
    assert existing["search_probe_pk"] == 1003
    assert "upsert" not in call_names
    assert any(
        call[0] == "delete" and "1000" in call[1]["filter"] for call in client.calls
    )
    assert any(
        call[0] == "search" and call[1]["filter"] == "id == 1003"
        for call in client.calls
    )


def test_existing_auto_id_phase_fails_when_insert_response_has_no_ids():
    client = MissingAutoIdResponseClient()
    report = ValidationReport()

    metrics = validate_phase_dml_dql._run_existing_collection_dml_dql(
        client,
        _dense_spec(auto_id=True),
        "qa_dense",
        rows=4,
        delete_rows=1,
        batch_size=2,
        start_id=50_000_000,
        seed=7,
        visibility_timeout_sec=0,
        visibility_interval_sec=0,
        report=report,
    )

    assert not report.passed
    assert metrics["inserted"] == 0
    assert report.failures[0]["type"] == validate_phase_dml_dql.PHASE_DML_FAILED
    assert "returned 0 primary keys for 2 rows" in report.failures[0]["error"]


def test_phase_dml_dql_mutates_carried_upgrade_collection_after_rollback(
    monkeypatch, tmp_path
):
    checkpoint = _checkpoint(tmp_path)
    client = PhaseClient()
    client.collections.add("qa_after_upgrade_dense")
    _patch_schema_helpers(monkeypatch, _dense_spec())
    monkeypatch.setattr(
        validate_phase_dml_dql,
        "create_client",
        lambda *args, **kwargs: client,
    )

    code = validate_phase_dml_dql.main(
        [
            *_args(tmp_path, checkpoint),
            "--phase",
            "after-rollback",
            "--new-collection-prefix",
            "qa_after_rollback",
            "--carried-collection-prefix",
            "qa_after_upgrade",
            "--existing-start-id",
            "70000000",
            "--new-start-id",
            "80000000",
        ]
    )

    result = json.loads((tmp_path / "result.json").read_text())
    assert code == 0
    assert result["status"] == "passed"
    assert result["metrics"]["existing_collections_total"] == 1
    assert result["metrics"]["carried_collections_total"] == 1
    assert result["metrics"]["new_collections_total"] == 1
    assert result["metrics"]["carried_inserted_total"] == 4
    assert result["metrics"]["carried_upserted_total"] == 4
    assert result["metrics"]["carried_deleted_total"] == 1
    assert any(
        call[0] == "insert" and call[1]["collection_name"] == "qa_after_upgrade_dense"
        for call in client.calls
    )
    assert any(
        call[0] == "search" and call[1]["collection_name"] == "qa_after_upgrade_dense"
        for call in client.calls
    )


def test_phase_dml_dql_reports_search_failure(monkeypatch, tmp_path):
    checkpoint = _checkpoint(tmp_path)
    client = PhaseClient(search_fails=True)
    _patch_schema_helpers(monkeypatch, _dense_spec())
    monkeypatch.setattr(
        validate_phase_dml_dql,
        "create_client",
        lambda *args, **kwargs: client,
    )

    code = validate_phase_dml_dql.main(_args(tmp_path, checkpoint))

    result = json.loads((tmp_path / "result.json").read_text())
    assert code == 1
    assert result["status"] == "failed"
    assert result["failures"][-1]["type"] == "PHASE_DQL_FAILED"
