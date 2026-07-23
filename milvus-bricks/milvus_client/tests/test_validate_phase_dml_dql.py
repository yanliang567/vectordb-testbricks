import json
import re

from milvus_client.common.schema import FieldSpec, IndexSpec, SchemaSpec
from milvus_client.requests import validate_phase_dml_dql


class PhaseClient:
    def __init__(self, *, auto_id: bool = False, search_fails: bool = False):
        self.auto_id = auto_id
        self.search_fails = search_fails
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
            if rows:
                return rows
        if collection_name == "qa_dense" and (
            "== 50000000" in filter_expr or "== 70000000" in filter_expr
        ):
            return []
        if collection_name == "qa_after_upgrade_dense" and "== 80000000" in filter_expr:
            return []
        if "== 1000" in filter_expr:
            return []
        equality = re.search(r"==\s*(\d+)", filter_expr)
        if equality:
            rows = self._project_rows(
                collection_name, [int(equality.group(1))], output_fields
            )
            if rows:
                return rows
        return [{"id": 1}]

    def search(self, **kwargs):
        self.calls.append(("search", kwargs))
        if self.search_fails:
            raise RuntimeError("search unavailable")
        return [[{"id": 1, "distance": 0.1}]]


class NoopUpsertPhaseClient(PhaseClient):
    def upsert(self, **kwargs):
        self.calls.append(("upsert", kwargs))
        return {"upsert_count": len(kwargs["data"])}


def _dense_spec(auto_id: bool = False) -> SchemaSpec:
    return SchemaSpec(
        name="dense",
        version="test",
        fields=[
            FieldSpec(name="id", dtype="INT64", primary=True, auto_id=auto_id),
            FieldSpec(name="category", dtype="INT64"),
            FieldSpec(name="embedding", dtype="FLOAT_VECTOR", dim=4),
        ],
        indexes=[
            IndexSpec(field="category", index_type="INVERTED"),
            IndexSpec(field="embedding", index_type="HNSW", metric_type="COSINE"),
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
    assert result["metrics"]["existing_inserted_total"] == 4
    assert result["metrics"]["existing_upserted_total"] == 4
    assert result["metrics"]["existing_deleted_total"] == 1
    assert result["metrics"]["new_collection_inserted_total"] == 4
    assert "create_collection" in call_names
    assert "upsert" in call_names
    assert "delete" in call_names
    assert "search" in call_names


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
        checkpoint_payload["new_collections"]["qa_after_upgrade_dense"]["inserted"] == 4
    )
    assert checkpoint_payload["new_collections"]["qa_after_upgrade_dense"][
        "sample_values"
    ] == [60000000, 60000003]


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
    assert "upsert" not in call_names
    assert any(
        call[0] == "delete" and "1000" in call[1]["filter"] for call in client.calls
    )


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
