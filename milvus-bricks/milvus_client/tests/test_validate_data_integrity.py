import json

from milvus_client.common.schema import FieldSpec, SchemaSpec
from milvus_client.requests import validate_data_integrity


class CapturingValidationClient:
    def __init__(self):
        self.query_calls = []

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        output_fields = kwargs.get("output_fields", [])
        if output_fields == ["count(*)"]:
            return [{"count(*)": 3}]
        return [{"pk": "pk_00000000000000000000"}]


class AutoIdValidationClient:
    def __init__(self):
        self.query_calls = []

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        output_fields = kwargs.get("output_fields", [])
        if output_fields == ["count(*)"]:
            return [{"count(*)": 2}]
        if " in " in kwargs.get("filter", ""):
            return [{"id": 1010, "category": 1}, {"id": 1011, "category": 2}]
        return [{"id": 1010}]


def test_validate_data_integrity_writes_structured_unexpected_failure(monkeypatch, tmp_path):
    checkpoint = tmp_path / "seed_data.json"
    checkpoint.write_text(json.dumps({"collections": {}}))
    output_json = tmp_path / "result.json"
    monkeypatch.setattr(validate_data_integrity, "load_schema_matrix", lambda path: (_ for _ in ()).throw(RuntimeError("bad schema")))

    code = validate_data_integrity.main(
        [
            "--uri",
            "http://localhost:19530",
            "--collection-prefix",
            "qa",
            "--schema-matrix",
            "bad.yaml",
            "--checkpoint-file",
            str(checkpoint),
            "--checkpoint-dir",
            str(tmp_path),
            "--output-json",
            str(output_json),
        ]
    )

    result = json.loads(output_json.read_text())
    assert code == 4
    assert result["status"] == "failed"
    assert result["failures"][0]["type"] == "VALIDATION_FAILED"


def test_validate_data_integrity_generates_string_primary_key_filters(monkeypatch, tmp_path):
    checkpoint = tmp_path / "seed_data.json"
    checkpoint.write_text(
        json.dumps(
            {
                "collections": {
                    "qa_string": {
                        "schema_name": "string_pk",
                        "expected_count": 3,
                        "primary_field": "pk",
                        "min_pk": 0,
                        "max_pk": 2,
                    }
                }
            }
        )
    )
    output_json = tmp_path / "result.json"
    spec = SchemaSpec(
        name="string_pk",
        version="test",
        fields=[
            FieldSpec(name="pk", dtype="VARCHAR", primary=True, max_length=64),
            FieldSpec(name="embedding", dtype="FLOAT_VECTOR", dim=2),
        ],
    )
    client = CapturingValidationClient()
    monkeypatch.setattr(validate_data_integrity, "load_schema_matrix", lambda path: [spec])
    monkeypatch.setattr(validate_data_integrity, "create_client", lambda *args, **kwargs: client)

    code = validate_data_integrity.main(
        [
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
            str(output_json),
        ]
    )

    result = json.loads(output_json.read_text())
    assert code == 0
    assert result["status"] == "passed"
    assert client.query_calls[0]["filter"] == 'pk >= "pk_00000000000000000000" && pk <= "pk_00000000000000000002"'
    assert client.query_calls[1]["filter"] == 'pk == "pk_00000000000000000000"'
    assert client.query_calls[2]["filter"] == 'pk == "pk_00000000000000000001"'
    assert client.query_calls[3]["filter"] == 'pk == "pk_00000000000000000002"'


def test_validate_data_integrity_uses_auto_id_pk_values_for_checksum(monkeypatch, tmp_path):
    from milvus_client.common.data import stable_checksum

    rows = [{"id": 1010, "category": 1}, {"id": 1011, "category": 2}]
    checkpoint = tmp_path / "seed_data.json"
    checkpoint.write_text(
        json.dumps(
            {
                "collections": {
                    "qa_auto": {
                        "schema_name": "auto",
                        "expected_count": 2,
                        "primary_field": "id",
                        "min_pk": 1010,
                        "max_pk": 1011,
                        "pk_samples": [1010, 1011],
                        "pk_values": [1010, 1011],
                        "checksum_fields": ["category"],
                        "checksum": stable_checksum(rows, fields=["category"], primary_field="id"),
                    }
                }
            }
        )
    )
    output_json = tmp_path / "result.json"
    spec = SchemaSpec(
        name="auto",
        version="test",
        fields=[
            FieldSpec(name="id", dtype="INT64", primary=True, auto_id=True),
            FieldSpec(name="category", dtype="INT64"),
            FieldSpec(name="embedding", dtype="FLOAT_VECTOR", dim=2),
        ],
    )
    client = AutoIdValidationClient()
    monkeypatch.setattr(validate_data_integrity, "load_schema_matrix", lambda path: [spec])
    monkeypatch.setattr(validate_data_integrity, "create_client", lambda *args, **kwargs: client)

    code = validate_data_integrity.main(
        [
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
            str(output_json),
        ]
    )

    result = json.loads(output_json.read_text())
    assert code == 0
    assert result["status"] == "passed"
    assert client.query_calls[0]["filter"] == "id >= 1010 && id <= 1011"
    assert client.query_calls[-1]["filter"] == "id in [1010, 1011]"
