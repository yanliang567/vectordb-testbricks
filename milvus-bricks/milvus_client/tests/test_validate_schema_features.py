import json

from milvus_client.common.schema import FieldSpec, SchemaSpec
from milvus_client.requests import validate_schema_features


def _args(tmp_path, checkpoint, output):
    return [
        "--uri",
        "http://localhost:19530",
        "--collection-prefix",
        "qa",
        "--checkpoint-dir",
        str(tmp_path),
        "--output-json",
        str(output),
        "--schema-matrix",
        "schema.yaml",
        "--checkpoint-file",
        str(checkpoint),
    ]


def test_validate_schema_features_executes_declared_feature_validators(
    monkeypatch, tmp_path
):
    checkpoint = tmp_path / "seed.json"
    checkpoint.write_text(
        json.dumps(
            {
                "collections": {
                    "qa_geometry": {
                        "schema_name": "geometry",
                        "min_pk": 0,
                        "max_pk": 2,
                    }
                }
            }
        )
    )
    output = tmp_path / "result.json"
    spec = SchemaSpec(
        name="geometry",
        version="2.6",
        fields=[
            FieldSpec(name="id", dtype="INT64", primary=True),
            FieldSpec(name="location", dtype="GEOMETRY"),
        ],
        validators=["count", "geometry_filter"],
    )

    class Client:
        def query(self, **kwargs):
            return [{"id": 0}]

    monkeypatch.setattr(
        validate_schema_features, "load_schema_matrix", lambda path: [spec]
    )
    monkeypatch.setattr(
        validate_schema_features, "create_client", lambda *args, **kwargs: Client()
    )

    code = validate_schema_features.main(_args(tmp_path, checkpoint, output))

    result = json.loads(output.read_text())
    assert code == 0
    assert result["status"] == "passed"
    assert result["metrics"]["feature_validators_executed"] == 1
    assert result["metrics"]["external_validators_skipped"] == 1


def test_validate_schema_features_rejects_unknown_validator(monkeypatch, tmp_path):
    checkpoint = tmp_path / "seed.json"
    checkpoint.write_text(json.dumps({"collections": {}}))
    output = tmp_path / "result.json"
    spec = SchemaSpec(
        name="unknown",
        version="3.0",
        fields=[FieldSpec(name="id", dtype="INT64", primary=True)],
        validators=["not_implemented"],
    )
    monkeypatch.setattr(
        validate_schema_features, "load_schema_matrix", lambda path: [spec]
    )

    code = validate_schema_features.main(_args(tmp_path, checkpoint, output))

    result = json.loads(output.read_text())
    assert code == 2
    assert result["failures"][0]["type"] == "UNKNOWN_SCHEMA_VALIDATOR"


def test_validate_schema_features_rejects_missing_matrix_collection(
    monkeypatch, tmp_path
):
    checkpoint = tmp_path / "seed.json"
    checkpoint.write_text(json.dumps({"collections": {}}))
    output = tmp_path / "result.json"
    spec = SchemaSpec(
        name="geometry",
        version="2.6",
        fields=[
            FieldSpec(name="id", dtype="INT64", primary=True),
            FieldSpec(name="location", dtype="GEOMETRY"),
        ],
        validators=["geometry_filter"],
    )
    monkeypatch.setattr(
        validate_schema_features, "load_schema_matrix", lambda path: [spec]
    )

    code = validate_schema_features.main(_args(tmp_path, checkpoint, output))

    result = json.loads(output.read_text())
    assert code == 2
    assert result["failures"][0]["type"] == "SCHEMA_COLLECTION_MISSING"
    assert result["failures"][0]["collections"] == ["qa_geometry"]
