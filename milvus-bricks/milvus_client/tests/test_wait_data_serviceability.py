import json

from milvus_client.common.schema import FieldSpec, SchemaSpec
from milvus_client.requests import wait_data_serviceability


class RecoveringClient:
    def __init__(self, fail_attempts: int):
        self.fail_attempts = fail_attempts
        self.query_calls = []

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        attempt = len([call for call in self.query_calls if call.get("output_fields") == ["count(*)"]])
        if kwargs.get("output_fields") == ["count(*)"] and attempt <= self.fail_attempts:
            raise RuntimeError("channel distribution is not serviceable: channel not available")
        if kwargs.get("output_fields") == ["count(*)"]:
            return [{"count(*)": 3}]
        return [{"id": 0}]


class CountDriftClient:
    def query(self, **kwargs):
        if kwargs.get("output_fields") == ["count(*)"]:
            return [{"count(*)": 2}]
        return [{"id": 0}]


def _checkpoint(path):
    checkpoint = path / "seed_data.json"
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


def _spec():
    return SchemaSpec(
        name="dense",
        version="test",
        fields=[
            FieldSpec(name="id", dtype="INT64", primary=True),
            FieldSpec(name="embedding", dtype="FLOAT_VECTOR", dim=2),
        ],
    )


def _args(tmp_path, checkpoint, output_json, *, timeout_sec="10", interval_sec="0"):
    return [
        "--uri",
        "http://localhost:19530",
        "--collection-prefix",
        "qa",
        "--schema-matrix",
        "schema.yaml",
        "--checkpoint-file",
        str(checkpoint),
        "--timeout-sec",
        timeout_sec,
        "--interval-sec",
        interval_sec,
        "--checkpoint-dir",
        str(tmp_path),
        "--output-json",
        str(output_json),
    ]


def test_wait_data_serviceability_records_recovery_duration(monkeypatch, tmp_path):
    checkpoint = _checkpoint(tmp_path)
    output_json = tmp_path / "result.json"
    client = RecoveringClient(fail_attempts=2)
    monotonic_values = iter([0, 0, 1, 2, 25])
    monkeypatch.setattr(wait_data_serviceability, "load_schema_matrix", lambda path: [_spec()])
    monkeypatch.setattr(wait_data_serviceability, "create_client", lambda *args, **kwargs: client)
    monkeypatch.setattr(wait_data_serviceability.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(wait_data_serviceability.time, "sleep", lambda seconds: None)

    code = wait_data_serviceability.main(_args(tmp_path, checkpoint, output_json))

    result = json.loads(output_json.read_text())
    assert code == 0
    assert result["status"] == "passed"
    assert result["brick"] == "wait_data_serviceability"
    assert result["metrics"]["recovered"] is True
    assert result["metrics"]["recovery_duration_sec"] == 25
    assert result["metrics"]["attempts"] == 3
    assert result["metrics"]["transient_failure_attempts"] == 2


def test_wait_data_serviceability_fails_fast_on_count_drift(monkeypatch, tmp_path):
    checkpoint = _checkpoint(tmp_path)
    output_json = tmp_path / "result.json"
    monkeypatch.setattr(wait_data_serviceability, "load_schema_matrix", lambda path: [_spec()])
    monkeypatch.setattr(wait_data_serviceability, "create_client", lambda *args, **kwargs: CountDriftClient())

    code = wait_data_serviceability.main(_args(tmp_path, checkpoint, output_json))

    result = json.loads(output_json.read_text())
    assert code == 1
    assert result["status"] == "failed"
    assert result["failures"][0]["type"] == "COUNT_DRIFT"
    assert result["metrics"]["recovered"] is False


def test_wait_data_serviceability_times_out_on_transient_errors(monkeypatch, tmp_path):
    checkpoint = _checkpoint(tmp_path)
    output_json = tmp_path / "result.json"
    client = RecoveringClient(fail_attempts=99)
    monotonic_values = iter([0, 0, 1, 1])
    monkeypatch.setattr(wait_data_serviceability, "load_schema_matrix", lambda path: [_spec()])
    monkeypatch.setattr(wait_data_serviceability, "create_client", lambda *args, **kwargs: client)
    monkeypatch.setattr(wait_data_serviceability.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(wait_data_serviceability.time, "sleep", lambda seconds: None)

    code = wait_data_serviceability.main(
        _args(tmp_path, checkpoint, output_json, timeout_sec="1", interval_sec="0")
    )

    result = json.loads(output_json.read_text())
    assert code == 1
    assert result["status"] == "failed"
    assert result["failures"][-1]["type"] == "SERVICEABILITY_TIMEOUT"
    assert result["metrics"]["recovered"] is False
    assert result["metrics"]["recovery_duration_sec"] == 1
