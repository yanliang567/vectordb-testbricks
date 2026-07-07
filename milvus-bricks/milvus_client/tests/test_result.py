import json

from milvus_client.common.result import BrickResult


def test_write_result_json(tmp_path):
    path = tmp_path / "result.json"
    result = BrickResult(
        brick="unit",
        feature_set="compat_2_6",
        compat_mode="rollback_safe",
        lifecycle_phase="steady_state",
        status="passed",
        target={"uri": "http://localhost:19530"},
        metrics={"requests_total": 1},
    )

    result.write(path)

    payload = json.loads(path.read_text())
    assert payload["brick"] == "unit"
    assert payload["status"] == "passed"
    assert payload["feature_set"] == "compat_2_6"
    assert payload["compat_mode"] == "rollback_safe"
    assert payload["lifecycle_phase"] == "steady_state"
    assert payload["skip_reason"] is None
    assert payload["metrics"]["requests_total"] == 1
    assert payload["failures"] == []
