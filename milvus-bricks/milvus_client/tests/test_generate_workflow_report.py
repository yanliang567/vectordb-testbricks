import json
from pathlib import Path

from milvus_client.requests import generate_workflow_report


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _base_args(tmp_path: Path, *, pressure_fail_on_error: str) -> list[str]:
    return [
        "--results-dir",
        str(tmp_path / "results"),
        "--pressure-summary",
        str(tmp_path / "pressure-summary.json"),
        "--k8s-dir",
        str(tmp_path / "k8s"),
        "--env-snapshot",
        str(tmp_path / "reports" / "env_snapshot.json"),
        "--flow-summary",
        str(tmp_path / "reports" / "flow_summary.json"),
        "--output-json",
        str(tmp_path / "reports" / "orchestrator_report.json"),
        "--output-md",
        str(tmp_path / "reports" / "final_report.md"),
        "--workflow-name",
        "wf-123",
        "--workflow-uid",
        "uid-123",
        "--workflow-namespace",
        "qa",
        "--milvus-release-name",
        "wf-123",
        "--milvus-namespace",
        "qa-milvus",
        "--milvus-host",
        "wf-123-milvus.qa-milvus.svc",
        "--base-milvus-image",
        "harbor.milvus.io/milvusdb/milvus:v2.6.18",
        "--target-milvus-image",
        "harbor.milvus.io/milvusdb/milvus:2.6-latest",
        "--base-version",
        "2.6.18",
        "--target-version",
        "2.6.99",
        "--repo-url",
        "https://github.com/yanliang567/vectordb-testbricks.git",
        "--repo-revision",
        "main",
        "--schema-matrix",
        "milvus_client/manifests/schema_matrix_2_6.yaml",
        "--collection-prefix",
        "qa_upgrade",
        "--rows-per-collection",
        "5000",
        "--batch-size",
        "500",
        "--pressure-modules",
        "search_pressure query_pressure",
        "--pressure-fail-on-error",
        pressure_fail_on_error,
        "--observe-after-upgrade-sec",
        "60",
        "--observe-after-rollback-sec",
        "60",
    ]


def _write_successful_validation(tmp_path: Path) -> None:
    _write_json(tmp_path / "results" / "validate_before_upgrade.json", {"status": "passed"})
    _write_json(tmp_path / "results" / "validate_after_upgrade.json", {"status": "passed"})
    _write_json(tmp_path / "results" / "validate_after_rollback.json", {"status": "passed"})


def test_generate_workflow_report_marks_pressure_failures_as_warning_when_not_strict(tmp_path):
    _write_successful_validation(tmp_path)
    _write_json(
        tmp_path / "pressure-summary.json",
        {
            "total": 3,
            "passed": 2,
            "failed": 1,
            "fail_on_error": False,
            "failed_results": [{"file": "search_pressure_2.json", "brick": "search_pressure", "status": "failed"}],
        },
    )
    _write_json(tmp_path / "reports" / "env_snapshot.json", {"client_namespace": "qa"})
    _write_json(tmp_path / "reports" / "flow_summary.json", {"cleanup_status": "pending"})
    (tmp_path / "k8s").mkdir()
    (tmp_path / "k8s" / "pods.txt").write_text("pods")

    rc = generate_workflow_report.main(_base_args(tmp_path, pressure_fail_on_error="false"))

    report = json.loads((tmp_path / "reports" / "orchestrator_report.json").read_text())
    markdown = (tmp_path / "reports" / "final_report.md").read_text()
    assert rc == 0
    assert report["status"] == "warning"
    assert report["validation"]["passed"] is True
    assert report["pressure"]["failed"] == 1
    assert report["parameters"]["pressure_fail_on_error"] is False
    assert report["k8s_snapshot"]["pods.txt"] == str(tmp_path / "k8s" / "pods.txt")
    assert "## Validation" in markdown
    assert "## Pressure" in markdown
    assert "warning `search_pressure_2.json` `search_pressure`: failed" in markdown


def test_generate_workflow_report_fails_pressure_failures_in_strict_mode(tmp_path):
    _write_successful_validation(tmp_path)
    _write_json(
        tmp_path / "pressure-summary.json",
        {
            "total": 1,
            "passed": 0,
            "failed": 1,
            "fail_on_error": True,
            "failed_results": [{"file": "query_pressure_1.json", "brick": "query_pressure", "status": "failed"}],
        },
    )
    (tmp_path / "k8s").mkdir()

    rc = generate_workflow_report.main(_base_args(tmp_path, pressure_fail_on_error="true"))

    report = json.loads((tmp_path / "reports" / "orchestrator_report.json").read_text())
    assert rc == 1
    assert report["status"] == "failed"
    assert report["validation"]["passed"] is True
    assert report["parameters"]["pressure_fail_on_error"] is True


def test_generate_workflow_report_fails_when_validation_is_missing(tmp_path):
    _write_json(
        tmp_path / "pressure-summary.json",
        {"total": 1, "passed": 1, "failed": 0, "fail_on_error": False, "failed_results": []},
    )
    (tmp_path / "k8s").mkdir()

    rc = generate_workflow_report.main(_base_args(tmp_path, pressure_fail_on_error="false"))

    report = json.loads((tmp_path / "reports" / "orchestrator_report.json").read_text())
    assert rc == 1
    assert report["status"] == "failed"
    assert report["validation"]["passed"] is False
