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
        "--rollback-milvus-image",
        "harbor.milvus.io/milvusdb/milvus:2.6-latest",
        "--target-milvus-image",
        "harbor.milvus.io/milvusdb/milvus:2.6-latest",
        "--base-version",
        "2.6.18",
        "--rollback-version",
        "2.6.0",
        "--target-version",
        "2.6.99",
        "--repo-url",
        "https://github.com/yanliang567/vectordb-testbricks.git",
        "--repo-revision",
        "main",
        "--scenario-id",
        "standalone-2-6-18-to-3-0-latest-rollback-2-6-latest",
        "--deploy-profile",
        "milvus_client/manifests/deploy_profiles/standalone-rocksmq.yaml",
        "--deploy-topology",
        str(tmp_path / "reports" / "deploy_topology.json"),
        "--schema-matrix",
        "milvus_client/manifests/schema_matrix_2_6.yaml",
        "--collection-prefix",
        "qa_upgrade",
        "--forward-collection-prefix",
        "qa_upgrade_forward",
        "--rows-per-collection",
        "5000",
        "--batch-size",
        "500",
        "--pressure-modules",
        "search_pressure query_pressure",
        "--pressure-fail-on-error",
        pressure_fail_on_error,
        "--observe-before-upgrade-sec",
        "300",
        "--observe-after-upgrade-sec",
        "300",
        "--observe-before-rollback-sec",
        "300",
        "--observe-after-rollback-sec",
        "300",
        "--rollback-serviceability-timeout-sec",
        "900",
        "--rollback-serviceability-interval-sec",
        "10",
        "--base-json-shredding-enabled",
        "true",
        "--target-json-shredding-enabled",
        "true",
        "--rollback-json-shredding-enabled",
        "true",
        "--base-loon-ffi-enabled",
        "false",
        "--target-loon-ffi-enabled",
        "false",
        "--rollback-loon-ffi-enabled",
        "false",
        "--base-vortex-enabled",
        "false",
        "--target-vortex-enabled",
        "false",
        "--rollback-vortex-enabled",
        "false",
        "--post-upgrade-config-toggle-enabled",
        "false",
        "--post-upgrade-json-shredding-enabled",
        "true",
        "--forward-workload-enabled",
        "false",
        "--forward-schema-matrix",
        "milvus_client/manifests/schema_matrix_3_0.yaml",
        "--rollback-enabled",
        "true",
        "--rollback-forward-validation-enabled",
        "false",
        "--index-compatibility-validation-enabled",
        "true",
        "--phase-dml-dql-validation-enabled",
        "true",
        "--phase-new-collection-rows",
        "1000",
        "--phase-existing-dml-rows",
        "1000",
        "--phase-existing-delete-rows",
        "100",
        "--schema-evolution-existing-enabled",
        "true",
        "--schema-evolution-forward-enabled",
        "false",
    ]


def _write_successful_validation(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "results" / "validate_before_upgrade.json", {"status": "passed"}
    )
    _write_json(
        tmp_path / "results" / "validate_after_upgrade.json", {"status": "passed"}
    )
    _write_json(
        tmp_path / "results" / "validate_index_compatibility_after_upgrade.json",
        {"status": "passed"},
    )
    _write_json(
        tmp_path / "results" / "validate_phase_dml_dql_after_upgrade.json",
        {"status": "passed"},
    )
    _write_json(
        tmp_path / "results" / "validate_forward_after_upgrade.json",
        {"status": "skipped"},
    )
    _write_json(
        tmp_path / "results" / "validate_index_compatibility_after_rollback.json",
        {"status": "passed"},
    )
    _write_json(
        tmp_path / "results" / "validate_phase_dml_dql_after_rollback.json",
        {"status": "passed"},
    )
    _write_json(
        tmp_path / "results" / "validate_after_rollback.json", {"status": "passed"}
    )
    _write_json(
        tmp_path / "results" / "wait_rollback_serviceability.json",
        {
            "brick": "wait_data_serviceability",
            "status": "passed",
            "metrics": {
                "recovered": True,
                "recovery_duration_sec": 37.5,
                "attempts": 5,
            },
        },
    )


def _write_successful_upgrade_only_validation(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "results" / "validate_before_upgrade.json", {"status": "passed"}
    )
    _write_json(
        tmp_path / "results" / "validate_after_upgrade.json", {"status": "passed"}
    )
    _write_json(
        tmp_path / "results" / "validate_forward_after_upgrade.json",
        {"status": "passed"},
    )


def _write_successful_upgrade_validation(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "results" / "validate_before_upgrade.json", {"status": "passed"}
    )
    _write_json(
        tmp_path / "results" / "validate_after_upgrade.json", {"status": "passed"}
    )


def _write_successful_index_compatibility_validation(
    tmp_path: Path,
    *,
    after_rollback: bool,
) -> None:
    _write_json(
        tmp_path / "results" / "validate_index_compatibility_after_upgrade.json",
        {"status": "passed"},
    )
    if after_rollback:
        _write_json(
            tmp_path / "results" / "validate_index_compatibility_after_rollback.json",
            {"status": "passed"},
        )


def _write_successful_phase_dml_dql_validation(
    tmp_path: Path,
    *,
    after_rollback: bool,
) -> None:
    _write_json(
        tmp_path / "results" / "validate_phase_dml_dql_after_upgrade.json",
        {"status": "passed"},
    )
    if after_rollback:
        _write_json(
            tmp_path / "results" / "validate_phase_dml_dql_after_rollback.json",
            {"status": "passed"},
        )


def test_generate_workflow_report_marks_pressure_failures_as_warning_when_not_strict(
    tmp_path,
):
    _write_successful_validation(tmp_path)
    _write_json(
        tmp_path / "pressure-summary.json",
        {
            "total": 3,
            "passed": 2,
            "failed": 1,
            "fail_on_error": False,
            "failed_results": [
                {
                    "file": "search_pressure_2.json",
                    "brick": "search_pressure",
                    "status": "failed",
                }
            ],
        },
    )
    _write_json(tmp_path / "reports" / "env_snapshot.json", {"client_namespace": "qa"})
    _write_json(
        tmp_path / "reports" / "flow_summary.json", {"cleanup_status": "pending"}
    )
    _write_json(
        tmp_path / "reports" / "deploy_topology.json",
        {
            "profile": "standalone-rocksmq",
            "mode": "standalone",
            "components": {"standalone": {"replicas": 1}},
        },
    )
    (tmp_path / "k8s").mkdir()
    (tmp_path / "k8s" / "pods.txt").write_text("pods")

    rc = generate_workflow_report.main(
        _base_args(tmp_path, pressure_fail_on_error="false")
    )

    report = json.loads((tmp_path / "reports" / "orchestrator_report.json").read_text())
    markdown = (tmp_path / "reports" / "final_report.md").read_text()
    assert rc == 0
    assert report["status"] == "warning"
    assert report["validation"]["passed"] is True
    assert report["pressure"]["failed"] == 1
    assert report["parameters"]["pressure_fail_on_error"] is False
    assert (
        report["parameters"]["scenario_id"]
        == "standalone-2-6-18-to-3-0-latest-rollback-2-6-latest"
    )
    assert (
        report["parameters"]["deploy_profile"]
        == "milvus_client/manifests/deploy_profiles/standalone-rocksmq.yaml"
    )
    assert report["deploy_topology"]["profile"] == "standalone-rocksmq"
    assert report["deploy_topology"]["mode"] == "standalone"
    assert report["parameters"]["observe_before_upgrade_sec"] == 300
    assert report["parameters"]["observe_before_rollback_sec"] == 300
    assert report["parameters"]["rollback_serviceability_timeout_sec"] == 900
    assert report["parameters"]["rollback_serviceability_interval_sec"] == 10
    assert report["parameters"]["forward_collection_prefix"] == "qa_upgrade_forward"
    assert (
        report["target"]["rollback_milvus_image"]
        == "harbor.milvus.io/milvusdb/milvus:2.6-latest"
    )
    assert report["target"]["rollback_version"] == "2.6.0"
    assert report["parameters"]["config_matrix"] == {
        "base_json_shredding_enabled": True,
        "target_json_shredding_enabled": True,
        "rollback_json_shredding_enabled": True,
        "base_loon_ffi_enabled": False,
        "target_loon_ffi_enabled": False,
        "rollback_loon_ffi_enabled": False,
        "base_vortex_enabled": False,
        "target_vortex_enabled": False,
        "rollback_vortex_enabled": False,
        "post_upgrade_config_toggle_enabled": False,
        "post_upgrade_json_shredding_enabled": True,
        "forward_workload_enabled": False,
        "forward_schema_matrix": "milvus_client/manifests/schema_matrix_3_0.yaml",
        "rollback_enabled": True,
        "rollback_forward_validation_enabled": False,
        "index_compatibility_validation_enabled": True,
        "phase_dml_dql_validation_enabled": True,
        "phase_new_collection_rows": 1000,
        "phase_existing_dml_rows": 1000,
        "phase_existing_delete_rows": 100,
        "schema_evolution_existing_enabled": True,
        "schema_evolution_forward_enabled": False,
    }
    assert report["k8s_snapshot"]["pods.txt"] == str(tmp_path / "k8s" / "pods.txt")
    assert "## Config Matrix" in markdown
    assert "- rollback version: `2.6.0`" in markdown
    assert (
        "- scenario: `standalone-2-6-18-to-3-0-latest-rollback-2-6-latest`" in markdown
    )
    assert (
        "- deploy profile: `milvus_client/manifests/deploy_profiles/standalone-rocksmq.yaml`"
        in markdown
    )
    assert "- rollback image: `harbor.milvus.io/milvusdb/milvus:2.6-latest`" in markdown
    assert "- forward collection prefix: `qa_upgrade_forward`" in markdown
    assert "- rollback enabled: `True`" in markdown
    assert "- index compatibility validation: `True`" in markdown
    assert "- phase DML/DQL validation: `True`" in markdown
    assert "- phase new collection rows/schema: `1000`" in markdown
    assert "- base jsonShredding: `True`" in markdown
    assert "- target LoonFFI/storage v3: `False`" in markdown
    assert "- target vortex: `False`" in markdown
    assert "## Validation" in markdown
    assert "## Serviceability Recovery" in markdown
    assert (
        "`wait_rollback_serviceability`: passed, recovered=`True`, recovery_duration_sec=`37.5`, attempts=`5`"
        in markdown
    )
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
            "failed_results": [
                {
                    "file": "query_pressure_1.json",
                    "brick": "query_pressure",
                    "status": "failed",
                }
            ],
        },
    )
    (tmp_path / "k8s").mkdir()

    rc = generate_workflow_report.main(
        _base_args(tmp_path, pressure_fail_on_error="true")
    )

    report = json.loads((tmp_path / "reports" / "orchestrator_report.json").read_text())
    assert rc == 1
    assert report["status"] == "failed"
    assert report["validation"]["passed"] is True
    assert report["parameters"]["pressure_fail_on_error"] is True


def test_generate_workflow_report_fails_when_required_rollback_validation_is_missing(
    tmp_path,
):
    _write_successful_upgrade_validation(tmp_path)
    _write_successful_index_compatibility_validation(tmp_path, after_rollback=True)
    _write_successful_phase_dml_dql_validation(tmp_path, after_rollback=True)
    _write_json(
        tmp_path / "results" / "wait_rollback_serviceability.json",
        {"brick": "wait_data_serviceability", "status": "passed"},
    )
    _write_json(
        tmp_path / "pressure-summary.json",
        {
            "total": 1,
            "passed": 1,
            "failed": 0,
            "fail_on_error": True,
            "failed_results": [],
        },
    )
    (tmp_path / "k8s").mkdir()

    rc = generate_workflow_report.main(
        _base_args(tmp_path, pressure_fail_on_error="true")
    )

    report = json.loads((tmp_path / "reports" / "orchestrator_report.json").read_text())
    assert rc == 1
    assert report["status"] == "failed"
    assert report["validation"]["passed"] is False
    assert (
        report["validation"]["results"]["validate_after_rollback"]["status"]
        == "missing"
    )
    assert (
        report["failed_results"]["validate_after_rollback"]["failures"][0]["type"]
        == "VALIDATION_RESULT_MISSING"
    )


def test_generate_workflow_report_fails_when_required_serviceability_result_is_missing(
    tmp_path,
):
    _write_successful_upgrade_validation(tmp_path)
    _write_successful_index_compatibility_validation(tmp_path, after_rollback=True)
    _write_successful_phase_dml_dql_validation(tmp_path, after_rollback=True)
    _write_json(
        tmp_path / "results" / "validate_after_rollback.json", {"status": "passed"}
    )
    _write_json(
        tmp_path / "pressure-summary.json",
        {
            "total": 1,
            "passed": 1,
            "failed": 0,
            "fail_on_error": True,
            "failed_results": [],
        },
    )
    (tmp_path / "k8s").mkdir()

    rc = generate_workflow_report.main(
        _base_args(tmp_path, pressure_fail_on_error="true")
    )

    report = json.loads((tmp_path / "reports" / "orchestrator_report.json").read_text())
    assert rc == 1
    assert report["status"] == "failed"
    assert (
        report["serviceability"]["results"]["wait_rollback_serviceability"]["status"]
        == "missing"
    )
    assert (
        report["failed_results"]["wait_rollback_serviceability"]["failures"][0]["type"]
        == "SERVICEABILITY_RESULT_MISSING"
    )


def test_generate_workflow_report_fails_when_index_compatibility_result_is_missing(
    tmp_path,
):
    _write_successful_upgrade_validation(tmp_path)
    _write_successful_phase_dml_dql_validation(tmp_path, after_rollback=True)
    _write_json(
        tmp_path / "results" / "validate_after_rollback.json", {"status": "passed"}
    )
    _write_json(
        tmp_path / "results" / "wait_rollback_serviceability.json",
        {"brick": "wait_data_serviceability", "status": "passed"},
    )
    _write_json(
        tmp_path / "pressure-summary.json",
        {
            "total": 1,
            "passed": 1,
            "failed": 0,
            "fail_on_error": True,
            "failed_results": [],
        },
    )
    (tmp_path / "k8s").mkdir()

    rc = generate_workflow_report.main(
        _base_args(tmp_path, pressure_fail_on_error="true")
    )

    report = json.loads((tmp_path / "reports" / "orchestrator_report.json").read_text())
    assert rc == 1
    assert report["status"] == "failed"
    assert report["validation"]["passed"] is False
    assert (
        report["validation"]["results"]["validate_index_compatibility_after_upgrade"][
            "status"
        ]
        == "missing"
    )
    assert (
        report["validation"]["results"]["validate_index_compatibility_after_rollback"][
            "status"
        ]
        == "missing"
    )


def test_generate_workflow_report_does_not_require_index_compatibility_when_disabled(
    tmp_path,
):
    _write_successful_upgrade_validation(tmp_path)
    _write_successful_phase_dml_dql_validation(tmp_path, after_rollback=True)
    _write_json(
        tmp_path / "results" / "validate_after_rollback.json", {"status": "passed"}
    )
    _write_json(
        tmp_path / "results" / "wait_rollback_serviceability.json",
        {"brick": "wait_data_serviceability", "status": "passed"},
    )
    _write_json(
        tmp_path / "pressure-summary.json",
        {
            "total": 1,
            "passed": 1,
            "failed": 0,
            "fail_on_error": True,
            "failed_results": [],
        },
    )
    (tmp_path / "k8s").mkdir()

    rc = generate_workflow_report.main(
        [
            *_base_args(tmp_path, pressure_fail_on_error="true"),
            "--index-compatibility-validation-enabled",
            "false",
        ]
    )

    report = json.loads((tmp_path / "reports" / "orchestrator_report.json").read_text())
    assert rc == 0
    assert report["status"] == "passed"
    assert (
        report["parameters"]["config_matrix"]["index_compatibility_validation_enabled"]
        is False
    )
    assert (
        "validate_index_compatibility_after_upgrade"
        not in report["validation"]["results"]
    )
    assert (
        "validate_index_compatibility_after_rollback"
        not in report["validation"]["results"]
    )


def test_generate_workflow_report_fails_when_phase_dml_dql_result_is_missing(
    tmp_path,
):
    _write_successful_upgrade_validation(tmp_path)
    _write_successful_index_compatibility_validation(tmp_path, after_rollback=True)
    _write_json(
        tmp_path / "results" / "validate_after_rollback.json", {"status": "passed"}
    )
    _write_json(
        tmp_path / "results" / "wait_rollback_serviceability.json",
        {"brick": "wait_data_serviceability", "status": "passed"},
    )
    _write_json(
        tmp_path / "pressure-summary.json",
        {
            "total": 1,
            "passed": 1,
            "failed": 0,
            "fail_on_error": True,
            "failed_results": [],
        },
    )
    (tmp_path / "k8s").mkdir()

    rc = generate_workflow_report.main(
        _base_args(tmp_path, pressure_fail_on_error="true")
    )

    report = json.loads((tmp_path / "reports" / "orchestrator_report.json").read_text())
    assert rc == 1
    assert report["status"] == "failed"
    assert (
        report["validation"]["results"]["validate_phase_dml_dql_after_upgrade"][
            "status"
        ]
        == "missing"
    )
    assert (
        report["validation"]["results"]["validate_phase_dml_dql_after_rollback"][
            "status"
        ]
        == "missing"
    )


def test_generate_workflow_report_does_not_require_phase_dml_dql_when_disabled(
    tmp_path,
):
    _write_successful_upgrade_validation(tmp_path)
    _write_successful_index_compatibility_validation(tmp_path, after_rollback=True)
    _write_json(
        tmp_path / "results" / "validate_after_rollback.json", {"status": "passed"}
    )
    _write_json(
        tmp_path / "results" / "wait_rollback_serviceability.json",
        {"brick": "wait_data_serviceability", "status": "passed"},
    )
    _write_json(
        tmp_path / "pressure-summary.json",
        {
            "total": 1,
            "passed": 1,
            "failed": 0,
            "fail_on_error": True,
            "failed_results": [],
        },
    )
    (tmp_path / "k8s").mkdir()

    rc = generate_workflow_report.main(
        [
            *_base_args(tmp_path, pressure_fail_on_error="true"),
            "--phase-dml-dql-validation-enabled",
            "false",
        ]
    )

    report = json.loads((tmp_path / "reports" / "orchestrator_report.json").read_text())
    assert rc == 0
    assert report["status"] == "passed"
    assert (
        report["parameters"]["config_matrix"]["phase_dml_dql_validation_enabled"]
        is False
    )
    assert "validate_phase_dml_dql_after_upgrade" not in report["validation"]["results"]
    assert (
        "validate_phase_dml_dql_after_rollback" not in report["validation"]["results"]
    )


def test_generate_workflow_report_fails_when_required_forward_validation_is_missing(
    tmp_path,
):
    _write_successful_upgrade_validation(tmp_path)
    _write_successful_index_compatibility_validation(tmp_path, after_rollback=True)
    _write_successful_phase_dml_dql_validation(tmp_path, after_rollback=True)
    _write_json(
        tmp_path / "results" / "validate_after_rollback.json", {"status": "passed"}
    )
    _write_json(
        tmp_path / "results" / "wait_rollback_serviceability.json",
        {"brick": "wait_data_serviceability", "status": "passed"},
    )
    _write_json(
        tmp_path / "pressure-summary.json",
        {
            "total": 1,
            "passed": 1,
            "failed": 0,
            "fail_on_error": True,
            "failed_results": [],
        },
    )
    (tmp_path / "k8s").mkdir()

    rc = generate_workflow_report.main(
        [
            *_base_args(tmp_path, pressure_fail_on_error="true"),
            "--forward-workload-enabled",
            "true",
        ]
    )

    report = json.loads((tmp_path / "reports" / "orchestrator_report.json").read_text())
    assert rc == 1
    assert report["status"] == "failed"
    assert (
        report["validation"]["results"]["validate_forward_after_upgrade"]["status"]
        == "missing"
    )


def test_generate_workflow_report_fails_when_required_forward_rollback_validation_is_missing(
    tmp_path,
):
    _write_successful_validation(tmp_path)
    _write_json(
        tmp_path / "pressure-summary.json",
        {
            "total": 1,
            "passed": 1,
            "failed": 0,
            "fail_on_error": True,
            "failed_results": [],
        },
    )
    (tmp_path / "k8s").mkdir()

    rc = generate_workflow_report.main(
        [
            *_base_args(tmp_path, pressure_fail_on_error="true"),
            "--forward-workload-enabled",
            "true",
            "--rollback-forward-validation-enabled",
            "true",
        ]
    )

    report = json.loads((tmp_path / "reports" / "orchestrator_report.json").read_text())
    assert rc == 1
    assert report["status"] == "failed"
    assert (
        report["validation"]["results"]["validate_forward_after_rollback"]["status"]
        == "missing"
    )


def test_generate_workflow_report_does_not_require_forward_rollback_without_forward_workload(
    tmp_path,
):
    _write_successful_validation(tmp_path)
    _write_json(
        tmp_path / "pressure-summary.json",
        {
            "total": 1,
            "passed": 1,
            "failed": 0,
            "fail_on_error": True,
            "failed_results": [],
        },
    )
    (tmp_path / "k8s").mkdir()

    rc = generate_workflow_report.main(
        [
            *_base_args(tmp_path, pressure_fail_on_error="true"),
            "--rollback-forward-validation-enabled",
            "true",
        ]
    )

    report = json.loads((tmp_path / "reports" / "orchestrator_report.json").read_text())
    assert rc == 0
    assert report["status"] == "passed"
    assert "validate_forward_after_rollback" not in report["validation"]["results"]


def test_generate_workflow_report_allows_strict_upgrade_only_gate_without_rollback_validation(
    tmp_path,
):
    _write_successful_upgrade_only_validation(tmp_path)
    _write_json(
        tmp_path / "pressure-summary.json",
        {
            "total": 1,
            "passed": 1,
            "failed": 0,
            "fail_on_error": True,
            "failed_results": [],
        },
    )
    (tmp_path / "k8s").mkdir()

    rc = generate_workflow_report.main(
        [
            *_base_args(tmp_path, pressure_fail_on_error="true"),
            "--rollback-enabled",
            "false",
            "--forward-workload-enabled",
            "true",
        ]
    )

    report = json.loads((tmp_path / "reports" / "orchestrator_report.json").read_text())
    assert rc == 0
    assert report["status"] == "passed"
    assert report["validation"]["passed"] is True
    assert report["parameters"]["config_matrix"]["rollback_enabled"] is False
    assert report["parameters"]["config_matrix"]["forward_workload_enabled"] is True
    assert "validate_after_rollback" not in report["validation"]["results"]


def test_generate_workflow_report_ignores_forward_rollback_when_rollback_disabled(
    tmp_path,
):
    _write_successful_upgrade_only_validation(tmp_path)
    _write_json(
        tmp_path / "pressure-summary.json",
        {
            "total": 1,
            "passed": 1,
            "failed": 0,
            "fail_on_error": True,
            "failed_results": [],
        },
    )
    (tmp_path / "k8s").mkdir()

    rc = generate_workflow_report.main(
        [
            *_base_args(tmp_path, pressure_fail_on_error="true"),
            "--rollback-enabled",
            "false",
            "--forward-workload-enabled",
            "true",
            "--rollback-forward-validation-enabled",
            "true",
        ]
    )

    report = json.loads((tmp_path / "reports" / "orchestrator_report.json").read_text())
    assert rc == 0
    assert report["status"] == "passed"
    assert "validate_forward_after_rollback" not in report["validation"]["results"]


def test_generate_workflow_report_can_soft_fail_after_writing_failed_report(tmp_path):
    _write_successful_validation(tmp_path)
    _write_json(
        tmp_path / "pressure-summary.json",
        {
            "total": 1,
            "passed": 0,
            "failed": 1,
            "fail_on_error": True,
            "failed_results": [
                {
                    "file": "query_pressure_1.json",
                    "brick": "query_pressure",
                    "status": "failed",
                }
            ],
        },
    )
    (tmp_path / "k8s").mkdir()

    rc = generate_workflow_report.main(
        [*_base_args(tmp_path, pressure_fail_on_error="true"), "--soft-fail"]
    )

    report = json.loads((tmp_path / "reports" / "orchestrator_report.json").read_text())
    assert rc == 0
    assert report["status"] == "failed"


def test_generate_workflow_report_fails_when_validation_is_missing(tmp_path):
    _write_json(
        tmp_path / "pressure-summary.json",
        {
            "total": 1,
            "passed": 1,
            "failed": 0,
            "fail_on_error": False,
            "failed_results": [],
        },
    )
    (tmp_path / "k8s").mkdir()

    rc = generate_workflow_report.main(
        _base_args(tmp_path, pressure_fail_on_error="false")
    )

    report = json.loads((tmp_path / "reports" / "orchestrator_report.json").read_text())
    assert rc == 1
    assert report["status"] == "failed"
    assert report["validation"]["passed"] is False
