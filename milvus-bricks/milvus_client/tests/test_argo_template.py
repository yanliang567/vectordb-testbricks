import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

from milvus_client.common.pressure_maintenance import (
    classify_pressure_result,
    maintenance_windows_from_workflow_nodes,
)

ROOT = Path(__file__).resolve().parents[2]


def test_argo_template_persists_checkpoint_state_and_exports_results():
    template = yaml.safe_load(
        (ROOT / "argo" / "upgrade-rollback-compatibility.yaml").read_text()
    )
    brick_runner = next(
        item for item in template["spec"]["templates"] if item["name"] == "brick-runner"
    )

    artifact_names = {
        artifact["name"] for artifact in brick_runner["outputs"]["artifacts"]
    }
    mounts = {
        mount["name"]: mount["mountPath"]
        for mount in brick_runner["container"]["volumeMounts"]
    }
    claim_names = {
        claim["metadata"]["name"] for claim in template["spec"]["volumeClaimTemplates"]
    }

    assert {"result-json", "checkpoints"} <= artifact_names
    assert mounts["milvus-bricks-state"] == "/tmp/milvus-bricks"
    assert "milvus-bricks-state" in claim_names


def test_argo_template_runs_compatibility_bricks():
    template = yaml.safe_load(
        (ROOT / "argo" / "upgrade-rollback-compatibility.yaml").read_text()
    )
    parameter_names = {
        parameter["name"] for parameter in template["spec"]["arguments"]["parameters"]
    }
    dag = next(
        item
        for item in template["spec"]["templates"]
        if item["name"] == "upgrade-rollback-compatibility"
    )
    tasks = {task["name"]: task for task in dag["dag"]["tasks"]}
    scenario_runner = next(
        item
        for item in template["spec"]["templates"]
        if item["name"] == "scenario-runner"
    )

    assert {
        "compat-schema-matrix",
        "forward-schema-matrix",
        "cycles",
        "validator-interval-sec",
    } <= parameter_names
    assert tasks["run-closed-loop-scenario"]["template"] == "scenario-runner"
    artifact_names = {
        artifact["name"] for artifact in scenario_runner["outputs"]["artifacts"]
    }
    assert {"result-json", "checkpoints", "results"} <= artifact_names
    command = scenario_runner["container"]["args"][0]
    assert "milvus_client.scenarios.upgrade_rollback_compatibility" in command
    assert "forward_schema_matrix" in command


def test_upgrade_rollback_pressure_results_exclude_rollout_connectivity_windows():
    for template_path in [
        ROOT / "argo" / "standalone-2-6-upgrade-rollback.yaml",
        ROOT / "argo" / "standalone-3-0-upgrade-rollback.yaml",
        ROOT / "argo" / "cluster-upgrade-rollback.yaml",
    ]:
        template = yaml.safe_load(template_path.read_text())
        templates = {item["name"]: item for item in template["spec"]["templates"]}
        check_command = templates["check-pressure-results"]["container"]["args"][0]

        assert "_maintenance_windows" in check_command
        assert "schema-evolution-existing" in check_command
        assert "schema-evolution-forward" in check_command
        assert "duration_sec" in check_command
        assert "enabled" in check_command
        assert "schema-evolution-existing-enabled" in check_command
        assert "schema-evolution-forward-enabled" in check_command
        assert "excluded_failed_results" in check_command
        assert "failed_all" in check_command
        assert "PRESSURE_ATTEMPT_PENDING" in check_command
        assert "PRESSURE_RESULT_MISSING" in check_command
        assert "classify_pressure_result" in check_command
        assert "maintenance_windows_from_workflow_nodes" in check_command
        assert "metrics_only_failure_without_error_details" not in check_command
        assert "rm -rf /tmp/repo" in check_command
        assert (
            'git clone --depth 1 --branch "{{workflow.parameters.repo-revision}}"'
            in check_command
        )
        assert "cd /tmp/repo/milvus-bricks" in check_command
        assert "PYTHONPATH=. python3 - <<'PY'" in check_command


def test_upgrade_rollback_templates_assert_storage_config_before_phase_validation():
    for template_path in [
        ROOT / "argo" / "standalone-2-6-upgrade-rollback.yaml",
        ROOT / "argo" / "standalone-3-0-upgrade-rollback.yaml",
        ROOT / "argo" / "cluster-upgrade-rollback.yaml",
    ]:
        template = yaml.safe_load(template_path.read_text())
        templates = {item["name"]: item for item in template["spec"]["templates"]}
        dag = next(item for item in templates.values() if "dag" in item)
        tasks = {task["name"]: task for task in dag["dag"]["tasks"]}

        assert "assert-milvus-storage-config" in templates
        assert tasks["assert-base-storage-config"]["dependencies"] == [
            "snapshot-base-config"
        ]
        assert tasks["assert-base-storage-config"]["arguments"]["parameters"] == [
            {"name": "phase", "value": "base"},
            {
                "name": "expected-loon-ffi-enabled",
                "value": "{{workflow.parameters.base-loon-ffi-enabled}}",
            },
            {
                "name": "expected-vortex-enabled",
                "value": "{{workflow.parameters.base-vortex-enabled}}",
            },
        ]
        assert tasks["precheck-base"]["dependencies"] == ["assert-base-storage-config"]
        assert tasks["assert-after-upgrade-storage-config"]["dependencies"] == [
            "snapshot-after-upgrade-config",
            "pressure-daemon",
        ]
        assert tasks["assert-after-rollback-storage-config"]["dependencies"] == [
            "snapshot-after-rollback-config",
            "pressure-daemon",
        ]
        assert (
            "assert-after-upgrade-storage-config"
            in tasks["precheck-after-upgrade"]["dependencies"]
        )
        assert (
            "assert-after-rollback-storage-config"
            in tasks["precheck-after-rollback"]["dependencies"]
        )

        command = templates["assert-milvus-storage-config"]["container"]["args"][0]
        assert "python3 -m pip install --disable-pip-version-check -q pyyaml" in command
        assert "common.storage.useLoonFFI" in command
        assert "dataNode.storage.format" in command
        assert "expected YAML boolean or absent" in command
        assert "/milvus/configs/milvus.yaml" in command
        assert "/milvus/configs/user.yaml" in command
        assert "cat /milvus/configs/user.yaml 2>/dev/null || true" in command
        assert "runtime-milvus-${phase}.yaml" in command
        assert "runtime-user-${phase}.yaml" in command
        assert "deep_merge(runtime_milvus_config, runtime_user_config)" in command
        assert "runtime_config, missing_loon_as_false=True" in command
        assert "kubectl -n" in command
        assert " exec " in command
        assert "declared_actual" in command
        assert "runtime_actual" in command
        assert "expected-loon-ffi-enabled" in command
        assert "expected-vortex-enabled" in command
        if template_path.name.startswith("cluster-"):
            assert "helm get values" in command
            assert "extraConfigFiles" in command
            assert 'labels.get("component") == "datanode"' in command
        else:
            assert "kubectl -n" in command
            assert "get mi" in command


def _run_storage_assertion_heredoc(
    template_path: Path, tmp_path: Path, *, runtime_format: str
) -> subprocess.CompletedProcess[str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    template = yaml.safe_load(template_path.read_text())
    templates = {item["name"]: item for item in template["spec"]["templates"]}
    command = templates["assert-milvus-storage-config"]["container"]["args"][0]
    heredocs = re.findall(
        r"python3 - <<'PY'\n(.*?)\n\s*PY",
        command,
        flags=re.DOTALL,
    )
    code = heredocs[-1]
    code = code.replace("{{inputs.parameters.phase}}", "base")
    code = code.replace("{{inputs.parameters.expected-loon-ffi-enabled}}", "false")
    code = code.replace("{{inputs.parameters.expected-vortex-enabled}}", "false")
    code = code.replace("/tmp/milvus-bricks/k8s", str(tmp_path))

    if template_path.name.startswith("cluster-"):
        (tmp_path / "helm-values-base-storage-config-source.yaml").write_text(
            yaml.safe_dump({"extraConfigFiles": {"user.yaml": "{}"}})
        )
    else:
        (tmp_path / "milvus-base-storage-config-source.json").write_text(
            json.dumps({"spec": {"config": {}}})
        )
    (tmp_path / "runtime-milvus-base.yaml").write_text(
        yaml.safe_dump(
            {
                "common": {"storage": {"useLoonFFI": False}},
                "dataNode": {"storage": {"format": runtime_format}},
            }
        )
    )
    (tmp_path / "runtime-user-base.yaml").write_text("{}")
    (tmp_path / "runtime-config-pod-base.txt").write_text("milvus-pod-0")

    return subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )


def test_storage_config_assertion_accepts_parquet_for_non_vortex_runtime(tmp_path):
    for template_path in [
        ROOT / "argo" / "standalone-2-6-upgrade-rollback.yaml",
        ROOT / "argo" / "standalone-3-0-upgrade-rollback.yaml",
        ROOT / "argo" / "cluster-upgrade-rollback.yaml",
    ]:
        result = _run_storage_assertion_heredoc(
            template_path, tmp_path / template_path.stem, runtime_format="parquet"
        )

        assert result.returncode == 0, result.stderr
        report = json.loads(
            (
                tmp_path / template_path.stem / "storage-config-assertion-base.json"
            ).read_text()
        )
        assert report["status"] == "passed"
        assert report["runtime_actual"]["dataNode.storage.format"] == "parquet"


def test_storage_config_assertion_rejects_vortex_for_non_vortex_runtime(tmp_path):
    for template_path in [
        ROOT / "argo" / "standalone-2-6-upgrade-rollback.yaml",
        ROOT / "argo" / "standalone-3-0-upgrade-rollback.yaml",
        ROOT / "argo" / "cluster-upgrade-rollback.yaml",
    ]:
        result = _run_storage_assertion_heredoc(
            template_path, tmp_path / template_path.stem, runtime_format="vortex"
        )

        assert result.returncode != 0
        report = json.loads(
            (
                tmp_path / template_path.stem / "storage-config-assertion-base.json"
            ).read_text()
        )
        assert report["status"] == "failed"
        assert "runtime dataNode.storage.format expected non-vortex" in "\n".join(
            report["failures"]
        )


def test_upgrade_rollback_embedded_python_heredocs_parse():
    for template_path in [
        ROOT / "argo" / "standalone-2-6-upgrade-rollback.yaml",
        ROOT / "argo" / "standalone-3-0-upgrade-rollback.yaml",
        ROOT / "argo" / "cluster-upgrade-rollback.yaml",
    ]:
        heredocs = re.findall(
            r"python3 - <<'PY'\n(.*?)\n\s*PY",
            template_path.read_text(),
            flags=re.DOTALL,
        )

        assert heredocs
        for index, heredoc in enumerate(heredocs, start=1):
            code = "\n".join(
                line.removeprefix("            ") for line in heredoc.splitlines()
            )
            ast.parse(code, filename=f"{template_path.name}:python-heredoc:{index}")


def test_maintenance_windows_skip_disabled_schema_evolution_noop_nodes():
    nodes = [
        {
            "displayName": "schema-evolution-existing",
            "phase": "Succeeded",
            "startedAt": "2026-07-23T20:42:00Z",
            "finishedAt": "2026-07-23T20:42:30Z",
        },
        {
            "displayName": "schema-evolution-forward",
            "phase": "Succeeded",
            "startedAt": "2026-07-23T20:43:00Z",
            "finishedAt": "2026-07-23T20:43:30Z",
        },
    ]

    windows = maintenance_windows_from_workflow_nodes(
        nodes,
        schema_evolution_existing_enabled=False,
        schema_evolution_forward_enabled=False,
    )

    assert windows == []


def test_maintenance_windows_include_successful_rollout_duration():
    nodes = [
        {
            "displayName": "patch-upgrade",
            "phase": "Succeeded",
            "startedAt": "2026-07-23T20:41:00Z",
            "finishedAt": "2026-07-23T20:41:10Z",
        },
        {
            "displayName": "wait-upgrade-ready",
            "phase": "Succeeded",
            "startedAt": "2026-07-23T20:41:11Z",
            "finishedAt": "2026-07-23T20:42:00Z",
        },
    ]

    windows = maintenance_windows_from_workflow_nodes(
        nodes,
        schema_evolution_existing_enabled=False,
        schema_evolution_forward_enabled=False,
    )

    assert len(windows) == 1
    assert windows[0]["label"] == "upgrade-rollout"
    assert windows[0]["duration_sec"] == 60.0


def test_maintenance_windows_include_enabled_successful_schema_evolution_duration():
    nodes = [
        {
            "displayName": "schema-evolution-existing",
            "phase": "Succeeded",
            "startedAt": "2026-07-23T20:42:00Z",
            "finishedAt": "2026-07-23T20:42:30Z",
        }
    ]

    windows = maintenance_windows_from_workflow_nodes(
        nodes,
        schema_evolution_existing_enabled=True,
        schema_evolution_forward_enabled=False,
    )

    assert len(windows) == 1
    assert windows[0]["label"] == "schema-evolution-existing"
    assert windows[0]["started_at"] == "2026-07-23T20:42:00+00:00"
    assert windows[0]["finished_at"] == "2026-07-23T20:42:30+00:00"
    assert windows[0]["duration_sec"] == 30.0


def test_maintenance_windows_skip_enabled_failed_schema_evolution_nodes():
    nodes = [
        {
            "displayName": "schema-evolution-existing",
            "phase": "Failed",
            "startedAt": "2026-07-23T20:42:00Z",
            "finishedAt": "2026-07-23T20:42:30Z",
        }
    ]

    windows = maintenance_windows_from_workflow_nodes(
        nodes,
        schema_evolution_existing_enabled=True,
        schema_evolution_forward_enabled=False,
    )

    assert windows == []


def test_pressure_maintenance_classifier_excludes_only_window_connectivity_failure():
    result = {
        "status": "failed",
        "brick": "mixed_rw_pressure",
        "started_at": "2026-07-23T20:41:59+00:00",
        "finished_at": "2026-07-23T20:42:01+00:00",
        "metrics": {"requests_failed": 1, "failed_query": 1},
        "failures": [
            {
                "type": "PRESSURE_OPERATION_FAILED",
                "operation": "query",
                "started_at": "2026-07-23T20:42:00+00:00",
                "finished_at": "2026-07-23T20:42:00+00:00",
                "error": "connection reset by peer",
                "connectivity_transient": True,
            }
        ],
    }
    windows = [
        {
            "label": "rollback-rollout",
            "started_at": "2026-07-23T20:41:50+00:00",
            "finished_at": "2026-07-23T20:42:10+00:00",
        }
    ]

    classification, entry = classify_pressure_result("mixed.json", result, windows)

    assert classification == "excluded"
    assert entry["status"] == "maintenance_window_excluded"
    assert entry["maintenance_window"]["label"] == "rollback-rollout"


def test_pressure_maintenance_classifier_keeps_metrics_only_failure_strict():
    result = {
        "status": "failed",
        "brick": "mixed_rw_pressure",
        "started_at": "2026-07-23T20:41:59+00:00",
        "finished_at": "2026-07-23T20:42:01+00:00",
        "metrics": {"requests_failed": 1, "failed_search": 1},
        "failures": [],
    }
    windows = [
        {
            "label": "rollback-rollout",
            "started_at": "2026-07-23T20:41:50+00:00",
            "finished_at": "2026-07-23T20:42:10+00:00",
        }
    ]

    classification, entry = classify_pressure_result("mixed.json", result, windows)

    assert classification == "failed"
    assert (
        entry["classification_reason"] == "metrics_only_failure_without_error_details"
    )


def test_pressure_maintenance_classifier_keeps_partially_explained_metrics_strict():
    result = {
        "status": "failed",
        "brick": "mixed_rw_pressure",
        "started_at": "2026-07-23T20:41:59+00:00",
        "finished_at": "2026-07-23T20:42:01+00:00",
        "metrics": {"requests_failed": 2, "failed_query": 2},
        "failures": [
            {
                "type": "PRESSURE_OPERATION_FAILED",
                "operation": "query",
                "started_at": "2026-07-23T20:42:00+00:00",
                "finished_at": "2026-07-23T20:42:00+00:00",
                "error": "connection reset by peer",
                "connectivity_transient": True,
            }
        ],
    }
    windows = [
        {
            "label": "rollback-rollout",
            "started_at": "2026-07-23T20:41:50+00:00",
            "finished_at": "2026-07-23T20:42:10+00:00",
        }
    ]

    classification, entry = classify_pressure_result("mixed.json", result, windows)

    assert classification == "failed"
    assert entry["classification_reason"] == "failed_metrics_exceed_failure_details"
    assert entry["failure_detail_count"] == 1
    assert entry["failed_metric_count"] == 2


def test_pressure_maintenance_classifier_keeps_correctness_failure_strict_inside_window():
    result = {
        "status": "failed",
        "brick": "mixed_rw_pressure",
        "started_at": "2026-07-23T20:41:59+00:00",
        "finished_at": "2026-07-23T20:42:01+00:00",
        "metrics": {"requests_failed": 1, "failed_search": 1},
        "failures": [
            {
                "type": "PRESSURE_OPERATION_FAILED",
                "operation": "search",
                "started_at": "2026-07-23T20:42:00+00:00",
                "finished_at": "2026-07-23T20:42:00+00:00",
                "error_type": "AssertionError",
                "error": "qa_dense.embedding: search returned no hits",
                "connectivity_transient": False,
            }
        ],
    }
    windows = [
        {
            "label": "rollback-rollout",
            "started_at": "2026-07-23T20:41:50+00:00",
            "finished_at": "2026-07-23T20:42:10+00:00",
        }
    ]

    classification, entry = classify_pressure_result("mixed.json", result, windows)

    assert classification == "failed"
    assert entry["failures"][0]["operation"] == "search"


def test_pressure_maintenance_classifier_keeps_connectivity_failure_outside_window():
    result = {
        "status": "failed",
        "brick": "mixed_rw_pressure",
        "started_at": "2026-07-23T20:50:00+00:00",
        "finished_at": "2026-07-23T20:50:01+00:00",
        "metrics": {"requests_failed": 1, "failed_query": 1},
        "failures": [
            {
                "type": "PRESSURE_OPERATION_FAILED",
                "operation": "query",
                "started_at": "2026-07-23T20:50:00+00:00",
                "finished_at": "2026-07-23T20:50:00+00:00",
                "error": "connection refused",
                "connectivity_transient": True,
            }
        ],
    }
    windows = [
        {
            "label": "rollback-rollout",
            "started_at": "2026-07-23T20:41:50+00:00",
            "finished_at": "2026-07-23T20:42:10+00:00",
        }
    ]

    classification, entry = classify_pressure_result("mixed.json", result, windows)

    assert classification == "failed"
    assert entry["failures"][0]["operation"] == "query"


def test_pressure_maintenance_classifier_excludes_schema_mismatch_inside_schema_evolution_window():
    result = {
        "status": "failed",
        "brick": "mixed_rw_pressure",
        "started_at": "2026-07-23T20:42:00+00:00",
        "finished_at": "2026-07-23T20:42:01+00:00",
        "metrics": {"requests_failed": 1, "failed_upsert": 1},
        "failures": [
            {
                "type": "PRESSURE_OPERATION_FAILED",
                "operation": "upsert",
                "started_at": "2026-07-23T20:42:00+00:00",
                "finished_at": "2026-07-23T20:42:00+00:00",
                "error_type": "SchemaMismatchRetryableException",
                "error": "<SchemaMismatchRetryableException: (code=collection schema mismatch[collection=qa], message=)>",
                "connectivity_transient": False,
            }
        ],
    }
    windows = [
        {
            "label": "schema-evolution-existing",
            "started_at": "2026-07-23T20:41:50+00:00",
            "finished_at": "2026-07-23T20:42:10+00:00",
        }
    ]

    classification, entry = classify_pressure_result("mixed.json", result, windows)

    assert classification == "excluded"
    assert entry["status"] == "maintenance_window_excluded"
    assert entry["maintenance_window"]["label"] == "schema-evolution-existing"
    assert entry["failures"][0]["error_type"] == "SchemaMismatchRetryableException"


def test_pressure_maintenance_classifier_keeps_schema_mismatch_outside_schema_evolution_window():
    result = {
        "status": "failed",
        "brick": "mixed_rw_pressure",
        "started_at": "2026-07-23T20:42:00+00:00",
        "finished_at": "2026-07-23T20:42:01+00:00",
        "metrics": {"requests_failed": 1, "failed_upsert": 1},
        "failures": [
            {
                "type": "PRESSURE_OPERATION_FAILED",
                "operation": "upsert",
                "started_at": "2026-07-23T20:42:00+00:00",
                "finished_at": "2026-07-23T20:42:00+00:00",
                "error_type": "SchemaMismatchRetryableException",
                "error": "<SchemaMismatchRetryableException: (code=collection schema mismatch[collection=qa], message=)>",
                "connectivity_transient": False,
            }
        ],
    }
    windows = [
        {
            "label": "rollback-rollout",
            "started_at": "2026-07-23T20:41:50+00:00",
            "finished_at": "2026-07-23T20:42:10+00:00",
        }
    ]

    classification, entry = classify_pressure_result("mixed.json", result, windows)

    assert classification == "failed"
    assert entry["failures"][0]["error_type"] == "SchemaMismatchRetryableException"


def test_pressure_maintenance_classifier_keeps_schema_mismatch_without_failure_timestamps_strict():
    result = {
        "status": "failed",
        "brick": "mixed_rw_pressure",
        "started_at": "2026-07-23T20:42:00+00:00",
        "finished_at": "2026-07-23T20:42:01+00:00",
        "metrics": {"requests_failed": 1, "failed_upsert": 1},
        "failures": [
            {
                "type": "PRESSURE_OPERATION_FAILED",
                "operation": "upsert",
                "error_type": "SchemaMismatchRetryableException",
                "error": "<SchemaMismatchRetryableException: (code=collection schema mismatch[collection=qa], message=)>",
                "connectivity_transient": False,
            }
        ],
    }
    windows = [
        {
            "label": "schema-evolution-existing",
            "started_at": "2026-07-23T20:41:50+00:00",
            "finished_at": "2026-07-23T20:42:10+00:00",
        }
    ]

    classification, entry = classify_pressure_result("mixed.json", result, windows)

    assert classification == "failed"
    assert entry["failures"][0]["error_type"] == "SchemaMismatchRetryableException"


def test_pressure_maintenance_classifier_keeps_non_schema_mismatch_inside_schema_evolution_window():
    result = {
        "status": "failed",
        "brick": "mixed_rw_pressure",
        "started_at": "2026-07-23T20:42:00+00:00",
        "finished_at": "2026-07-23T20:42:01+00:00",
        "metrics": {"requests_failed": 1, "failed_search": 1},
        "failures": [
            {
                "type": "PRESSURE_OPERATION_FAILED",
                "operation": "search",
                "started_at": "2026-07-23T20:42:00+00:00",
                "finished_at": "2026-07-23T20:42:00+00:00",
                "error_type": "AssertionError",
                "error": "search returned no hits",
                "connectivity_transient": False,
            }
        ],
    }
    windows = [
        {
            "label": "schema-evolution-existing",
            "started_at": "2026-07-23T20:41:50+00:00",
            "finished_at": "2026-07-23T20:42:10+00:00",
        }
    ]

    classification, entry = classify_pressure_result("mixed.json", result, windows)

    assert classification == "failed"
    assert entry["failures"][0]["operation"] == "search"


def test_pressure_maintenance_classifier_keeps_mixed_failure_strict():
    result = {
        "status": "failed",
        "brick": "mixed_rw_pressure",
        "started_at": "2026-07-23T20:41:59+00:00",
        "finished_at": "2026-07-23T20:42:01+00:00",
        "metrics": {"requests_failed": 2, "failed_query": 1, "failed_search": 1},
        "failures": [
            {
                "type": "PRESSURE_OPERATION_FAILED",
                "operation": "query",
                "started_at": "2026-07-23T20:42:00+00:00",
                "finished_at": "2026-07-23T20:42:00+00:00",
                "error": "deadline exceeded",
                "connectivity_transient": True,
            },
            {
                "type": "PRESSURE_OPERATION_FAILED",
                "operation": "search",
                "started_at": "2026-07-23T20:42:00+00:00",
                "finished_at": "2026-07-23T20:42:00+00:00",
                "error": "search returned no hits",
                "connectivity_transient": False,
            },
        ],
    }
    windows = [
        {
            "label": "rollback-rollout",
            "started_at": "2026-07-23T20:41:50+00:00",
            "finished_at": "2026-07-23T20:42:10+00:00",
        }
    ]

    classification, entry = classify_pressure_result("mixed.json", result, windows)

    assert classification == "failed"
    assert [failure["operation"] for failure in entry["failures"]] == ["search"]
    assert [failure["operation"] for failure in entry["excluded_failures"]] == ["query"]


def test_upgrade_rollback_templates_retry_repo_clone():
    for template_path in [
        ROOT / "argo" / "standalone-2-6-upgrade-rollback.yaml",
        ROOT / "argo" / "standalone-3-0-upgrade-rollback.yaml",
        ROOT / "argo" / "cluster-upgrade-rollback.yaml",
    ]:
        template = yaml.safe_load(template_path.read_text())
        for template_item in template["spec"]["templates"]:
            container = template_item.get("container")
            if not container:
                continue
            command = "\n".join(str(arg) for arg in container.get("args", []))
            if "git clone --depth 1 --branch" not in command:
                continue
            assert "for attempt in 1 2 3 4 5; do" in command
            assert 'if [ "$attempt" = "5" ]; then' in command
            assert "sleep $((attempt * 5))" in command


def test_standalone_2_6_upgrade_rollback_template_is_2_6_only():
    template = yaml.safe_load(
        (ROOT / "argo" / "standalone-2-6-upgrade-rollback.yaml").read_text()
    )

    assert template["kind"] == "WorkflowTemplate"
    assert template["metadata"]["name"] == "milvus-standalone-2-6-upgrade-rollback"
    assert template["metadata"]["namespace"] == "qa"
    assert template["spec"]["serviceAccountName"] == "milvus-upgrade-rollback-runner"
    parameter_values = {
        parameter["name"]: parameter["value"]
        for parameter in template["spec"]["arguments"]["parameters"]
    }

    assert parameter_values["client-namespace"] == "qa"
    assert parameter_values["milvus-namespace"] == "qa-milvus"
    assert parameter_values["client-image"] == "harbor.milvus.io/qa/fouram:2.1"
    assert parameter_values["repo-revision"] == "main"
    assert (
        parameter_values["scenario-id"]
        == "standalone-2-6-18-to-3-0-latest-rollback-2-6-latest"
    )
    assert (
        parameter_values["deploy-profile"]
        == "milvus_client/manifests/deploy_profiles/standalone-rocksmq.yaml"
    )
    assert (
        parameter_values["base-milvus-image"]
        == "harbor.milvus.io/milvusdb/milvus:v2.6.18"
    )
    assert (
        parameter_values["rollback-milvus-image"]
        == "harbor.milvus.io/milvusdb/milvus:2.6-latest-placeholder"
    )
    assert parameter_values["rollback-version"] == "2.6.0"
    assert (
        parameter_values["target-milvus-image"]
        == "harbor.milvus.io/milvusdb/milvus:3.0-latest-placeholder"
    )
    assert parameter_values["target-version"] == "3.0.0"
    assert (
        parameter_values["schema-matrix"]
        == "milvus_client/manifests/schema_matrix_2_6.yaml"
    )
    assert (
        parameter_values["forward-schema-matrix"]
        == "milvus_client/manifests/schema_matrix_3_0.yaml"
    )
    assert parameter_values["forward-collection-prefix"] == "qa_upgrade_2618_forward"
    assert parameter_values["base-json-shredding-enabled"] == "false"
    assert parameter_values["target-json-shredding-enabled"] == "false"
    assert parameter_values["rollback-json-shredding-enabled"] == "false"
    assert parameter_values["base-loon-ffi-enabled"] == "false"
    assert parameter_values["target-loon-ffi-enabled"] == "false"
    assert parameter_values["rollback-loon-ffi-enabled"] == "false"
    assert parameter_values["base-vortex-enabled"] == "false"
    assert parameter_values["target-vortex-enabled"] == "false"
    assert parameter_values["rollback-vortex-enabled"] == "false"
    assert parameter_values["post-upgrade-config-toggle-enabled"] == "false"
    assert parameter_values["forward-workload-enabled"] == "false"
    assert parameter_values["rollback-enabled"] == "true"
    assert parameter_values["index-compatibility-validation-enabled"] == "true"
    assert parameter_values["phase-dml-dql-validation-enabled"] == "true"
    assert parameter_values["phase-new-collection-rows"] == "3000"
    assert parameter_values["phase-existing-dml-rows"] == "1000"
    assert parameter_values["phase-existing-delete-rows"] == "100"
    assert parameter_values["schema-evolution-existing-enabled"] == "false"
    assert parameter_values["schema-evolution-forward-enabled"] == "false"
    assert parameter_values["standalone-cpu-request"] == "2"
    assert parameter_values["standalone-memory-request"] == "4Gi"
    assert parameter_values["standalone-cpu"] == "4"
    assert parameter_values["standalone-memory"] == "8Gi"
    assert parameter_values["observe-before-upgrade-sec"] == "300"
    assert parameter_values["observe-after-upgrade-sec"] == "300"
    assert parameter_values["observe-before-rollback-sec"] == "300"
    assert parameter_values["observe-after-rollback-sec"] == "300"
    assert parameter_values["rollback-serviceability-timeout-sec"] == "900"
    assert parameter_values["rollback-serviceability-interval-sec"] == "10"
    assert parameter_values["pressure-fail-on-error"] == "false"
    assert parameter_values["gate-allow-warning"] == "true"
    assert parameter_values["allow-unsafe-negative-coverage"] == "false"
    assert parameter_values["keep-milvus"] == "false"


def test_standalone_2_6_upgrade_rollback_template_runs_full_closed_loop_with_pressure():
    template = yaml.safe_load(
        (ROOT / "argo" / "standalone-2-6-upgrade-rollback.yaml").read_text()
    )
    main = next(
        item for item in template["spec"]["templates"] if item["name"] == "main"
    )
    tasks = {task["name"]: task for task in main["dag"]["tasks"]}
    templates = {item["name"]: item for item in template["spec"]["templates"]}
    resolve_inputs = templates["resolve-inputs"]

    expected_tasks = {
        "resolve-inputs",
        "deploy-base",
        "wait-base-ready",
        "snapshot-base-config",
        "precheck-base",
        "create-compat-schema",
        "seed-compat-data",
        "validate-before-upgrade",
        "strict-pressure-before-upgrade",
        "pressure-daemon",
        "observe-before-upgrade",
        "patch-upgrade",
        "wait-upgrade-ready",
        "snapshot-after-upgrade-config",
        "observe-after-upgrade",
        "precheck-after-upgrade",
        "validate-after-upgrade",
        "validate-index-compatibility-after-upgrade",
        "validate-phase-dml-dql-after-upgrade",
        "strict-pressure-after-upgrade",
        "schema-evolution-existing",
        "patch-post-upgrade-config",
        "wait-post-upgrade-config-ready",
        "snapshot-post-upgrade-config",
        "create-forward-schema",
        "seed-forward-data",
        "validate-forward-after-upgrade",
        "schema-evolution-forward",
        "observe-before-rollback",
        "strict-pressure-before-rollback",
        "patch-rollback",
        "wait-rollback-ready",
        "snapshot-after-rollback-config",
        "observe-after-rollback",
        "precheck-after-rollback",
        "wait-rollback-serviceability",
        "validate-after-rollback",
        "validate-index-compatibility-after-rollback",
        "validate-phase-dml-dql-after-rollback",
        "wait-forward-rollback-serviceability",
        "validate-forward-after-rollback",
        "strict-pressure-after-rollback",
        "stop-pressure",
        "check-pressure-results",
        "collect-artifacts",
        "generate-final-report",
        "gate-final-status",
    }
    assert expected_tasks <= set(tasks)

    assert templates["pressure-daemon"]["daemon"] is True
    assert resolve_inputs["container"]["resources"] == {
        "requests": {"cpu": "1", "memory": "2Gi"},
        "limits": {"cpu": "2", "memory": "4Gi"},
    }
    assert resolve_inputs["container"]["volumeMounts"][0] == {
        "name": "milvus-test-state",
        "mountPath": "/tmp/milvus-bricks",
    }
    assert "readinessProbe" in templates["pressure-daemon"]["container"]
    assert "volumeMounts" not in templates["run-pressure-suite"]["container"]
    assert "validator-daemon" not in templates
    assert tasks["strict-pressure-before-upgrade"]["dependencies"] == [
        "validate-before-upgrade"
    ]
    assert tasks["pressure-daemon"]["dependencies"] == [
        "strict-pressure-before-upgrade"
    ]
    assert tasks["observe-before-upgrade"]["dependencies"] == ["pressure-daemon"]
    assert tasks["patch-upgrade"]["dependencies"] == [
        "observe-before-upgrade",
        "pressure-daemon",
    ]
    assert tasks["assert-base-storage-config"]["dependencies"] == [
        "snapshot-base-config"
    ]
    assert tasks["precheck-base"]["dependencies"] == ["assert-base-storage-config"]
    assert tasks["snapshot-after-upgrade-config"]["dependencies"] == [
        "wait-upgrade-ready",
        "pressure-daemon",
    ]
    assert tasks["snapshot-after-rollback-config"]["dependencies"] == [
        "wait-rollback-ready",
        "pressure-daemon",
    ]
    pressure_covered_tasks = [
        "observe-before-upgrade",
        "patch-upgrade",
        "wait-upgrade-ready",
        "snapshot-after-upgrade-config",
        "observe-after-upgrade",
        "precheck-after-upgrade",
        "validate-after-upgrade",
        "validate-index-compatibility-after-upgrade",
        "validate-phase-dml-dql-after-upgrade",
        "schema-evolution-existing",
        "patch-post-upgrade-config",
        "wait-post-upgrade-config-ready",
        "snapshot-post-upgrade-config",
        "create-forward-schema",
        "seed-forward-data",
        "validate-forward-after-upgrade",
        "schema-evolution-forward",
        "observe-before-rollback",
        "patch-rollback",
        "wait-rollback-ready",
        "snapshot-after-rollback-config",
        "observe-after-rollback",
        "precheck-after-rollback",
        "wait-rollback-serviceability",
        "validate-after-rollback",
        "validate-index-compatibility-after-rollback",
        "validate-phase-dml-dql-after-rollback",
        "wait-forward-rollback-serviceability",
        "validate-forward-after-rollback",
        "strict-pressure-after-rollback",
        "stop-pressure",
    ]
    for task_name in pressure_covered_tasks:
        assert "pressure-daemon" in tasks[task_name]["dependencies"]
    assert tasks["validate-index-compatibility-after-upgrade"]["dependencies"] == [
        "validate-after-upgrade",
        "pressure-daemon",
    ]
    assert tasks["strict-pressure-after-upgrade"]["dependencies"] == [
        "validate-phase-dml-dql-after-upgrade",
        "pressure-daemon",
    ]
    assert tasks["validate-phase-dml-dql-after-upgrade"]["dependencies"] == [
        "validate-index-compatibility-after-upgrade",
        "pressure-daemon",
    ]
    assert tasks["schema-evolution-existing"]["dependencies"] == [
        "strict-pressure-after-upgrade",
        "pressure-daemon",
    ]
    assert tasks["schema-evolution-existing"]["template"] == "optional-run-brick"
    assert tasks["patch-post-upgrade-config"]["dependencies"] == [
        "schema-evolution-existing",
        "pressure-daemon",
    ]
    assert tasks["wait-post-upgrade-config-ready"]["dependencies"] == [
        "patch-post-upgrade-config",
        "pressure-daemon",
    ]
    assert tasks["create-forward-schema"]["template"] == "optional-run-brick"
    assert tasks["validate-forward-after-upgrade"]["template"] == "optional-run-brick"
    assert tasks["schema-evolution-forward"]["dependencies"] == [
        "validate-forward-after-upgrade",
        "pressure-daemon",
    ]
    assert tasks["schema-evolution-forward"]["template"] == "optional-run-brick"
    assert tasks["observe-before-rollback"]["dependencies"] == [
        "schema-evolution-forward",
        "pressure-daemon",
    ]
    assert tasks["strict-pressure-before-rollback"]["dependencies"] == [
        "observe-before-rollback",
        "pressure-daemon",
    ]
    assert tasks["patch-rollback"]["dependencies"] == [
        "strict-pressure-before-rollback",
        "pressure-daemon",
    ]
    assert tasks["wait-rollback-serviceability"]["dependencies"] == [
        "precheck-after-rollback",
        "pressure-daemon",
    ]
    assert tasks["wait-forward-rollback-serviceability"]["dependencies"] == [
        "wait-rollback-serviceability",
        "pressure-daemon",
    ]
    assert tasks["observe-after-rollback"]["dependencies"] == [
        "wait-forward-rollback-serviceability",
        "assert-after-rollback-storage-config",
        "pressure-daemon",
    ]
    assert tasks["validate-index-compatibility-after-rollback"]["dependencies"] == [
        "observe-after-rollback",
        "pressure-daemon",
    ]
    assert tasks["validate-phase-dml-dql-after-rollback"]["dependencies"] == [
        "validate-index-compatibility-after-rollback",
        "pressure-daemon",
    ]
    assert tasks["validate-after-rollback"]["dependencies"] == [
        "validate-phase-dml-dql-after-rollback",
        "pressure-daemon",
    ]
    assert (
        tasks["wait-forward-rollback-serviceability"]["template"]
        == "optional-run-brick"
    )
    assert tasks["validate-forward-after-rollback"]["dependencies"] == [
        "validate-after-rollback",
        "pressure-daemon",
    ]
    assert tasks["strict-pressure-after-rollback"]["dependencies"] == [
        "validate-forward-after-rollback",
        "pressure-daemon",
    ]
    assert tasks["stop-pressure"]["dependencies"] == [
        "strict-pressure-after-rollback",
        "pressure-daemon",
    ]
    rollback_gated_tasks = [
        "observe-before-rollback",
        "strict-pressure-before-rollback",
        "patch-rollback",
        "wait-rollback-ready",
        "snapshot-after-rollback-config",
        "observe-after-rollback",
        "precheck-after-rollback",
        "wait-rollback-serviceability",
        "validate-after-rollback",
        "validate-index-compatibility-after-rollback",
        "validate-phase-dml-dql-after-rollback",
        "strict-pressure-after-rollback",
    ]
    for task_name in rollback_gated_tasks:
        assert (
            tasks[task_name]["when"]
            == "{{workflow.parameters.rollback-enabled}} == true"
        )
    assert tasks["validate-forward-after-rollback"]["when"] == (
        "{{workflow.parameters.rollback-enabled}} == true && {{workflow.parameters.forward-workload-enabled}} == true"
    )
    assert tasks["wait-forward-rollback-serviceability"]["when"] == (
        "{{workflow.parameters.rollback-enabled}} == true && {{workflow.parameters.forward-workload-enabled}} == true"
    )
    seed_args = {
        parameter["name"]: parameter["value"]
        for parameter in tasks["seed-compat-data"]["arguments"]["parameters"]
    }
    assert (
        "--checkpoint-file /tmp/milvus-bricks/checkpoints/baseline/seed_data.json"
        in seed_args["args"]
    )
    validate_before_args = {
        parameter["name"]: parameter["value"]
        for parameter in tasks["validate-before-upgrade"]["arguments"]["parameters"]
    }
    validate_after_upgrade_args = {
        parameter["name"]: parameter["value"]
        for parameter in tasks["validate-after-upgrade"]["arguments"]["parameters"]
    }
    validate_after_rollback_args = {
        parameter["name"]: parameter["value"]
        for parameter in tasks["validate-after-rollback"]["arguments"]["parameters"]
    }
    validate_index_after_upgrade_args = {
        parameter["name"]: parameter["value"]
        for parameter in tasks["validate-index-compatibility-after-upgrade"][
            "arguments"
        ]["parameters"]
    }
    validate_index_after_rollback_args = {
        parameter["name"]: parameter["value"]
        for parameter in tasks["validate-index-compatibility-after-rollback"][
            "arguments"
        ]["parameters"]
    }
    validate_phase_after_upgrade_args = {
        parameter["name"]: parameter["value"]
        for parameter in tasks["validate-phase-dml-dql-after-upgrade"]["arguments"][
            "parameters"
        ]
    }
    validate_phase_after_rollback_args = {
        parameter["name"]: parameter["value"]
        for parameter in tasks["validate-phase-dml-dql-after-rollback"]["arguments"][
            "parameters"
        ]
    }
    for validate_args in [
        validate_before_args,
        validate_after_upgrade_args,
        validate_after_rollback_args,
    ]:
        assert (
            "--checkpoint-file /tmp/milvus-bricks/checkpoints/baseline/seed_data.json"
            in validate_args["args"]
        )
    assert (
        validate_index_after_upgrade_args["enabled"]
        == "{{workflow.parameters.index-compatibility-validation-enabled}}"
    )
    assert (
        validate_index_after_upgrade_args["module"]
        == "milvus_client.requests.validate_index_compatibility"
    )
    assert "--phase after-upgrade" in validate_index_after_upgrade_args["args"]
    assert "--rebuild-index false" in validate_index_after_upgrade_args["args"]
    assert (
        "--index-checkpoint-file /tmp/milvus-bricks/checkpoints/index_compatibility.json"
        in validate_index_after_upgrade_args["args"]
    )
    assert (
        validate_index_after_rollback_args["enabled"]
        == "{{workflow.parameters.index-compatibility-validation-enabled}}"
    )
    assert (
        validate_index_after_rollback_args["module"]
        == "milvus_client.requests.validate_index_compatibility"
    )
    assert "--phase after-rollback" in validate_index_after_rollback_args["args"]
    assert "--rebuild-index false" in validate_index_after_rollback_args["args"]
    assert (
        "--index-checkpoint-file /tmp/milvus-bricks/checkpoints/index_compatibility.json"
        in validate_index_after_rollback_args["args"]
    )
    assert (
        validate_phase_after_upgrade_args["enabled"]
        == "{{workflow.parameters.phase-dml-dql-validation-enabled}}"
    )
    assert (
        validate_phase_after_upgrade_args["module"]
        == "milvus_client.requests.validate_phase_dml_dql"
    )
    assert "--phase after-upgrade" in validate_phase_after_upgrade_args["args"]
    assert (
        "--new-collection-prefix {{workflow.parameters.collection-prefix}}_after_upgrade"
        in validate_phase_after_upgrade_args["args"]
    )
    assert (
        "--new-collection-rows {{workflow.parameters.phase-new-collection-rows}}"
        in validate_phase_after_upgrade_args["args"]
    )
    assert (
        "--existing-dml-rows {{workflow.parameters.phase-existing-dml-rows}}"
        in validate_phase_after_upgrade_args["args"]
    )
    assert (
        "--existing-delete-rows {{workflow.parameters.phase-existing-delete-rows}}"
        in validate_phase_after_upgrade_args["args"]
    )
    assert (
        validate_phase_after_rollback_args["enabled"]
        == "{{workflow.parameters.phase-dml-dql-validation-enabled}}"
    )
    assert (
        validate_phase_after_rollback_args["module"]
        == "milvus_client.requests.validate_phase_dml_dql"
    )
    assert "--phase after-rollback" in validate_phase_after_rollback_args["args"]
    assert (
        "--new-collection-prefix {{workflow.parameters.collection-prefix}}_after_rollback"
        in validate_phase_after_rollback_args["args"]
    )
    assert (
        "--carried-collection-prefix {{workflow.parameters.collection-prefix}}_after_upgrade"
        in validate_phase_after_rollback_args["args"]
    )
    seed_forward_args = {
        parameter["name"]: parameter["value"]
        for parameter in tasks["seed-forward-data"]["arguments"]["parameters"]
    }
    assert (
        "--checkpoint-file /tmp/milvus-bricks/checkpoints/forward/seed_data.json"
        in seed_forward_args["args"]
    )
    validate_forward_after_upgrade_args = {
        parameter["name"]: parameter["value"]
        for parameter in tasks["validate-forward-after-upgrade"]["arguments"][
            "parameters"
        ]
    }
    assert (
        "--checkpoint-file /tmp/milvus-bricks/checkpoints/forward/seed_data.json"
        in validate_forward_after_upgrade_args["args"]
    )
    schema_evolution_args = {
        parameter["name"]: parameter["value"]
        for parameter in tasks["schema-evolution-existing"]["arguments"]["parameters"]
    }
    assert (
        schema_evolution_args["module"]
        == "milvus_client.requests.schema_evolution_workload"
    )
    strict_pressure_args = {
        parameter["name"]: parameter["value"]
        for parameter in tasks["strict-pressure-before-upgrade"]["arguments"][
            "parameters"
        ]
    }
    assert (
        strict_pressure_args["collection-prefix"]
        == "{{workflow.parameters.collection-prefix}}"
    )
    assert (
        strict_pressure_args["schema-matrix"] == "{{workflow.parameters.schema-matrix}}"
    )
    strict_pressure_command = templates["run-pressure-suite"]["container"]["args"][0]
    assert (
        "for module in {{workflow.parameters.pressure-modules}}"
        in strict_pressure_command
    )
    assert (
        "--checkpoint-dir /tmp/strict-pressure-checkpoints" in strict_pressure_command
    )
    assert 'if [ -f "$result" ]; then' in strict_pressure_command
    assert (
        'python3 -m json.tool "$result" || cat "$result" || true'
        in strict_pressure_command
    )
    assert "strict pressure result file not found: $result" in strict_pressure_command
    assert 'exit "$failed"' in strict_pressure_command
    assert (
        schema_evolution_args["collection-prefix"]
        == "{{workflow.parameters.collection-prefix}}"
    )
    assert (
        "--schema-matrix {{workflow.parameters.schema-matrix}}"
        in schema_evolution_args["args"]
    )
    forward_evolution_args = {
        parameter["name"]: parameter["value"]
        for parameter in tasks["schema-evolution-forward"]["arguments"]["parameters"]
    }
    assert (
        forward_evolution_args["collection-prefix"]
        == "{{workflow.parameters.forward-collection-prefix}}"
    )
    assert (
        "--schema-matrix {{workflow.parameters.forward-schema-matrix}}"
        in forward_evolution_args["args"]
    )
    forward_validate_args = {
        parameter["name"]: parameter["value"]
        for parameter in tasks["validate-forward-after-rollback"]["arguments"][
            "parameters"
        ]
    }
    assert (
        forward_validate_args["collection-prefix"]
        == "{{workflow.parameters.forward-collection-prefix}}"
    )
    assert (
        "--checkpoint-file /tmp/milvus-bricks/checkpoints/forward/seed_data.json"
        in forward_validate_args["args"]
    )
    serviceability_args = {
        parameter["name"]: parameter["value"]
        for parameter in tasks["wait-rollback-serviceability"]["arguments"][
            "parameters"
        ]
    }
    assert (
        serviceability_args["module"]
        == "milvus_client.requests.wait_data_serviceability"
    )
    assert (
        serviceability_args["collection-prefix"]
        == "{{workflow.parameters.collection-prefix}}"
    )
    assert (
        "--checkpoint-file /tmp/milvus-bricks/checkpoints/baseline/seed_data.json"
        in serviceability_args["args"]
    )
    assert (
        "--timeout-sec {{workflow.parameters.rollback-serviceability-timeout-sec}}"
        in serviceability_args["args"]
    )
    assert (
        "--interval-sec {{workflow.parameters.rollback-serviceability-interval-sec}}"
        in serviceability_args["args"]
    )
    forward_serviceability_args = {
        parameter["name"]: parameter["value"]
        for parameter in tasks["wait-forward-rollback-serviceability"]["arguments"][
            "parameters"
        ]
    }
    assert (
        forward_serviceability_args["enabled"]
        == "{{workflow.parameters.rollback-forward-validation-enabled}}"
    )
    assert (
        forward_serviceability_args["module"]
        == "milvus_client.requests.wait_data_serviceability"
    )
    assert (
        forward_serviceability_args["collection-prefix"]
        == "{{workflow.parameters.forward-collection-prefix}}"
    )
    assert (
        "--checkpoint-file /tmp/milvus-bricks/checkpoints/forward/seed_data.json"
        in forward_serviceability_args["args"]
    )
    patch_rollback_args = {
        parameter["name"]: parameter["value"]
        for parameter in tasks["patch-rollback"]["arguments"]["parameters"]
    }
    assert (
        patch_rollback_args["image"] == "{{workflow.parameters.rollback-milvus-image}}"
    )
    assert patch_rollback_args["version"] == "{{workflow.parameters.rollback-version}}"
    wait_rollback_args = {
        parameter["name"]: parameter["value"]
        for parameter in tasks["wait-rollback-ready"]["arguments"]["parameters"]
    }
    assert (
        wait_rollback_args["expected-image"]
        == "{{workflow.parameters.rollback-milvus-image}}"
    )
    assert tasks["deploy-base"]["dependencies"] == ["resolve-inputs"]
    assert tasks["check-pressure-results"]["dependencies"] == ["stop-pressure"]
    assert tasks["collect-artifacts"]["dependencies"] == ["check-pressure-results"]
    assert tasks["generate-final-report"]["dependencies"] == ["collect-artifacts"]
    assert tasks["gate-final-status"]["dependencies"] == ["generate-final-report"]

    parameter_values = {
        parameter["name"]: parameter["value"]
        for parameter in template["spec"]["arguments"]["parameters"]
    }
    pressure_modules = parameter_values["pressure-modules"]
    assert parameter_values["pressure-fail-on-error"] == "false"
    assert parameter_values["gate-allow-warning"] == "true"
    assert "search_pressure" in pressure_modules
    assert "query_pressure" in pressure_modules
    assert "query_iterator_scan" in pressure_modules
    assert "count_pressure" in pressure_modules
    assert "upsert_pressure" in pressure_modules
    assert "delete_pressure" in pressure_modules
    assert "mixed_rw_pressure" in pressure_modules

    pressure_template = templates["pressure-daemon"]
    assert "volumeMounts" not in pressure_template["container"]
    pressure_command = pressure_template["container"]["args"][0]
    assert (
        'if [ "$rc" = "0" ] && [ ! -f /tmp/pressure-ready ]; then' in pressure_command
    )
    assert "pressure-stop" in pressure_command
    assert (
        'kubectl -n "$pressure_ns" create configmap "$attempt_cm"' in pressure_command
    )
    assert "zilliz.com/pressure-result=true" in pressure_command
    assert "PRESSURE_PROCESS_FAILED" in pressure_command
    assert 'python3 -m json.tool "$result"' in pressure_command
    assert "python -m" not in pressure_command
    assert "python3 -m" in pressure_command
    assert "--baseline-rows-per-collection" in pressure_command
    assert "{{workflow.parameters.rows-per-collection}}" in pressure_command

    stop_pressure = templates["stop-pressure"]
    assert "volumeMounts" not in stop_pressure["container"]
    stop_command = stop_pressure["container"]["args"][0]
    assert "{{workflow.name}}-pressure-stop" in stop_command
    assert "zilliz.com/pressure-stop=true" in stop_command

    cleanup = templates["maybe-cleanup"]
    cleanup_artifacts = {
        artifact["name"] for artifact in cleanup["outputs"]["artifacts"]
    }
    assert {
        "orchestrator-report",
        "final-report-md",
        "flow-summary",
        "env-snapshot",
        "k8s-snapshot",
    } <= cleanup_artifacts
    assert cleanup["container"]["volumeMounts"][0] == {
        "name": "milvus-test-state",
        "mountPath": "/tmp/milvus-bricks",
    }

    check_pressure = templates["check-pressure-results"]
    check_artifacts = {
        artifact["name"] for artifact in check_pressure["outputs"]["artifacts"]
    }
    assert "pressure-summary" in check_artifacts
    check_command = check_pressure["container"]["args"][0]
    assert "NO_PRESSURE_RESULTS" in check_command
    assert "PRESSURE_RESULT_MISSING" in check_command
    assert "PRESSURE_ATTEMPT_PENDING" in check_command
    assert "kubectl" in check_command
    assert "zilliz.com/pressure-result=true" in check_command
    assert 'summary["fail_on_error"] and failed' not in check_command

    final_report = templates["generate-final-report"]
    final_artifacts = {
        artifact["name"] for artifact in final_report["outputs"]["artifacts"]
    }
    assert {
        "orchestrator-report",
        "final-report-md",
        "env-snapshot",
        "flow-summary",
    } <= final_artifacts
    final_command = final_report["container"]["args"][0]
    assert "milvus_client.requests.generate_workflow_report" in final_command
    assert "pressure-summary.json" in final_command
    assert "final_report.md" in final_command
    assert "orchestrator_report.json" in final_command
    assert "--soft-fail" in final_command
    assert "--base-json-shredding-enabled" in final_command
    assert "--base-loon-ffi-enabled" in final_command
    assert "--target-loon-ffi-enabled" in final_command
    assert "--rollback-loon-ffi-enabled" in final_command
    assert "--base-vortex-enabled" in final_command
    assert "--target-vortex-enabled" in final_command
    assert "--rollback-vortex-enabled" in final_command
    assert "--rollback-milvus-image" in final_command
    assert "--rollback-version" in final_command
    assert "--target-json-shredding-enabled" in final_command
    assert "--forward-workload-enabled" in final_command
    assert "--rollback-enabled" in final_command
    assert "--observe-before-upgrade-sec" in final_command
    assert "--observe-before-rollback-sec" in final_command
    assert "--rollback-serviceability-timeout-sec" in final_command
    assert "--rollback-serviceability-interval-sec" in final_command
    assert "--index-compatibility-validation-enabled" in final_command
    assert "--phase-dml-dql-validation-enabled" in final_command
    assert "--phase-new-collection-rows" in final_command
    assert "--phase-existing-dml-rows" in final_command
    assert "--phase-existing-delete-rows" in final_command
    assert "--scenario-id" in final_command
    assert "--deploy-profile" in final_command
    assert (
        "--deploy-topology /tmp/milvus-bricks/reports/deploy_topology.json"
        in final_command
    )
    assert "--schema-evolution-existing-enabled" in final_command
    assert "--schema-evolution-forward-enabled" in final_command
    resolve_command = resolve_inputs["container"]["args"][0]
    assert "invalid Milvus collection prefix parameters" in resolve_command
    assert "forward-collection-prefix" in resolve_command
    assert "invalid 2.6 -> 3.0 -> 2.6 rollback gate" in resolve_command
    assert "enabled phase flags" in resolve_command
    assert "allow_unsafe_negative_coverage" in resolve_command
    assert "and not allow_unsafe_negative_coverage" in resolve_command
    assert "approved_unsafe_negative_scenarios" in resolve_command
    assert "standalone-3-0-loon-vortex-to-2-6-negative" in resolve_command
    assert "invalid unsafe negative coverage bypass" in resolve_command

    gate = templates["gate-final-status"]
    gate_command = gate["container"]["args"][0]
    assert "orchestrator_report.json" in gate_command
    assert "allow_warning" in gate_command
    assert 'status != "passed"' in gate_command
    assert 'allow_warning and status == "warning"' in gate_command


def test_standalone_2_6_upgrade_rollback_template_renders_milvus_cr_from_deploy_profile():
    template = yaml.safe_load(
        (ROOT / "argo" / "standalone-2-6-upgrade-rollback.yaml").read_text()
    )
    deploy = next(
        item
        for item in template["spec"]["templates"]
        if item["name"] == "deploy-milvus"
    )
    command = deploy["container"]["args"][0]

    assert "milvus_client.requests.render_milvus_cr" in command
    assert '--deploy-profile "{{workflow.parameters.deploy-profile}}"' in command
    assert '--namespace "{{workflow.parameters.milvus-namespace}}"' in command
    assert '--image "{{workflow.parameters.base-milvus-image}}"' in command
    assert '--version "{{workflow.parameters.base-version}}"' in command
    assert (
        '--json-shredding-enabled "{{workflow.parameters.base-json-shredding-enabled}}"'
        in command
    )
    assert (
        '--loon-ffi-enabled "{{workflow.parameters.base-loon-ffi-enabled}}"' in command
    )
    assert '--vortex-enabled "{{workflow.parameters.base-vortex-enabled}}"' in command
    assert "--output-yaml /tmp/milvus-cr.yaml" in command
    assert "--summary-json /tmp/milvus-bricks/reports/deploy_topology.json" in command
    assert "kubectl apply -f /tmp/milvus-cr.yaml" in command
    assert "cat <<EOF | kubectl apply -f -" not in command


def test_standalone_2_6_upgrade_rollback_template_patches_config_matrix():
    template = yaml.safe_load(
        (ROOT / "argo" / "standalone-2-6-upgrade-rollback.yaml").read_text()
    )
    templates = {item["name"]: item for item in template["spec"]["templates"]}
    patch_command = templates["patch-milvus-image"]["container"]["args"][0]
    snapshot = templates["snapshot-milvus-config"]

    assert (
        'json.loads("""{{inputs.parameters.json-shredding-enabled}}""")'
        in patch_command
    )
    assert 'json.loads("""{{inputs.parameters.loon-ffi-enabled}}""")' in patch_command
    assert 'json.loads("""{{inputs.parameters.vortex-enabled}}""")' in patch_command
    assert '"useLoonFFI": loon_ffi_enabled' in patch_command
    assert "storageV3Enabled" not in patch_command
    assert "clear_storage_v3_enabled" not in patch_command
    assert "clear_vortex_enabled" not in patch_command
    assert '"dataNode"] = {"storage": {"format": "vortex"}}' in patch_command
    assert '"dataNode"] = {"storage": {"format": None}}' in patch_command
    assert "--patch-file /tmp/milvus-patch.json" in patch_command
    config_patch_command = templates["patch-milvus-config"]["container"]["args"][0]
    assert (
        'if [ "{{inputs.parameters.enabled}}" != "true" ]; then' in config_patch_command
    )
    assert '"dataNode"] = {"storage": {"format": "vortex"}}' in config_patch_command
    assert '"dataNode"] = {"storage": {"format": None}}' in config_patch_command
    assert "--patch-file /tmp/milvus-config-patch.json" in config_patch_command
    optional_command = templates["optional-run-brick"]["container"]["args"][0]
    assert '"status": "skipped"' in optional_command
    assert "python3 -m {{inputs.parameters.module}}" in optional_command
    assert (
        snapshot["outputs"]["artifacts"][0]["path"]
        == "/tmp/milvus-bricks/k8s/config-{{inputs.parameters.phase}}.json"
    )
    assert snapshot["container"]["volumeMounts"][0]["name"] == "milvus-test-state"


def test_standalone_3_0_upgrade_rollback_template_defaults_to_3_0_matrix():
    template = yaml.safe_load(
        (ROOT / "argo" / "standalone-3-0-upgrade-rollback.yaml").read_text()
    )

    assert template["kind"] == "WorkflowTemplate"
    assert template["metadata"]["name"] == "milvus-standalone-3-0-upgrade-rollback"
    assert template["metadata"]["namespace"] == "qa"
    assert template["spec"]["serviceAccountName"] == "milvus-upgrade-rollback-runner"
    parameter_values = {
        parameter["name"]: parameter["value"]
        for parameter in template["spec"]["arguments"]["parameters"]
    }

    assert parameter_values["client-image"] == "harbor.milvus.io/qa/fouram:2.1"
    assert parameter_values["scenario-id"] == "standalone-3-0-upgrade-rollback"
    assert (
        parameter_values["deploy-profile"]
        == "milvus_client/manifests/deploy_profiles/standalone-rocksmq.yaml"
    )
    assert parameter_values["base-milvus-image"] == (
        "harbor.milvus.io/milvusdb/milvus:3.0-20260701-d19d8484-47f6c14"
    )
    assert parameter_values["rollback-milvus-image"] == (
        "harbor.milvus.io/milvusdb/milvus:3.0-20260701-d19d8484-47f6c14"
    )
    assert parameter_values["rollback-version"] == "3.0.0"
    assert parameter_values["target-milvus-image"].startswith(
        "harbor.milvus.io/milvusdb/milvus:master-"
    )
    assert (
        parameter_values["schema-matrix"]
        == "milvus_client/manifests/schema_matrix_3_0.yaml"
    )
    assert (
        parameter_values["forward-schema-matrix"]
        == "milvus_client/manifests/schema_matrix_3_0.yaml"
    )
    assert parameter_values["collection-prefix"] == "qa_upgrade_30"
    assert parameter_values["forward-collection-prefix"] == "qa_upgrade_30_forward"
    assert parameter_values["rollback-enabled"] == "true"
    assert parameter_values["rollback-forward-validation-enabled"] == "true"
    assert parameter_values["index-compatibility-validation-enabled"] == "true"
    assert parameter_values["phase-dml-dql-validation-enabled"] == "true"
    assert parameter_values["phase-new-collection-rows"] == "3000"
    assert parameter_values["phase-existing-dml-rows"] == "1000"
    assert parameter_values["phase-existing-delete-rows"] == "100"
    assert parameter_values["schema-evolution-existing-enabled"] == "true"
    assert parameter_values["schema-evolution-forward-enabled"] == "false"
    assert parameter_values["standalone-cpu-request"] == "2"
    assert parameter_values["standalone-memory-request"] == "4Gi"
    assert parameter_values["standalone-cpu"] == "4"
    assert parameter_values["standalone-memory"] == "8Gi"
    assert parameter_values["observe-before-upgrade-sec"] == "300"
    assert parameter_values["observe-after-upgrade-sec"] == "300"
    assert parameter_values["observe-before-rollback-sec"] == "300"
    assert parameter_values["observe-after-rollback-sec"] == "300"
    assert parameter_values["rollback-serviceability-timeout-sec"] == "900"
    assert parameter_values["rollback-serviceability-interval-sec"] == "10"
    assert parameter_values["pressure-fail-on-error"] == "false"
    assert parameter_values["gate-allow-warning"] == "true"

    templates = {item["name"]: item for item in template["spec"]["templates"]}
    main = next(
        item for item in template["spec"]["templates"] if item["name"] == "main"
    )
    tasks = {task["name"]: task for task in main["dag"]["tasks"]}
    assert templates["pressure-daemon"]["daemon"] is True
    assert "volumeMounts" not in templates["pressure-daemon"]["container"]
    assert "volumeMounts" not in templates["run-pressure-suite"]["container"]
    assert templates["maybe-cleanup"]["container"]["volumeMounts"][0] == {
        "name": "milvus-test-state",
        "mountPath": "/tmp/milvus-bricks",
    }
    assert "patch-milvus-config" in templates
    assert tasks["validate-index-compatibility-after-upgrade"]["dependencies"] == [
        "validate-after-upgrade",
        "pressure-daemon",
    ]
    assert tasks["validate-phase-dml-dql-after-upgrade"]["dependencies"] == [
        "validate-index-compatibility-after-upgrade",
        "pressure-daemon",
    ]
    assert tasks["strict-pressure-after-upgrade"]["dependencies"] == [
        "validate-phase-dml-dql-after-upgrade",
        "pressure-daemon",
    ]
    assert tasks["validate-index-compatibility-after-rollback"]["dependencies"] == [
        "observe-after-rollback",
        "pressure-daemon",
    ]
    assert tasks["validate-phase-dml-dql-after-rollback"]["dependencies"] == [
        "validate-index-compatibility-after-rollback",
        "pressure-daemon",
    ]
    assert tasks["validate-after-rollback"]["dependencies"] == [
        "validate-phase-dml-dql-after-rollback",
        "pressure-daemon",
    ]
    index_after_upgrade_args = {
        parameter["name"]: parameter["value"]
        for parameter in tasks["validate-index-compatibility-after-upgrade"][
            "arguments"
        ]["parameters"]
    }
    index_after_rollback_args = {
        parameter["name"]: parameter["value"]
        for parameter in tasks["validate-index-compatibility-after-rollback"][
            "arguments"
        ]["parameters"]
    }
    assert (
        index_after_upgrade_args["module"]
        == "milvus_client.requests.validate_index_compatibility"
    )
    assert "--phase after-upgrade" in index_after_upgrade_args["args"]
    assert "--rebuild-index false" in index_after_upgrade_args["args"]
    assert "--phase after-rollback" in index_after_rollback_args["args"]
    assert "--rebuild-index false" in index_after_rollback_args["args"]
    phase_after_upgrade_args = {
        parameter["name"]: parameter["value"]
        for parameter in tasks["validate-phase-dml-dql-after-upgrade"]["arguments"][
            "parameters"
        ]
    }
    phase_after_rollback_args = {
        parameter["name"]: parameter["value"]
        for parameter in tasks["validate-phase-dml-dql-after-rollback"]["arguments"][
            "parameters"
        ]
    }
    assert (
        phase_after_upgrade_args["module"]
        == "milvus_client.requests.validate_phase_dml_dql"
    )
    assert "--phase after-upgrade" in phase_after_upgrade_args["args"]
    assert (
        "--new-collection-prefix {{workflow.parameters.collection-prefix}}_after_upgrade"
        in phase_after_upgrade_args["args"]
    )
    assert (
        phase_after_rollback_args["module"]
        == "milvus_client.requests.validate_phase_dml_dql"
    )
    assert "--phase after-rollback" in phase_after_rollback_args["args"]
    assert (
        "--carried-collection-prefix {{workflow.parameters.collection-prefix}}_after_upgrade"
        in phase_after_rollback_args["args"]
    )
    final_command = templates["generate-final-report"]["container"]["args"][0]
    assert "--index-compatibility-validation-enabled" in final_command
    assert "--phase-dml-dql-validation-enabled" in final_command
    assert "--phase-new-collection-rows" in final_command
    assert "--phase-existing-dml-rows" in final_command
    assert "--phase-existing-delete-rows" in final_command
    resolve_command = templates["resolve-inputs"]["container"]["args"][0]
    assert "approved_unsafe_negative_scenarios = set()" in resolve_command
    assert "invalid unsafe negative coverage bypass" in resolve_command


def test_cluster_upgrade_rollback_template_uses_cluster_deploy_profile_and_shared_dag():
    template = yaml.safe_load(
        (ROOT / "argo" / "cluster-upgrade-rollback.yaml").read_text()
    )

    assert template["kind"] == "WorkflowTemplate"
    assert template["metadata"]["name"] == "milvus-cluster-upgrade-rollback"
    assert template["metadata"]["namespace"] == "qa"
    assert template["spec"]["serviceAccountName"] == "milvus-upgrade-rollback-runner"
    parameter_values = {
        parameter["name"]: parameter["value"]
        for parameter in template["spec"]["arguments"]["parameters"]
    }
    assert (
        parameter_values["scenario-id"]
        == "cluster-3-0-baseline-to-3-0-latest-rollback-3-0-baseline"
    )
    assert (
        parameter_values["deploy-profile"]
        == "milvus_client/manifests/deploy_profiles/cluster-woodpecker-1cu.yaml"
    )
    assert (
        parameter_values["schema-matrix"]
        == "milvus_client/manifests/schema_matrix_3_0.yaml"
    )
    assert parameter_values["collection-prefix"] == "qa_gate_cluster_30_to_30latest"
    assert (
        parameter_values["forward-collection-prefix"]
        == "qa_gate_cluster_30_to_30latest_forward"
    )
    assert parameter_values["rows-per-collection"] == "5000"
    assert parameter_values["pressure-fail-on-error"] == "true"
    assert parameter_values["gate-allow-warning"] == "false"
    assert parameter_values["rollback-enabled"] == "true"
    assert parameter_values["index-compatibility-validation-enabled"] == "true"
    assert parameter_values["phase-dml-dql-validation-enabled"] == "true"
    assert parameter_values["phase-new-collection-rows"] == "3000"
    assert parameter_values["phase-existing-dml-rows"] == "1000"
    assert parameter_values["phase-existing-delete-rows"] == "100"
    assert parameter_values["schema-evolution-existing-enabled"] == "true"

    templates = {item["name"]: item for item in template["spec"]["templates"]}
    main = templates["main"]
    tasks = {task["name"]: task for task in main["dag"]["tasks"]}
    assert {
        "deploy-base",
        "wait-base-ready",
        "create-compat-schema",
        "seed-compat-data",
        "validate-before-upgrade",
        "pressure-daemon",
        "patch-upgrade",
        "wait-upgrade-ready",
        "wait-upgrade-serviceability",
        "patch-rollback",
        "wait-rollback-ready",
        "wait-rollback-serviceability",
        "validate-index-compatibility-after-upgrade",
        "validate-index-compatibility-after-rollback",
        "validate-phase-dml-dql-after-upgrade",
        "validate-phase-dml-dql-after-rollback",
        "validate-after-rollback",
        "generate-final-report",
        "gate-final-status",
    } <= set(tasks)
    deploy_command = templates["deploy-milvus"]["container"]["args"][0]
    assert "milvus_client.requests.render_milvus_helm_values" in deploy_command
    assert "helm upgrade --install" in deploy_command
    assert '--deploy-profile "{{workflow.parameters.deploy-profile}}"' in deploy_command
    assert "--app-name milvus-cluster-upgrade-rollback" in deploy_command
    assert (
        'for key in ["repo_name", "repo_url", "chart", "chart_version"]'
        in deploy_command
    )
    assert 'helm upgrade --install "{{workflow.name}}" "$chart"' in deploy_command
    assert '--version "$chart_version"' in deploy_command
    wait_command = templates["wait-milvus-ready"]["container"]["args"][0]
    assert "helm status" in wait_command
    assert 'get svc "$name"' in wait_command
    patch_command = templates["patch-milvus-image"]["container"]["args"][0]
    assert "helm upgrade" in patch_command
    assert "--reuse-values" in patch_command
    assert "--set-file extraConfigFiles.user\\\\.yaml=/tmp/user.yaml" in patch_command
    assert (
        'git clone --depth 1 --branch "{{workflow.parameters.repo-revision}}"'
        in patch_command
    )
    assert (
        'for key in ["repo_name", "repo_url", "chart", "chart_version"]'
        in patch_command
    )
    assert 'helm upgrade "{{workflow.name}}" "$chart"' in patch_command
    assert '--version "$chart_version"' in patch_command
    assert "zilliztech/milvus" not in patch_command
    config_patch_command = templates["patch-milvus-config"]["container"]["args"][0]
    assert (
        "--set-file extraConfigFiles.user\\\\.yaml=/tmp/user.yaml"
        in config_patch_command
    )
    assert (
        'for key in ["repo_name", "repo_url", "chart", "chart_version"]'
        in config_patch_command
    )
    assert 'helm upgrade "{{workflow.name}}" "$chart"' in config_patch_command
    assert '--version "$chart_version"' in config_patch_command
    assert "zilliztech/milvus" not in config_patch_command
    snapshot_command = templates["snapshot-milvus-config"]["container"]["args"][0]
    assert "helm get values" in snapshot_command
    assert "workloads-${phase}.json" in snapshot_command
    assert 'export SNAPSHOT_VERSION="$version"' in snapshot_command
    assert 'version = os.environ["SNAPSHOT_VERSION"]' in snapshot_command
    assert '"version": "$version"' not in snapshot_command
    assert '"current_version": "$version"' not in snapshot_command
    cleanup_command = templates["maybe-cleanup"]["container"]["args"][0]
    assert (
        'helm uninstall "{{workflow.name}}" -n "$ns" --wait --timeout 5m || true'
        not in cleanup_command
    )
    assert (
        'if ! helm uninstall "{{workflow.name}}" -n "$ns" --wait --timeout 5m; then'
        in cleanup_command
    )
    assert (
        'echo "helm uninstall failed for release {{workflow.name}}"' in cleanup_command
    )
    assert "cleanup_failed=true" in cleanup_command
    assert "pvcs-before-explicit-cleanup.json" in cleanup_command
    assert "failed to list PVCs before explicit cleanup" in cleanup_command
    assert "explicit-pvcs-to-delete.txt" in cleanup_command
    assert "data-{re.escape(release)}-etcd-[0-9]+" in cleanup_command
    assert (
        "woodpecker-storage-{re.escape(release)}-woodpecker-[0-9]+" in cleanup_command
    )
    assert (
        'kubectl -n "$ns" delete "$pvc_resource" --ignore-not-found=true'
        in cleanup_command
    )
    assert (
        'delete services,persistentvolumeclaims,deployments,statefulsets,configmaps,secrets,serviceaccounts,poddisruptionbudgets,jobs,roles,rolebindings -l release="{{workflow.name}}"'
        in cleanup_command
    )
    assert (
        'if helm status "{{workflow.name}}" -n "$ns" >/dev/null 2>&1; then'
        in cleanup_command
    )
    assert "pvcs-after-cleanup.json" in cleanup_command
    assert "failed to list PVCs after cleanup" in cleanup_command
    assert "remaining-pvcs-after-cleanup.txt" in cleanup_command
    assert 'labels.get("release") == release' in cleanup_command
    assert 'labels.get("app.kubernetes.io/instance") == release' in cleanup_command
    assert '[ "$cleanup_failed" = "false" ]' in cleanup_command
    collect_command = templates["collect-artifacts"]["container"]["args"][0]
    assert "release_resources.txt" in collect_command
    assert (
        'get pods,services,persistentvolumeclaims,deployments,statefulsets,replicasets,jobs,configmaps,secrets,serviceaccounts,roles,rolebindings -l release="{{workflow.name}}"'
        in collect_command
    )
    assert "release_resources.txt" in cleanup_command
    final_command = templates["generate-final-report"]["container"]["args"][0]
    assert "--scenario-id" in final_command
    assert "--deploy-profile" in final_command
    assert "--index-compatibility-validation-enabled" in final_command
    assert "--phase-dml-dql-validation-enabled" in final_command
    assert "--phase-new-collection-rows" in final_command
    assert "--phase-existing-dml-rows" in final_command
    assert "--phase-existing-delete-rows" in final_command
    assert (
        "--deploy-topology /tmp/milvus-bricks/reports/deploy_topology.json"
        in final_command
    )
    assert "optional-run-brick" in templates
    main = next(
        item for item in template["spec"]["templates"] if item["name"] == "main"
    )
    tasks = {task["name"]: task for task in main["dag"]["tasks"]}
    assert tasks["strict-pressure-before-upgrade"]["dependencies"] == [
        "validate-before-upgrade"
    ]
    assert tasks["pressure-daemon"]["dependencies"] == [
        "strict-pressure-before-upgrade"
    ]
    assert tasks["schema-evolution-existing"]["template"] == "optional-run-brick"
    assert tasks["wait-upgrade-serviceability"]["dependencies"] == [
        "precheck-after-upgrade",
        "pressure-daemon",
    ]
    assert tasks["validate-after-upgrade"]["dependencies"] == [
        "wait-upgrade-serviceability",
        "pressure-daemon",
    ]
    assert tasks["validate-index-compatibility-after-upgrade"]["dependencies"] == [
        "validate-after-upgrade",
        "pressure-daemon",
    ]
    assert tasks["validate-phase-dml-dql-after-upgrade"]["dependencies"] == [
        "validate-index-compatibility-after-upgrade",
        "pressure-daemon",
    ]
    assert tasks["strict-pressure-after-upgrade"]["dependencies"] == [
        "validate-phase-dml-dql-after-upgrade",
        "pressure-daemon",
    ]
    assert tasks["schema-evolution-existing"]["dependencies"] == [
        "strict-pressure-after-upgrade",
        "pressure-daemon",
    ]
    assert tasks["validate-forward-after-rollback"]["when"] == (
        "{{workflow.parameters.rollback-enabled}} == true && {{workflow.parameters.forward-workload-enabled}} == true"
    )
    assert tasks["strict-pressure-before-rollback"]["dependencies"] == [
        "observe-before-rollback",
        "pressure-daemon",
    ]
    assert tasks["patch-rollback"]["dependencies"] == [
        "strict-pressure-before-rollback",
        "pressure-daemon",
    ]
    assert tasks["wait-rollback-serviceability"]["dependencies"] == [
        "precheck-after-rollback",
        "pressure-daemon",
    ]
    assert tasks["wait-forward-rollback-serviceability"]["dependencies"] == [
        "wait-rollback-serviceability",
        "pressure-daemon",
    ]
    assert tasks["observe-after-rollback"]["dependencies"] == [
        "wait-forward-rollback-serviceability",
        "assert-after-rollback-storage-config",
        "pressure-daemon",
    ]
    assert tasks["validate-index-compatibility-after-rollback"]["dependencies"] == [
        "observe-after-rollback",
        "pressure-daemon",
    ]
    assert tasks["validate-phase-dml-dql-after-rollback"]["dependencies"] == [
        "validate-index-compatibility-after-rollback",
        "pressure-daemon",
    ]
    assert tasks["validate-after-rollback"]["dependencies"] == [
        "validate-phase-dml-dql-after-rollback",
        "pressure-daemon",
    ]
    assert tasks["validate-forward-after-rollback"]["dependencies"] == [
        "validate-after-rollback",
        "pressure-daemon",
    ]
    assert tasks["strict-pressure-after-rollback"]["dependencies"] == [
        "validate-forward-after-rollback",
        "pressure-daemon",
    ]
    assert tasks["stop-pressure"]["dependencies"] == [
        "strict-pressure-after-rollback",
        "pressure-daemon",
    ]
    seed_args = {
        parameter["name"]: parameter["value"]
        for parameter in tasks["seed-compat-data"]["arguments"]["parameters"]
    }
    forward_seed_args = {
        parameter["name"]: parameter["value"]
        for parameter in tasks["seed-forward-data"]["arguments"]["parameters"]
    }
    assert (
        "--checkpoint-file /tmp/milvus-bricks/checkpoints/baseline/seed_data.json"
        in seed_args["args"]
    )
    assert (
        "--checkpoint-file /tmp/milvus-bricks/checkpoints/forward/seed_data.json"
        in forward_seed_args["args"]
    )
    validate_index_after_upgrade_args = {
        parameter["name"]: parameter["value"]
        for parameter in tasks["validate-index-compatibility-after-upgrade"][
            "arguments"
        ]["parameters"]
    }
    validate_index_after_rollback_args = {
        parameter["name"]: parameter["value"]
        for parameter in tasks["validate-index-compatibility-after-rollback"][
            "arguments"
        ]["parameters"]
    }
    assert (
        validate_index_after_upgrade_args["module"]
        == "milvus_client.requests.validate_index_compatibility"
    )
    assert (
        validate_index_after_upgrade_args["enabled"]
        == "{{workflow.parameters.index-compatibility-validation-enabled}}"
    )
    assert "--phase after-upgrade" in validate_index_after_upgrade_args["args"]
    assert "--rebuild-index false" in validate_index_after_upgrade_args["args"]
    assert (
        "--index-checkpoint-file /tmp/milvus-bricks/checkpoints/index_compatibility.json"
        in validate_index_after_upgrade_args["args"]
    )
    assert (
        validate_index_after_rollback_args["module"]
        == "milvus_client.requests.validate_index_compatibility"
    )
    assert (
        validate_index_after_rollback_args["enabled"]
        == "{{workflow.parameters.index-compatibility-validation-enabled}}"
    )
    assert "--phase after-rollback" in validate_index_after_rollback_args["args"]
    assert "--rebuild-index false" in validate_index_after_rollback_args["args"]
    assert (
        "--index-checkpoint-file /tmp/milvus-bricks/checkpoints/index_compatibility.json"
        in validate_index_after_rollback_args["args"]
    )
    upgrade_serviceability_args = {
        parameter["name"]: parameter["value"]
        for parameter in tasks["wait-upgrade-serviceability"]["arguments"]["parameters"]
    }
    assert (
        upgrade_serviceability_args["module"]
        == "milvus_client.requests.wait_data_serviceability"
    )
    assert (
        upgrade_serviceability_args["collection-prefix"]
        == "{{workflow.parameters.collection-prefix}}"
    )
    assert (
        "--checkpoint-file /tmp/milvus-bricks/checkpoints/baseline/seed_data.json"
        in upgrade_serviceability_args["args"]
    )
    assert (
        "--timeout-sec {{workflow.parameters.rollback-serviceability-timeout-sec}}"
        in upgrade_serviceability_args["args"]
    )
    assert (
        "--interval-sec {{workflow.parameters.rollback-serviceability-interval-sec}}"
        in upgrade_serviceability_args["args"]
    )
    resolve_command = templates["resolve-inputs"]["container"]["args"][0]
    assert "approved_unsafe_negative_scenarios = set()" in resolve_command
    assert "invalid unsafe negative coverage bypass" in resolve_command


def test_standalone_2_6_upgrade_rollback_rbac_is_namespace_scoped():
    docs = [
        doc
        for doc in yaml.safe_load_all(
            (ROOT / "argo" / "standalone-2-6-upgrade-rollback-rbac.yaml").read_text()
        )
        if doc
    ]
    kinds = [doc["kind"] for doc in docs]
    assert kinds == ["ServiceAccount", "Role", "RoleBinding", "Role", "RoleBinding"]

    service_account = docs[0]
    qa_role = docs[1]
    milvus_role = docs[3]
    milvus_binding = docs[4]

    assert service_account["metadata"]["namespace"] == "qa"
    assert service_account["metadata"]["name"] == "milvus-upgrade-rollback-runner"
    assert qa_role["metadata"]["namespace"] == "qa"
    assert milvus_role["metadata"]["namespace"] == "qa-milvus"
    assert milvus_binding["subjects"][0] == {
        "kind": "ServiceAccount",
        "name": "milvus-upgrade-rollback-runner",
        "namespace": "qa",
    }

    milvus_resources = {
        resource for rule in milvus_role["rules"] for resource in rule["resources"]
    }
    qa_resources = {
        resource for rule in qa_role["rules"] for resource in rule["resources"]
    }
    assert {
        "milvuses",
        "persistentvolumeclaims",
        "pods/log",
        "events",
    } <= milvus_resources
    assert "configmaps" in qa_resources
    assert "workflowtaskresults" in qa_resources
    assert "workflows" in qa_resources
    assert "pod logs" not in milvus_resources
    assert any(
        "" in rule["apiGroups"]
        and "pods/exec" in rule["resources"]
        and {"create"} <= set(rule["verbs"])
        for rule in milvus_role["rules"]
    )


def test_cluster_upgrade_rollback_rbac_allows_helm_pulsar_chart_resources():
    docs = [
        doc
        for doc in yaml.safe_load_all(
            (ROOT / "argo" / "cluster-upgrade-rollback-rbac.yaml").read_text()
        )
        if doc
    ]
    kinds = [doc["kind"] for doc in docs]
    assert kinds == ["ServiceAccount", "Role", "RoleBinding", "Role", "RoleBinding"]

    service_account = docs[0]
    qa_role = docs[1]
    milvus_role = docs[3]
    milvus_binding = docs[4]

    assert service_account["metadata"]["namespace"] == "qa"
    assert service_account["metadata"]["name"] == "milvus-upgrade-rollback-runner"
    assert qa_role["metadata"]["namespace"] == "qa"
    assert "workflows" in {
        resource for rule in qa_role["rules"] for resource in rule["resources"]
    }
    assert milvus_role["metadata"]["namespace"] == "qa-milvus"
    assert milvus_binding["subjects"][0] == {
        "kind": "ServiceAccount",
        "name": "milvus-upgrade-rollback-runner",
        "namespace": "qa",
    }

    def has_verbs(api_group: str, resource: str, required_verbs: set[str]) -> bool:
        for rule in milvus_role["rules"]:
            if (
                api_group in rule["apiGroups"]
                and resource in rule["resources"]
                and required_verbs <= set(rule["verbs"])
            ):
                return True
        return False

    write_verbs = {"create", "patch", "update", "delete"}
    read_write_verbs = {"get", "list", "watch", "create", "patch", "update", "delete"}
    assert has_verbs("batch", "jobs", write_verbs)
    assert has_verbs("rbac.authorization.k8s.io", "roles", write_verbs)
    assert has_verbs("rbac.authorization.k8s.io", "rolebindings", write_verbs)
    assert has_verbs("", "pods", write_verbs)
    for api_group in ("", "extensions", "apps"):
        for resource in ("pods", "services", "deployments", "secrets", "statefulsets"):
            assert has_verbs(api_group, resource, read_write_verbs), (
                api_group,
                resource,
            )
    assert "pods/log" in {
        resource for rule in milvus_role["rules"] for resource in rule["resources"]
    }
    assert any(
        "" in rule["apiGroups"]
        and "pods/exec" in rule["resources"]
        and {"create"} <= set(rule["verbs"])
        for rule in milvus_role["rules"]
    )
