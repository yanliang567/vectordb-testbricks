from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_argo_template_persists_checkpoint_state_and_exports_results():
    template = yaml.safe_load((ROOT / "argo" / "upgrade-rollback-compatibility.yaml").read_text())
    brick_runner = next(item for item in template["spec"]["templates"] if item["name"] == "brick-runner")

    artifact_names = {artifact["name"] for artifact in brick_runner["outputs"]["artifacts"]}
    mounts = {mount["name"]: mount["mountPath"] for mount in brick_runner["container"]["volumeMounts"]}
    claim_names = {claim["metadata"]["name"] for claim in template["spec"]["volumeClaimTemplates"]}

    assert {"result-json", "checkpoints"} <= artifact_names
    assert mounts["milvus-bricks-state"] == "/tmp/milvus-bricks"
    assert "milvus-bricks-state" in claim_names


def test_argo_template_runs_compatibility_bricks():
    template = yaml.safe_load((ROOT / "argo" / "upgrade-rollback-compatibility.yaml").read_text())
    parameter_names = {parameter["name"] for parameter in template["spec"]["arguments"]["parameters"]}
    dag = next(item for item in template["spec"]["templates"] if item["name"] == "upgrade-rollback-compatibility")
    tasks = {task["name"]: task for task in dag["dag"]["tasks"]}
    scenario_runner = next(item for item in template["spec"]["templates"] if item["name"] == "scenario-runner")

    assert {"compat-schema-matrix", "forward-schema-matrix", "cycles", "validator-interval-sec"} <= parameter_names
    assert tasks["run-closed-loop-scenario"]["template"] == "scenario-runner"
    artifact_names = {artifact["name"] for artifact in scenario_runner["outputs"]["artifacts"]}
    assert {"result-json", "checkpoints", "results"} <= artifact_names
    command = scenario_runner["container"]["args"][0]
    assert "milvus_client.scenarios.upgrade_rollback_compatibility" in command
    assert "forward_schema_matrix" in command


def test_standalone_2_6_upgrade_rollback_template_is_2_6_only():
    template = yaml.safe_load((ROOT / "argo" / "standalone-2-6-upgrade-rollback.yaml").read_text())

    assert template["kind"] == "WorkflowTemplate"
    assert template["metadata"]["name"] == "milvus-standalone-2-6-upgrade-rollback"
    assert template["metadata"]["namespace"] == "qa"
    assert template["spec"]["serviceAccountName"] == "milvus-upgrade-rollback-runner"
    parameter_values = {parameter["name"]: parameter["value"] for parameter in template["spec"]["arguments"]["parameters"]}

    assert parameter_values["client-namespace"] == "qa"
    assert parameter_values["milvus-namespace"] == "qa-milvus"
    assert parameter_values["client-image"] == "harbor.milvus.io/qa/fouram:2.1"
    assert parameter_values["repo-revision"] == "main"
    assert parameter_values["base-milvus-image"] == "harbor.milvus.io/milvusdb/milvus:v2.6.18"
    assert parameter_values["rollback-milvus-image"] == "harbor.milvus.io/milvusdb/milvus:v2.6.18"
    assert parameter_values["target-milvus-image"].startswith("harbor.milvus.io/milvusdb/milvus:2.6-")
    assert parameter_values["schema-matrix"] == "milvus_client/manifests/schema_matrix_2_6.yaml"
    assert parameter_values["forward-schema-matrix"] == "milvus_client/manifests/schema_matrix_3_0.yaml"
    assert parameter_values["forward-collection-prefix"] == "qa_upgrade_2618_forward"
    assert parameter_values["base-json-shredding-enabled"] == "false"
    assert parameter_values["target-json-shredding-enabled"] == "false"
    assert parameter_values["rollback-json-shredding-enabled"] == "false"
    assert parameter_values["target-loon-ffi-enabled"] == "false"
    assert parameter_values["post-upgrade-config-toggle-enabled"] == "false"
    assert parameter_values["forward-workload-enabled"] == "false"
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
    assert parameter_values["keep-milvus"] == "false"


def test_standalone_2_6_upgrade_rollback_template_runs_full_closed_loop_with_pressure():
    template = yaml.safe_load((ROOT / "argo" / "standalone-2-6-upgrade-rollback.yaml").read_text())
    main = next(item for item in template["spec"]["templates"] if item["name"] == "main")
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
        "pressure-daemon",
        "observe-before-upgrade",
        "patch-upgrade",
        "wait-upgrade-ready",
        "snapshot-after-upgrade-config",
        "observe-after-upgrade",
        "precheck-after-upgrade",
        "validate-after-upgrade",
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
        "validate-after-rollback",
        "validate-forward-after-rollback",
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
    assert "validator-daemon" not in templates
    assert tasks["observe-before-upgrade"]["dependencies"] == ["pressure-daemon"]
    assert tasks["patch-upgrade"]["dependencies"] == ["observe-before-upgrade", "pressure-daemon"]
    assert tasks["precheck-base"]["dependencies"] == ["snapshot-base-config"]
    assert tasks["snapshot-after-upgrade-config"]["dependencies"] == ["wait-upgrade-ready", "pressure-daemon"]
    assert tasks["snapshot-after-rollback-config"]["dependencies"] == ["wait-rollback-ready", "pressure-daemon"]
    pressure_covered_tasks = [
        "observe-before-upgrade",
        "patch-upgrade",
        "wait-upgrade-ready",
        "snapshot-after-upgrade-config",
        "observe-after-upgrade",
        "precheck-after-upgrade",
        "validate-after-upgrade",
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
        "validate-after-rollback",
        "validate-forward-after-rollback",
        "stop-pressure",
    ]
    for task_name in pressure_covered_tasks:
        assert "pressure-daemon" in tasks[task_name]["dependencies"]
    assert tasks["schema-evolution-existing"]["dependencies"] == ["validate-after-upgrade", "pressure-daemon"]
    assert tasks["schema-evolution-existing"]["template"] == "optional-run-brick"
    assert tasks["patch-post-upgrade-config"]["dependencies"] == ["schema-evolution-existing", "pressure-daemon"]
    assert tasks["wait-post-upgrade-config-ready"]["dependencies"] == ["patch-post-upgrade-config", "pressure-daemon"]
    assert tasks["create-forward-schema"]["template"] == "optional-run-brick"
    assert tasks["validate-forward-after-upgrade"]["template"] == "optional-run-brick"
    assert tasks["schema-evolution-forward"]["dependencies"] == ["validate-forward-after-upgrade", "pressure-daemon"]
    assert tasks["schema-evolution-forward"]["template"] == "optional-run-brick"
    assert tasks["observe-before-rollback"]["dependencies"] == ["schema-evolution-forward", "pressure-daemon"]
    assert tasks["patch-rollback"]["dependencies"] == ["observe-before-rollback", "pressure-daemon"]
    schema_evolution_args = {
        parameter["name"]: parameter["value"]
        for parameter in tasks["schema-evolution-existing"]["arguments"]["parameters"]
    }
    assert schema_evolution_args["module"] == "milvus_client.requests.schema_evolution_workload"
    assert schema_evolution_args["collection-prefix"] == "{{workflow.parameters.collection-prefix}}"
    assert "--schema-matrix {{workflow.parameters.schema-matrix}}" in schema_evolution_args["args"]
    forward_evolution_args = {
        parameter["name"]: parameter["value"]
        for parameter in tasks["schema-evolution-forward"]["arguments"]["parameters"]
    }
    assert forward_evolution_args["collection-prefix"] == "{{workflow.parameters.forward-collection-prefix}}"
    assert "--schema-matrix {{workflow.parameters.forward-schema-matrix}}" in forward_evolution_args["args"]
    forward_validate_args = {
        parameter["name"]: parameter["value"]
        for parameter in tasks["validate-forward-after-rollback"]["arguments"]["parameters"]
    }
    assert forward_validate_args["collection-prefix"] == "{{workflow.parameters.forward-collection-prefix}}"
    patch_rollback_args = {
        parameter["name"]: parameter["value"]
        for parameter in tasks["patch-rollback"]["arguments"]["parameters"]
    }
    assert patch_rollback_args["image"] == "{{workflow.parameters.rollback-milvus-image}}"
    wait_rollback_args = {
        parameter["name"]: parameter["value"]
        for parameter in tasks["wait-rollback-ready"]["arguments"]["parameters"]
    }
    assert wait_rollback_args["expected-image"] == "{{workflow.parameters.rollback-milvus-image}}"
    assert tasks["deploy-base"]["dependencies"] == ["resolve-inputs"]
    assert tasks["validate-forward-after-rollback"]["dependencies"] == ["validate-after-rollback", "pressure-daemon"]
    assert tasks["stop-pressure"]["dependencies"] == ["validate-forward-after-rollback", "pressure-daemon"]
    assert tasks["check-pressure-results"]["dependencies"] == ["stop-pressure"]
    assert tasks["collect-artifacts"]["dependencies"] == ["check-pressure-results"]
    assert tasks["generate-final-report"]["dependencies"] == ["collect-artifacts"]
    assert tasks["gate-final-status"]["dependencies"] == ["generate-final-report"]

    parameter_values = {parameter["name"]: parameter["value"] for parameter in template["spec"]["arguments"]["parameters"]}
    pressure_modules = parameter_values["pressure-modules"]
    assert parameter_values["pressure-fail-on-error"] == "true"
    assert parameter_values["gate-allow-warning"] == "false"
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
    assert 'if [ "$rc" = "0" ] && [ ! -f /tmp/pressure-ready ]; then' in pressure_command
    assert "pressure-stop" in pressure_command
    assert "kubectl -n \"$pressure_ns\" create configmap \"$attempt_cm\"" in pressure_command
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
    cleanup_artifacts = {artifact["name"] for artifact in cleanup["outputs"]["artifacts"]}
    assert {"orchestrator-report", "final-report-md", "flow-summary", "env-snapshot", "k8s-snapshot"} <= cleanup_artifacts

    check_pressure = templates["check-pressure-results"]
    check_artifacts = {artifact["name"] for artifact in check_pressure["outputs"]["artifacts"]}
    assert "pressure-summary" in check_artifacts
    check_command = check_pressure["container"]["args"][0]
    assert "NO_PRESSURE_RESULTS" in check_command
    assert "PRESSURE_RESULT_MISSING" in check_command
    assert "PRESSURE_ATTEMPT_PENDING" in check_command
    assert "kubectl" in check_command
    assert "zilliz.com/pressure-result=true" in check_command
    assert "summary[\"fail_on_error\"] and failed" not in check_command

    final_report = templates["generate-final-report"]
    final_artifacts = {artifact["name"] for artifact in final_report["outputs"]["artifacts"]}
    assert {"orchestrator-report", "final-report-md", "env-snapshot", "flow-summary"} <= final_artifacts
    final_command = final_report["container"]["args"][0]
    assert "milvus_client.requests.generate_workflow_report" in final_command
    assert "pressure-summary.json" in final_command
    assert "final_report.md" in final_command
    assert "orchestrator_report.json" in final_command
    assert "--soft-fail" in final_command
    assert "--base-json-shredding-enabled" in final_command
    assert "--rollback-milvus-image" in final_command
    assert "--target-json-shredding-enabled" in final_command
    assert "--target-loon-ffi-enabled" in final_command
    assert "--forward-workload-enabled" in final_command
    assert "--observe-before-upgrade-sec" in final_command
    assert "--observe-before-rollback-sec" in final_command
    assert "--schema-evolution-existing-enabled" in final_command
    assert "--schema-evolution-forward-enabled" in final_command

    gate = templates["gate-final-status"]
    gate_command = gate["container"]["args"][0]
    assert "orchestrator_report.json" in gate_command
    assert "allow_warning" in gate_command
    assert 'status != "passed"' in gate_command
    assert 'allow_warning and status == "warning"' in gate_command


def test_standalone_2_6_upgrade_rollback_template_creates_configurable_standalone_resources():
    template = yaml.safe_load((ROOT / "argo" / "standalone-2-6-upgrade-rollback.yaml").read_text())
    deploy = next(item for item in template["spec"]["templates"] if item["name"] == "deploy-milvus")
    command = deploy["container"]["args"][0]

    assert "mode: standalone" in command
    assert 'cpu: "{{workflow.parameters.standalone-cpu-request}}"' in command
    assert 'memory: "{{workflow.parameters.standalone-memory-request}}"' in command
    assert 'cpu: "{{workflow.parameters.standalone-cpu}}"' in command
    assert 'memory: "{{workflow.parameters.standalone-memory}}"' in command
    assert "namespace: {{workflow.parameters.milvus-namespace}}" in command
    assert "jsonShreddingEnabled: {{workflow.parameters.base-json-shredding-enabled}}" in command
    assert "zilliz.com/workflow-run-id" in command
    assert "msgStreamType: rocksmq" in command
    assert "pvcDeletion: true" in command


def test_standalone_2_6_upgrade_rollback_template_patches_config_matrix():
    template = yaml.safe_load((ROOT / "argo" / "standalone-2-6-upgrade-rollback.yaml").read_text())
    templates = {item["name"]: item for item in template["spec"]["templates"]}
    patch_command = templates["patch-milvus-image"]["container"]["args"][0]
    snapshot = templates["snapshot-milvus-config"]

    assert "json.loads(\"\"\"{{inputs.parameters.json-shredding-enabled}}\"\"\")" in patch_command
    assert "json.loads(\"\"\"{{inputs.parameters.loon-ffi-enabled}}\"\"\")" in patch_command
    assert '"dataNode"] = {"storage": {"format": "vortex"}}' in patch_command
    assert '"dataNode"] = {"storage": {"format": None}}' in patch_command
    assert "--patch-file /tmp/milvus-patch.json" in patch_command
    config_patch_command = templates["patch-milvus-config"]["container"]["args"][0]
    assert 'if [ "{{inputs.parameters.enabled}}" != "true" ]; then' in config_patch_command
    assert '"dataNode"] = {"storage": {"format": "vortex"}}' in config_patch_command
    assert '"dataNode"] = {"storage": {"format": None}}' in config_patch_command
    assert "--patch-file /tmp/milvus-config-patch.json" in config_patch_command
    optional_command = templates["optional-run-brick"]["container"]["args"][0]
    assert '"status": "skipped"' in optional_command
    assert "python3 -m {{inputs.parameters.module}}" in optional_command
    assert snapshot["outputs"]["artifacts"][0]["path"] == "/tmp/milvus-bricks/k8s/config-{{inputs.parameters.phase}}.json"
    assert snapshot["container"]["volumeMounts"][0]["name"] == "milvus-test-state"


def test_standalone_3_0_upgrade_rollback_template_defaults_to_3_0_matrix():
    template = yaml.safe_load((ROOT / "argo" / "standalone-3-0-upgrade-rollback.yaml").read_text())

    assert template["kind"] == "WorkflowTemplate"
    assert template["metadata"]["name"] == "milvus-standalone-3-0-upgrade-rollback"
    assert template["metadata"]["namespace"] == "qa"
    assert template["spec"]["serviceAccountName"] == "milvus-upgrade-rollback-runner"
    parameter_values = {parameter["name"]: parameter["value"] for parameter in template["spec"]["arguments"]["parameters"]}

    assert parameter_values["client-image"] == "harbor.milvus.io/qa/fouram:2.1"
    assert parameter_values["base-milvus-image"] == (
        "harbor.milvus.io/milvusdb/milvus:3.0-20260701-d19d8484-47f6c14"
    )
    assert parameter_values["rollback-milvus-image"] == (
        "harbor.milvus.io/milvusdb/milvus:3.0-20260701-d19d8484-47f6c14"
    )
    assert parameter_values["target-milvus-image"].startswith("harbor.milvus.io/milvusdb/milvus:master-")
    assert parameter_values["schema-matrix"] == "milvus_client/manifests/schema_matrix_3_0.yaml"
    assert parameter_values["forward-schema-matrix"] == "milvus_client/manifests/schema_matrix_3_0.yaml"
    assert parameter_values["collection-prefix"] == "qa_upgrade_30"
    assert parameter_values["forward-collection-prefix"] == "qa_upgrade_30_forward"
    assert parameter_values["rollback-forward-validation-enabled"] == "true"
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
    assert parameter_values["pressure-fail-on-error"] == "true"

    templates = {item["name"]: item for item in template["spec"]["templates"]}
    assert templates["pressure-daemon"]["daemon"] is True
    assert "volumeMounts" not in templates["pressure-daemon"]["container"]
    assert "patch-milvus-config" in templates
    assert "optional-run-brick" in templates
    main = next(item for item in template["spec"]["templates"] if item["name"] == "main")
    tasks = {task["name"]: task for task in main["dag"]["tasks"]}
    assert tasks["schema-evolution-existing"]["template"] == "optional-run-brick"
    assert tasks["schema-evolution-existing"]["dependencies"] == ["validate-after-upgrade", "pressure-daemon"]


def test_standalone_2_6_upgrade_rollback_rbac_is_namespace_scoped():
    docs = [
        doc
        for doc in yaml.safe_load_all((ROOT / "argo" / "standalone-2-6-upgrade-rollback-rbac.yaml").read_text())
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
        resource
        for rule in milvus_role["rules"]
        for resource in rule["resources"]
    }
    qa_resources = {
        resource
        for rule in qa_role["rules"]
        for resource in rule["resources"]
    }
    assert {"milvuses", "persistentvolumeclaims", "pods/log", "events"} <= milvus_resources
    assert "configmaps" in qa_resources
    assert "workflowtaskresults" in qa_resources
    assert "pod logs" not in milvus_resources
