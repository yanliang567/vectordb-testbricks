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
    assert parameter_values["target-milvus-image"].startswith("harbor.milvus.io/milvusdb/milvus:2.6-")
    assert parameter_values["schema-matrix"] == "milvus_client/manifests/schema_matrix_2_6.yaml"
    assert "forward-schema-matrix" not in parameter_values
    assert parameter_values["keep-milvus"] == "false"


def test_standalone_2_6_upgrade_rollback_template_runs_full_closed_loop_with_pressure():
    template = yaml.safe_load((ROOT / "argo" / "standalone-2-6-upgrade-rollback.yaml").read_text())
    main = next(item for item in template["spec"]["templates"] if item["name"] == "main")
    tasks = {task["name"]: task for task in main["dag"]["tasks"]}
    templates = {item["name"]: item for item in template["spec"]["templates"]}

    expected_tasks = {
        "resolve-inputs",
        "deploy-base",
        "wait-base-ready",
        "precheck-base",
        "create-compat-schema",
        "seed-compat-data",
        "validate-before-upgrade",
        "pressure-daemon",
        "patch-upgrade",
        "wait-upgrade-ready",
        "observe-after-upgrade",
        "precheck-after-upgrade",
        "validate-after-upgrade",
        "patch-rollback",
        "wait-rollback-ready",
        "observe-after-rollback",
        "precheck-after-rollback",
        "validate-after-rollback",
        "collect-artifacts",
    }
    assert expected_tasks <= set(tasks)

    assert templates["pressure-daemon"]["daemon"] is True
    assert "readinessProbe" in templates["pressure-daemon"]["container"]
    assert "validator-daemon" not in templates
    assert tasks["patch-upgrade"]["dependencies"] == ["pressure-daemon"]
    assert tasks["patch-rollback"]["dependencies"] == ["validate-after-upgrade"]
    assert tasks["deploy-base"]["dependencies"] == ["resolve-inputs"]
    assert tasks["collect-artifacts"]["dependencies"] == ["validate-after-rollback"]

    parameter_values = {parameter["name"]: parameter["value"] for parameter in template["spec"]["arguments"]["parameters"]}
    pressure_modules = parameter_values["pressure-modules"]
    assert parameter_values["pressure-fail-on-error"] == "true"
    assert "search_pressure" in pressure_modules
    assert "query_pressure" in pressure_modules
    assert "query_iterator_scan" in pressure_modules
    assert "upsert_pressure" in pressure_modules
    assert "delete_pressure" in pressure_modules
    assert "mixed_rw_pressure" in pressure_modules

    pressure_template = templates["pressure-daemon"]
    assert "volumeMounts" not in pressure_template["container"]
    pressure_command = pressure_template["container"]["args"][0]
    assert 'if [ "$rc" = "0" ] && [ ! -f /tmp/pressure-ready ]; then' in pressure_command
    assert "python -m" not in pressure_command
    assert "python3 -m" in pressure_command
    pressure_artifacts = {artifact["name"] for artifact in pressure_template["outputs"]["artifacts"]}
    assert "pressure-results" in pressure_artifacts

    cleanup = templates["maybe-cleanup"]
    cleanup_artifacts = {artifact["name"] for artifact in cleanup["outputs"]["artifacts"]}
    assert {"orchestrator-report", "flow-summary", "env-snapshot", "k8s-snapshot"} <= cleanup_artifacts


def test_standalone_2_6_upgrade_rollback_template_creates_4c16g_standalone():
    template = yaml.safe_load((ROOT / "argo" / "standalone-2-6-upgrade-rollback.yaml").read_text())
    deploy = next(item for item in template["spec"]["templates"] if item["name"] == "deploy-milvus")
    command = deploy["container"]["args"][0]

    assert "mode: standalone" in command
    assert 'cpu: "{{workflow.parameters.standalone-cpu}}"' in command
    assert 'memory: "{{workflow.parameters.standalone-memory}}"' in command
    assert "namespace: {{workflow.parameters.milvus-namespace}}" in command
    assert "zilliz.com/workflow-run-id" in command
    assert "msgStreamType: rocksmq" in command
    assert "pvcDeletion: true" in command


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
    assert "workflowtaskresults" in qa_resources
    assert "pod logs" not in milvus_resources
