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
