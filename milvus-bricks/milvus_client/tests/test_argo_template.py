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
    dag = next(item for item in template["spec"]["templates"] if item["name"] == "upgrade-rollback-compatibility")
    tasks = {task["name"]: task for task in dag["dag"]["tasks"]}

    assert {
        "precheck",
        "create-compat-schema",
        "seed-compat-data",
        "validate-before-upgrade",
        "mixed-rw-pressure",
        "wait-upgrade",
        "observe-after-upgrade",
        "validate-after-upgrade",
        "wait-rollback",
        "observe-after-rollback",
        "validate-after-rollback",
    } <= set(tasks)
    modules = {
        task["arguments"]["parameters"][0]["value"]
        for task in tasks.values()
        if task["template"] == "brick-runner"
    }
    assert "milvus_client.requests.create_schema_matrix" in modules
    assert "milvus_client.requests.seed_data" in modules
    assert "milvus_client.requests.mixed_rw_pressure" in modules
    assert "milvus_client.requests.validate_data_integrity" in modules
