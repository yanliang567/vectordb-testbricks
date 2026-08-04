import json
from pathlib import Path

from milvus_client.requests import create_schema_matrix
from milvus_client.requests.create_schema_matrix import run_dry_run


ROOT = Path(__file__).resolve().parents[1]


def test_create_schema_matrix_dry_run_loads_manifest():
    result = run_dry_run(
        str(ROOT / "manifests" / "schema_matrix_2_6.yaml"),
        str(ROOT / "manifests" / "feature_inventory.yaml"),
        str(ROOT / "manifests" / "capability_catalog.yaml"),
    )

    assert result["schemas_total"] > 0
    assert result["errors"] == []


def _contract_args(tmp_path, matrix: Path, rollback_version: str) -> list[str]:
    return [
        "--uri",
        "http://milvus:19530",
        "--collection-prefix",
        "qa_forward",
        "--checkpoint-dir",
        str(tmp_path / "checkpoints"),
        "--output-json",
        str(tmp_path / f"result-{rollback_version}.json"),
        "--schema-matrix",
        str(matrix),
        "--feature-inventory",
        str(ROOT / "manifests" / "feature_inventory.yaml"),
        "--capability-catalog",
        str(ROOT / "manifests" / "capability_catalog.yaml"),
        "--rollback-version",
        rollback_version,
        "--rollback-forward-validation-enabled",
        "true",
        "--dry-run",
    ]


def test_create_schema_matrix_rejects_renamed_forward_only_matrix_for_older_rollback(
    tmp_path,
):
    renamed_matrix = tmp_path / "renamed-capabilities.yaml"
    renamed_matrix.write_text(
        (ROOT / "manifests" / "schema_matrix_3_0.yaml").read_text()
    )

    code = create_schema_matrix.main(_contract_args(tmp_path, renamed_matrix, "2.6.18"))

    result = json.loads((tmp_path / "result-2.6.18.json").read_text())
    assert code == 2
    assert result["failures"][0]["type"] == "ROLLBACK_CONTRACT_INVALID"
    assert "nullable_vector" in result["failures"][0]["incompatible_schemas"]


def test_create_schema_matrix_allows_forward_only_matrix_for_same_family_rollback(
    tmp_path,
):
    renamed_matrix = tmp_path / "renamed-capabilities.yaml"
    renamed_matrix.write_text(
        (ROOT / "manifests" / "schema_matrix_3_0.yaml").read_text()
    )

    code = create_schema_matrix.main(_contract_args(tmp_path, renamed_matrix, "3.0.0"))

    result = json.loads((tmp_path / "result-3.0.0.json").read_text())
    assert code == 0
    assert result["status"] == "passed"
