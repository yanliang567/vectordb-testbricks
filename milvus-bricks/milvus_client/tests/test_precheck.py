import json

from milvus_client.requests import precheck


class PrecheckClient:
    def __init__(self, version: str):
        self.version = version

    def list_collections(self):
        return ["collection_a"]

    def get_server_version(self):
        return self.version


def _args(tmp_path, expected_version: str, *extra: str) -> list[str]:
    return [
        "--uri",
        "http://milvus:19530",
        "--collection-prefix",
        "qa",
        "--checkpoint-dir",
        str(tmp_path / "checkpoints"),
        "--output-json",
        str(tmp_path / "result.json"),
        "--expected-server-version",
        expected_version,
        *extra,
    ]


def test_precheck_accepts_matching_server_version_family(monkeypatch, tmp_path):
    monkeypatch.setattr(
        precheck, "create_client", lambda *args: PrecheckClient("v3.0.2-dev")
    )

    code = precheck.main(_args(tmp_path, "3.0.1"))

    result = json.loads((tmp_path / "result.json").read_text())
    assert code == 0
    assert result["status"] == "passed"
    assert result["capabilities"]["server_version"] == "v3.0.2-dev"
    assert result["metrics"]["server_version_family"] == "3.0"


def test_precheck_rejects_mismatched_server_version_family(monkeypatch, tmp_path):
    monkeypatch.setattr(
        precheck, "create_client", lambda *args: PrecheckClient("v2.6.18")
    )

    code = precheck.main(_args(tmp_path, "3.0.1"))

    result = json.loads((tmp_path / "result.json").read_text())
    assert code == 1
    assert result["status"] == "failed"
    assert result["failures"] == [
        {
            "type": "SERVER_VERSION_MISMATCH",
            "message": "Milvus server version family differs from the expected phase version",
            "expected_version": "3.0.1",
            "expected_family": "3.0",
            "actual_version": "v2.6.18",
            "actual_family": "2.6",
        }
    ]


def test_precheck_rejects_server_version_below_expected_patch(monkeypatch, tmp_path):
    monkeypatch.setattr(
        precheck, "create_client", lambda *args: PrecheckClient("v3.0.0")
    )

    code = precheck.main(_args(tmp_path, "3.0.1"))

    result = json.loads((tmp_path / "result.json").read_text())
    assert code == 1
    assert result["failures"] == [
        {
            "type": "SERVER_VERSION_TOO_OLD",
            "message": "Milvus server version is below the expected phase version",
            "expected_version": "3.0.1",
            "actual_version": "v3.0.0",
        }
    ]


def test_precheck_rejects_unparseable_server_version(monkeypatch, tmp_path):
    monkeypatch.setattr(
        precheck, "create_client", lambda *args: PrecheckClient("unknown")
    )

    code = precheck.main(_args(tmp_path, "3.0.1"))

    result = json.loads((tmp_path / "result.json").read_text())
    assert code == 1
    assert result["failures"][0]["type"] == "SERVER_VERSION_UNAVAILABLE"


def test_precheck_accepts_digest_pinned_candidate_build_identity(monkeypatch, tmp_path):
    monkeypatch.setattr(
        precheck,
        "create_client",
        lambda *args: PrecheckClient("master-20260810-eaec01bc71"),
    )

    code = precheck.main(
        _args(
            tmp_path,
            "3.0.0",
            "--expected-server-image",
            "harbor.milvus.io/milvusdb/milvus:master-20260810-eaec01bc@sha256:"
            + "9" * 64,
            "--release-gate-eligible",
            "false",
        )
    )

    result = json.loads((tmp_path / "result.json").read_text())
    assert code == 0
    assert result["status"] == "passed"
    assert result["capabilities"]["server_version"] == ("master-20260810-eaec01bc71")
    assert result["capabilities"]["effective_server_version"] == "3.0.0"
    assert result["metrics"]["server_version_family"] == "3.0"
    assert result["metrics"]["server_version_validation_mode"] == (
        "immutable_image_build_identity"
    )


def test_precheck_rejects_opaque_build_for_release_gate(monkeypatch, tmp_path):
    monkeypatch.setattr(
        precheck,
        "create_client",
        lambda *args: PrecheckClient("master-20260810-eaec01bc71"),
    )

    code = precheck.main(
        _args(
            tmp_path,
            "3.0.0",
            "--expected-server-image",
            "harbor.milvus.io/milvusdb/milvus:master-20260810-eaec01bc@sha256:"
            + "9" * 64,
            "--release-gate-eligible",
            "true",
        )
    )

    result = json.loads((tmp_path / "result.json").read_text())
    assert code == 1
    assert result["failures"][0]["type"] == "SERVER_VERSION_UNAVAILABLE"


def test_precheck_rejects_unpinned_opaque_build(monkeypatch, tmp_path):
    monkeypatch.setattr(
        precheck,
        "create_client",
        lambda *args: PrecheckClient("master-20260810-eaec01bc71"),
    )

    code = precheck.main(
        _args(
            tmp_path,
            "3.0.0",
            "--expected-server-image",
            "harbor.milvus.io/milvusdb/milvus:master-20260810-eaec01bc",
            "--release-gate-eligible",
            "false",
        )
    )

    result = json.loads((tmp_path / "result.json").read_text())
    assert code == 1
    assert result["failures"][0]["type"] == "SERVER_VERSION_UNAVAILABLE"


def test_precheck_rejects_mismatched_opaque_build_identity(monkeypatch, tmp_path):
    monkeypatch.setattr(
        precheck,
        "create_client",
        lambda *args: PrecheckClient("master-20260810-deadbeef"),
    )

    code = precheck.main(
        _args(
            tmp_path,
            "3.0.0",
            "--expected-server-image",
            "harbor.milvus.io/milvusdb/milvus:master-20260810-eaec01bc@sha256:"
            + "9" * 64,
            "--release-gate-eligible",
            "false",
        )
    )

    result = json.loads((tmp_path / "result.json").read_text())
    assert code == 1
    assert result["failures"][0]["type"] == "SERVER_VERSION_UNAVAILABLE"
