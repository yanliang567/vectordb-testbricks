import json

from milvus_client.requests import precheck


class PrecheckClient:
    def __init__(self, version: str):
        self.version = version

    def list_collections(self):
        return ["collection_a"]

    def get_server_version(self):
        return self.version


def _args(tmp_path, expected_version: str) -> list[str]:
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


def test_precheck_rejects_unparseable_server_version(monkeypatch, tmp_path):
    monkeypatch.setattr(
        precheck, "create_client", lambda *args: PrecheckClient("unknown")
    )

    code = precheck.main(_args(tmp_path, "3.0.1"))

    result = json.loads((tmp_path / "result.json").read_text())
    assert code == 1
    assert result["failures"][0]["type"] == "SERVER_VERSION_UNAVAILABLE"
