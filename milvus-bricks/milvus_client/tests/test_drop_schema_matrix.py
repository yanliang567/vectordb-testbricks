import json

from milvus_client.requests import drop_schema_matrix


def _write_minimal_matrix(tmp_path):
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(
        "version: \"3.0\"\n"
        "schemas:\n"
        "  - name: alpha\n"
        "  - name: beta\n"
    )
    return matrix


def _args(tmp_path, matrix):
    return [
        "--uri",
        "http://milvus:19530",
        "--collection-prefix",
        "qa_forward",
        "--checkpoint-dir",
        str(tmp_path / "checkpoints"),
        "--output-json",
        str(tmp_path / "result.json"),
        "--schema-matrix",
        str(matrix),
    ]


class FakeClient:
    def __init__(self, existing=(), drop_error=None):
        self.existing = set(existing)
        self.dropped = []
        self.drop_error = drop_error

    def has_collection(self, name):
        return name in self.existing

    def drop_collection(self, name):
        if self.drop_error is not None:
            raise self.drop_error
        self.dropped.append(name)


def test_drop_schema_matrix_drops_existing_collections(monkeypatch, tmp_path):
    matrix = _write_minimal_matrix(tmp_path)
    client = FakeClient(existing={"qa_forward_alpha", "qa_forward_beta"})
    monkeypatch.setattr(drop_schema_matrix, "create_client", lambda *a, **k: client)

    code = drop_schema_matrix.main(_args(tmp_path, matrix))

    result = json.loads((tmp_path / "result.json").read_text())
    assert code == 0
    assert result["status"] == "passed"
    assert result["metrics"]["dropped_total"] == 2
    assert result["metrics"]["missing_total"] == 0
    assert sorted(client.dropped) == ["qa_forward_alpha", "qa_forward_beta"]


def test_drop_schema_matrix_all_missing_marks_skipped(monkeypatch, tmp_path):
    matrix = _write_minimal_matrix(tmp_path)
    client = FakeClient(existing=())
    monkeypatch.setattr(drop_schema_matrix, "create_client", lambda *a, **k: client)

    code = drop_schema_matrix.main(_args(tmp_path, matrix))

    result = json.loads((tmp_path / "result.json").read_text())
    assert code == 0
    assert result["status"] == "skipped"
    assert result["skip_reason"] == "no forward collections to drop"
    assert result["metrics"]["dropped_total"] == 0
    assert result["metrics"]["missing_total"] == 2


def test_drop_schema_matrix_drop_failure_returns_1(monkeypatch, tmp_path):
    matrix = _write_minimal_matrix(tmp_path)
    client = FakeClient(
        existing={"qa_forward_alpha"}, drop_error=RuntimeError("drop boom")
    )
    monkeypatch.setattr(drop_schema_matrix, "create_client", lambda *a, **k: client)

    code = drop_schema_matrix.main(_args(tmp_path, matrix))

    result = json.loads((tmp_path / "result.json").read_text())
    assert code == 1
    assert result["status"] == "failed"
    assert result["failures"][0]["type"] == "DROP_COLLECTION_FAILED"
    assert result["failures"][0]["collection"] == "qa_forward_alpha"


def test_drop_schema_matrix_env_unavailable_returns_3(monkeypatch, tmp_path):
    matrix = _write_minimal_matrix(tmp_path)

    def boom(*args, **kwargs):
        raise RuntimeError("connect refused")

    monkeypatch.setattr(drop_schema_matrix, "create_client", boom)

    code = drop_schema_matrix.main(_args(tmp_path, matrix))

    result = json.loads((tmp_path / "result.json").read_text())
    assert code == 3
    assert result["status"] == "failed"
    assert result["failures"][0]["type"] == "ENV_UNAVAILABLE"
