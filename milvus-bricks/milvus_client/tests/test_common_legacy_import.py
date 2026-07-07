import pytest

import milvus_client.common as common


def test_legacy_import_errors_are_not_swallowed(monkeypatch):
    common._legacy_module = None

    def fail_import(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("optional dependency failed")

    monkeypatch.setattr(common, "import_module", fail_import)

    with pytest.raises(ImportError) as exc:
        common.__getattr__("create_n_insert")

    assert "legacy common helpers could not be imported" in str(exc.value)
    assert isinstance(exc.value.__cause__, RuntimeError)
