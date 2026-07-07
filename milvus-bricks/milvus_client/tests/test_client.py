from milvus_client.common.client import normalize_uri


def test_normalize_uri_adds_http_for_host():
    assert normalize_uri("localhost") == "http://localhost:19530"


def test_normalize_uri_keeps_explicit_host_port():
    assert normalize_uri("localhost:19531") == "http://localhost:19531"
    assert normalize_uri("10.104.27.64:19531") == "http://10.104.27.64:19531"


def test_normalize_uri_keeps_full_uri():
    assert normalize_uri("https://example.com") == "https://example.com"
