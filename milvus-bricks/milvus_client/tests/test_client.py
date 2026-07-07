from milvus_client.common.client import normalize_uri


def test_normalize_uri_adds_http_for_host():
    assert normalize_uri("localhost") == "http://localhost:19530"


def test_normalize_uri_keeps_full_uri():
    assert normalize_uri("https://example.com") == "https://example.com"

