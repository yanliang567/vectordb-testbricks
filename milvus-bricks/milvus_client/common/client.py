from __future__ import annotations

from typing import Any


def normalize_uri(uri: str) -> str:
    if uri.startswith("http://") or uri.startswith("https://"):
        return uri
    return f"http://{uri}:19530"


def create_client(uri: str, token: str = "", db_name: str = "default") -> Any:
    from pymilvus import MilvusClient

    normalized = normalize_uri(uri)
    kwargs = {"uri": normalized, "db_name": db_name}
    if token:
        kwargs["token"] = token
    return MilvusClient(**kwargs)


def get_server_version(client: Any) -> str:
    if hasattr(client, "get_server_version"):
        return str(client.get_server_version())
    return "unknown"

