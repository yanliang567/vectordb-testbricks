from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_deploy_profile(path: str | Path) -> dict[str, Any]:
    profile_path = Path(path)
    payload = yaml.safe_load(profile_path.read_text()) or {}
    validate_deploy_profile(payload, source=str(profile_path))
    return payload


def validate_deploy_profile(profile: dict[str, Any], source: str = "<profile>") -> None:
    mode = profile.get("mode")
    if mode not in {"standalone", "cluster"}:
        raise ValueError(f"{source}: mode must be standalone or cluster")
    components = profile.get("components")
    if not isinstance(components, dict) or not components:
        raise ValueError(f"{source}: components must be a non-empty mapping")
    dependencies = profile.get("dependencies")
    if not isinstance(dependencies, dict) or not dependencies.get("msgStreamType"):
        raise ValueError(f"{source}: dependencies.msgStreamType is required")
    if mode == "standalone" and "standalone" not in components:
        raise ValueError(f"{source}: standalone profile must define components.standalone")
    if mode == "cluster":
        required = {"mixCoord", "proxy", "queryNode", "dataNode", "streamingNode"}
        missing = sorted(required - set(components))
        if missing:
            raise ValueError(f"{source}: cluster profile missing components: {', '.join(missing)}")
    for dependency in ("etcd", "storage"):
        in_cluster = dependencies.get(dependency, {}).get("inCluster")
        if not isinstance(in_cluster, dict):
            raise ValueError(f"{source}: dependencies.{dependency}.inCluster is required")
        if in_cluster.get("deletionPolicy") != "Delete":
            raise ValueError(f"{source}: dependencies.{dependency}.inCluster.deletionPolicy must be Delete")
        if in_cluster.get("pvcDeletion") is not True:
            raise ValueError(f"{source}: dependencies.{dependency}.inCluster.pvcDeletion must be true")


def _storage_config(
    *,
    json_shredding_enabled: bool,
    loon_ffi_enabled: bool,
    vortex_enabled: bool,
) -> dict[str, Any]:
    storage: dict[str, Any] = {
        "jsonShreddingEnabled": json_shredding_enabled,
    }
    if loon_ffi_enabled:
        storage["useLoonFFI"] = True
    config: dict[str, Any] = {"common": {"storage": storage}}
    if vortex_enabled:
        config["dataNode"] = {"storage": {"format": "vortex"}}
    return config


def render_milvus_cr(
    *,
    profile: dict[str, Any],
    name: str,
    namespace: str,
    image: str,
    version: str,
    image_update_mode: str,
    image_pull_policy: str = "IfNotPresent",
    json_shredding_enabled: bool = False,
    loon_ffi_enabled: bool = False,
    vortex_enabled: bool = False,
    labels: dict[str, str] | None = None,
    annotations: dict[str, str] | None = None,
) -> dict[str, Any]:
    validate_deploy_profile(profile)
    components = deepcopy(profile["components"])
    components.update(
        {
            "image": image,
            "version": version,
            "imagePullPolicy": image_pull_policy,
            "imageUpdateMode": image_update_mode,
        }
    )
    return {
        "apiVersion": "milvus.io/v1beta1",
        "kind": "Milvus",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": labels or {},
            "annotations": annotations or {},
        },
        "spec": {
            "mode": profile["mode"],
            "config": _storage_config(
                json_shredding_enabled=json_shredding_enabled,
                loon_ffi_enabled=loon_ffi_enabled,
                vortex_enabled=vortex_enabled,
            ),
            "components": components,
            "dependencies": deepcopy(profile["dependencies"]),
        },
    }


def deploy_topology_summary(profile: dict[str, Any], cr: dict[str, Any]) -> dict[str, Any]:
    spec = cr.get("spec", {})
    components = {
        name: {
            "replicas": payload.get("replicas", 1),
            "resources": payload.get("resources", {}),
        }
        for name, payload in spec.get("components", {}).items()
        if isinstance(payload, dict)
    }
    dependencies = spec.get("dependencies", {})
    return {
        "profile": profile.get("name"),
        "mode": spec.get("mode"),
        "image": spec.get("components", {}).get("image"),
        "version": spec.get("components", {}).get("version"),
        "image_update_mode": spec.get("components", {}).get("imageUpdateMode"),
        "components": components,
        "dependencies": {
            "msgStreamType": dependencies.get("msgStreamType"),
            "etcd": dependencies.get("etcd", {}),
            "storage": dependencies.get("storage", {}),
        },
        "config": spec.get("config", {}),
    }


def dump_yaml(payload: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(payload, sort_keys=False))
