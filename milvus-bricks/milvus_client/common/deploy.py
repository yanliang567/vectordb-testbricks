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
    deployer = profile.get("deployer", "operator")
    if deployer not in {"operator", "helm"}:
        raise ValueError(f"{source}: deployer must be operator or helm")
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
        raise ValueError(
            f"{source}: standalone profile must define components.standalone"
        )
    if mode == "cluster":
        required = {"mixCoord", "proxy", "queryNode", "dataNode", "streamingNode"}
        missing = sorted(required - set(components))
        if missing:
            raise ValueError(
                f"{source}: cluster profile missing components: {', '.join(missing)}"
            )
    if deployer == "helm":
        helm = profile.get("helm")
        if not isinstance(helm, dict):
            raise ValueError(
                f"{source}: helm settings are required when deployer is helm"
            )
        for field in ("repo_name", "repo_url", "chart", "chart_version"):
            if not helm.get(field):
                raise ValueError(f"{source}: helm.{field} is required")
        if mode != "cluster":
            raise ValueError(
                f"{source}: helm deployer is currently supported only for cluster profiles"
            )
    for dependency in ("etcd", "storage"):
        in_cluster = dependencies.get(dependency, {}).get("inCluster")
        if not isinstance(in_cluster, dict):
            raise ValueError(
                f"{source}: dependencies.{dependency}.inCluster is required"
            )
        if in_cluster.get("deletionPolicy") != "Delete":
            raise ValueError(
                f"{source}: dependencies.{dependency}.inCluster.deletionPolicy must be Delete"
            )
        if in_cluster.get("pvcDeletion") is not True:
            raise ValueError(
                f"{source}: dependencies.{dependency}.inCluster.pvcDeletion must be true"
            )


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


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def split_image_ref(image: str) -> tuple[str, str]:
    if ":" not in image.rsplit("/", 1)[-1]:
        raise ValueError(f"image must include a tag: {image}")
    repository, tag = image.rsplit(":", 1)
    if not repository or not tag:
        raise ValueError(f"image must include repository and tag: {image}")
    return repository, tag


_HELM_COMPONENT_KEYS = {
    "mixCoord": "mixCoordinator",
    "proxy": "proxy",
    "queryNode": "queryNode",
    "dataNode": "dataNode",
    "streamingNode": "streamingNode",
}

_HELM_CHART_MANAGED_LABELS = {
    "app.kubernetes.io/name",
    "app.kubernetes.io/instance",
    "app.kubernetes.io/managed-by",
    "app.kubernetes.io/version",
    "helm.sh/chart",
    "component",
}


def _storage_user_yaml(
    *,
    json_shredding_enabled: bool,
    loon_ffi_enabled: bool,
    vortex_enabled: bool,
) -> str:
    config = _storage_config(
        json_shredding_enabled=json_shredding_enabled,
        loon_ffi_enabled=loon_ffi_enabled,
        vortex_enabled=vortex_enabled,
    )
    return yaml.safe_dump(config, sort_keys=False)


def _validate_helm_custom_labels(labels: dict[str, str] | None) -> None:
    chart_managed = sorted(_HELM_CHART_MANAGED_LABELS & set(labels or {}))
    if chart_managed:
        raise ValueError(
            "Helm custom labels must not override chart-managed selector labels: "
            + ", ".join(chart_managed)
        )


def render_milvus_helm_values(
    *,
    profile: dict[str, Any],
    name: str,
    namespace: str,
    image: str,
    version: str,
    image_pull_policy: str = "Always",
    json_shredding_enabled: bool = False,
    loon_ffi_enabled: bool = False,
    vortex_enabled: bool = False,
    labels: dict[str, str] | None = None,
    annotations: dict[str, str] | None = None,
) -> dict[str, Any]:
    validate_deploy_profile(profile)
    if profile.get("deployer", "operator") != "helm":
        raise ValueError("render_milvus_helm_values requires deployer: helm")
    _validate_helm_custom_labels(labels)
    repository, tag = split_image_ref(image)
    values = _deep_merge(
        {
            "cluster": {"enabled": profile["mode"] == "cluster"},
            "service": {"type": "ClusterIP"},
            "image": {
                "all": {
                    "repository": repository,
                    "tag": tag,
                    "pullPolicy": image_pull_policy,
                }
            },
            "extraConfigFiles": {
                "user.yaml": _storage_user_yaml(
                    json_shredding_enabled=json_shredding_enabled,
                    loon_ffi_enabled=loon_ffi_enabled,
                    vortex_enabled=vortex_enabled,
                )
            },
        },
        profile.get("helm_values", {}),
    )
    values["cluster"] = _deep_merge(
        values.get("cluster", {}), {"enabled": profile["mode"] == "cluster"}
    )
    values["image"] = _deep_merge(
        values.get("image", {}),
        {
            "all": {
                "repository": repository,
                "tag": tag,
                "pullPolicy": image_pull_policy,
            }
        },
    )
    values["extraConfigFiles"] = _deep_merge(
        values.get("extraConfigFiles", {}),
        {
            "user.yaml": _storage_user_yaml(
                json_shredding_enabled=json_shredding_enabled,
                loon_ffi_enabled=loon_ffi_enabled,
                vortex_enabled=vortex_enabled,
            )
        },
    )
    values["service"] = _deep_merge(values.get("service", {}), {"type": "ClusterIP"})
    values.setdefault("fullnameOverride", name)
    values.setdefault("nameOverride", name)
    values["labels"] = _deep_merge(values.get("labels", {}), labels or {})
    _validate_helm_custom_labels(values.get("labels"))
    values["annotations"] = _deep_merge(
        values.get("annotations", {}), annotations or {}
    )
    values.setdefault("extraConfigFiles", {})["upgrade-rollback.yaml"] = yaml.safe_dump(
        {
            "workflow": {
                "name": annotations.get("zilliz.com/workflow-name")
                if annotations
                else name
            },
            "deployment": {
                "namespace": namespace,
                "profile": profile.get("name"),
                "deployer": "helm",
                "image": image,
                "version": version,
            },
        },
        sort_keys=False,
    )
    for profile_key, helm_key in _HELM_COMPONENT_KEYS.items():
        component = profile.get("components", {}).get(profile_key)
        if isinstance(component, dict):
            values[helm_key] = _deep_merge(values.get(helm_key, {}), component)
    dependencies = profile.get("dependencies", {})
    etcd_values = dependencies.get("etcd", {}).get("inCluster", {}).get("values", {})
    storage_values = (
        dependencies.get("storage", {}).get("inCluster", {}).get("values", {})
    )
    if etcd_values:
        values["etcd"] = _deep_merge(values.get("etcd", {}), etcd_values)
    if storage_values:
        values["minio"] = _deep_merge(values.get("minio", {}), storage_values)
    return values


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
    if profile.get("deployer", "operator") != "operator":
        raise ValueError("render_milvus_cr requires deployer: operator")
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


def deploy_topology_summary(
    profile: dict[str, Any], cr: dict[str, Any]
) -> dict[str, Any]:
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


def helm_deploy_topology_summary(
    profile: dict[str, Any], values: dict[str, Any], *, image: str, version: str
) -> dict[str, Any]:
    components = {}
    for profile_key, helm_key in _HELM_COMPONENT_KEYS.items():
        payload = values.get(helm_key, {})
        if isinstance(payload, dict):
            components[profile_key] = {
                "replicas": payload.get("replicas", 1),
                "resources": payload.get("resources", {}),
            }
    return {
        "profile": profile.get("name"),
        "mode": profile.get("mode"),
        "deployer": "helm",
        "helm": profile.get("helm", {}),
        "image": image,
        "version": version,
        "image_update_mode": "helm-upgrade",
        "components": components,
        "dependencies": {
            "msgStreamType": profile.get("dependencies", {}).get("msgStreamType"),
            "etcd": values.get("etcd", {}),
            "storage": values.get("minio", {}),
            "woodpecker": values.get("woodpecker", {}),
            "pulsarv3": values.get("pulsarv3", {}),
        },
        "config": yaml.safe_load(
            values.get("extraConfigFiles", {}).get("user.yaml", "{}")
        )
        or {},
    }


def dump_yaml(payload: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(payload, sort_keys=False))
