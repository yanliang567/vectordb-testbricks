from __future__ import annotations

import sys

from milvus_client.common.args import build_common_parser, parse_bool
from milvus_client.common.capability import (
    evaluate_capabilities,
    load_capability_catalog,
)
from milvus_client.common.client import create_client, get_server_version
from milvus_client.common.result import FAILED, PASSED, SKIPPED, result_from_args
from milvus_client.common.schema import (
    build_index_params,
    build_milvus_schema,
    collection_name,
    create_collection_kwargs,
    load_feature_inventory,
    load_schema_matrix,
    rollback_incompatible_specs,
    validate_schema_matrix,
)
from milvus_client.common.version import server_version_for_feature_detection


def add_args(parser):
    parser.add_argument("--schema-matrix", required=True)
    parser.add_argument(
        "--feature-inventory", default="milvus_client/manifests/feature_inventory.yaml"
    )
    parser.add_argument(
        "--capability-catalog",
        default="milvus_client/manifests/capability_catalog.yaml",
    )
    parser.add_argument("--drop-if-exists", type=parse_bool, default=False)
    parser.add_argument("--load-after-create", type=parse_bool, default=True)
    parser.add_argument("--rollback-version", default="")
    parser.add_argument("--rollback-enabled", type=parse_bool, default=True)
    parser.add_argument(
        "--rollback-forward-validation-enabled", type=parse_bool, default=False
    )
    parser.add_argument("--dry-run", action="store_true")


def run_dry_run(
    schema_matrix: str, feature_inventory: str, capability_catalog: str
) -> dict:
    features = load_feature_inventory(feature_inventory)
    capabilities = load_capability_catalog(capability_catalog)
    specs = load_schema_matrix(schema_matrix)
    errors = validate_schema_matrix(specs, features, set(capabilities))
    return {"schemas_total": len(specs), "errors": errors}


def main(argv: list[str] | None = None) -> int:
    parser = build_common_parser("Create Milvus collections from schema matrix")
    add_args(parser)
    args = parser.parse_args(argv)
    result = result_from_args(args, "create_schema_matrix")

    try:
        dry_run = run_dry_run(
            args.schema_matrix, args.feature_inventory, args.capability_catalog
        )
    except Exception as exc:
        result.status = FAILED
        result.mark_failed(
            "MANIFEST_INVALID", "failed to load schema matrix", error=str(exc)
        )
        result.write(args.output_json)
        return 2
    if dry_run["errors"]:
        result.status = FAILED
        for error in dry_run["errors"]:
            result.mark_failed("MANIFEST_INVALID", error)
        result.write(args.output_json)
        return 2
    if args.rollback_enabled and args.rollback_forward_validation_enabled:
        incompatible_specs = rollback_incompatible_specs(
            load_schema_matrix(args.schema_matrix),
            args.rollback_version,
        )
        if incompatible_specs:
            result.status = FAILED
            result.mark_failed(
                "ROLLBACK_CONTRACT_INVALID",
                "forward schema matrix contains schemas incompatible with the rollback version",
                rollback_version=args.rollback_version,
                incompatible_schemas=[spec.name for spec in incompatible_specs],
            )
            result.write(args.output_json)
            return 2
    if args.dry_run:
        result.metrics = dry_run
        result.write(args.output_json)
        return 0

    client = create_client(args.uri, args.token, args.db_name)
    actual_server_version = get_server_version(client)
    server_version = server_version_for_feature_detection(
        actual_server_version,
        args.server_version_hint,
        expected_image=args.expected_server_image,
        release_gate_eligible=args.release_gate_eligible,
    )
    capabilities = load_capability_catalog(args.capability_catalog)
    specs = load_schema_matrix(args.schema_matrix)
    created = []
    skipped = []
    loaded = []
    failed = []
    for spec in specs:
        capability_result = evaluate_capabilities(
            spec.required_capabilities, capabilities, server_version
        )
        if capability_result["unsupported"] and args.skip_unsupported:
            skipped.append(
                {
                    "schema": spec.name,
                    "reason": "unsupported capabilities",
                    "capabilities": capability_result,
                }
            )
            continue
        name = collection_name(args.collection_prefix, spec)
        try:
            if client.has_collection(name):
                if args.drop_if_exists:
                    client.drop_collection(name)
                else:
                    if args.load_after_create:
                        client.load_collection(name)
                        loaded.append(
                            {
                                "schema": spec.name,
                                "collection": name,
                                "reason": "collection exists",
                            }
                        )
                    skipped.append(
                        {
                            "schema": spec.name,
                            "collection": name,
                            "reason": "collection exists",
                        }
                    )
                    continue
            client.create_collection(
                collection_name=name,
                schema=build_milvus_schema(spec),
                **create_collection_kwargs(spec),
            )
            for partition in spec.partitions:
                if not hasattr(client, "create_partition"):
                    raise RuntimeError("client does not support create_partition")
                has_partition = False
                if hasattr(client, "has_partition"):
                    has_partition = client.has_partition(
                        collection_name=name, partition_name=partition
                    )
                if not has_partition:
                    client.create_partition(
                        collection_name=name, partition_name=partition
                    )
            if spec.indexes:
                client.create_index(
                    collection_name=name, index_params=build_index_params(spec)
                )
            if args.load_after_create:
                client.load_collection(name)
                loaded.append(
                    {"schema": spec.name, "collection": name, "reason": "created"}
                )
            created.append({"schema": spec.name, "collection": name})
        except Exception as exc:
            failed.append({"schema": spec.name, "collection": name, "error": str(exc)})

    result.capabilities = {
        "server_version": actual_server_version,
        "effective_server_version": server_version,
    }
    result.metrics = {
        "schemas_total": len(specs),
        "created_total": len(created),
        "skipped_total": len(skipped),
        "failed_total": len(failed),
        "loaded_total": len(loaded),
        "created": created,
        "skipped": skipped,
        "loaded": loaded,
    }
    if failed:
        result.status = FAILED
        for failure in failed:
            result.mark_failed(
                "CREATE_COLLECTION_FAILED",
                "failed to create schema collection",
                **failure,
            )
    else:
        result.status = PASSED if created else SKIPPED
        if not created:
            result.skip_reason = "all schemas skipped"
    result.write(args.output_json)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
