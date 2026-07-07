from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import threading
import time

import yaml

from milvus_client.common.args import build_common_parser
from milvus_client.common.result import FAILED, PASSED
from milvus_client.common.result import result_from_args


def add_args(parser):
    parser.add_argument("--scenario-manifest", default="milvus_client/manifests/scenario_upgrade_rollback.yaml")
    parser.add_argument("--dry-run", action="store_true")


def build_plan(manifest: dict) -> list[dict]:
    cycles = int(manifest.get("cycles", 1))
    steps = [
        {"name": "precheck", "phase": "before_upgrade"},
        {"name": "create_compat_schema", "phase": "before_upgrade"},
        {"name": "seed_compat_data", "phase": "before_upgrade"},
        {"name": "start_mixed_rw_pressure", "phase": "before_upgrade"},
        {"name": "start_validator_loop", "phase": "before_upgrade"},
    ]
    for cycle in range(1, cycles + 1):
        steps.extend(
            [
                {"name": "wait_upgrade", "cycle": cycle, "phase": "before_upgrade"},
                {"name": "observe_after_upgrade", "cycle": cycle, "phase": "after_upgrade"},
                {"name": "validate_compat_after_upgrade", "cycle": cycle, "phase": "after_upgrade"},
                {"name": "create_forward_schema", "cycle": cycle, "phase": "after_upgrade"},
                {"name": "seed_forward_data", "cycle": cycle, "phase": "after_upgrade"},
                {"name": "validate_forward_after_upgrade", "cycle": cycle, "phase": "after_upgrade"},
                {"name": "wait_rollback", "cycle": cycle, "phase": "before_rollback"},
                {"name": "observe_after_rollback", "cycle": cycle, "phase": "after_rollback"},
                {"name": "validate_compat_only", "cycle": cycle, "phase": "after_rollback"},
            ]
        )
    steps.append({"name": "stop_background_workloads", "phase": "steady_state"})
    steps.append({"name": "final_validate_compat", "phase": "steady_state"})
    return steps


def _results_dir(args) -> Path:
    path = Path(args.checkpoint_dir).parent / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _brick_common_args(args, output_json: Path, lifecycle_phase: str, checkpoint_dir: str | None = None) -> list[str]:
    return [
        "--uri",
        args.uri,
        "--token",
        args.token,
        "--db-name",
        args.db_name,
        "--collection-prefix",
        args.collection_prefix,
        "--feature-set",
        args.feature_set,
        "--compat-mode",
        args.compat_mode,
        "--lifecycle-phase",
        lifecycle_phase,
        "--checkpoint-dir",
        checkpoint_dir or args.checkpoint_dir,
        "--output-json",
        str(output_json),
    ]


def run_brick(args, module: str, name: str, extra_args: list[str], lifecycle_phase: str, checkpoint_dir: str | None = None) -> dict:
    output_json = _results_dir(args) / f"{name}.json"
    cmd = [
        sys.executable,
        "-m",
        module,
        *_brick_common_args(args, output_json, lifecycle_phase, checkpoint_dir=checkpoint_dir),
        *extra_args,
    ]
    completed = subprocess.run(cmd, check=False)
    return {
        "name": name,
        "module": module,
        "return_code": completed.returncode,
        "output_json": str(output_json),
        "phase": lifecycle_phase,
    }


def wait_for_action(action: dict, action_name: str, step_results: list[dict]) -> None:
    wait_file = action.get("wait_file", "")
    if not wait_file:
        step_results.append({"name": f"wait_{action_name}", "return_code": 0, "wait_file": "", "skipped": True})
        return
    path = Path(wait_file)
    while not path.exists():
        time.sleep(10)
    step_results.append({"name": f"wait_{action_name}", "return_code": 0, "wait_file": str(path)})


def observe(seconds: int, phase: str, step_results: list[dict]) -> None:
    if seconds > 0:
        time.sleep(seconds)
    step_results.append({"name": f"observe_{phase}", "return_code": 0, "seconds": seconds})


def _failed_steps(step_results: list[dict], include_background: bool = True) -> list[dict]:
    return [
        step
        for step in step_results
        if (include_background or not step.get("background")) and int(step.get("return_code", 0)) != 0
    ]


def _start_pressure_loop(args, manifest: dict, stop_event: threading.Event, step_results: list[dict]) -> threading.Thread:
    workload = manifest.get("workloads", {}).get("mixed_rw", {})
    max_workers = str(workload.get("max_workers", 4))
    batch_size = str(workload.get("batch_size", 10))
    slice_duration = str(max(1, int(workload.get("slice_duration_sec", 30))))
    baseline_rows = str(manifest.get("rows_per_collection", 0))

    def run_loop() -> None:
        iteration = 0
        while not stop_event.is_set():
            iteration += 1
            result = run_brick(
                args,
                "milvus_client.requests.mixed_rw_pressure",
                f"mixed_rw_pressure_loop_{iteration}",
                [
                    "--schema-matrix",
                    manifest["compat_schema_matrix"],
                    "--duration-sec",
                    slice_duration,
                    "--max-workers",
                    max_workers,
                    "--batch-size",
                    batch_size,
                    "--baseline-start-id",
                    "0",
                    "--baseline-rows-per-collection",
                    baseline_rows,
                ],
                "steady_state",
            )
            result["background"] = "mixed_rw_pressure"
            step_results.append(result)

    thread = threading.Thread(target=run_loop, name="mixed-rw-pressure-loop", daemon=True)
    thread.start()
    return thread


def _start_validator_loop(args, manifest: dict, stop_event: threading.Event, step_results: list[dict]) -> threading.Thread:
    validator = manifest.get("workloads", {}).get("validator", {})
    interval_sec = max(1, int(validator.get("interval_sec", 60)))

    def run_loop() -> None:
        iteration = 0
        while not stop_event.is_set():
            iteration += 1
            result = run_brick(
                args,
                "milvus_client.requests.validate_data_integrity",
                f"validate_compat_loop_{iteration}",
                ["--schema-matrix", manifest["compat_schema_matrix"]],
                "steady_state",
            )
            result["background"] = "validator"
            step_results.append(result)
            stop_event.wait(interval_sec)

    thread = threading.Thread(target=run_loop, name="validator-loop", daemon=True)
    thread.start()
    return thread


def execute_scenario(args, manifest: dict) -> tuple[bool, list[dict]]:
    step_results: list[dict] = []
    cycles = int(manifest.get("cycles", 1))
    fail_on_background_failure = bool(manifest.get("fail_on_background_failure", True))
    compat_schema = manifest["compat_schema_matrix"]
    forward_schema = manifest.get("forward_schema_matrix", "")
    rows_per_collection = str(manifest.get("rows_per_collection", 1000))
    batch_size = str(manifest.get("batch_size", 100))

    step_results.append(run_brick(args, "milvus_client.requests.precheck", "precheck", [], "before_upgrade"))
    step_results.append(
        run_brick(
            args,
            "milvus_client.requests.create_schema_matrix",
            "create_compat_schema",
            ["--schema-matrix", compat_schema, "--drop-if-exists", "true", "--load-after-create", "true"],
            "before_upgrade",
        )
    )
    step_results.append(
        run_brick(
            args,
            "milvus_client.requests.seed_data",
            "seed_compat_data",
            ["--schema-matrix", compat_schema, "--rows-per-collection", rows_per_collection, "--batch-size", batch_size],
            "before_upgrade",
        )
    )
    step_results.append(
        run_brick(args, "milvus_client.requests.validate_data_integrity", "validate_before_upgrade", ["--schema-matrix", compat_schema], "before_upgrade")
    )
    if _failed_steps(step_results):
        return False, step_results

    stop_event = threading.Event()
    pressure_thread = _start_pressure_loop(args, manifest, stop_event, step_results)
    validator_thread = _start_validator_loop(args, manifest, stop_event, step_results)
    try:
        for cycle in range(1, cycles + 1):
            wait_for_action(manifest.get("actions", {}).get("upgrade", {}), "upgrade", step_results)
            observe(int(manifest.get("observe_after_upgrade_sec", 0)), "after_upgrade", step_results)
            step_results.append(
                run_brick(args, "milvus_client.requests.validate_data_integrity", f"validate_compat_after_upgrade_{cycle}", ["--schema-matrix", compat_schema], "after_upgrade")
            )
            if forward_schema:
                forward_checkpoint_dir = str(Path(args.checkpoint_dir) / f"forward_cycle_{cycle}")
                start_id = str(cycle * 1_000_000)
                step_results.append(
                    run_brick(
                        args,
                        "milvus_client.requests.create_schema_matrix",
                        f"create_forward_schema_{cycle}",
                        ["--schema-matrix", forward_schema, "--drop-if-exists", "true", "--load-after-create", "true"],
                        "after_upgrade",
                        checkpoint_dir=forward_checkpoint_dir,
                    )
                )
                step_results.append(
                    run_brick(
                        args,
                        "milvus_client.requests.seed_data",
                        f"seed_forward_data_{cycle}",
                        [
                            "--schema-matrix",
                            forward_schema,
                            "--rows-per-collection",
                            rows_per_collection,
                            "--batch-size",
                            batch_size,
                            "--start-id",
                            start_id,
                        ],
                        "after_upgrade",
                        checkpoint_dir=forward_checkpoint_dir,
                    )
                )
                step_results.append(
                    run_brick(
                        args,
                        "milvus_client.requests.validate_data_integrity",
                        f"validate_forward_after_upgrade_{cycle}",
                        ["--schema-matrix", forward_schema],
                        "after_upgrade",
                        checkpoint_dir=forward_checkpoint_dir,
                    )
                )
            wait_for_action(manifest.get("actions", {}).get("rollback", {}), "rollback", step_results)
            observe(int(manifest.get("observe_after_rollback_sec", 0)), "after_rollback", step_results)
            step_results.append(
                run_brick(args, "milvus_client.requests.validate_data_integrity", f"validate_compat_after_rollback_{cycle}", ["--schema-matrix", compat_schema], "after_rollback")
            )
    finally:
        stop_event.set()
        pressure_thread.join(timeout=120)
        validator_thread.join(timeout=120)

    step_results.append(
        run_brick(args, "milvus_client.requests.validate_data_integrity", "final_validate_compat", ["--schema-matrix", compat_schema], "steady_state")
    )
    required_failures = _failed_steps(step_results, include_background=fail_on_background_failure)
    return not required_failures, step_results


def main(argv: list[str] | None = None) -> int:
    parser = build_common_parser("Upgrade/rollback compatibility scenario")
    add_args(parser)
    args = parser.parse_args(argv)
    result = result_from_args(args, "upgrade_rollback_compatibility")
    manifest = yaml.safe_load(Path(args.scenario_manifest).read_text()) or {}
    plan = build_plan(manifest)
    result.metrics = {
        "scenario": manifest.get("name", "upgrade_rollback_compatibility"),
        "cycles": int(manifest.get("cycles", 1)),
        "planned_steps_total": len(plan),
        "planned_steps": plan,
    }
    if args.dry_run:
        result.write(args.output_json)
        return 0

    passed, step_results = execute_scenario(args, manifest)
    result.status = PASSED if passed else FAILED
    result.metrics["step_results"] = step_results
    result.metrics["steps_failed"] = len([step for step in step_results if int(step.get("return_code", 0)) != 0])
    if not passed:
        result.mark_failed("SCENARIO_STEP_FAILED", "one or more required scenario steps failed")
    result.write(args.output_json)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
