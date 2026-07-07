from argparse import Namespace

from milvus_client.scenarios import upgrade_rollback_compatibility
from milvus_client.scenarios.upgrade_rollback_compatibility import build_plan, execute_scenario


def test_upgrade_rollback_plan_expands_cycles():
    plan = build_plan({"cycles": 2})
    names = [step["name"] for step in plan]

    assert names.count("wait_upgrade") == 2
    assert names.count("wait_rollback") == 2
    assert names[-1] == "final_validate_compat"


def test_execute_scenario_runs_forward_schema_only_after_upgrade(monkeypatch, tmp_path):
    calls = []

    class FakeThread:
        def join(self, timeout=None):
            del timeout

    def fake_run_brick(args, module, name, extra_args, lifecycle_phase, checkpoint_dir=None):
        del args
        calls.append(
            {
                "module": module,
                "name": name,
                "extra_args": extra_args,
                "phase": lifecycle_phase,
                "checkpoint_dir": checkpoint_dir,
            }
        )
        return {"name": name, "module": module, "return_code": 0, "phase": lifecycle_phase}

    monkeypatch.setattr(upgrade_rollback_compatibility, "run_brick", fake_run_brick)
    monkeypatch.setattr(upgrade_rollback_compatibility, "_start_pressure_loop", lambda *args, **kwargs: FakeThread())
    monkeypatch.setattr(upgrade_rollback_compatibility, "_start_validator_loop", lambda *args, **kwargs: FakeThread())

    args = Namespace(
        uri="http://localhost:19530",
        token="",
        db_name="default",
        collection_prefix="qa",
        feature_set="compat_2_6",
        compat_mode="rollback_safe",
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )
    manifest = {
        "cycles": 1,
        "compat_schema_matrix": "compat.yaml",
        "forward_schema_matrix": "forward.yaml",
        "rows_per_collection": 10,
        "batch_size": 5,
        "observe_after_upgrade_sec": 0,
        "observe_after_rollback_sec": 0,
        "actions": {"upgrade": {"wait_file": ""}, "rollback": {"wait_file": ""}},
        "workloads": {"mixed_rw": {"slice_duration_sec": 1}, "validator": {"interval_sec": 1}},
    }

    passed, step_results = execute_scenario(args, manifest)

    assert passed
    assert all(step["return_code"] == 0 for step in step_results)
    names = [call["name"] for call in calls]
    assert "create_forward_schema_1" in names
    assert "seed_forward_data_1" in names
    assert "validate_forward_after_upgrade_1" in names
    forward_calls = [call for call in calls if "forward" in call["name"]]
    assert forward_calls
    assert all(call["phase"] == "after_upgrade" for call in forward_calls)
    assert all(call["checkpoint_dir"] and call["checkpoint_dir"].endswith("forward_cycle_1") for call in forward_calls)
