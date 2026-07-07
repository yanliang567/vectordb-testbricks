from milvus_client.scenarios.upgrade_rollback_compatibility import build_plan


def test_upgrade_rollback_plan_expands_cycles():
    plan = build_plan({"cycles": 2})
    names = [step["name"] for step in plan]

    assert names.count("wait_upgrade") == 2
    assert names.count("wait_rollback") == 2
    assert names[-1] == "final_validate_compat"

