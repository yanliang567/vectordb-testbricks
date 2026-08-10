from milvus_client.common.capability import CapabilitySpec, evaluate_capabilities


def test_evaluate_capabilities_uses_version_gate_for_config_backed_capability():
    catalog = {
        "StorageV3": CapabilitySpec(
            id="StorageV3",
            detect={
                "server_version_min": "3.0.0",
                "config_probe": "common.storage.useLoonFFI",
            },
        ),
        "NoProbe": CapabilitySpec(id="NoProbe", detect={}),
        "Versioned": CapabilitySpec(
            id="Versioned", detect={"server_version_min": "3.0.0"}
        ),
    }

    result = evaluate_capabilities(
        ["StorageV3", "NoProbe", "Versioned"], catalog, "3.0.0"
    )

    assert "StorageV3" in result["supported"]
    assert "NoProbe" in result["supported"]
    assert "Versioned" in result["supported"]

    old_server = evaluate_capabilities(["StorageV3"], catalog, "2.6.18")
    assert old_server["unsupported"] == ["StorageV3"]
