from milvus_client.common.capability import CapabilitySpec, evaluate_capabilities


def test_evaluate_capabilities_does_not_assume_unimplemented_probes_are_supported():
    catalog = {
        "StorageV3": CapabilitySpec(
            id="StorageV3",
            detect={"config_probe": "common.storage.useLoonFFI"},
        ),
        "NoProbe": CapabilitySpec(id="NoProbe", detect={}),
        "Versioned": CapabilitySpec(id="Versioned", detect={"server_version_min": "3.0.0"}),
    }

    result = evaluate_capabilities(["StorageV3", "NoProbe", "Versioned"], catalog, "3.0.0")

    assert "StorageV3" in result["unsupported"]
    assert "NoProbe" in result["supported"]
    assert "Versioned" in result["supported"]
