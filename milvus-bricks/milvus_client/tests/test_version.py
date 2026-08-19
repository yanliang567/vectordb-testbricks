import pytest

from milvus_client.common.version import (
    image_is_immutable,
    is_daily_build_image,
    server_version_for_feature_detection,
    version_at_least,
    version_core,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("3.0.1", (3, 0, 1)),
        ("v3.0.1-dev", (3, 0, 1)),
        ("3.0", (3, 0, 0)),
    ],
)
def test_version_core_parses_numeric_release_components(value, expected):
    assert version_core(value) == expected


def test_version_core_rejects_unparseable_values():
    with pytest.raises(ValueError, match="numeric major.minor"):
        version_core("master")


@pytest.mark.parametrize(
    ("value", "minimum", "expected"),
    [
        ("3.0.0", "3.0.1", False),
        ("3.0.1", "3.0.1", True),
        ("3.0.2-dev", "3.0.1", True),
        ("2.6.18", "3.0.1", False),
    ],
)
def test_version_at_least_compares_core_versions(value, minimum, expected):
    assert version_at_least(value, minimum) is expected


def test_feature_detection_uses_hint_only_for_opaque_server_version():
    assert server_version_for_feature_detection("v3.0.1", "3.0.0") == "v3.0.1"
    assert (
        server_version_for_feature_detection(
            "master-20260810-eaec01bc71",
            "3.0.0",
            expected_image=(
                "harbor.milvus.io/milvusdb/milvus:master-20260810-eaec01bc"
                "@sha256:" + "9" * 64
            ),
            release_gate_eligible=False,
        )
        == "3.0.0"
    )
    assert (
        server_version_for_feature_detection("master-20260810-eaec01bc71")
        == "master-20260810-eaec01bc71"
    )


def test_feature_detection_rejects_unverified_opaque_hint():
    with pytest.raises(ValueError, match="release-eligible"):
        server_version_for_feature_detection("master-20260810-eaec01bc71", "3.0.0")


@pytest.mark.parametrize(
    "image",
    [
        "harbor.milvus.io/milvusdb/milvus:v3.0.1-placeholder",
        "harbor.milvus.io/milvusdb/milvus:3.0-latest-placeholder",
        "harbor.milvus.io/milvusdb/milvus:2.6-latest-placeholder",
    ],
)
def test_image_is_immutable_treats_placeholder_tags_as_mutable(image):
    assert image_is_immutable(image) is False


def test_image_is_immutable_treats_digest_pinned_images_as_immutable():
    assert (
        image_is_immutable("harbor.milvus.io/milvusdb/milvus:v3.0.1@sha256:" + "a" * 64)
        is True
    )


@pytest.mark.parametrize(
    "image",
    [
        "harbor.milvus.io/milvusdb/milvus:3.0-20260817-beb93ec2",
        "harbor.milvus.io/milvusdb/milvus:3.0-20260805-ad3ba1ea-amd64",
        "harbor.milvus.io/milvusdb/milvus:3.0-20260817-beb93ec2@sha256:" + "b" * 64,
    ],
)
def test_is_daily_build_image_matches_daily_build_tags(image):
    assert is_daily_build_image(image) is True


@pytest.mark.parametrize(
    "image",
    [
        "harbor.milvus.io/milvusdb/milvus:v3.0.0",
        "harbor.milvus.io/milvusdb/milvus:3.0-latest-placeholder",
        "harbor.milvus.io/milvusdb/milvus:master-latest",
    ],
)
def test_is_daily_build_image_rejects_release_and_mutable_tags(image):
    assert is_daily_build_image(image) is False
