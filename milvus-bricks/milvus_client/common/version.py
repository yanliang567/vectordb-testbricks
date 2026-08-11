from __future__ import annotations

import re


VERSION_PREFIX = re.compile(r"^v?(?P<major>\d+)\.(?P<minor>\d+)(?:[.\-+]|$)")
VERSION_CORE = re.compile(
    r"^v?(?P<major>\d+)\.(?P<minor>\d+)(?:\.(?P<patch>\d+))?(?:[\-+]|$)"
)
SHA256_DIGEST = re.compile(r"@sha256:[0-9a-fA-F]{64}$")
COMMIT_SUFFIX = re.compile(r"(?:^|[-_.])(?P<commit>[0-9a-fA-F]{8,40})$")
HEX_SUFFIX = re.compile(r"[0-9a-fA-F]{1,32}")


def version_family(value: str) -> str:
    match = VERSION_PREFIX.match(str(value).strip())
    if match is None:
        raise ValueError(f"Milvus version must start with numeric major.minor: {value}")
    return f"{match.group('major')}.{match.group('minor')}"


def version_core(value: str) -> tuple[int, int, int]:
    match = VERSION_CORE.match(str(value).strip())
    if match is None:
        raise ValueError(f"Milvus version must start with numeric major.minor: {value}")
    return (
        int(match.group("major")),
        int(match.group("minor")),
        int(match.group("patch") or 0),
    )


def version_at_least(value: str, minimum: str) -> bool:
    return version_core(value) >= version_core(minimum)


def matching_pinned_image_build_tag(image: str, actual_version: str) -> str | None:
    image_value = str(image).strip()
    if SHA256_DIGEST.search(image_value) is None:
        return None
    image_without_digest = image_value.split("@", 1)[0]
    image_name = image_without_digest.rsplit("/", 1)[-1]
    if ":" not in image_name:
        return None
    tag = image_name.rsplit(":", 1)[-1]
    commit_match = COMMIT_SUFFIX.search(tag)
    if (
        commit_match is None
        or re.search(r"[a-fA-F]", commit_match.group("commit")) is None
        or not actual_version.startswith(tag)
    ):
        return None
    suffix = actual_version[len(tag) :]
    if suffix and HEX_SUFFIX.fullmatch(suffix) is None:
        return None
    return tag


def server_version_for_feature_detection(
    actual: str,
    hint: str = "",
    *,
    expected_image: str = "",
    release_gate_eligible: bool = True,
) -> str:
    try:
        version_family(actual)
        return actual
    except ValueError:
        if not hint:
            return actual
        version_family(hint)
        if release_gate_eligible:
            raise ValueError(
                "release-eligible gates require a parseable Milvus API version"
            )
        if matching_pinned_image_build_tag(expected_image, actual) is None:
            raise ValueError(
                "opaque Milvus API version does not match a digest-pinned image build"
            )
        return hint


def image_version_family(image: str) -> str | None:
    image_name = str(image).rsplit("/", 1)[-1]
    image_name = image_name.split("@", 1)[0]
    if ":" not in image_name:
        return None
    tag = image_name.rsplit(":", 1)[-1]
    match = VERSION_PREFIX.match(tag)
    if match is None:
        return None
    return f"{match.group('major')}.{match.group('minor')}"


def image_is_immutable(image: str) -> bool:
    image_value = str(image).strip()
    if SHA256_DIGEST.search(image_value):
        return True
    image_name = image_value.rsplit("/", 1)[-1]
    if ":" not in image_name:
        return False
    tag = image_name.rsplit(":", 1)[-1]
    mutable_tokens = {"latest", "head", "edge"}
    tokens = {token for token in re.split(r"[-_.]", tag.lower()) if token}
    return tag.lower() not in {"master", "main", "nightly", "dev"} and not (
        tokens & mutable_tokens
    )
