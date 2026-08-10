from __future__ import annotations

import re


VERSION_PREFIX = re.compile(r"^v?(?P<major>\d+)\.(?P<minor>\d+)(?:[.\-+]|$)")
VERSION_CORE = re.compile(
    r"^v?(?P<major>\d+)\.(?P<minor>\d+)(?:\.(?P<patch>\d+))?(?:[\-+]|$)"
)
SHA256_DIGEST = re.compile(r"@sha256:[0-9a-fA-F]{64}$")


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
