from pathlib import Path
from struct import pack, unpack

from milvus_client.common.data import (
    canonical_float32,
    checksum_fields_for_spec,
    generate_rows,
    stable_checksum,
    text_payload_metadata,
)
from milvus_client.common.schema import (
    FieldSpec,
    SchemaSpec,
    StructArraySpec,
    load_schema_matrix,
)

ROOT = Path(__file__).resolve().parents[1]


def test_generate_rows_is_deterministic():
    spec = load_schema_matrix(ROOT / "manifests" / "schema_matrix_2_6.yaml")[0]
    rows1 = generate_rows(spec, start_id=0, count=10, seed=7)
    rows2 = generate_rows(spec, start_id=0, count=10, seed=7)

    assert rows1 == rows2
    assert rows1[0]["id"] == 0
    assert "embedding" in rows1[0]
    assert rows1[0]["dyn_bucket"] == 0


def test_stable_checksum_uses_selected_fields_and_is_order_independent():
    rows = [
        {"id": 2, "category": 20, "embedding": [0.2]},
        {"id": 1, "category": 10, "embedding": [0.1]},
    ]
    reordered_rows = list(reversed(rows))

    checksum = stable_checksum(rows, fields=["id", "category"], primary_field="id")

    assert checksum == stable_checksum(
        reordered_rows, fields=["id", "category"], primary_field="id"
    )
    assert checksum != stable_checksum(
        rows, fields=["id", "category", "embedding"], primary_field="id"
    )


def test_stable_checksum_sorts_by_primary_even_when_primary_is_not_digested():
    rows = [
        {"id": 2, "category": 20},
        {"id": 1, "category": 10},
    ]
    queried_rows = list(reversed(rows))

    assert stable_checksum(
        rows, fields=["category"], primary_field="id"
    ) == stable_checksum(
        queried_rows,
        fields=["category"],
        primary_field="id",
    )


def test_stable_checksum_normalizes_repeated_scalar_containers():
    class RepeatedScalarLike:
        def __iter__(self):
            return iter(["tag_0", "tag_1"])

    rows = [{"id": 1, "tags": RepeatedScalarLike()}]
    list_rows = [{"id": 1, "tags": ["tag_0", "tag_1"]}]

    assert stable_checksum(
        rows, fields=["id", "tags"], primary_field="id"
    ) == stable_checksum(
        list_rows,
        fields=["id", "tags"],
        primary_field="id",
    )


def test_stable_checksum_normalizes_float32_round_trip_precision():
    inserted_rows = [{"id": 1, "score": 16.2}]
    queried_rows = [{"id": 1, "score": 16.200000762939453}]

    assert stable_checksum(
        inserted_rows, fields=["id", "score"], primary_field="id"
    ) == stable_checksum(
        queried_rows,
        fields=["id", "score"],
        primary_field="id",
    )


def test_stable_checksum_normalizes_binary_vector_round_trip_representation():
    inserted_rows = [{"id": 1, "binary_flat": b"\x00\x01\x02\x03\x04\x05\x06\x07"}]
    queried_rows = [{"id": 1, "binary_flat": [b"\x00\x01\x02\x03\x04\x05\x06\x07"]}]

    assert stable_checksum(
        inserted_rows, fields=["id", "binary_flat"], primary_field="id"
    ) == stable_checksum(
        queried_rows,
        fields=["id", "binary_flat"],
        primary_field="id",
    )


def test_struct_array_vector_checksum_survives_float32_storage_round_trip():
    spec = next(
        spec
        for spec in load_schema_matrix(ROOT / "manifests" / "schema_matrix_2_6.yaml")
        if spec.name == "struct_array_element_rollback_safe"
    )
    inserted_rows = generate_rows(spec, start_id=0, count=100, seed=0)
    queried_rows = []
    for row in inserted_rows:
        queried_rows.append(
            {
                **row,
                "items": [
                    {
                        **item,
                        "embedding": [
                            unpack("!f", pack("!f", value))[0]
                            for value in item["embedding"]
                        ],
                    }
                    for item in row["items"]
                ],
            }
        )
    fields = checksum_fields_for_spec(spec)

    assert stable_checksum(
        inserted_rows, fields=fields, primary_field="id"
    ) == stable_checksum(queried_rows, fields=fields, primary_field="id")


def test_struct_array_scalar_float_uses_storage_precision_before_checksum():
    spec = next(
        spec
        for spec in load_schema_matrix(ROOT / "manifests" / "schema_matrix_2_6.yaml")
        if spec.name == "struct_array_numeric_autoindex_rollback_safe"
    )

    row = generate_rows(spec, start_id=129, count=1, seed=0)[0]
    score = row["items"][1]["score"]

    assert score == canonical_float32(129.1)
    assert score != 129.1
    assert stable_checksum(
        [row], fields=["id", "items"], primary_field="id"
    ) == stable_checksum(
        [
            {
                **row,
                "items": [
                    {
                        **item,
                        "score": canonical_float32(item["score"]),
                    }
                    for item in row["items"]
                ],
            }
        ],
        fields=["id", "items"],
        primary_field="id",
    )


def test_checksum_fields_exclude_vectors():
    spec = load_schema_matrix(ROOT / "manifests" / "schema_matrix_2_6.yaml")[0]

    assert "id" in checksum_fields_for_spec(spec)
    assert "embedding" not in checksum_fields_for_spec(spec)
    assert "dyn_bucket" not in checksum_fields_for_spec(spec)


def test_generate_rows_uses_sdk_compatible_vector_values():
    spec = next(
        spec
        for spec in load_schema_matrix(ROOT / "manifests" / "schema_matrix_2_6.yaml")
        if spec.name == "vector_autoid_bm25"
    )
    row = generate_rows(spec, start_id=1, count=1, seed=7)[0]

    assert "id" not in row
    assert row["float16_hnsw"].dtype == "float16"
    assert isinstance(row["bfloat16_diskann"], bytes)
    assert len(row["bfloat16_diskann"]) == 256
    assert row["int8_autoindex"].dtype == "int8"
    assert isinstance(row["binary_ivf"], bytes)
    assert "sparse_bm25" not in row


def test_generate_rows_uses_timestamptz_string():
    spec = load_schema_matrix(ROOT / "manifests" / "schema_matrix_3_0.yaml")[2]
    row = generate_rows(spec, start_id=1, count=1, seed=7)[0]

    assert row["event_time"] == "2100-01-01T00:00:01Z"


def test_generate_rows_uses_canonical_geometry_wkt():
    spec = load_schema_matrix(ROOT / "manifests" / "schema_matrix_3_0.yaml")[1]
    row = generate_rows(spec, start_id=0, count=1, seed=7)[0]

    assert row["location"] == "POINT (-122 37)"


def test_generate_rows_builds_nested_json_shredding_payload():
    spec = load_schema_matrix(ROOT / "manifests" / "schema_matrix_json_shredding.yaml")[
        0
    ]

    row = generate_rows(spec, start_id=1, count=1, seed=7)[0]

    assert row["json_nested"] == {
        "pk": 1,
        "bucket": 1,
        "nested": {"score": 0.1, "active": False},
        "labels": ["label_1", "label_2"],
    }
    assert row["dyn_json"] == {"pk_mod": 1, "active": False}


def test_json_shredding_checksum_fields_include_selected_dynamic_fields():
    spec = load_schema_matrix(ROOT / "manifests" / "schema_matrix_json_shredding.yaml")[
        0
    ]

    assert checksum_fields_for_spec(spec) == [
        "id",
        "tenant",
        "category",
        "json_profile",
        "json_nested",
        "tags",
        "dyn_bucket",
        "dyn_text",
        "dyn_json",
    ]


def test_generate_rows_supports_string_primary_key():
    spec = SchemaSpec(
        name="string_pk",
        version="test",
        fields=[
            FieldSpec(name="pk", dtype="VARCHAR", primary=True, max_length=64),
            FieldSpec(name="embedding", dtype="FLOAT_VECTOR", dim=2),
        ],
    )

    row = generate_rows(spec, start_id=7, count=1, seed=1)[0]

    assert row["pk"] == "pk_00000000000000000007"


def test_generate_rows_supports_numeric_arrays():
    spec = SchemaSpec(
        name="numeric_array",
        version="test",
        fields=[
            FieldSpec(name="id", dtype="INT64", primary=True),
            FieldSpec(name="ints", dtype="ARRAY", element_type="INT64"),
            FieldSpec(name="floats", dtype="ARRAY", element_type="FLOAT"),
            FieldSpec(name="bools", dtype="ARRAY", element_type="BOOL"),
        ],
    )

    row = generate_rows(spec, start_id=3, count=1, seed=1)[0]

    assert row["ints"] == [3, 4]
    assert row["floats"] == [3.0, 4.0]
    assert row["bools"] == [False, True]


def test_generate_rows_caps_int64_partition_key_by_field_attribute():
    spec = SchemaSpec(
        name="int64_partition_key",
        version="test",
        fields=[
            FieldSpec(name="id", dtype="INT64", primary=True),
            FieldSpec(name="tenant_id", dtype="INT64", is_partition_key=True),
        ],
    )

    row = generate_rows(spec, start_id=2049, count=1, seed=1)[0]

    assert row["tenant_id"] == 1


def test_generate_rows_builds_deterministic_struct_array_values():
    spec = SchemaSpec(
        name="struct_array",
        version="3.0",
        fields=[FieldSpec(name="id", dtype="INT64", primary=True)],
        struct_arrays=[
            StructArraySpec(
                name="attributes",
                max_capacity=8,
                fields=[
                    FieldSpec(name="embedding", dtype="FLOAT_VECTOR", dim=4),
                    FieldSpec(name="score_sort", dtype="FLOAT"),
                    FieldSpec(name="category_inverted", dtype="VARCHAR"),
                    FieldSpec(name="tag_bitmap", dtype="VARCHAR"),
                    FieldSpec(name="rank_sort", dtype="INT64"),
                    FieldSpec(name="enabled_bitmap", dtype="BOOL"),
                ],
            )
        ],
    )

    first = generate_rows(spec, start_id=3, count=1, seed=7)[0]
    second = generate_rows(spec, start_id=3, count=1, seed=7)[0]

    assert first == second
    assert len(first["attributes"]) == 4
    assert first["attributes"][0]["score_sort"] == 3.0
    assert first["attributes"][0]["category_inverted"] == "category_3"
    assert first["attributes"][0]["tag_bitmap"] == "tag_3"
    assert first["attributes"][1]["rank_sort"] == 31
    assert first["attributes"][1]["enabled_bitmap"] is True
    assert first["attributes"][0]["embedding"] != first["attributes"][1]["embedding"]


def test_generate_rows_supports_nullable_struct_array():
    spec = SchemaSpec(
        name="nullable_struct_array",
        version="3.0",
        fields=[FieldSpec(name="id", dtype="INT64", primary=True)],
        struct_arrays=[
            StructArraySpec(
                name="attributes",
                max_capacity=4,
                nullable=True,
                fields=[FieldSpec(name="score", dtype="FLOAT")],
            )
        ],
    )

    rows = generate_rows(spec, start_id=9, count=2, seed=1)

    assert rows[0]["attributes"] is not None
    assert rows[1]["attributes"] is None


def test_text_lob_boundary_profile_covers_storage_v3_boundaries():
    spec = SchemaSpec(
        name="text_lob",
        version="3.0",
        fields=[
            FieldSpec(name="id", dtype="INT64", primary=True),
            FieldSpec(
                name="text",
                dtype="TEXT",
                nullable=True,
                value_profile="text_lob_boundary",
            ),
        ],
    )

    rows = generate_rows(spec, start_id=0, count=7, seed=1)

    assert rows[0]["text"] is None
    assert text_payload_metadata(rows[1]["text"])["state"] == "empty"
    assert "\u4e2d\u6587" in rows[2]["text"]
    assert text_payload_metadata(rows[3]["text"])["bytes"] == 64 * 1024 - 1
    assert text_payload_metadata(rows[4]["text"])["bytes"] == 64 * 1024
    assert text_payload_metadata(rows[5]["text"])["bytes"] == 64 * 1024 + 1
    assert text_payload_metadata(rows[6]["text"])["bytes"] == 1024 * 1024


def test_struct_array_payload_is_included_in_default_checksum_fields():
    spec = SchemaSpec(
        name="struct",
        version="3.0",
        fields=[FieldSpec(name="id", dtype="INT64", primary=True)],
        struct_arrays=[
            StructArraySpec(
                name="items",
                max_capacity=4,
                fields=[FieldSpec(name="score", dtype="FLOAT")],
            )
        ],
    )

    assert checksum_fields_for_spec(spec) == ["id", "items"]
