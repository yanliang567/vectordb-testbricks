from milvus_client.common.feature_validators import (
    _struct_scalar_filter,
    known_validator_names,
    unknown_validators,
    validate_index_engine_version,
)
from milvus_client.common.schema import (
    FieldSpec,
    IndexSpec,
    SchemaSpec,
    StructArraySpec,
)
from milvus_client.common.validators import ValidationReport


def test_declared_validator_registry_fails_closed():
    spec = SchemaSpec(
        name="unknown",
        version="3.0",
        fields=[FieldSpec(name="id", dtype="INT64", primary=True)],
        validators=["count", "not_implemented"],
    )

    assert "struct_array_scalar_index_queries" in known_validator_names()
    assert unknown_validators(spec) == ["not_implemented"]


def test_struct_scalar_filter_covers_float_and_varchar_indexes():
    spec = SchemaSpec(
        name="struct",
        version="3.0",
        fields=[FieldSpec(name="id", dtype="INT64", primary=True)],
        struct_arrays=[
            StructArraySpec(
                name="attributes",
                max_capacity=8,
                fields=[
                    FieldSpec(name="score_sort", dtype="FLOAT"),
                    FieldSpec(name="category_inverted", dtype="VARCHAR"),
                ],
            )
        ],
        indexes=[
            IndexSpec(field="attributes[score_sort]", index_type="STL_SORT"),
            IndexSpec(field="attributes[category_inverted]", index_type="INVERTED"),
        ],
    )

    assert (
        _struct_scalar_filter(
            spec, spec.indexes[0], spec.struct_arrays[0].fields[0], 7, 3
        )
        == "MATCH_ANY(attributes, $[score_sort] >= 7.0)"
    )
    assert (
        _struct_scalar_filter(
            spec, spec.indexes[1], spec.struct_arrays[0].fields[1], 7, 3
        )
        == 'MATCH_ANY(attributes, $[category_inverted] == "category_7")'
    )


def test_index_engine_version_reads_runtime_config_and_fails_closed():
    spec = SchemaSpec(
        name="index_v10_v4",
        version="3.0",
        fields=[FieldSpec(name="id", dtype="INT64", primary=True)],
        validator_params={
            "target_vec_index_version": 10,
            "target_scalar_index_version": 4,
        },
    )
    report = ValidationReport()

    validate_index_engine_version(
        "qa_index",
        spec,
        {
            "dataCoord": {
                "targetVecIndexVersion": 10,
                "targetScalarIndexVersion": 4,
            }
        },
        report,
    )

    assert report.passed
    assert report.metrics["qa_index.index_engine_version.passed"] == 2

    mismatch = ValidationReport()
    validate_index_engine_version("qa_index", spec, {}, mismatch)
    assert not mismatch.passed
    assert len(mismatch.failures) == 2
