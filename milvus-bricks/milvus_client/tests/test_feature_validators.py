from milvus_client.common.feature_validators import (
    _normalize,
    _struct_scalar_filter,
    known_validator_names,
    run_feature_validator,
    unknown_validators,
    validate_index_engine_version,
    validate_entity_ttl,
    validate_minhash_search,
    validate_nullable_vector_semantics,
    validate_struct_array_element_search,
    validate_text_match_phrase_match,
)
from milvus_client.common.pk_namespaces import (
    ENTITY_TTL_BASE,
    PRESSURE_DELETE_BASE,
    PRESSURE_INSERT_BASE,
    PRESSURE_UPSERT_BASE,
)
from milvus_client.common.data import generate_field_value
from milvus_client.common.schema import (
    FieldSpec,
    FunctionSpec,
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


def test_float_normalization_accepts_float32_round_trip_noise():
    assert _normalize(499.10001) == _normalize(499.1)
    assert _normalize(999.09998) == _normalize(999.1)
    assert _normalize(499.11) != _normalize(499.1)


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


def test_struct_scalar_index_query_is_version_gated():
    class Client:
        calls = []

        def query(self, **kwargs):
            self.calls.append(kwargs)
            return [{"id": 0}]

    spec = SchemaSpec(
        name="struct",
        version="2.6",
        fields=[FieldSpec(name="id", dtype="INT64", primary=True)],
        struct_arrays=[
            StructArraySpec(
                name="items",
                max_capacity=4,
                fields=[FieldSpec(name="category", dtype="VARCHAR")],
            )
        ],
        indexes=[IndexSpec(field="items[category]", index_type="AUTOINDEX")],
        validator_params={"min_struct_scalar_index_queries": 1},
    )
    meta = {"min_pk": 0, "max_pk": 0, "data_min_pk": 0, "data_max_pk": 0}
    client = Client()
    base_report = ValidationReport()

    run_feature_validator(
        "struct_array_scalar_index_queries",
        client,
        "qa_struct",
        spec,
        meta,
        0,
        base_report,
        server_version="v2.6.18",
    )

    assert base_report.passed
    assert client.calls == []
    assert (
        base_report.metrics[
            "qa_struct.struct_array_scalar_index_queries.skipped_unsupported_total"
        ]
        == 1
    )

    target_report = ValidationReport()
    run_feature_validator(
        "struct_array_scalar_index_queries",
        client,
        "qa_struct",
        spec,
        meta,
        0,
        target_report,
        server_version="v3.0.0",
    )
    assert target_report.passed
    assert (
        target_report.metrics["qa_struct.struct_array_scalar_index_queries.total"] == 1
    )
    assert len(client.calls) == 1


def test_struct_scalar_index_query_fails_closed_for_unknown_runtime_version():
    spec = SchemaSpec(
        name="struct",
        version="2.6",
        fields=[FieldSpec(name="id", dtype="INT64", primary=True)],
        struct_arrays=[
            StructArraySpec(
                name="items",
                max_capacity=4,
                fields=[FieldSpec(name="category", dtype="VARCHAR")],
            )
        ],
        indexes=[IndexSpec(field="items[category]", index_type="AUTOINDEX")],
    )
    report = ValidationReport()

    run_feature_validator(
        "struct_array_scalar_index_queries",
        object(),
        "qa_struct",
        spec,
        {"min_pk": 0, "max_pk": 0},
        0,
        report,
        server_version="unknown",
    )

    assert not report.passed
    assert (
        report.failures[0]["type"]
        == "STRUCT_ARRAY_SCALAR_INDEX_RUNTIME_VERSION_UNKNOWN"
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


def test_struct_max_sim_search_uses_embedding_list_and_does_not_require_offset():
    class Client:
        search_kwargs = None

        def search(self, **kwargs):
            self.search_kwargs = kwargs
            return [[{"id": 3, "distance": 1.0}]]

    spec = SchemaSpec(
        name="struct_max_sim",
        version="3.0",
        fields=[FieldSpec(name="id", dtype="INT64", primary=True)],
        struct_arrays=[
            StructArraySpec(
                name="attributes",
                max_capacity=4,
                fields=[FieldSpec(name="embedding", dtype="FLOAT_VECTOR", dim=4)],
            )
        ],
        indexes=[
            IndexSpec(
                field="attributes[embedding]",
                index_type="HNSW",
                metric_type="MAX_SIM_COSINE",
            )
        ],
    )
    meta = {"min_pk": 3, "max_pk": 3, "data_min_pk": 3, "data_max_pk": 3}
    client = Client()
    report = ValidationReport()

    validate_struct_array_element_search(client, "qa_struct", spec, meta, 7, report)

    query = client.search_kwargs["data"][0]
    assert type(query).__name__ == "EmbeddingList"
    assert len(query) == 1
    assert report.passed


def test_feature_validator_fails_when_declared_target_is_absent():
    spec = SchemaSpec(
        name="no_nullable_vectors",
        version="3.0",
        fields=[FieldSpec(name="id", dtype="INT64", primary=True)],
    )
    report = ValidationReport()

    validate_nullable_vector_semantics(
        object(),
        "qa_no_nullable_vectors",
        spec,
        {"min_pk": 0, "max_pk": 1},
        7,
        report,
    )

    assert not report.passed
    assert report.failures[0]["type"] == "FEATURE_VALIDATION_TARGET_MISSING"


def test_nullable_vector_search_uses_primary_key_filter_without_is_not_null():
    class Client:
        search_kwargs = None

        def query(self, **kwargs):
            if kwargs["filter"] == "id == 0":
                return [{"id": 0, "embedding": None}]
            return [{"id": 1, "embedding": [1.0, 0.0, 0.0, 0.0]}]

        def search(self, **kwargs):
            self.search_kwargs = kwargs
            return [[{"id": 1, "distance": 1.0}]]

    spec = SchemaSpec(
        name="nullable_vector",
        version="2.6",
        fields=[
            FieldSpec(name="id", dtype="INT64", primary=True),
            FieldSpec(name="embedding", dtype="FLOAT_VECTOR", dim=4, nullable=True),
        ],
        indexes=[IndexSpec(field="embedding", index_type="HNSW", metric_type="COSINE")],
    )
    client = Client()
    report = ValidationReport()

    validate_nullable_vector_semantics(
        client,
        "qa_nullable_vector",
        spec,
        {"min_pk": 0, "max_pk": 1},
        7,
        report,
    )

    assert report.passed
    assert client.search_kwargs["filter"] == "id == 1"


def _minhash_spec() -> SchemaSpec:
    return SchemaSpec(
        name="minhash",
        version="3.0",
        fields=[
            FieldSpec(name="id", dtype="INT64", primary=True),
            FieldSpec(
                name="document",
                dtype="VARCHAR",
                max_length=65535,
                value_profile="minhash_documents",
            ),
            FieldSpec(name="minhash", dtype="BINARY_VECTOR", dim=4096),
        ],
        functions=[
            FunctionSpec(
                name="text_to_minhash",
                function_type="MINHASH",
                input_fields=["document"],
                output_fields=["minhash"],
            )
        ],
        indexes=[
            IndexSpec(
                field="minhash",
                index_type="MINHASH_LSH",
                metric_type="MHJACCARD",
                search_params={"mh_search_with_jaccard": True},
            )
        ],
    )


def test_minhash_validator_allows_approximate_near_duplicate_omission():
    class Client:
        def search(self, **kwargs):
            return [[{"id": 0, "distance": 1.0}]]

    report = ValidationReport()

    validate_minhash_search(
        Client(),
        "qa_minhash",
        _minhash_spec(),
        {"min_pk": 0, "max_pk": 2},
        7,
        report,
    )

    assert report.passed
    assert (
        report.metrics["qa_minhash.minhash_search.coverage_mode"]
        == "exact_self_search_with_observational_near_duplicate"
    )
    assert report.metrics["qa_minhash.minhash_search.exact_self_search_enforced"]
    assert not report.metrics["qa_minhash.minhash_search.near_duplicate_gate_enforced"]
    assert (
        report.metrics["qa_minhash.minhash_search.ranking_gate_mode"]
        == "conditional_when_both_observational_hits_returned"
    )
    assert report.metrics["qa_minhash.minhash_search.near_duplicate_returned"] == 0
    assert report.metrics["qa_minhash.minhash_search.unrelated_returned"] == 0


def test_minhash_validator_still_requires_exact_document():
    class Client:
        def search(self, **kwargs):
            return [[{"id": 1, "distance": 0.9}]]

    report = ValidationReport()

    validate_minhash_search(
        Client(),
        "qa_minhash",
        _minhash_spec(),
        {"min_pk": 0, "max_pk": 2},
        7,
        report,
    )

    assert not report.passed
    assert report.failures[0]["type"] == "MINHASH_SEARCH_FAILED"
    assert report.failures[0]["expected_exact"] == 0


def test_minhash_validator_requires_exact_document_to_rank_first():
    class Client:
        def search(self, **kwargs):
            return [
                [
                    {"id": 1, "distance": 0.8},
                    {"id": 2, "distance": 0.2},
                    {"id": 0, "distance": 1.0},
                ]
            ]

    report = ValidationReport()

    validate_minhash_search(
        Client(),
        "qa_minhash",
        _minhash_spec(),
        {"min_pk": 0, "max_pk": 2},
        7,
        report,
    )

    assert not report.passed
    assert report.failures[0]["expected_exact"] == 0
    assert report.failures[0]["actual_rank"] == 2


def test_text_feature_validator_rejects_unrelated_rows():
    class Client:
        def query(self, **kwargs):
            return [{"id": 3, "text": "a" * 16}]

        def search(self, **kwargs):
            return [[{"id": 3, "distance": 1.0, "entity": {"text": "a" * 16}}]]

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
            FieldSpec(name="sparse_bm25", dtype="SPARSE_FLOAT_VECTOR"),
        ],
        functions=[
            FunctionSpec(
                name="text_bm25",
                function_type="BM25",
                input_fields=["text"],
                output_fields=["sparse_bm25"],
            )
        ],
        indexes=[
            IndexSpec(
                field="sparse_bm25",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25",
            )
        ],
    )
    report = ValidationReport()

    validate_text_match_phrase_match(
        Client(),
        "qa_text",
        spec,
        {"min_pk": 0, "max_pk": 9},
        7,
        report,
    )

    assert not report.passed
    assert any(failure["type"] == "TEXT_FILTER_FAILED" for failure in report.failures)


def _text_lob_spec() -> SchemaSpec:
    return SchemaSpec(
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
            FieldSpec(name="sparse_bm25", dtype="SPARSE_FLOAT_VECTOR"),
        ],
        functions=[
            FunctionSpec(
                name="text_bm25",
                function_type="BM25",
                input_fields=["text"],
                output_fields=["sparse_bm25"],
            )
        ],
        indexes=[
            IndexSpec(
                field="sparse_bm25",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25",
            )
        ],
    )


def test_text_feature_validator_rejects_incomplete_postings_count():
    class Client:
        def query(self, **kwargs):
            if kwargs["output_fields"] == ["count(*)"]:
                return [{"count(*)": 1}]
            return [{"id": 0, "text": generate_field_value(spec.fields[1], 0, 7)}]

        def search(self, **kwargs):
            return [[{"id": 0, "distance": 1.0}]]

    spec = _text_lob_spec()
    report = ValidationReport()

    validate_text_match_phrase_match(
        Client(),
        "qa_text",
        spec,
        {"min_pk": 0, "max_pk": 9},
        7,
        report,
    )

    assert not report.passed
    assert any(
        failure["type"] == "TEXT_FILTER_FAILED"
        and failure.get("actual_count") == 1
        and failure.get("expected_count") == 4
        for failure in report.failures
    )


def test_text_feature_validator_rejects_incomplete_sample_query():
    class Client:
        def query(self, **kwargs):
            filter_expr = kwargs["filter"]
            if kwargs["output_fields"] == ["count(*)"]:
                if "definitely_absent" in filter_expr:
                    return [{"count(*)": 0}]
                return [{"count(*)": 3 if "PHRASE_MATCH" in filter_expr else 4}]
            return [{"id": 7, "text": generate_field_value(spec.fields[1], 7, 7)}]

        def search(self, **kwargs):
            return [[{"id": 7, "distance": 1.0}]]

    spec = _text_lob_spec()
    report = ValidationReport()

    validate_text_match_phrase_match(
        Client(),
        "qa_text",
        spec,
        {"min_pk": 0, "max_pk": 9},
        7,
        report,
    )

    assert not report.passed
    sample_failures = [
        failure
        for failure in report.failures
        if failure["type"] == "TEXT_FILTER_FAILED" and "actual_sample_count" in failure
    ]
    assert {failure["expected_sample_count"] for failure in sample_failures} == {3, 4}
    assert all(failure["actual_sample_count"] == 1 for failure in sample_failures)


def test_text_feature_validator_accepts_complete_count_and_valid_samples():
    class Client:
        def query(self, **kwargs):
            filter_expr = kwargs["filter"]
            if kwargs["output_fields"] == ["count(*)"]:
                if "zzzzqvnotpresenttokenzzzz" in filter_expr:
                    return [{"count(*)": 0}]
                if "definitely_absent_upgrade_gate_token" in filter_expr:
                    return [{"count(*)": 3}]
                return [{"count(*)": 3 if "PHRASE_MATCH" in filter_expr else 4}]
            sample_pks = (7, 8, 9) if "PHRASE_MATCH" in filter_expr else (2, 7, 8, 9)
            return [
                {"id": pk, "text": generate_field_value(spec.fields[1], pk, 7)}
                for pk in sample_pks
            ]

        def search(self, **kwargs):
            return [[{"id": 7, "distance": 1.0}]]

    spec = _text_lob_spec()
    report = ValidationReport()

    validate_text_match_phrase_match(
        Client(),
        "qa_text",
        spec,
        {"min_pk": 0, "max_pk": 9},
        7,
        report,
    )

    assert report.passed


def test_text_feature_validator_scopes_queries_to_checkpoint_pk_range():
    class Client:
        filters = []

        def query(self, **kwargs):
            filter_expr = kwargs["filter"]
            self.filters.append(filter_expr)
            if kwargs["output_fields"] == ["count(*)"]:
                if "zzzzqvnotpresenttokenzzzz" in filter_expr:
                    return [{"count(*)": 0}]
                return [{"count(*)": 3 if "PHRASE_MATCH" in filter_expr else 4}]
            sample_pks = (7, 8, 9) if "PHRASE_MATCH" in filter_expr else (2, 7, 8, 9)
            return [
                {"id": pk, "text": generate_field_value(spec.fields[1], pk, 7)}
                for pk in sample_pks
            ]

        def search(self, **kwargs):
            return [[{"id": 7, "distance": 1.0}]]

    spec = _text_lob_spec()
    client = Client()
    report = ValidationReport()

    validate_text_match_phrase_match(
        client,
        "qa_text",
        spec,
        {"min_pk": 0, "max_pk": 9},
        7,
        report,
    )

    text_filters = [
        filter_expr
        for filter_expr in client.filters
        if "TEXT_MATCH" in filter_expr or "PHRASE_MATCH" in filter_expr
    ]
    assert report.passed
    assert text_filters
    assert all("id >= 0 && id <= 9" in filter_expr for filter_expr in text_filters)


def test_entity_ttl_uses_reserved_pk_namespace_outside_pressure_ranges():
    class Client:
        inserted = []

        def insert(self, **kwargs):
            self.inserted = kwargs["data"]

        def flush(self, **kwargs):
            return None

        def query(self, **kwargs):
            return [
                {
                    "id": self.inserted[1]["id"],
                    "event_time": self.inserted[1]["event_time"],
                },
                {
                    "id": self.inserted[2]["id"],
                    "event_time": self.inserted[2]["event_time"],
                },
            ]

        def delete(self, **kwargs):
            return None

    spec = SchemaSpec(
        name="ttl",
        version="3.0",
        fields=[
            FieldSpec(name="id", dtype="INT64", primary=True),
            FieldSpec(name="event_time", dtype="TIMESTAMPTZ", nullable=True),
            FieldSpec(name="embedding", dtype="FLOAT_VECTOR", dim=4),
        ],
        properties={"ttl_field": "event_time"},
    )
    client = Client()
    report = ValidationReport()

    validate_entity_ttl(
        client,
        "qa_ttl",
        spec,
        {"min_pk": 0, "max_pk": 5000},
        7,
        report,
    )

    assert report.passed
    assert [row["id"] for row in client.inserted] == [
        ENTITY_TTL_BASE + 1,
        ENTITY_TTL_BASE + 2,
        ENTITY_TTL_BASE + 3,
    ]
    pressure_bases = {
        PRESSURE_INSERT_BASE,
        PRESSURE_UPSERT_BASE,
        PRESSURE_DELETE_BASE,
    }
    assert not pressure_bases.intersection(row["id"] for row in client.inserted)
    assert ENTITY_TTL_BASE > PRESSURE_DELETE_BASE + 10_000_000
