from pathlib import Path

from milvus_client.common.schema import FieldSpec, SchemaSpec, load_schema_matrix
from milvus_client.common.workload import run_operation


ROOT = Path(__file__).resolve().parents[1]


def test_delete_operation_uses_pressure_pk_range_not_seed_baseline():
    spec = load_schema_matrix(ROOT / "manifests" / "schema_matrix_2_6.yaml")[0]

    class DeleteClient:
        def __init__(self):
            self.delete_calls = []

        def delete(self, **kwargs):
            self.delete_calls.append(kwargs)
            return {"delete_count": 0}

    client = DeleteClient()

    op, count = run_operation(client, spec, "qa_dense", "delete", 7, 10, 2)

    assert op == "delete"
    assert count == 10
    assert client.delete_calls[0]["filter"] == "id >= 30000020 && id <= 30000029"


def test_query_iterator_operation_closes_iterator():
    spec = SchemaSpec(
        name="string_pk",
        version="test",
        fields=[
            FieldSpec(name="pk", dtype="VARCHAR", primary=True, max_length=64),
            FieldSpec(name="embedding", dtype="FLOAT_VECTOR", dim=2),
        ],
    )

    class FakeIterator:
        def __init__(self):
            self.closed = False
            self.calls = 0

        def next(self):
            self.calls += 1
            if self.calls == 1:
                return [{"pk": "pk_00000000000000000000"}]
            return []

        def close(self):
            self.closed = True

    class IteratorClient:
        def __init__(self):
            self.iterator = FakeIterator()
            self.query_iterator_calls = []

        def query_iterator(self, **kwargs):
            self.query_iterator_calls.append(kwargs)
            return self.iterator

    client = IteratorClient()

    op, count = run_operation(client, spec, "qa_string", "query_iterator", 7, 10, 1)

    assert op == "query_iterator"
    assert count == 1
    assert client.iterator.closed
    assert client.query_iterator_calls[0]["filter"] == 'pk >= "pk_00000000000000000000"'
