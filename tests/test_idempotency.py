"""Idempotency: same Idempotency-Key returns same run_id."""

import os
import tempfile
import pytest

# Set DB to a temp path so tests don't touch real DB
@pytest.fixture(autouse=True)
def temp_db(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setenv("GENIE_DB_PATH", os.path.join(d, "test.db"))
        yield


def test_idempotency_key_maps_to_same_run_id():
    from genie.storage.repositories import (
        create_run,
        get_run_id_by_idempotency_key,
        get_run_by_id,
    )
    run_id = "run-123"
    key = "idem-key-456"
    create_run(run_id, "conv-1", idempotency_key=key)
    found = get_run_id_by_idempotency_key(key)
    assert found == run_id
    run = get_run_by_id(run_id)
    assert run is not None
    assert run["id"] == run_id


def test_different_key_different_run():
    from genie.storage.repositories import create_run, get_run_id_by_idempotency_key
    create_run("run-a", "conv-1", idempotency_key="key-a")
    create_run("run-b", "conv-1", idempotency_key="key-b")
    assert get_run_id_by_idempotency_key("key-a") == "run-a"
    assert get_run_id_by_idempotency_key("key-b") == "run-b"
