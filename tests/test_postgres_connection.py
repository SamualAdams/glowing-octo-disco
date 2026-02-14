from __future__ import annotations

from uuid import uuid4

import psycopg


def test_postgres_connectivity_smoke(postgres_connection: psycopg.Connection) -> None:
    with postgres_connection.cursor() as cursor:
        cursor.execute("SELECT 1")
        assert cursor.fetchone() == (1,)


def test_postgres_write_read_roundtrip(postgres_connection: psycopg.Connection) -> None:
    payload = f"payload-{uuid4()}"

    with postgres_connection.cursor() as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS db_connection_test (
                id BIGSERIAL PRIMARY KEY,
                payload TEXT NOT NULL
            )
            """
        )
        cursor.execute("INSERT INTO db_connection_test (payload) VALUES (%s)", (payload,))
        cursor.execute("SELECT payload FROM db_connection_test WHERE payload = %s", (payload,))
        row = cursor.fetchone()

    assert row == (payload,)
