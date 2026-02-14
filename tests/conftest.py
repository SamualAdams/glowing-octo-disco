from __future__ import annotations

import os
from pathlib import Path
import subprocess
import time
from typing import Iterator

import psycopg
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
STARTUP_TIMEOUT_SECONDS = 45
RETRY_INTERVAL_SECONDS = 1


def _database_url() -> str:
    pg_host = os.getenv("PGHOST", "127.0.0.1")
    pg_port = os.getenv("PGPORT", "5442")
    pg_database = os.getenv("PGDATABASE", "app")
    pg_user = os.getenv("PGUSER", "app")
    pg_password = os.getenv("PGPASSWORD", "app")
    return os.getenv(
        "DATABASE_URL",
        f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_database}",
    )


def _run_docker_compose(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    command = ["docker", "compose", *args]
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        pytest.fail("Docker CLI is not available. Install Docker Desktop and retry.")

    if check and completed.returncode != 0:
        combined_output = "\n".join(
            part for part in [completed.stdout.strip(), completed.stderr.strip()] if part
        )
        daemon_hint = ""
        lowered = combined_output.lower()
        if "cannot connect to the docker daemon" in lowered or "is the docker daemon running" in lowered:
            daemon_hint = "\nStart Docker Desktop (or your Docker daemon) and retry."
        pytest.fail(f"`{' '.join(command)}` failed.\n{combined_output}{daemon_hint}")

    return completed


def _wait_for_database(database_url: str) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        try:
            with psycopg.connect(database_url, connect_timeout=2) as connection:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
            return
        except psycopg.Error as error:
            last_error = error
            time.sleep(RETRY_INTERVAL_SECONDS)

    message = f"Postgres was not reachable within {STARTUP_TIMEOUT_SECONDS}s."
    if last_error:
        message += f" Last error: {last_error}"
    raise TimeoutError(message)


@pytest.fixture(scope="session")
def postgres_database_url() -> Iterator[str]:
    database_url = _database_url()
    _run_docker_compose(["up", "-d", "postgres"])

    try:
        _wait_for_database(database_url)
    except TimeoutError as error:
        logs = _run_docker_compose(["logs", "postgres"], check=False)
        _run_docker_compose(["down", "-v", "--remove-orphans"], check=False)
        pytest.fail(f"{error}\nDocker logs:\n{logs.stdout}\n{logs.stderr}")

    yield database_url

    _run_docker_compose(["down", "-v", "--remove-orphans"], check=False)


@pytest.fixture()
def postgres_connection(postgres_database_url: str) -> Iterator[psycopg.Connection]:
    with psycopg.connect(postgres_database_url, autocommit=True) as connection:
        yield connection
