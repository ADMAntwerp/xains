"""Shared pytest-recording (VCR) config for integration tests.

Scrubs Anthropic auth + version headers from recorded cassettes so that the
committed YAML carries no secrets. ``record_mode="once"`` means the cassette
is recorded on first run if missing, and thereafter replayed strictly — any
new, unrecorded request fails the test rather than silently hitting the API.
"""

import pytest


@pytest.fixture
def vcr_config() -> dict[str, object]:
    return {
        "filter_headers": ["authorization", "x-api-key", "anthropic-version"],
        "record_mode": "once",
    }
