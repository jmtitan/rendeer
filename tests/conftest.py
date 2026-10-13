"""Tests never write caches, figures or temporary data into the checkout."""

import os
from contextlib import nullcontext
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_config():
    import rendeer as rd

    with getattr(rd, "context", nullcontext)():
        yield


@pytest.fixture
def output_dir():
    path = Path(os.environ["RENDEER_TEST_OUTPUT"]).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path
