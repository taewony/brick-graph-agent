"""Shared pytest fixtures (workspace-local tmp dir — pytest's default
tmp_path lives outside the sandbox). Each test gets a unique subdir so a
stale dir can never leak events between tests."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest

WS_TMP_ROOT = Path(__file__).resolve().parent.parent / ".runtime-tests"


@pytest.fixture
def ws_tmp():
    d = WS_TMP_ROOT / uuid.uuid4().hex[:8]
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)
