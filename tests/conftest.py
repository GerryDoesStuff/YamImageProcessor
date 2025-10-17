from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


# Ensure the project root is on ``sys.path`` so tests can import ``yam_processor``
# without requiring the package to be installed in the environment.  This mirrors
# how the application is executed locally while keeping the tests self-contained.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    # ``insert`` keeps the repo ahead of any site-packages entry so that the
    # in-tree modules are exercised.
    sys.path.insert(0, str(_PROJECT_ROOT))


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-performance",
        action="store_true",
        default=False,
        help="run tests marked as performance",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--run-performance"):
        return
    skip_marker = pytest.mark.skip(reason="need --run-performance option to run")
    for item in items:
        if "performance" in item.keywords:
            item.add_marker(skip_marker)
