from __future__ import annotations
import pytest


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
