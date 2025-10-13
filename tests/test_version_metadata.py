"""Basic metadata exposure tests for the package."""

import importlib
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator


@contextmanager
def _real_package() -> Iterator[SimpleNamespace]:
    saved: dict[str, object] = {}
    project_root = str(Path(__file__).resolve().parents[1])
    path_added = False
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        path_added = True
    to_remove = [
        name
        for name in list(sys.modules)
        if name == "yam_processor" or name.startswith("yam_processor.")
    ]
    for name in to_remove:
        saved[name] = sys.modules.pop(name)
    try:
        module = importlib.import_module("yam_processor")
        yield SimpleNamespace(
            __version__=module.__version__,
            get_version=module.get_version,
        )
    finally:
        for name in list(sys.modules):
            if name == "yam_processor" or name.startswith("yam_processor."):
                sys.modules.pop(name)
        sys.modules.update(saved)
        if path_added:
            sys.path.remove(project_root)


def test_version_constant_matches_helper() -> None:
    with _real_package() as module:
        version = module.get_version()
        assert isinstance(module.__version__, str)
        assert isinstance(version, str)
        assert version == module.__version__
        assert version  # non-empty string

