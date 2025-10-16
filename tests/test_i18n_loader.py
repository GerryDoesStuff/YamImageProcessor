"""Smoke tests for the translation bootstrap helpers."""
from __future__ import annotations

from core.i18n import TranslationConfig, TranslationLoader


def test_translation_loader_installs_packaged_catalogues(qapp) -> None:
    """The loader should discover the packaged Qt translations by default."""

    loader = TranslationLoader(qapp, TranslationConfig())
    loaded_catalogues = loader.install()

    assert loaded_catalogues, "expected at least one translation catalogue to load"
    assert all(path.suffix == ".qm" for path in loaded_catalogues)

    # Clean up to avoid bleeding translators into other tests
    loader.remove()
